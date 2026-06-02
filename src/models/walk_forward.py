"""
Walk-forward validation for the ML strategy predictor.

Eliminates lookahead bias by training only on PAST data before each prediction:
  - Year 1-2 (2019-2020): warm-up, no ML predictions
  - Year 3 (2021): train on 2019-2020, predict 2021
  - Year 4 (2022): train on 2019-2021, predict 2022
  - Year 5 (2023): train on 2019-2022, predict 2023
  - Year 6 (2024): train on 2019-2023, predict 2024
  - 2025: train on 2019-2024, predict 2025

This ensures the model NEVER sees future data during training.
"""
import pandas as pd
import numpy as np
from typing import Optional

from .feature_engineering import build_weekly_features
from .strategy_predictor import StrategyPredictor, STRATEGY_MAP


def build_walk_forward_predictions(
    df: pd.DataFrame,
    regime_probs: np.ndarray,
    trades: list,
    warmup_years: int = 2,
    retrain_freq: str = "yearly",  # "yearly" or "quarterly"
    verbose: bool = True,
) -> pd.Series:
    """
    Generate out-of-sample ML predictions via walk-forward expanding window.

    Returns a pd.Series indexed by entry date (Mondays) with predicted strategy name.
    Dates in the warmup period have NO prediction (rule-based selector used instead).
    """
    features = build_weekly_features(df, regime_probs)
    if features.empty:
        return pd.Series(dtype=str)

    start_year = features.index.year.min()
    end_year   = features.index.year.max()
    predict_start = start_year + warmup_years

    # Map trades to weekly labels for training
    trade_labels = {}
    for t in trades:
        if t.pnl is None:
            continue
        pnl_per_lot = t.pnl / max(t.lots, 1)
        entry = str(t.entry_date)
        if entry not in trade_labels or pnl_per_lot > trade_labels[entry]["pnl"]:
            trade_labels[entry] = {"strategy": t.strategy, "pnl": pnl_per_lot}

    all_predictions = {}
    feature_importance_by_year = {}

    if retrain_freq == "yearly":
        periods = list(range(predict_start, end_year + 1))
    else:  # quarterly
        periods = []
        for y in range(predict_start, end_year + 1):
            for q in range(1, 5):
                periods.append((y, q))

    for period in periods:
        if isinstance(period, int):
            year = period
            train_mask = features.index.year < year
            test_mask  = features.index.year == year
        else:
            year, quarter = period
            month_end = quarter * 3
            train_mask = (features.index.year < year) | \
                         ((features.index.year == year) & (features.index.month < (quarter - 1) * 3 + 1))
            test_mask  = (features.index.year == year) & \
                         (features.index.month >= (quarter - 1) * 3 + 1) & \
                         (features.index.month <= month_end)

        train_feat = features[train_mask]
        test_feat  = features[test_mask]

        if len(train_feat) < 40 or len(test_feat) == 0:
            continue

        # Build training labels
        X_train, y_train = [], []
        for date in train_feat.index:
            key = str(date.date())
            if key in trade_labels:
                label_info = trade_labels[key]
                label = _strat_to_label(label_info["strategy"], label_info["pnl"])
                X_train.append(train_feat.loc[date].values)
                y_train.append(label)

        if len(X_train) < 20:
            continue

        predictor = StrategyPredictor(use_xgboost=True)
        predictor.fit(
            pd.DataFrame(X_train, columns=train_feat.columns),
            _make_mock_trades(y_train, train_feat.index[:len(X_train)]),
            verbose=False,
        )

        if not predictor.fitted:
            continue

        preds = predictor.predict(test_feat)
        for date, pred in preds.items():
            all_predictions[date] = pred

        imp = predictor.feature_importance(top_n=5)
        if not imp.empty:
            feature_importance_by_year[str(period)] = imp.to_dict()

    result = pd.Series(all_predictions, name="ml_prediction")
    result.index = pd.to_datetime(result.index)

    if verbose and len(result) > 0:
        print(f"\n  Walk-forward ML predictions: {len(result)} weeks")
        print(f"  Distribution: {result.value_counts().to_dict()}")
        if feature_importance_by_year:
            last_year = list(feature_importance_by_year.keys())[-1]
            print(f"  Top features ({last_year}): {list(feature_importance_by_year[last_year].keys())[:3]}")

    return result


def _strat_to_label(strategy: str, pnl: float) -> int:
    if pnl <= 0:
        return 0  # skip
    mapping = {"iron_condor": 1, "short_straddle": 2, "calendar_spread": 3}
    return mapping.get(strategy, 1)


def _make_mock_trades(labels: list, dates) -> list:
    """Create minimal mock trade objects for StrategyPredictor.fit()."""
    from ..strategy.base import TradeSetup
    label_to_strat = {0: "skip", 1: "iron_condor", 2: "short_straddle", 3: "calendar_spread"}
    trades = []
    for i, (label, date) in enumerate(zip(labels, dates)):
        strat = label_to_strat.get(label, "iron_condor")
        t = TradeSetup(
            strategy=strat,
            entry_date=str(date.date()),
            expiry_date=str(date.date()),
            spot_entry=22000, vix_entry=15,
            regime="normal", regime_confidence=0.7,
        )
        t.pnl = 1000 if label > 0 else -500
        t.lots = 1
        trades.append(t)
    return trades
