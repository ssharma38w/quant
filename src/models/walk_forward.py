"""
Walk-forward validation for the ML strategy predictor.

Two modes (both available, no overlap):

1. EXPANDING window (--ml):
   - Year 1-2: warmup, no predictions
   - Year 3: train 2019-2020 → predict all of 2021
   - Year 4: train 2019-2021 → predict all of 2022
   - Best for: stable long-term patterns

2. ROLLING window (--ml --rolling, default 12 weeks):
   - Week 13: train weeks 1-12 → predict week 13
   - Week 14: train weeks 2-13 → predict week 14
   - Uses only RECENT data — adapts faster to regime changes
   - Best for: recent market conditions dominate
   - Simpler model (depth-3 GBM) since training set is small

Both produce out-of-sample predictions with no lookahead bias.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .feature_engineering import build_weekly_features
from .strategy_predictor import StrategyPredictor, STRATEGY_MAP


# ─────────────────────────────────────────────
# EXPANDING WINDOW (existing, yearly batches)
# ─────────────────────────────────────────────

def build_walk_forward_predictions(
    df: pd.DataFrame,
    regime_probs: np.ndarray,
    trades: list,
    warmup_years: int = 2,
    retrain_freq: str = "yearly",
    verbose: bool = True,
) -> pd.Series:
    """
    Expanding window walk-forward (yearly or quarterly retraining).
    Train on all past data before each period, predict that period.
    """
    features = build_weekly_features(df, regime_probs)
    if features.empty:
        return pd.Series(dtype=str)

    trade_labels = _build_trade_labels(trades)
    start_year = features.index.year.min()
    end_year   = features.index.year.max()
    predict_start = start_year + warmup_years

    periods = list(range(predict_start, end_year + 1)) if retrain_freq == "yearly" else [
        (y, q) for y in range(predict_start, end_year + 1) for q in range(1, 5)
    ]

    all_predictions = {}
    feature_importance_by_period = {}

    for period in periods:
        if isinstance(period, int):
            year = period
            train_mask = features.index.year < year
            test_mask  = features.index.year == year
        else:
            year, quarter = period
            m_start = (quarter - 1) * 3 + 1
            m_end   = quarter * 3
            train_mask = (features.index.year < year) | \
                         ((features.index.year == year) & (features.index.month < m_start))
            test_mask  = (features.index.year == year) & \
                         (features.index.month.isin(range(m_start, m_end + 1)))

        train_feat = features[train_mask]
        test_feat  = features[test_mask]

        if len(train_feat) < 40 or len(test_feat) == 0:
            continue

        train_df, y_train = _align_features_labels(train_feat, trade_labels)
        if len(train_df) < 20:
            continue

        predictor = StrategyPredictor(use_xgboost=True)
        predictor.fit(train_df, _mock_trades(y_train, train_df.index), verbose=False)
        if not predictor.fitted:
            continue

        for date, pred in predictor.predict(test_feat).items():
            all_predictions[date] = pred

        imp = predictor.feature_importance(top_n=5)
        if not imp.empty:
            feature_importance_by_period[str(period)] = imp.to_dict()

    result = pd.Series(all_predictions, name="ml_prediction")
    result.index = pd.to_datetime(result.index)

    if verbose and len(result) > 0:
        print(f"\n  Walk-forward ML predictions (expanding): {len(result)} weeks")
        print(f"  Distribution: {result.value_counts().to_dict()}")
        if feature_importance_by_period:
            last = list(feature_importance_by_period.keys())[-1]
            print(f"  Top features ({last}): {list(feature_importance_by_period[last].keys())[:3]}")

    return result


# ─────────────────────────────────────────────
# ROLLING WINDOW (new, week-by-week)
# ─────────────────────────────────────────────

def build_rolling_predictions(
    df: pd.DataFrame,
    regime_probs: np.ndarray,
    trades: list,
    window_weeks: int = 12,
    min_train_samples: int = 8,
    verbose: bool = True,
) -> pd.Series:
    """
    Rolling window walk-forward: train on last `window_weeks`, predict next week.

    Week-by-week loop:
      For each week t (starting at t = window_weeks + 1):
        - train on weeks [t - window_weeks, t - 1]
        - predict week t

    Uses a lightweight model (depth-3 GBM) suited for small training sets.
    Falls back to None (→ rule-based) if not enough labeled samples.
    """
    features = build_weekly_features(df, regime_probs)
    if features.empty:
        return pd.Series(dtype=str)

    trade_labels = _build_trade_labels(trades)
    all_dates = features.index.tolist()
    all_predictions = {}
    retrain_count = 0

    for i in range(window_weeks, len(all_dates)):
        predict_date = all_dates[i]
        window_dates = all_dates[i - window_weeks: i]

        window_feat = features.loc[window_dates]
        train_df, y_train = _align_features_labels(window_feat, trade_labels)

        if len(train_df) < min_train_samples:
            continue  # not enough data in this window → rule-based handles it

        pred = _rolling_predict_single(train_df, y_train, features.loc[[predict_date]])
        if pred is not None:
            all_predictions[predict_date] = pred
            retrain_count += 1

    result = pd.Series(all_predictions, name="ml_rolling_prediction")
    result.index = pd.to_datetime(result.index)

    if verbose and len(result) > 0:
        print(f"\n  Rolling ML predictions ({window_weeks}w window): {len(result)} weeks")
        print(f"  Distribution: {result.value_counts().to_dict()}")
        print(f"  Retrained every week ({retrain_count} models fit total)")

    return result


def _rolling_predict_single(
    train_df: pd.DataFrame,
    y_train: list,
    test_df: pd.DataFrame,
) -> str | None:
    """Fit a lightweight model on small training window and predict one week."""
    if len(set(y_train)) < 2:
        # Only one class → predict that class directly
        label = y_train[0]
        return STRATEGY_MAP.get(label, "iron_condor")

    try:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df.values)
        X_te = scaler.transform(test_df.values)

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=np.array(y_train))
        w_map = dict(zip(classes, weights))
        sample_w = np.array([w_map[y] for y in y_train])

        # Lightweight model: depth-3 GBM, few estimators — fast on 12 samples
        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        model.fit(X_tr, y_train, sample_weight=sample_w)
        pred_label = int(model.predict(X_te)[0])
        return STRATEGY_MAP.get(pred_label, "iron_condor")
    except Exception:
        return None


# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

def _build_trade_labels(trades: list) -> dict:
    """Map entry date str → {strategy, pnl} keeping best P&L per week."""
    labels = {}
    for t in trades:
        if t.pnl is None:
            continue
        pnl_per_lot = t.pnl / max(t.lots, 1)
        key = str(t.entry_date)
        if key not in labels or pnl_per_lot > labels[key]["pnl"]:
            labels[key] = {"strategy": t.strategy, "pnl": pnl_per_lot}
    return labels


def _align_features_labels(feat: pd.DataFrame, trade_labels: dict):
    """Return (DataFrame with DatetimeIndex, y list) aligned to trade labels."""
    X, y, dates = [], [], []
    for date in feat.index:
        key = str(date.date())
        if key in trade_labels:
            info = trade_labels[key]
            X.append(feat.loc[date].values)
            y.append(_strat_to_label(info["strategy"], info["pnl"]))
            dates.append(date)
    df_out = pd.DataFrame(X, index=dates, columns=feat.columns) if X else pd.DataFrame()
    return df_out, y


def _strat_to_label(strategy: str, pnl: float) -> int:
    if pnl <= 0:
        return 0
    return {"iron_condor": 1, "short_straddle": 2, "calendar_spread": 3}.get(strategy, 1)


def _mock_trades(labels: list, dates) -> list:
    from ..strategy.base import TradeSetup
    label_to_strat = {0: "skip", 1: "iron_condor", 2: "short_straddle", 3: "calendar_spread"}
    trades = []
    for label, date in zip(labels, dates):
        t = TradeSetup(
            strategy=label_to_strat.get(label, "iron_condor"),
            entry_date=str(date.date()), expiry_date=str(date.date()),
            spot_entry=22000, vix_entry=15,
            regime="normal", regime_confidence=0.7,
        )
        t.pnl = 1000 if label > 0 else -500
        t.lots = 1
        trades.append(t)
    return trades
