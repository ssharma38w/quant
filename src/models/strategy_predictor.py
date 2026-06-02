"""
ML Strategy Predictor — XGBoost classifier trained on historical weekly data.

For each week it predicts:
  - Whether to trade (skip vs trade)
  - Which strategy will be most profitable

Training approach: walk-forward validation
  - Train on first N years, predict year N+1
  - Expand window each year (no lookahead bias)

Labels (from backtested P&L):
  0 = skip    (best action: don't trade — all strategies lose)
  1 = iron_condor
  2 = short_straddle
  3 = calendar_spread (future extension)

Features: VIX, IV/HV, momentum, regime probs, vol spread, etc.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .feature_engineering import build_weekly_features, FEATURE_NAMES

STRATEGY_MAP = {
    0: "skip",
    1: "iron_condor",
    2: "short_straddle",
    3: "calendar_spread",
}
STRATEGY_MAP_INV = {v: k for k, v in STRATEGY_MAP.items()}

MODEL_PATH = Path("models/strategy_predictor.pkl")


class StrategyPredictor:
    """
    Predicts the best options strategy for each week.
    Trained on historical (features → actual best strategy) pairs.
    """

    def __init__(self, use_xgboost: bool = True):
        self.fitted = False
        self.scaler = StandardScaler()

        if use_xgboost and XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            )
            self.model_type = "xgboost"
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            self.model_type = "gbm"

    def _label_weeks(self, trades: list) -> dict:
        """
        Convert backtest trade results to weekly labels.
        For each week: which strategy produced the best P&L per lot?
        label = that strategy's class ID.
        If all strategies lose (or skipped): label = 0 (skip).
        """
        labels = {}
        for t in trades:
            if t.pnl is None:
                continue
            pnl_per_lot = t.pnl / max(t.lots, 1)
            entry = str(t.entry_date)
            if entry not in labels or pnl_per_lot > labels[entry]["pnl"]:
                labels[entry] = {"strategy": t.strategy, "pnl": pnl_per_lot}

        result = {}
        for date, info in labels.items():
            strat = info["strategy"]
            pnl = info["pnl"]
            # If the best strategy still lost money, label as skip
            label = STRATEGY_MAP_INV.get(strat, 1) if pnl > 0 else 0
            result[date] = label
        return result

    def build_training_data(
        self,
        features: pd.DataFrame,
        trades: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align features with trade labels."""
        labels_dict = self._label_weeks(trades)
        X, y = [], []
        for date in features.index:
            date_str = str(date.date())
            if date_str in labels_dict:
                X.append(features.loc[date].values)
                y.append(labels_dict[date_str])
        return np.array(X), np.array(y)

    def fit(self, features: pd.DataFrame, trades: list, verbose: bool = True) -> "StrategyPredictor":
        X, y = self.build_training_data(features, trades)
        if len(X) < 30:
            print(f"  [predictor] Not enough samples ({len(X)}) to train.")
            return self

        X_scaled = self.scaler.fit_transform(X)

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        sample_weights = np.array([weights[np.where(classes == yi)[0][0]] for yi in y])

        if self.model_type == "xgboost":
            self.model.fit(X_scaled, y, sample_weight=sample_weights)
        else:
            self.model.fit(X_scaled, y, sample_weight=sample_weights)

        self.fitted = True

        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"\n  Strategy Predictor trained ({self.model_type}) on {len(X)} weeks")
            print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
            print(f"  In-sample accuracy: {(y_pred == y).mean():.1%}")

        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return predicted strategy name for each week in features."""
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        X = self.scaler.transform(features.values)
        preds = self.model.predict(X)
        return pd.Series(
            [STRATEGY_MAP[p] for p in preds],
            index=features.index,
            name="predicted_strategy",
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities."""
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        X = self.scaler.transform(features.values)
        probs = self.model.predict_proba(X)
        classes = self.model.classes_
        cols = [STRATEGY_MAP.get(c, str(c)) for c in classes]
        return pd.DataFrame(probs, index=features.index, columns=cols)

    def feature_importance(self, top_n: int = 10) -> pd.Series:
        if not self.fitted:
            return pd.Series()
        imp = self.model.feature_importances_
        return pd.Series(imp, index=FEATURE_NAMES).nlargest(top_n)

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "StrategyPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)


def walk_forward_train(
    df: pd.DataFrame,
    regime_probs: np.ndarray,
    trades_by_year: dict,
    test_start_year: int = 2022,
) -> tuple["StrategyPredictor", dict]:
    """
    Walk-forward training: train on all data before test_start_year,
    evaluate on test_start_year onwards.
    Returns trained predictor + evaluation report.
    """
    features = build_weekly_features(df, regime_probs)
    train_mask = features.index.year < test_start_year
    test_mask  = features.index.year >= test_start_year

    train_features = features[train_mask]
    test_features  = features[test_mask]

    # Collect all trades for training period
    all_trades = []
    for year, trades in trades_by_year.items():
        if year < test_start_year:
            all_trades.extend(trades)

    predictor = StrategyPredictor()
    predictor.fit(train_features, all_trades)

    report = {}
    if test_features.shape[0] > 0 and predictor.fitted:
        preds = predictor.predict(test_features)
        report["test_predictions"] = preds.value_counts().to_dict()
        imp = predictor.feature_importance()
        if not imp.empty:
            report["top_features"] = imp.to_dict()

    return predictor, report
