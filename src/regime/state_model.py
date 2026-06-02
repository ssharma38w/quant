import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


REGIME_LABELS = {0: "low_vol", 1: "normal", 2: "high_vol"}


class RegimeModel:
    """
    Hidden Markov Model to classify weekly market regimes.
    States: 0=low_vol, 1=normal, 2=high_vol
    Features: VIX level, HV20, IV/HV ratio, 5d Nifty return, 5d VIX change
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200, random_state: int = 42):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self._state_map = {}

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        cols = ["IV", "HV20", "IV_HV_ratio", "nifty_return_5d", "vix_change_5d"]
        return df[cols].values

    def fit(self, df: pd.DataFrame) -> "RegimeModel":
        X = self._features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True
        self._build_state_map(df)
        return self

    def _build_state_map(self, df: pd.DataFrame):
        """Map HMM states to low/normal/high vol labels by mean VIX level."""
        X = self._features(df)
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        mean_vix = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                mean_vix[s] = df["VIX"].values[mask].mean()
            else:
                mean_vix[s] = 0
        sorted_states = sorted(mean_vix, key=lambda s: mean_vix[s])
        labels = ["low_vol", "normal", "high_vol"]
        self._state_map = {s: labels[i] for i, s in enumerate(sorted_states)}

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._features(df)
        X_scaled = self.scaler.transform(X)
        raw_states = self.model.predict(X_scaled)
        regimes = [self._state_map.get(s, "normal") for s in raw_states]
        return pd.Series(regimes, index=df.index, name="regime")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        X = self._features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def current_regime(self, df: pd.DataFrame) -> dict:
        """Return regime + confidence for the most recent row."""
        regimes = self.predict(df)
        probs = self.predict_proba(df)
        last_regime = regimes.iloc[-1]
        last_probs = probs[-1]
        confidence = last_probs.max()
        return {
            "regime": last_regime,
            "confidence": round(confidence, 4),
            "state_probs": {self._state_map.get(i, i): round(p, 4) for i, p in enumerate(last_probs)},
        }
