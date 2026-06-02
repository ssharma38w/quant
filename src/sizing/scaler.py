"""
Position sizing (scaler) agent.
Determines number of lots based on:
  - Effective max loss per lot (stop-loss-aware)
  - IV/HV premium (edge strength)
  - Regime confidence
  - Current drawdown
"""
import numpy as np


LOT_SIZE = 75
DEFAULT_CAPITAL = 1_000_000
MAX_LOTS = 15
MIN_LOTS = 1


class ScalerAgent:
    def __init__(self, capital: float = DEFAULT_CAPITAL, risk_per_trade: float = 0.06):
        self.capital = capital
        self.risk_per_trade = risk_per_trade  # 6% default — calibrated to real backtest
        self.peak_capital = capital
        self.current_drawdown = 0.0

    def update_capital(self, new_capital: float):
        self.capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital

    def _edge_multiplier(self, iv_hv_ratio: float) -> float:
        """Scale up when IV is meaningfully rich vs realized vol."""
        if iv_hv_ratio < 1.0:
            return 0.0
        elif iv_hv_ratio < 1.15:
            return 0.75
        elif iv_hv_ratio < 1.35:
            return 1.0
        elif iv_hv_ratio < 1.6:
            return 1.25
        else:
            return 1.5

    def _drawdown_multiplier(self) -> float:
        """Reduce size proportionally with drawdown depth."""
        if self.current_drawdown < 0.05:
            return 1.0
        elif self.current_drawdown < 0.10:
            return 0.75
        elif self.current_drawdown < 0.18:
            return 0.50
        else:
            return 0.25

    def _confidence_multiplier(self, confidence: float) -> float:
        if confidence >= 0.80:
            return 1.0
        elif confidence >= 0.65:
            return 0.85
        else:
            return 0.65

    def size(
        self,
        max_loss_per_lot: float,
        iv_hv_ratio: float,
        regime_confidence: float,
        regime: str = "normal",
    ) -> int:
        """
        Returns number of lots to trade.
        max_loss_per_lot: effective worst-case loss in points (stop-loss-aware)
        """
        if max_loss_per_lot <= 0 or iv_hv_ratio < 1.0:
            return 0

        risk_amount = self.capital * self.risk_per_trade
        base_lots = risk_amount / (max_loss_per_lot * LOT_SIZE)
        base_lots = max(MIN_LOTS, min(base_lots, MAX_LOTS))

        scale = (
            self._edge_multiplier(iv_hv_ratio)
            * self._drawdown_multiplier()
            * self._confidence_multiplier(regime_confidence)
        )

        lots = max(MIN_LOTS, round(base_lots * scale))
        return min(lots, MAX_LOTS)

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 0
        b = avg_win / abs(avg_loss)
        kelly = (b * win_rate - (1 - win_rate)) / b
        return max(0, min(kelly, 0.25))
