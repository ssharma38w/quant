"""
Position sizing (scaler) agent.
Determines number of lots based on:
  - IV/HV premium (edge strength)
  - Regime confidence
  - Current drawdown
  - Kelly criterion
"""
import numpy as np


LOT_SIZE = 75
DEFAULT_CAPITAL = 1_000_000  # 10 Lakh INR
MAX_LOTS = 10
MIN_LOTS = 1


class ScalerAgent:
    """
    Rule-based position scaler. Can be replaced with RL agent later.
    Outputs number of lots to trade given current conditions.
    """

    def __init__(self, capital: float = DEFAULT_CAPITAL, risk_per_trade: float = 0.02):
        self.capital = capital
        self.risk_per_trade = risk_per_trade  # Max 2% of capital at risk per trade
        self.peak_capital = capital
        self.current_drawdown = 0.0

    def update_capital(self, new_capital: float):
        self.capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital

    def _edge_multiplier(self, iv_hv_ratio: float) -> float:
        """Scale size with IV richness. Cap at 2x for very rich IV."""
        if iv_hv_ratio < 1.0:
            return 0.0
        elif iv_hv_ratio < 1.2:
            return 0.5
        elif iv_hv_ratio < 1.5:
            return 1.0
        elif iv_hv_ratio < 2.0:
            return 1.5
        else:
            return 2.0

    def _drawdown_multiplier(self) -> float:
        """Reduce size in drawdown."""
        if self.current_drawdown < 0.05:
            return 1.0
        elif self.current_drawdown < 0.10:
            return 0.75
        elif self.current_drawdown < 0.15:
            return 0.5
        else:
            return 0.25

    def _confidence_multiplier(self, confidence: float) -> float:
        if confidence >= 0.80:
            return 1.0
        elif confidence >= 0.65:
            return 0.75
        else:
            return 0.5

    def size(
        self,
        max_loss_per_lot: float,
        iv_hv_ratio: float,
        regime_confidence: float,
        regime: str = "normal",
    ) -> int:
        """
        Returns number of lots to trade.
        max_loss_per_lot: worst-case loss for 1 lot (in INR)
        """
        if max_loss_per_lot <= 0 or iv_hv_ratio < 1.0:
            return 0

        risk_amount = self.capital * self.risk_per_trade
        base_lots = int(risk_amount / (max_loss_per_lot * LOT_SIZE))
        base_lots = max(MIN_LOTS, min(base_lots, MAX_LOTS))

        scale = (
            self._edge_multiplier(iv_hv_ratio)
            * self._drawdown_multiplier()
            * self._confidence_multiplier(regime_confidence)
        )

        # High vol regime: never go above 50% of base
        if regime == "high_vol":
            scale *= 0.5

        lots = max(MIN_LOTS, round(base_lots * scale))
        return min(lots, MAX_LOTS)

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly criterion for optional reference."""
        if avg_loss == 0:
            return 0
        b = avg_win / avg_loss
        kelly = (b * win_rate - (1 - win_rate)) / b
        return max(0, min(kelly, 0.25))  # cap at 25% Kelly
