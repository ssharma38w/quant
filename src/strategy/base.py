from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TradeSetup:
    strategy: str
    entry_date: str
    expiry_date: str
    spot_entry: float
    vix_entry: float
    regime: str
    regime_confidence: float

    # Legs: (strike, option_type, action, premium)
    legs: list = field(default_factory=list)

    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_lower: float = 0.0
    breakeven_upper: float = 0.0

    # Filled at exit
    exit_date: Optional[str] = None
    spot_exit: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    outcome: Optional[str] = None  # "profit" | "loss" | "max_profit"

    lots: int = 1
    lot_size: int = 75
    margin_required: float = 0.0

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "entry_date": self.entry_date,
            "expiry_date": self.expiry_date,
            "spot": self.spot_entry,
            "vix": self.vix_entry,
            "regime": self.regime,
            "confidence": self.regime_confidence,
            "max_profit": round(self.max_profit, 2),
            "max_loss": round(self.max_loss, 2),
            "be_lower": round(self.breakeven_lower, 2),
            "be_upper": round(self.breakeven_upper, 2),
            "legs": self.legs,
        }
