"""Real bankroll derived from bookmaker wallets (Decimal-safe)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from betting_app.core.db import query_df
from betting_app.services.wallet_service import accounts

DEFAULT_BANKROLL_CURRENCY = "PLN"


@dataclass(frozen=True)
class WalletBankroll:
    """Wallet cash plus the stakes currently locked in open bets.

    ``place_wallet_bet`` debits the wallet when a bet is placed, so ``cash`` already
    excludes open stakes. Kelly sizing takes the total nominal bankroll and subtracts
    reserved (open) stakes itself, hence ``total = cash + open_stakes`` is what
    callers pass as ``bankroll``.
    """

    cash: Decimal
    open_stakes: Decimal

    @property
    def total(self) -> Decimal:
        return self.cash + self.open_stakes


def wallet_bankroll(currency: str = DEFAULT_BANKROLL_CURRENCY) -> WalletBankroll | None:
    """Return the real wallet bankroll, or None when no active wallet exists.

    Callers must treat None as "no stake suggestion"; never substitute a default.
    """
    wallets = accounts(active_only=True)
    if wallets.empty:
        return None
    wallets = wallets[wallets["currency"].fillna(DEFAULT_BANKROLL_CURRENCY) == currency]
    if wallets.empty:
        return None
    cash = sum((Decimal(str(value)) for value in wallets["current_balance"]), Decimal("0"))
    open_df = query_df("SELECT COALESCE(SUM(stake), 0) AS open_stake FROM bets WHERE status = 'open'")
    open_stakes = Decimal(str(open_df.iloc[0]["open_stake"])) if not open_df.empty else Decimal("0")
    if not cash.is_finite() or cash < 0 or not open_stakes.is_finite() or open_stakes < 0:
        return None
    return WalletBankroll(cash=cash, open_stakes=open_stakes)
