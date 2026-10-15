"""
Hold-out vault.

* Each symbol has one ``research_end`` date, set once and never moved. A symbol with
  no date cannot start any run, so a missing date fails closed.
* A research run may not request or receive bars after ``research_end``. Requests
  past it are refused at submit time and again when the runner loads data, and a
  loaded (possibly cached) frame that extends past it is refused too.
* A hold-out run needs a pre-registered frozen config hash. The first hold-out load
  per family is the one allowed access; a second marks the family contaminated
  permanently.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from .ledger import Ledger, get_ledger, utcnow


class VaultError(PermissionError):
    pass


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def get_research_end(symbol: str, ledger: Optional[Ledger] = None) -> Optional[date]:
    ledger = ledger or get_ledger()
    row = ledger.one("SELECT research_end FROM research_end WHERE symbol = ?", (symbol,))
    return _as_date(row["research_end"]) if row else None


def set_research_end(symbol: str, research_end: Any, note: str = "",
                     ledger: Optional[Ledger] = None) -> date:
    """Set a symbol's hold-out boundary. Setting it again to a different date fails."""
    ledger = ledger or get_ledger()
    new = _as_date(research_end)
    current = get_research_end(symbol, ledger)
    if current is not None:
        if current != new:
            raise VaultError(
                f"research_end for {symbol} is {current} and is never moved (requested {new})"
            )
        return current
    latest = ledger.one(
        "SELECT MAX(end) AS latest FROM data_access WHERE symbol = ?", (symbol,)
    )
    if latest and latest["latest"] and _as_date(latest["latest"]) > new:
        raise VaultError(
            f"research_end {new} for {symbol} is before the latest bar already examined "
            f"({latest['latest'][:10]}); it must fall after it"
        )
    with ledger.transaction() as con:
        con.execute(
            "INSERT INTO research_end(symbol, research_end, set_at, note) VALUES (?, ?, ?, ?)",
            (symbol, new.isoformat(), utcnow(), note),
        )
    return new


def all_research_ends(ledger: Optional[Ledger] = None) -> Dict[str, date]:
    ledger = ledger or get_ledger()
    return {r["symbol"]: _as_date(r["research_end"])
            for r in ledger.query("SELECT * FROM research_end ORDER BY symbol")}


def preregister(family_id: str, config_hash: str, payload: Optional[Dict] = None,
                ledger: Optional[Ledger] = None) -> None:
    ledger = ledger or get_ledger()
    with ledger.transaction() as con:
        con.execute(
            "INSERT INTO preregistrations(family_id, config_hash, payload_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (family_id, config_hash, json.dumps(payload or {}, default=str), utcnow()),
        )
        con.execute(
            "UPDATE families SET frozen_config_hash = ? WHERE family_id = ?",
            (config_hash, family_id),
        )


def is_preregistered(family_id: str, config_hash: str, ledger: Optional[Ledger] = None) -> bool:
    ledger = ledger or get_ledger()
    return ledger.one(
        "SELECT 1 FROM preregistrations WHERE family_id = ? AND config_hash = ?",
        (family_id, config_hash),
    ) is not None


def check_request(symbol: str, start: Any, end: Any, purpose: str, *, family_id: str,
                  config_hash: str, ledger: Optional[Ledger] = None) -> date:
    """
    Validate a data request before any bar is loaded. Returns the symbol's
    ``research_end``. Raises ``VaultError`` when the request is not allowed.
    """
    ledger = ledger or get_ledger()
    boundary = get_research_end(symbol, ledger)
    if boundary is None:
        raise VaultError(
            f"No research_end set for {symbol}. Set one before any run on this symbol "
            f"(python -m afml.studio set-research-end {symbol} YYYY-MM-DD)."
        )
    if purpose == "holdout":
        if not is_preregistered(family_id, config_hash, ledger):
            raise VaultError(
                f"Hold-out run refused: config {config_hash[:12]} is not pre-registered "
                f"for family {family_id}"
            )
        return boundary
    if _as_date(end) > boundary:
        raise VaultError(
            f"{symbol} research window ends {boundary}; requested end {_as_date(end)} "
            f"reaches into the hold-out"
        )
    if _as_date(start) > boundary:
        raise VaultError(f"{symbol} start {_as_date(start)} is after research_end {boundary}")
    return boundary


def record_holdout_access(family_id: str, run_id: str, ledger: Optional[Ledger] = None) -> bool:
    """
    Log a hold-out load. Returns True when this was the family's first access; a
    later access marks the family contaminated permanently.
    """
    ledger = ledger or get_ledger()
    with ledger.transaction() as con:
        prior = con.execute(
            "SELECT COUNT(*) AS n FROM holdout_access WHERE family_id = ?", (family_id,)
        ).fetchone()["n"]
        con.execute(
            "INSERT INTO holdout_access(family_id, run_id, at) VALUES (?, ?, ?)",
            (family_id, run_id, utcnow()),
        )
    if prior:
        ledger.mark_contaminated(family_id, f"hold-out accessed again by run {run_id}")
        return False
    return True


def check_frame(df: pd.DataFrame, symbol: str, purpose: str,
                ledger: Optional[Ledger] = None) -> None:
    """Refuse a loaded frame (fresh or cached) whose index passes ``research_end``."""
    if purpose == "holdout" or df.empty:
        return
    boundary = get_research_end(symbol, ledger)
    if boundary is None:
        raise VaultError(f"No research_end set for {symbol}")
    limit = pd.Timestamp(boundary + timedelta(days=1))
    last = pd.Timestamp(df.index[-1])
    if last.tzinfo is not None:
        limit = limit.tz_localize(last.tzinfo)
    if last >= limit:
        raise VaultError(
            f"Loaded bars for {symbol} end {last}, past research_end {boundary}; "
            f"refusing (possibly a stale cache entry)"
        )
