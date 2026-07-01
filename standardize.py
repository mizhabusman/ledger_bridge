"""
LedgerBridge AI — Standardization & Cleaning Engine

Converts a raw DataFrame + confirmed mapping into the canonical 9-field schema.

Gross Amount is COMPUTED here (not directly mapped), uniformly for every
ledger, with no buyer/seller distinction:

    Gross Amount = Debit - Credit

Because the two parties in a reconciliation are double-entry counterparties,
the SAME real-world transaction lands with OPPOSITE signs on the two ledgers
under this uniform rule (one party's receivable is the other's payable) —
this is expected and is handled by reconcile.py's mirror-sign matching
(see the "signed sum ~ 0" predicates there), not by flipping a sign here.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import parser as date_parser

from config import CANONICAL_FIELDS

_BALANCE_WORDS = [
    'opening balance', 'closing balance', 'grand total',
    'account closed', 'brought forward', 'carried forward',
    'total', 'totals', 'sub total', 'subtotal',
]

_ISO_DATE_RE = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}")


# ─────────────────────────── cleaning primitives ─────────────────────────────

def clean_amount(value) -> float:
    """Strip currency symbols, commas, parens-as-negative. Return 0 for blanks."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "-", "—"):
        return 0.0

    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]

    s = re.sub(r"[₹$€£¥,\s?]", "", s)

    if s.startswith("-"):
        negative = True
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]

    if not s:
        return 0.0

    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return 0.0


def clean_date(value) -> pd.Timestamp | None:
    """ISO (YYYY-MM-DD) auto-detected; otherwise dayfirst (Indian DD-MM-YYYY)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).normalize()

    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return pd.NaT

    s = s.split(" ")[0] if " " in s else s
    dayfirst = not bool(_ISO_DATE_RE.match(s))

    try:
        dt = date_parser.parse(s, dayfirst=dayfirst, fuzzy=True)
        return pd.Timestamp(dt).normalize()
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def clean_text(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return re.sub(r"\s+", " ", s)


def clean_voucher(value) -> str:
    """Uppercase + strip separators for matching: JV-001 → JV001."""
    if value is None:
        return ""
    s = str(value).strip().upper()
    if s.lower() in ("nan", "none", "-"):
        return ""
    return re.sub(r"[\s\-_/.]", "", s)


def _is_balance_row(row: pd.Series) -> bool:
    for v in row:
        s = str(v).strip().lower()
        if any(bw in s for bw in _BALANCE_WORDS):
            return True
    return False


# ─────────────────────────── canonical conversion ────────────────────────────

def standardize(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Convert a raw DataFrame to the canonical 9-field schema.

    Args:
        df:      Raw ledger DataFrame.
        mapping: Dict of canonical field -> {source, confidence}.
                 Includes 'Debit' and 'Credit' separately (NOT 'Gross Amount').

    Returns:
        DataFrame with exactly CANONICAL_FIELDS columns, cleaned values.
    """
    out = pd.DataFrame(index=df.index)

    def get_src(field):
        return (mapping.get(field, {}) or {}).get("source")

    # Date
    src = get_src("Date")
    out["Date"] = df[src].apply(clean_date) if (src and src in df.columns) else pd.NaT

    # Voucher Type
    src = get_src("Voucher Type")
    out["Voucher Type"] = df[src].apply(clean_text) if (src and src in df.columns) else ""

    # Voucher No
    src = get_src("Voucher No")
    out["Voucher No"] = df[src].apply(clean_voucher) if (src and src in df.columns) else ""

    # Invoice Ref (primary match key). Falls back to the already-computed
    # Voucher No when no separate external reference column was mapped — many
    # ledger pairs use the same voucher/document number as the shared
    # cross-party identifier, with no distinct "invoice number" column at all.
    # Safe when the fallback doesn't apply: if Voucher No differs between the
    # two ledgers too, ref-based matching finds nothing either way, same as
    # leaving this blank — it can only help, never hurt.
    src = get_src("Invoice Ref")
    out["Invoice Ref"] = (
        df[src].apply(clean_voucher) if (src and src in df.columns) else out["Voucher No"]
    )

    # Description
    src = get_src("Description")
    out["Description"] = df[src].apply(clean_text) if (src and src in df.columns) else ""

    # Gross Amount — COMPUTED from Debit and Credit, uniformly, no role.
    # Any TDS-caused gap between what one party billed and what the other
    # received now surfaces as a visible AMOUNT_MISMATCH row (explained by
    # tds_reconciliation.py's TDS sheet) instead of being silently absorbed
    # by a gross-up heuristic.
    d_src = get_src("Debit")
    c_src = get_src("Credit")

    debit  = df[d_src].apply(clean_amount) if (d_src and d_src in df.columns) else pd.Series(0.0, index=df.index)
    credit = df[c_src].apply(clean_amount) if (c_src and c_src in df.columns) else pd.Series(0.0, index=df.index)

    tds_src = get_src("TDS Amount")
    tds = df[tds_src].apply(clean_amount) if (tds_src and tds_src in df.columns) else pd.Series(0.0, index=df.index)
    out["TDS Amount"] = tds

    out["Gross Amount"] = debit - credit

    # Notes & Rec Code — filled by reconciliation engine later
    out["Notes"]    = ""
    out["Rec Code"] = ""

    # Drop balance / summary rows
    balance_mask = df.apply(_is_balance_row, axis=1)
    out = out[~balance_mask]

    # Drop rows with no date AND no amount
    out = out[~((out["Date"].isna()) & (out["Gross Amount"] == 0))].reset_index(drop=True)

    return out[CANONICAL_FIELDS]