"""
LedgerBridge AI — Standardization & Cleaning Engine

Converts a raw DataFrame + confirmed mapping + role into the canonical 9-field schema.

Gross Amount is COMPUTED here (not directly mapped):
    if role == "buyer":  gross = credit - debit
    else (seller):       gross = debit - credit

This makes the same transaction produce the SAME signed Gross Amount in both
ledgers regardless of which side records it in Debit vs Credit.
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


# ─────────────────────────── role auto-detection ─────────────────────────────

def detect_role_from_data(df: pd.DataFrame, mapping: dict) -> tuple[str, str]:
    """
    Inspect the data to guess role: 'buyer' or 'seller'.
    Returns (role, reasoning).

    Signals checked, in order of reliability:
      1. Voucher Type column keywords (Sales/Purchase/Receipt/Payment)
      2. Voucher No prefixes (SV, BR, CN = seller; BP, PV = buyer)
      3. Description keywords
      4. Debit/Credit column distribution
    """
    debit_src  = (mapping.get("Debit",  {}) or {}).get("source")
    credit_src = (mapping.get("Credit", {}) or {}).get("source")
    vt_src     = (mapping.get("Voucher Type", {}) or {}).get("source")
    vn_src     = (mapping.get("Voucher No", {}) or {}).get("source")
    desc_src   = (mapping.get("Description", {}) or {}).get("source")

    seller_score = 0
    buyer_score = 0
    reasons = []

    # Signal 1: Voucher Type keywords
    if vt_src and vt_src in df.columns:
        text = " ".join(str(v).lower() for v in df[vt_src].fillna(""))
        s = sum(text.count(k) for k in ["sales", " sv", "bank receipt", " br ", "receipt", "credit note"])
        b = sum(text.count(k) for k in ["purchase", "bank-payment", "bank payment", " bp ", "payment voucher"])
        if s > b:
            seller_score += 2
            reasons.append(f"voucher-type keywords seller={s} buyer={b}")
        elif b > s:
            buyer_score += 2
            reasons.append(f"voucher-type keywords buyer={b} seller={s}")

    # Signal 2: Voucher No prefixes (Tally exports use SV/BR/CN/BP codes)
    if vn_src and vn_src in df.columns:
        vns = df[vn_src].astype(str).str.upper().str.replace(r"[\s\-_/]", "", regex=True)
        # Extract first 2-3 alpha chars after any leading digits/year code
        import re as _re
        codes = vns.apply(lambda s: "".join(_re.findall(r"[A-Z]+", s)))
        s_codes = codes.apply(lambda c: any(k in c for k in ["SV", "BR", "RC", "CN", "SR"])).sum()
        b_codes = codes.apply(lambda c: any(k in c for k in ["BP", "PV", "PI", "PR"])).sum()
        if s_codes > b_codes:
            seller_score += 2
            reasons.append(f"voucher-no codes seller={s_codes} buyer={b_codes}")
        elif b_codes > s_codes:
            buyer_score += 2
            reasons.append(f"voucher-no codes buyer={b_codes} seller={s_codes}")

    # Signal 3: Description keywords
    if desc_src and desc_src in df.columns:
        text = " ".join(str(v).lower() for v in df[desc_src].fillna(""))
        s = sum(text.count(k) for k in ["sales", "bill raised", "invoice raised", "receipt"])
        b = sum(text.count(k) for k in ["payment to vendor", "purchase", "bill received", "vr. payments"])
        if s > b:
            seller_score += 1
        elif b > s:
            buyer_score += 1

    # Signal 4: Debit/Credit distribution (weakest signal — keep tie-breaker only)
    if debit_src and credit_src and debit_src in df.columns and credit_src in df.columns:
        d_filled = df[debit_src].apply(clean_amount).abs() > 0
        c_filled = df[credit_src].apply(clean_amount).abs() > 0
        d_count = int(d_filled.sum())
        c_count = int(c_filled.sum())
        if d_count > c_count * 1.5:
            seller_score += 1
            reasons.append(f"debit-heavy distribution ({d_count} vs {c_count})")
        elif c_count > d_count * 1.5:
            buyer_score += 1
            reasons.append(f"credit-heavy distribution ({c_count} vs {d_count})")

    if seller_score > buyer_score:
        return "seller", "; ".join(reasons) if reasons else "seller-side signals"
    if buyer_score > seller_score:
        return "buyer", "; ".join(reasons) if reasons else "buyer-side signals"
    return "buyer", "could not determine — defaulted to buyer"


# ─────────────────────────── canonical conversion ────────────────────────────

def standardize(df: pd.DataFrame, mapping: dict, role: str = "buyer") -> pd.DataFrame:
    """
    Convert a raw DataFrame to the canonical 9-field schema.

    Args:
        df:      Raw ledger DataFrame.
        mapping: Dict of canonical field -> {source, confidence}.
                 Includes 'Debit' and 'Credit' separately (NOT 'Gross Amount').
        role:    "buyer" or "seller". Determines sign convention for Gross Amount.

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

    # Invoice Ref (primary match key)
    src = get_src("Invoice Ref")
    out["Invoice Ref"] = df[src].apply(clean_voucher) if (src and src in df.columns) else ""

    # Description
    src = get_src("Description")
    out["Description"] = df[src].apply(clean_text) if (src and src in df.columns) else ""

    # Gross Amount — COMPUTED from Debit and Credit using the role
    d_src = get_src("Debit")
    c_src = get_src("Credit")

    debit  = df[d_src].apply(clean_amount) if (d_src and d_src in df.columns) else pd.Series(0.0, index=df.index)
    credit = df[c_src].apply(clean_amount) if (c_src and c_src in df.columns) else pd.Series(0.0, index=df.index)

    # TDS Amount (read first — we need it for the Gross calculation)
    tds_src = get_src("TDS Amount")
    tds = df[tds_src].apply(clean_amount) if (tds_src and tds_src in df.columns) else pd.Series(0.0, index=df.index)
    out["TDS Amount"] = tds

    if role == "buyer":
        # In a buyer's vendor ledger:
        #   Credit = net invoice amount (after the buyer withheld TDS) → POSITIVE
        #   Debit  = payment / CN (we owe less) → NEGATIVE
        # We add the buyer's withheld TDS back to invoice rows so Gross matches
        # the gross figure on the seller's side, which is the full amount billed.
        # Only add TDS on rows that look like invoice rows (Credit > 0).
        base = credit - debit
        # Add TDS back only on PURE invoice rows — credit booked with no matching
        # debit on the same line. A row carrying both debit and credit is an
        # adjustment/net-off, not a fresh invoice, so grossing it up would inflate
        # the balance and break the match against the seller's gross figure.
        invoice_mask = (credit > 0) & (debit == 0)
        out["Gross Amount"] = base + (tds * invoice_mask.astype(float))
    else:
        # In a seller's customer ledger:
        #   Debit  = full invoice amount (gross, before any TDS) → POSITIVE
        #   Credit = receipt / CN (they owe less) → NEGATIVE
        # Seller's Gross already reflects the gross billed value; no TDS adjustment.
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