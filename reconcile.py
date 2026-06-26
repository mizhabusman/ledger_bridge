"""
LedgerBridge AI — Reconciliation Engine

The deterministic core. Takes two standardized DataFrames and produces:
- Matched pairs (with L1/L2/L3 confidence)
- Missing-in-ours, Missing-in-theirs
- Amount mismatches
- Timing differences

Matching strategy (each Our-row tries each level in order, stops at first hit):
- L1: Invoice Ref + Date + Amount (highest confidence)
- L2: Invoice Ref + Amount (date differs → timing)
- L3: Date + Amount (when refs are blank/different)

Each Their-row can only be claimed once. Duplicates pair in occurrence order
(1st A row of ₹5000 ↔ 1st B row of ₹5000).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    DEFAULT_AMOUNT_TOLERANCE,
    DEFAULT_DATE_TOLERANCE_DAYS,
    REC_CODES,
)


@dataclass
class ReconciliationResult:
    """All outputs of a reconciliation run."""
    our_ledger: pd.DataFrame              # Standardized + Rec Code filled
    their_ledger: pd.DataFrame            # Standardized + Rec Code filled
    matched: pd.DataFrame                 # Paired rows with match level
    missing_in_ours: pd.DataFrame         # Their rows with no match
    missing_in_theirs: pd.DataFrame       # Our rows with no match
    amount_mismatches: pd.DataFrame       # Same Date+Ref, different amount
    timing_differences: pd.DataFrame      # L2 matches (date gap)
    summary: dict                         # Counts and totals


def _add_occurrence_index(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """
    For each row, count how many earlier rows share the same key tuple.
    Used to pair duplicates 1st↔1st, 2nd↔2nd.
    """
    return df.groupby(keys, dropna=False).cumcount()


def _amounts_match(a: float, b: float, tolerance: float) -> bool:
    """Check if two amounts match within tolerance (absolute value comparison)."""
    return abs(abs(a) - abs(b)) <= tolerance


def reconcile(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    amount_tolerance: float = DEFAULT_AMOUNT_TOLERANCE,
    date_tolerance_days: int = DEFAULT_DATE_TOLERANCE_DAYS,
    opening_balance_ours: float = 0.0,
    opening_balance_theirs: float = 0.0,
) -> ReconciliationResult:
    """
    Run the full reconciliation. Returns a ReconciliationResult.

    The two input DataFrames must already be in canonical schema (use standardize.py).
    """
    # Defensive copy — never mutate the caller's data.
    ours = ours.reset_index(drop=True).copy()
    theirs = theirs.reset_index(drop=True).copy()

    # Track which their-rows have been claimed.
    their_claimed = np.zeros(len(theirs), dtype=bool)

    # Track per-row match info on both sides.
    ours["_rec_code"] = ""
    ours["_match_idx"] = -1   # index into theirs of the matched row
    ours["_rec_id"] = ""
    ours["_match_level"] = ""

    theirs["_rec_code"] = ""
    theirs["_match_idx"] = -1
    theirs["_rec_id"] = ""
    theirs["_match_level"] = ""

    # Add occurrence indices for duplicate handling.
    ours["_occ_l1"] = _add_occurrence_index(ours, ["Date", "Invoice Ref", "_amt_key"]) if False else _add_occurrence_index(
        ours.assign(_amt_key=ours["Gross Amount"].abs().round(2)), ["Date", "Invoice Ref", "_amt_key"]
    )
    theirs["_occ_l1"] = _add_occurrence_index(
        theirs.assign(_amt_key=theirs["Gross Amount"].abs().round(2)), ["Date", "Invoice Ref", "_amt_key"]
    )

    # ---------------- LEVEL 1: Date + Invoice Ref + Amount ----------------
    for i, our_row in ours.iterrows():
        if our_row["_match_idx"] != -1:
            continue
        if not our_row["Invoice Ref"]:  # L1 requires a ref
            continue
        if pd.isna(our_row["Date"]):
            continue

        candidates = theirs[
            (~their_claimed)
            & (theirs["Invoice Ref"] == our_row["Invoice Ref"])
            & (theirs["Date"] == our_row["Date"])
        ]
        # Apply amount + occurrence filter
        for j, their_row in candidates.iterrows():
            if _amounts_match(our_row["Gross Amount"], their_row["Gross Amount"], amount_tolerance):
                if our_row["_occ_l1"] == their_row["_occ_l1"]:
                    _record_match(ours, theirs, i, j, "L1")
                    their_claimed[j] = True
                    break

    # ---------------- LEVEL 2: Invoice Ref + Amount (dates differ) ----------------
    for i, our_row in ours.iterrows():
        if ours.at[i, "_match_idx"] != -1:
            continue
        if not our_row["Invoice Ref"]:
            continue

        candidates = theirs[
            (~their_claimed)
            & (theirs["Invoice Ref"] == our_row["Invoice Ref"])
        ]
        for j, their_row in candidates.iterrows():
            if _amounts_match(our_row["Gross Amount"], their_row["Gross Amount"], amount_tolerance):
                # Optionally check date is within reasonable window (timing window)
                if pd.notna(our_row["Date"]) and pd.notna(their_row["Date"]):
                    days_apart = abs((our_row["Date"] - their_row["Date"]).days)
                    if days_apart > date_tolerance_days:
                        continue
                _record_match(ours, theirs, i, j, "L2")
                their_claimed[j] = True
                break

    # ---------------- AMOUNT MISMATCH: Same Date + Ref, different amount ----------------
    # These are NOT matches but they're "close" — flag them before L3.
    for i, our_row in ours.iterrows():
        if ours.at[i, "_match_idx"] != -1:
            continue
        if not our_row["Invoice Ref"] or pd.isna(our_row["Date"]):
            continue

        candidates = theirs[
            (~their_claimed)
            & (theirs["Invoice Ref"] == our_row["Invoice Ref"])
            & (theirs["Date"] == our_row["Date"])
        ]
        if len(candidates) > 0:
            j = candidates.index[0]
            ours.at[i, "_rec_code"] = REC_CODES["AMOUNT_MISMATCH"]
            ours.at[i, "_match_idx"] = j
            ours.at[i, "_match_level"] = "MISMATCH"
            theirs.at[j, "_rec_code"] = REC_CODES["AMOUNT_MISMATCH"]
            theirs.at[j, "_match_idx"] = i
            theirs.at[j, "_match_level"] = "MISMATCH"
            rec_id = _new_rec_id()
            ours.at[i, "_rec_id"] = rec_id
            theirs.at[j, "_rec_id"] = rec_id
            their_claimed[j] = True

    # ---------------- LEVEL 3: Date + Amount (no/different refs) ----------------
    # Recompute occurrence index over (Date, abs amount) for L3.
    ours_l3_key = ours.assign(_amt=ours["Gross Amount"].abs().round(2))
    theirs_l3_key = theirs.assign(_amt=theirs["Gross Amount"].abs().round(2))
    ours["_occ_l3"] = ours_l3_key.groupby(["Date", "_amt"], dropna=False).cumcount()
    theirs["_occ_l3"] = theirs_l3_key.groupby(["Date", "_amt"], dropna=False).cumcount()

    for i, our_row in ours.iterrows():
        if ours.at[i, "_match_idx"] != -1:
            continue
        if pd.isna(our_row["Date"]):
            continue

        candidates = theirs[
            (~their_claimed)
            & (theirs["Date"] == our_row["Date"])
        ]
        for j, their_row in candidates.iterrows():
            if _amounts_match(our_row["Gross Amount"], their_row["Gross Amount"], amount_tolerance):
                if ours.at[i, "_occ_l3"] == theirs.at[j, "_occ_l3"]:
                    _record_match(ours, theirs, i, j, "L3")
                    their_claimed[j] = True
                    break

    # ---------------- Unmatched: flag missing ----------------
    for i in ours.index:
        if ours.at[i, "_match_idx"] == -1:
            ours.at[i, "_rec_code"] = REC_CODES["MISSING_THEIRS"]

    for j in theirs.index:
        if theirs.at[j, "_match_idx"] == -1:
            theirs.at[j, "_rec_code"] = REC_CODES["MISSING_OURS"]

    # ---------------- Build result frames ----------------
    # Copy Rec Code into the public column.
    ours["Rec Code"] = ours["_rec_code"]
    theirs["Rec Code"] = theirs["_rec_code"]

    # Notes column gets a human-readable hint.
    def _make_note(row):
        lvl = row["_match_level"]
        if lvl == "L1": return "Matched on Date + Invoice Ref + Amount"
        if lvl == "L2": return "Matched on Invoice Ref + Amount; dates differ (possible timing)"
        if lvl == "L3": return "Matched on Date + Amount only (weak — please review)"
        if lvl == "MISMATCH": return "Same Date + Invoice Ref but amounts differ"
        if row["_rec_code"] == REC_CODES["MISSING_THEIRS"]: return "Not found in counterparty ledger"
        if row["_rec_code"] == REC_CODES["MISSING_OURS"]: return "Not found in our ledger"
        return ""

    ours["Notes"] = ours.apply(_make_note, axis=1)
    theirs["Notes"] = theirs.apply(_make_note, axis=1)

    # Build matched table (side-by-side view)
    matched_pairs = []
    for i in ours.index:
        if ours.at[i, "_match_idx"] != -1 and ours.at[i, "_match_level"] in ("L1", "L2", "L3"):
            j = ours.at[i, "_match_idx"]
            matched_pairs.append({
                "Rec ID": ours.at[i, "_rec_id"],
                "Match Level": ours.at[i, "_match_level"],
                "Our Date": ours.at[i, "Date"],
                "Their Date": theirs.at[j, "Date"],
                "Invoice Ref": ours.at[i, "Invoice Ref"] or theirs.at[j, "Invoice Ref"],
                "Our Description": ours.at[i, "Description"],
                "Their Description": theirs.at[j, "Description"],
                "Our Amount": ours.at[i, "Gross Amount"],
                "Their Amount": theirs.at[j, "Gross Amount"],
                "Difference": abs(ours.at[i, "Gross Amount"]) - abs(theirs.at[j, "Gross Amount"]),
            })
    matched = pd.DataFrame(matched_pairs)

    # Amount mismatches table
    mismatch_pairs = []
    for i in ours.index:
        if ours.at[i, "_match_level"] == "MISMATCH":
            j = ours.at[i, "_match_idx"]
            mismatch_pairs.append({
                "Rec ID": ours.at[i, "_rec_id"],
                "Date": ours.at[i, "Date"],
                "Invoice Ref": ours.at[i, "Invoice Ref"],
                "Description": ours.at[i, "Description"],
                "Our Amount": ours.at[i, "Gross Amount"],
                "Their Amount": theirs.at[j, "Gross Amount"],
                "Difference": ours.at[i, "Gross Amount"] - theirs.at[j, "Gross Amount"],
            })
    amount_mismatches = pd.DataFrame(mismatch_pairs)

    # Timing differences = L2 matches
    timing = matched[matched["Match Level"] == "L2"].copy() if not matched.empty else pd.DataFrame()
    if not timing.empty:
        timing["Days Apart"] = (timing["Their Date"] - timing["Our Date"]).dt.days

    # Missing tables — drop internal helper columns
    helper_cols = [c for c in ours.columns if c.startswith("_")]
    missing_theirs = ours[ours["_rec_code"] == REC_CODES["MISSING_THEIRS"]].drop(columns=helper_cols).reset_index(drop=True)
    missing_ours = theirs[theirs["_rec_code"] == REC_CODES["MISSING_OURS"]].drop(columns=helper_cols).reset_index(drop=True)

    # Final cleanup of ledgers — drop helper columns from public output
    our_clean = ours.drop(columns=helper_cols).reset_index(drop=True)
    their_clean = theirs.drop(columns=helper_cols).reset_index(drop=True)

    # ---------------- Summary / balance walk ----------------
    sum_ours = ours["Gross Amount"].sum()
    sum_theirs = theirs["Gross Amount"].sum()
    sum_tds_ours = ours["TDS Amount"].sum()
    sum_tds_theirs = theirs["TDS Amount"].sum()

    closing_ours = opening_balance_ours + sum_ours
    closing_theirs = opening_balance_theirs + sum_theirs
    difference = closing_ours - closing_theirs

    # Reconciling items: sum of unmatched + TDS difference
    reconciling_item = (
        ours[ours["_rec_code"] == REC_CODES["MISSING_THEIRS"]]["Gross Amount"].sum()
        - theirs[theirs["_rec_code"] == REC_CODES["MISSING_OURS"]]["Gross Amount"].sum()
    )

    residual = difference - reconciling_item
    reconciled = abs(residual) <= amount_tolerance

    summary = {
        "total_our_records": len(ours),
        "total_their_records": len(theirs),
        "matched_l1": int((matched["Match Level"] == "L1").sum()) if not matched.empty else 0,
        "matched_l2": int((matched["Match Level"] == "L2").sum()) if not matched.empty else 0,
        "matched_l3": int((matched["Match Level"] == "L3").sum()) if not matched.empty else 0,
        "amount_mismatches": len(amount_mismatches),
        "missing_in_theirs": len(missing_theirs),
        "missing_in_ours": len(missing_ours),
        "opening_balance_ours": opening_balance_ours,
        "opening_balance_theirs": opening_balance_theirs,
        "sum_our_transactions": float(sum_ours),
        "sum_their_transactions": float(sum_theirs),
        "closing_balance_ours": float(closing_ours),
        "closing_balance_theirs": float(closing_theirs),
        "difference": float(difference),
        "reconciling_item": float(reconciling_item),
        "residual": float(residual),
        "tds_ours": float(sum_tds_ours),
        "tds_theirs": float(sum_tds_theirs),
        "tds_difference": float(sum_tds_ours - sum_tds_theirs),
        "reconciled": bool(reconciled),
        "amount_tolerance": amount_tolerance,
    }

    return ReconciliationResult(
        our_ledger=our_clean,
        their_ledger=their_clean,
        matched=matched,
        missing_in_ours=missing_ours,
        missing_in_theirs=missing_theirs,
        amount_mismatches=amount_mismatches,
        timing_differences=timing if isinstance(timing, pd.DataFrame) else pd.DataFrame(),
        summary=summary,
    )


# ---------- Helpers ----------

def _new_rec_id() -> str:
    """Generate a short, human-friendly reconciliation ID."""
    return "REC-" + uuid.uuid4().hex[:6].upper()


def _record_match(ours: pd.DataFrame, theirs: pd.DataFrame, i: int, j: int, level: str) -> None:
    """Mark a matched pair on both sides with a shared Rec ID."""
    rec_id = _new_rec_id()
    code_key = level  # "L1", "L2", or "L3"
    code = REC_CODES[code_key]

    ours.at[i, "_match_idx"] = j
    ours.at[i, "_rec_code"] = code
    ours.at[i, "_rec_id"] = rec_id
    ours.at[i, "_match_level"] = level

    theirs.at[j, "_match_idx"] = i
    theirs.at[j, "_rec_code"] = code
    theirs.at[j, "_rec_id"] = rec_id
    theirs.at[j, "_match_level"] = level
