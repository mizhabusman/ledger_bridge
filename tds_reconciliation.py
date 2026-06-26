"""
LedgerBridge AI — TDS Reconciliation Post-Processor

Runs after the main reconciliation engine. Purpose:

  In real Indian B2B accounting, TDS (Tax Deducted at Source) is stored
  differently on the buyer's and seller's books:

  * BUYER side: usually has a dedicated TDS Amount column, with the TDS for
    each invoice on the same row as the invoice.

  * SELLER side: often has TDS booked as one (or several) consolidated
    journal entries — a single row with description like "TDS Cut 2025-26".

  When matched naively, these structurally-different representations cause
  the seller's journal entry to appear as "Missing in Their Books" — but it's
  not missing, just structured differently.

  This module:
    1. Detects TDS-described journal entries in the unmatched rows
    2. Re-classifies them with a TDS_ENTRY rec code
    3. Computes side-by-side TDS totals (column vs journal)
    4. Assigns a status (MATCHED / PARTIAL / EXCESS / UNVERIFIED)
    5. Returns the data needed for the new "TDS Reconciliation" sheet

  IMPORTANT: This module does NOT change:
    - The closing-balance walk
    - The residual calculation
    - The matched / amount mismatch tables
    - The L1/L2/L3 matching logic
    - Any amount values
  It only RECLASSIFIES rows that were already "missing" and provides a new
  reporting view. The reconciliation math is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import REC_CODES, DEFAULT_AMOUNT_TOLERANCE


# Keywords (lower-case) that identify a row as a TDS-related journal entry.
# Conservative list — designed to avoid false positives.
_TDS_KEYWORDS = [
    "tds cut",
    "tds payable",
    "tds receivable",
    "tds deducted",
    "tax deducted at source",
    "tax deducted",
    "withholding tax",
    "wht ",
    " tds ",
    "section 194",
    "u/s 194",
]

# A row is TDS-related if its description contains ANY of these keywords
# OR if it's a single-token "TDS" (with word boundaries).
_TDS_SINGLE_TOKENS = {"tds"}


@dataclass
class TdsReconciliationResult:
    """Output of the TDS post-processor."""
    # Per-row flagged TDS journal entries (combined from both sides)
    flagged_entries: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Totals comparison
    our_tds_column_total: float = 0.0       # Sum of TDS Amount column on our side
    their_tds_column_total: float = 0.0     # Sum of TDS Amount column on their side
    our_tds_journal_total: float = 0.0      # Sum of TDS-flagged journal entries on our side
    their_tds_journal_total: float = 0.0    # Sum of TDS-flagged journal entries on their side

    # Overall TDS reconciliation status
    overall_status: str = "NO_TDS_ACTIVITY"  # NO_TDS_ACTIVITY / MATCHED / PARTIAL / EXCESS / UNVERIFIED
    status_message: str = ""

    # Indices of rows in the original missing tables that should be removed
    # (because they've been reclassified as TDS entries)
    removed_from_missing_theirs: list[int] = field(default_factory=list)
    removed_from_missing_ours: list[int] = field(default_factory=list)


def _row_looks_like_tds(description: str) -> bool:
    """Return True if a row's description looks like a TDS journal entry."""
    if not description:
        return False
    desc = str(description).lower()

    # Multi-word keyword match
    if any(kw in desc for kw in _TDS_KEYWORDS):
        return True

    # Single-token "tds" with word boundaries (avoid matching "tdsabc")
    # Split on common separators and check each token
    import re
    tokens = re.split(r"[\s,.\-:;/()]+", desc)
    for tok in tokens:
        if tok.lower() in _TDS_SINGLE_TOKENS:
            return True

    return False


def _classify_status(
    journal_total: float,
    opposite_column_total: float,
    tolerance: float,
) -> tuple[str, str]:
    """
    Given a journal-entry total and the opposite side's TDS-column total,
    return (status_code, human_readable_message).
    """
    j = abs(journal_total)
    o = abs(opposite_column_total)

    if o == 0:
        return ("UNVERIFIED",
                "Counterparty has no TDS column to verify against.")

    diff = j - o
    if abs(diff) <= tolerance:
        return ("MATCHED",
                f"Journal entry matches opposite side's TDS column total exactly.")

    if j < o:
        pct = (j / o * 100) if o else 0
        return ("PARTIAL",
                f"Counterparty has booked only {pct:.0f}% of withheld TDS. "
                f"Gap of ₹{o - j:,.2f} not yet booked.")

    # j > o
    return ("EXCESS",
            f"Journal entry is ₹{j - o:,.2f} more than opposite side's "
            f"TDS column total. Needs investigation.")


def classify_tds_entries(
    missing_in_theirs: pd.DataFrame,
    missing_in_ours: pd.DataFrame,
    our_full_ledger: pd.DataFrame,
    their_full_ledger: pd.DataFrame,
    tolerance: float = DEFAULT_AMOUNT_TOLERANCE,
) -> TdsReconciliationResult:
    """
    Identify TDS-related journal entries in the unmatched rows and produce
    the data for the TDS Reconciliation sheet.

    Args:
        missing_in_theirs: rows from ours that didn't match anything in theirs
        missing_in_ours:   rows from theirs that didn't match anything in ours
        our_full_ledger:   the full standardized our-ledger (for TDS column sum)
        their_full_ledger: the full standardized their-ledger (for TDS column sum)
        tolerance:         amount tolerance for status classification

    Returns:
        TdsReconciliationResult with flagged entries, totals, and status.
    """
    result = TdsReconciliationResult()

    # ─── Step 1: Compute TDS column totals from full ledgers ──────────────────
    if "TDS Amount" in our_full_ledger.columns:
        result.our_tds_column_total = float(our_full_ledger["TDS Amount"].sum())
    if "TDS Amount" in their_full_ledger.columns:
        result.their_tds_column_total = float(their_full_ledger["TDS Amount"].sum())

    # ─── Step 2: Find TDS-flagged rows in each missing table ──────────────────
    flagged_rows = []

    # On OUR side (missing_in_theirs = rows in ours with no counterpart)
    if not missing_in_theirs.empty and "Description" in missing_in_theirs.columns:
        for idx, row in missing_in_theirs.iterrows():
            if _row_looks_like_tds(row.get("Description", "")):
                flagged_rows.append({
                    "Source": "Ours",
                    "Date": row.get("Date"),
                    "Voucher No": row.get("Voucher No", ""),
                    "Description": row.get("Description", ""),
                    "Amount": float(row.get("Gross Amount", 0.0)),
                    "_original_index": idx,
                })
                result.removed_from_missing_theirs.append(idx)

    # On THEIR side (missing_in_ours = rows in theirs with no counterpart)
    if not missing_in_ours.empty and "Description" in missing_in_ours.columns:
        for idx, row in missing_in_ours.iterrows():
            if _row_looks_like_tds(row.get("Description", "")):
                flagged_rows.append({
                    "Source": "Theirs",
                    "Date": row.get("Date"),
                    "Voucher No": row.get("Voucher No", ""),
                    "Description": row.get("Description", ""),
                    "Amount": float(row.get("Gross Amount", 0.0)),
                    "_original_index": idx,
                })
                result.removed_from_missing_ours.append(idx)

    # ─── Step 3: Compute journal totals from flagged rows ─────────────────────
    for r in flagged_rows:
        if r["Source"] == "Ours":
            result.our_tds_journal_total += abs(r["Amount"])
        else:
            result.their_tds_journal_total += abs(r["Amount"])

    # ─── Step 4: Assign status to each flagged row ────────────────────────────
    for r in flagged_rows:
        if r["Source"] == "Ours":
            # Compare our journal entry against their TDS column
            status, msg = _classify_status(
                r["Amount"], result.their_tds_column_total, tolerance
            )
        else:
            # Compare their journal entry against our TDS column
            status, msg = _classify_status(
                r["Amount"], result.our_tds_column_total, tolerance
            )
        r["Status"] = status
        r["Notes"] = msg

    # ─── Step 5: Build the flagged_entries DataFrame ──────────────────────────
    if flagged_rows:
        df = pd.DataFrame(flagged_rows)
        # Drop the internal index column from the user-facing output
        result.flagged_entries = df.drop(columns=["_original_index"])
    else:
        result.flagged_entries = pd.DataFrame(
            columns=["Source", "Date", "Voucher No", "Description", "Amount", "Status", "Notes"]
        )

    # ─── Step 6: Overall status of the whole TDS reconciliation ──────────────
    result.overall_status, result.status_message = _overall_tds_status(result, tolerance)

    return result


def _overall_tds_status(
    result: TdsReconciliationResult,
    tolerance: float,
) -> tuple[str, str]:
    """Produce a one-liner summary status for the TDS sheet header."""
    has_any_tds = (
        result.our_tds_column_total
        or result.their_tds_column_total
        or result.our_tds_journal_total
        or result.their_tds_journal_total
    )

    if not has_any_tds:
        return ("NO_TDS_ACTIVITY", "No TDS activity detected in either ledger.")

    # Two views:
    #   Our column total should approximately equal Their journal total
    #   Their column total should approximately equal Our journal total
    our_col_vs_their_journal = result.our_tds_column_total - result.their_tds_journal_total
    their_col_vs_our_journal = result.their_tds_column_total - result.our_tds_journal_total

    # If one side has a column and the other has journal entries (the common case)
    if result.our_tds_column_total > 0 and result.their_tds_journal_total > 0:
        diff = our_col_vs_their_journal
        if abs(diff) <= tolerance:
            return ("MATCHED",
                    f"TDS fully reconciled: our records show ₹{result.our_tds_column_total:,.2f} "
                    f"withheld, counterparty has booked ₹{result.their_tds_journal_total:,.2f}.")
        if diff > 0:
            return ("PARTIAL",
                    f"Partial TDS posting: we withheld ₹{result.our_tds_column_total:,.2f}, "
                    f"counterparty has booked only ₹{result.their_tds_journal_total:,.2f}. "
                    f"Gap: ₹{diff:,.2f} not yet posted by counterparty.")
        return ("EXCESS",
                f"Counterparty's TDS journal (₹{result.their_tds_journal_total:,.2f}) "
                f"exceeds TDS we withheld (₹{result.our_tds_column_total:,.2f}) "
                f"by ₹{-diff:,.2f}. Needs investigation.")

    if result.their_tds_column_total > 0 and result.our_tds_journal_total > 0:
        diff = their_col_vs_our_journal
        if abs(diff) <= tolerance:
            return ("MATCHED",
                    f"TDS fully reconciled: counterparty's records show ₹{result.their_tds_column_total:,.2f} "
                    f"withheld, we have booked ₹{result.our_tds_journal_total:,.2f}.")
        if diff > 0:
            return ("PARTIAL",
                    f"Partial TDS posting: counterparty withheld ₹{result.their_tds_column_total:,.2f}, "
                    f"we have booked only ₹{result.our_tds_journal_total:,.2f}. "
                    f"Gap: ₹{diff:,.2f} not yet posted by us.")
        return ("EXCESS",
                f"Our TDS journal (₹{result.our_tds_journal_total:,.2f}) "
                f"exceeds TDS counterparty withheld (₹{result.their_tds_column_total:,.2f}) "
                f"by ₹{-diff:,.2f}. Needs investigation.")

    # Both sides have only column entries (no journals)
    if result.our_tds_column_total > 0 and result.their_tds_column_total > 0:
        diff = result.our_tds_column_total - result.their_tds_column_total
        if abs(diff) <= tolerance:
            return ("MATCHED",
                    f"TDS columns agree: ₹{result.our_tds_column_total:,.2f} on both sides.")
        return ("PARTIAL",
                f"TDS columns differ by ₹{diff:,.2f}. "
                f"Ours: ₹{result.our_tds_column_total:,.2f}, "
                f"Theirs: ₹{result.their_tds_column_total:,.2f}.")

    # One side has activity but other has nothing
    return ("UNVERIFIED",
            "Only one side has TDS activity. Cannot fully reconcile — "
            "counterparty may not be tracking TDS in this ledger.")


def apply_tds_reclassification(
    our_ledger: pd.DataFrame,
    their_ledger: pd.DataFrame,
    tds_result: TdsReconciliationResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given the full standardized ledgers and the TDS result, update the
    Rec Code on TDS-flagged rows so they're labeled TDS_ENTRY instead of
    MISSING. This affects the Rec Code column only — amounts are untouched.

    Returns the updated ledgers (copies — caller's input is not mutated).
    """
    our_updated = our_ledger.copy()
    their_updated = their_ledger.copy()

    tds_code = REC_CODES.get("TDS_ENTRY", "TDS_ENTRY_OTHER_SIDE")

    # Reclassify on our side
    if not tds_result.flagged_entries.empty:
        ours_flagged = tds_result.flagged_entries[tds_result.flagged_entries["Source"] == "Ours"]
        for _, fr in ours_flagged.iterrows():
            # Match by description + amount + voucher no
            mask = (
                (our_updated["Description"].astype(str) == str(fr["Description"]))
                & (our_updated["Voucher No"].astype(str) == str(fr["Voucher No"]))
                & (our_updated["Gross Amount"].round(2) == round(fr["Amount"], 2))
                & (our_updated["Rec Code"] == REC_CODES["MISSING_THEIRS"])
            )
            our_updated.loc[mask, "Rec Code"] = tds_code
            our_updated.loc[mask, "Notes"] = "TDS journal entry — see TDS Reconciliation sheet"

        theirs_flagged = tds_result.flagged_entries[tds_result.flagged_entries["Source"] == "Theirs"]
        for _, fr in theirs_flagged.iterrows():
            mask = (
                (their_updated["Description"].astype(str) == str(fr["Description"]))
                & (their_updated["Voucher No"].astype(str) == str(fr["Voucher No"]))
                & (their_updated["Gross Amount"].round(2) == round(fr["Amount"], 2))
                & (their_updated["Rec Code"] == REC_CODES["MISSING_OURS"])
            )
            their_updated.loc[mask, "Rec Code"] = tds_code
            their_updated.loc[mask, "Notes"] = "TDS journal entry — see TDS Reconciliation sheet"

    return our_updated, their_updated