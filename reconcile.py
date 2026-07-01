"""
LedgerBridge AI — Core Reconciliation Engine

Performs a deterministic, multi-pass match between two standardized ledgers.
Strictly relies on Pandas for 100% mathematical accuracy.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

from config import REC_CODES, DEFAULT_AMOUNT_TOLERANCE, DEFAULT_DATE_TOLERANCE_DAYS


@dataclass
class ReconciliationResult:
    our_ledger: pd.DataFrame
    their_ledger: pd.DataFrame
    summary: dict
    matched: pd.DataFrame
    amount_mismatches: pd.DataFrame
    missing_in_theirs: pd.DataFrame
    missing_in_ours: pd.DataFrame
    timing_differences: pd.DataFrame


def _add_occurrence_index(df: pd.DataFrame, subset: list[str]) -> pd.Series:
    """
    Returns a series representing the occurrence number (0, 1, 2...) of a row
    within a group defined by `subset`. Crucial for matching duplicates (e.g.,
    two identical payments on the same day) exactly 1-to-1.
    """
    return df.groupby(subset, dropna=False).cumcount()


def reconcile(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    amount_tolerance: float = DEFAULT_AMOUNT_TOLERANCE,
    date_tolerance_days: int = DEFAULT_DATE_TOLERANCE_DAYS,
    opening_balance_ours: float = 0.0,
    opening_balance_theirs: float = 0.0,
) -> ReconciliationResult:
    """
    Core matching logic. Runs in cascading levels:
      L1: Exact Date + Ref + Amount
      L2: Ref + Amount (Timing difference within tolerance)
      L3: Date + Amount (Missing Ref, optional fallback)
    """
    # Working copies
    ours = ours.copy()
    theirs = theirs.copy()

    # We need an internal ID to track rows across passes
    ours["_internal_id"] = range(len(ours))
    theirs["_internal_id"] = range(len(theirs))

    # Initialize tracking columns
    ours["_match_idx"] = -1
    ours["_rec_code"] = ""
    ours["_rec_id"] = ""
    ours["_match_level"] = ""

    theirs["_rec_code"] = ""
    theirs["_match_idx"] = -1
    theirs["_rec_id"] = ""
    theirs["_match_level"] = ""

    # Add occurrence indices for duplicate handling.
    ours["_occ_l1"] = _add_occurrence_index(
        ours.assign(_amt_key=ours["Gross Amount"].abs().round(2)), ["Date", "Invoice Ref", "_amt_key"]
    )
    theirs["_occ_l1"] = _add_occurrence_index(
        theirs.assign(_amt_key=theirs["Gross Amount"].abs().round(2)), ["Date", "Invoice Ref", "_amt_key"]
    )

    # ---------------- LEVEL 1: Date + Invoice Ref + Amount ----------------
    for i, our_row in ours.iterrows():
        if not pd.isna(our_row["Invoice Ref"]) and our_row["Invoice Ref"] != "":
            mask = (
                (theirs["_match_idx"] == -1) &
                (theirs["Date"] == our_row["Date"]) &
                (theirs["Invoice Ref"] == our_row["Invoice Ref"]) &
                (abs(theirs["Gross Amount"] - our_row["Gross Amount"]) <= amount_tolerance) &
                (theirs["_occ_l1"] == our_row["_occ_l1"])
            )
            matches = theirs[mask]
            if not matches.empty:
                match_idx = matches.index[0]
                rec_id = f"M1-{i}"
                ours.at[i, "_match_idx"] = match_idx
                ours.at[i, "_rec_code"] = REC_CODES["L1"]
                ours.at[i, "_rec_id"] = rec_id
                ours.at[i, "_match_level"] = "L1"

                theirs.at[match_idx, "_match_idx"] = i
                theirs.at[match_idx, "_rec_code"] = REC_CODES["L1"]
                theirs.at[match_idx, "_rec_id"] = rec_id
                theirs.at[match_idx, "_match_level"] = "L1"

    # ---------------- LEVEL 2: Invoice Ref + Amount (Timing Diff) ----------------
    ours["_occ_l2"] = _add_occurrence_index(
        ours.assign(_amt_key=ours["Gross Amount"].abs().round(2)), ["Invoice Ref", "_amt_key"]
    )
    theirs["_occ_l2"] = _add_occurrence_index(
        theirs.assign(_amt_key=theirs["Gross Amount"].abs().round(2)), ["Invoice Ref", "_amt_key"]
    )

    for i, our_row in ours[ours["_match_idx"] == -1].iterrows():
        if not pd.isna(our_row["Invoice Ref"]) and our_row["Invoice Ref"] != "":
            mask = (
                (theirs["_match_idx"] == -1) &
                (theirs["Invoice Ref"] == our_row["Invoice Ref"]) &
                (abs(theirs["Gross Amount"] - our_row["Gross Amount"]) <= amount_tolerance) &
                (theirs["_occ_l2"] == our_row["_occ_l2"])
            )
            # Check date tolerance
            valid_dates = mask & (abs((theirs["Date"] - our_row["Date"]).dt.days) <= date_tolerance_days)
            matches = theirs[valid_dates]

            if not matches.empty:
                match_idx = matches.index[0]
                rec_id = f"M2-{i}"
                ours.at[i, "_match_idx"] = match_idx
                ours.at[i, "_rec_code"] = REC_CODES["L2"]
                ours.at[i, "_rec_id"] = rec_id
                ours.at[i, "_match_level"] = "L2"

                theirs.at[match_idx, "_match_idx"] = i
                theirs.at[match_idx, "_rec_code"] = REC_CODES["L2"]
                theirs.at[match_idx, "_rec_id"] = rec_id
                theirs.at[match_idx, "_match_level"] = "L2"

    # ---------------- AMOUNT MISMATCHES (Ref match, amount differs) ----------------
    for i, our_row in ours[ours["_match_idx"] == -1].iterrows():
        if not pd.isna(our_row["Invoice Ref"]) and our_row["Invoice Ref"] != "":
            mask = (
                (theirs["_match_idx"] == -1) &
                (theirs["Invoice Ref"] == our_row["Invoice Ref"])
            )
            matches = theirs[mask]
            if not matches.empty:
                match_idx = matches.index[0]
                rec_id = f"AM-{i}"
                ours.at[i, "_match_idx"] = match_idx
                ours.at[i, "_rec_code"] = REC_CODES["AMOUNT_MISMATCH"]
                ours.at[i, "_rec_id"] = rec_id
                ours.at[i, "_match_level"] = "AM"

                theirs.at[match_idx, "_match_idx"] = i
                theirs.at[match_idx, "_rec_code"] = REC_CODES["AMOUNT_MISMATCH"]
                theirs.at[match_idx, "_rec_id"] = rec_id
                theirs.at[match_idx, "_match_level"] = "AM"

    # ---------------- LEVEL 3: Date + Amount (Missing/Mismatched Ref) ----------------
    ours["_occ_l3"] = _add_occurrence_index(
        ours.assign(_amt_key=ours["Gross Amount"].abs().round(2)), ["Date", "_amt_key"]
    )
    theirs["_occ_l3"] = _add_occurrence_index(
        theirs.assign(_amt_key=theirs["Gross Amount"].abs().round(2)), ["Date", "_amt_key"]
    )

    for i, our_row in ours[ours["_match_idx"] == -1].iterrows():
        mask = (
            (theirs["_match_idx"] == -1) &
            (theirs["Date"] == our_row["Date"]) &
            (abs(theirs["Gross Amount"] - our_row["Gross Amount"]) <= amount_tolerance) &
            (theirs["_occ_l3"] == our_row["_occ_l3"])
        )
        matches = theirs[mask]
        if not matches.empty:
            match_idx = matches.index[0]
            rec_id = f"M3-{i}"
            ours.at[i, "_match_idx"] = match_idx
            ours.at[i, "_rec_code"] = REC_CODES["L3"]
            ours.at[i, "_rec_id"] = rec_id
            ours.at[i, "_match_level"] = "L3"

            theirs.at[match_idx, "_match_idx"] = i
            theirs.at[match_idx, "_rec_code"] = REC_CODES["L3"]
            theirs.at[match_idx, "_rec_id"] = rec_id
            theirs.at[match_idx, "_match_level"] = "L3"

    # ---------------- LEFTOVERS: MISSING ----------------
    ours.loc[ours["_match_idx"] == -1, "_rec_code"] = REC_CODES["MISSING_THEIRS"]
    theirs.loc[theirs["_match_idx"] == -1, "_rec_code"] = REC_CODES["MISSING_OURS"]

    # ---------------- CALCULATE METRICS ----------------
    sum_ours = ours["Gross Amount"].sum()
    sum_theirs = theirs["Gross Amount"].sum()
    cb_ours = opening_balance_ours + sum_ours
    cb_theirs = opening_balance_theirs + sum_theirs
    diff = cb_ours - cb_theirs

    # Reconciling items = (Missing in Theirs) - (Missing in Ours) + (Amount Mismatches diff)
    missing_theirs_sum = ours[ours["_rec_code"] == REC_CODES["MISSING_THEIRS"]]["Gross Amount"].sum()
    missing_ours_sum = theirs[theirs["_rec_code"] == REC_CODES["MISSING_OURS"]]["Gross Amount"].sum()
    
    am_ours_sum = ours[ours["_rec_code"] == REC_CODES["AMOUNT_MISMATCH"]]["Gross Amount"].sum()
    am_theirs_sum = theirs[theirs["_rec_code"] == REC_CODES["AMOUNT_MISMATCH"]]["Gross Amount"].sum()
    am_diff = am_ours_sum - am_theirs_sum

    reconciling_item = missing_theirs_sum - missing_ours_sum + am_diff
    residual = diff - reconciling_item

    # TDS column totals (used by the report Summary sheet + AI insights)
    tds_ours = float(ours["TDS Amount"].sum()) if "TDS Amount" in ours.columns else 0.0
    tds_theirs = float(theirs["TDS Amount"].sum()) if "TDS Amount" in theirs.columns else 0.0

    summary = {
        "total_our_records": len(ours),
        "total_their_records": len(theirs),
        "matched_l1": len(ours[ours["_rec_code"] == REC_CODES["L1"]]),
        "matched_l2": len(ours[ours["_rec_code"] == REC_CODES["L2"]]),
        "matched_l3": len(ours[ours["_rec_code"] == REC_CODES["L3"]]),
        "amount_mismatches": len(ours[ours["_rec_code"] == REC_CODES["AMOUNT_MISMATCH"]]),
        "missing_in_theirs": len(ours[ours["_rec_code"] == REC_CODES["MISSING_THEIRS"]]),
        "missing_in_ours": len(theirs[theirs["_rec_code"] == REC_CODES["MISSING_OURS"]]),
        
        "opening_balance_ours": opening_balance_ours,
        "opening_balance_theirs": opening_balance_theirs,
        "sum_our_transactions": sum_ours,
        "sum_their_transactions": sum_theirs,
        "closing_balance_ours": cb_ours,
        "closing_balance_theirs": cb_theirs,
        "difference": diff,
        "reconciling_item": reconciling_item,
        "residual": residual,
        "reconciled": abs(residual) <= (amount_tolerance * 2), # Minor buffer for accum rounding
        "amount_tolerance": amount_tolerance,
        "tds_ours": tds_ours,
        "tds_theirs": tds_theirs,
        "tds_difference": tds_ours - tds_theirs,
    }

    # Extract clean dataframes for the UI/Excel report
    display_cols = ["Date", "Voucher Type", "Invoice Ref", "Description", "Gross Amount", "TDS Amount", "_rec_code", "_rec_id"]
    
    # Amount Mismatches (Join them side-by-side)
    am_ours = ours[ours["_rec_code"] == REC_CODES["AMOUNT_MISMATCH"]].copy()
    am_theirs = theirs[theirs["_rec_code"] == REC_CODES["AMOUNT_MISMATCH"]].copy()
    
    am_merged = pd.DataFrame()
    if not am_ours.empty:
        am_merged = am_ours.merge(am_theirs, left_on="_match_idx", right_on="_internal_id", suffixes=("_ours", "_theirs"))
        # Clean up columns for display
        am_merged = am_merged[[
            "Date_ours", "Invoice Ref_ours", "Gross Amount_ours", "Description_ours",
            "Date_theirs", "Invoice Ref_theirs", "Gross Amount_theirs", "Description_theirs"
        ]]
        am_merged["Difference"] = am_merged["Gross Amount_ours"] - am_merged["Gross Amount_theirs"]

    # Timing Differences (L2 Matches)
    l2_ours = ours[ours["_rec_code"] == REC_CODES["L2"]].copy()
    l2_theirs = theirs[theirs["_rec_code"] == REC_CODES["L2"]].copy()
    
    l2_merged = pd.DataFrame()
    if not l2_ours.empty:
        l2_merged = l2_ours.merge(l2_theirs, left_on="_match_idx", right_on="_internal_id", suffixes=("_ours", "_theirs"))
        l2_merged = l2_merged[[
            "Date_ours", "Invoice Ref_ours", "Gross Amount_ours",
            "Date_theirs", "Invoice Ref_theirs", "Gross Amount_theirs"
        ]]
        l2_merged["Days Difference"] = (l2_merged["Date_theirs"] - l2_merged["Date_ours"]).dt.days

    # Combined Matched table (L1 + L2 + L3) — side-by-side pairs for the report/UI
    matched_cols = [
        "Match Level", "Invoice Ref", "Date (Ours)", "Date (Theirs)",
        "Gross Amount (Ours)", "Gross Amount (Theirs)", "Description",
    ]
    matched_frames = []
    for lvl in ("L1", "L2", "L3"):
        o = ours[ours["_rec_code"] == REC_CODES[lvl]].copy()
        if o.empty:
            continue
        m = o.merge(theirs, left_on="_match_idx", right_on="_internal_id", suffixes=("_ours", "_theirs"))
        if m.empty:
            continue
        matched_frames.append(pd.DataFrame({
            "Match Level":           lvl,
            "Invoice Ref":           m["Invoice Ref_ours"],
            "Date (Ours)":           m["Date_ours"],
            "Date (Theirs)":         m["Date_theirs"],
            "Gross Amount (Ours)":   m["Gross Amount_ours"],
            "Gross Amount (Theirs)": m["Gross Amount_theirs"],
            "Description":           m["Description_ours"],
        }))
    matched_df = (
        pd.concat(matched_frames, ignore_index=True)
        if matched_frames else pd.DataFrame(columns=matched_cols)
    )

    # Missing
    missing_theirs_df = ours[ours["_rec_code"] == REC_CODES["MISSING_THEIRS"]][display_cols].copy()
    missing_ours_df = theirs[theirs["_rec_code"] == REC_CODES["MISSING_OURS"]][display_cols].copy()

    # Cleanup temp columns
    drop_cols = ["_internal_id", "_match_idx", "_occ_l1", "_occ_l2", "_occ_l3"]
    ours = ours.drop(columns=[c for c in drop_cols if c in ours.columns])
    theirs = theirs.drop(columns=[c for c in drop_cols if c in theirs.columns])

    # Final sort
    ours = ours.sort_values(by=["Date", "Invoice Ref"], na_position="last").reset_index(drop=True)
    theirs = theirs.sort_values(by=["Date", "Invoice Ref"], na_position="last").reset_index(drop=True)

    return ReconciliationResult(
        our_ledger=ours,
        their_ledger=theirs,
        summary=summary,
        matched=matched_df,
        amount_mismatches=am_merged,
        missing_in_theirs=missing_theirs_df,
        missing_in_ours=missing_ours_df,
        timing_differences=l2_merged
    )
