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

    # Occurrence indices for duplicate handling — same keys as the reference
    # implementation, kept so tolerance-band duplicates still pair 1-to-1.
    occ_l1_o = _add_occurrence_index(
        ours.assign(_amt_key=ours["Gross Amount"].abs().round(2)), ["Date", "Invoice Ref", "_amt_key"]
    ).to_numpy()
    occ_l1_t = _add_occurrence_index(
        theirs.assign(_amt_key=theirs["Gross Amount"].abs().round(2)), ["Date", "Invoice Ref", "_amt_key"]
    ).to_numpy()
    occ_l2_o = _add_occurrence_index(
        ours.assign(_amt_key=ours["Gross Amount"].abs().round(2)), ["Invoice Ref", "_amt_key"]
    ).to_numpy()
    occ_l2_t = _add_occurrence_index(
        theirs.assign(_amt_key=theirs["Gross Amount"].abs().round(2)), ["Invoice Ref", "_amt_key"]
    ).to_numpy()

    # ---------------- VECTORIZED MATCHING ----------------
    # The reference implementation scanned ALL of `theirs` (an O(m) boolean mask)
    # for every row of `ours`, i.e. O(n·m). Here we bucket `theirs` once by the
    # natural candidate key of each pass — (Date, Ref) for L1, Ref for L2/AM,
    # Date for L3 — so each row only examines its own small candidate group.
    # The per-candidate predicates below are byte-for-byte the same conditions
    # as the reference, so classifications are identical (locked by diff tests).
    o_date = ours["Date"].to_numpy()                       # datetime64[ns]; NaT != NaT
    t_date = theirs["Date"].to_numpy()
    o_amt = ours["Gross Amount"].to_numpy(dtype=float)
    t_amt = theirs["Gross Amount"].to_numpy(dtype=float)
    o_ref = ours["Invoice Ref"].to_numpy(dtype=object)
    t_ref = theirs["Invoice Ref"].to_numpy(dtype=object)
    o_dkey = o_date.view("int64")                          # stable hashable date key
    t_dkey = t_date.view("int64")

    n_o, n_t = len(ours), len(theirs)
    o_match = np.full(n_o, -1, dtype=np.int64)
    t_match = np.full(n_t, -1, dtype=np.int64)
    o_code = [""] * n_o; o_recid = [""] * n_o; o_level = [""] * n_o
    t_code = [""] * n_t; t_recid = [""] * n_t; t_level = [""] * n_t

    day = np.timedelta64(1, "D")

    def _has_ref(ref) -> bool:
        # Mirrors: not pd.isna(ref) and ref != ""
        return not (ref is None or (isinstance(ref, float) and np.isnan(ref)) or ref == "")

    # Bucket theirs rows into candidate groups (positions in ascending row order,
    # so "first candidate" == the reference's matches.index[0]).
    grp_dr: dict = {}   # (date_key, ref) -> [theirs positions]   (L1)
    grp_r: dict = {}    # ref            -> [theirs positions]   (L2, AM)
    grp_td: dict = {}   # date_key       -> [theirs positions]   (L3)
    for c in range(n_t):
        grp_td.setdefault(t_dkey[c], []).append(c)
        r = t_ref[c]
        if _has_ref(r):
            grp_dr.setdefault((t_dkey[c], r), []).append(c)
            grp_r.setdefault(r, []).append(c)
    grp_od: dict = {}   # date_key -> [ours positions]  (L3 reverse-uniqueness)
    for k in range(n_o):
        grp_od.setdefault(o_dkey[k], []).append(k)

    def _assign(i, c, code, prefix, level):
        rid = f"{prefix}-{i}"
        o_match[i] = c; o_code[i] = code; o_recid[i] = rid; o_level[i] = level
        t_match[c] = i; t_code[c] = code; t_recid[c] = rid; t_level[c] = level

    # ---------------- LEVEL 1: Date + Invoice Ref + Amount ----------------
    for i in range(n_o):
        r = o_ref[i]
        if not _has_ref(r):
            continue
        for c in grp_dr.get((o_dkey[i], r), ()):
            if t_match[c] != -1:
                continue
            if not (o_date[i] == t_date[c]):              # NaT-safe date equality
                continue
            if abs(t_amt[c] - o_amt[i]) > amount_tolerance:
                continue
            if occ_l1_t[c] != occ_l1_o[i]:
                continue
            _assign(i, c, REC_CODES["L1"], "M1", "L1")
            break

    # ---------------- LEVEL 2: Invoice Ref + Amount (Timing Diff) ----------------
    for i in range(n_o):
        if o_match[i] != -1:
            continue
        r = o_ref[i]
        if not _has_ref(r):
            continue
        for c in grp_r.get(r, ()):
            if t_match[c] != -1:
                continue
            if abs(t_amt[c] - o_amt[i]) > amount_tolerance:
                continue
            if occ_l2_t[c] != occ_l2_o[i]:
                continue
            dd = (t_date[c] - o_date[i]) / day            # NaN if either NaT
            if not (abs(dd) <= date_tolerance_days):
                continue
            _assign(i, c, REC_CODES["L2"], "M2", "L2")
            break

    # ---------------- AMOUNT MISMATCHES (Ref match, amount differs) ----------------
    for i in range(n_o):
        if o_match[i] != -1:
            continue
        r = o_ref[i]
        if not _has_ref(r):
            continue
        # Pair the duplicated ref with the CLOSEST-amount candidate (ties → first
        # by row order, matching .abs().idxmin()).
        best_c, best_diff = -1, None
        for c in grp_r.get(r, ()):
            if t_match[c] != -1:
                continue
            d = abs(t_amt[c] - o_amt[i])
            if best_diff is None or d < best_diff:
                best_diff, best_c = d, c
        if best_c != -1:
            _assign(i, best_c, REC_CODES["AMOUNT_MISMATCH"], "AM", "AM")

    # ---------------- LEVEL 3: Date + Amount (Missing/Mismatched Ref) ----------------
    # L3 is a fuzzy fallback: it matches on Date + Amount alone (no Invoice Ref),
    # so any pairing risks joining two unrelated transactions that merely share a
    # date and value. To keep that risk low we only accept an UNAMBIGUOUS match:
    #   - exactly one unmatched candidate on their side within tolerance, AND
    #   - this is the only unmatched row on our side that would claim it.
    # Groups with several equally-plausible candidates are left as "missing" for
    # human review rather than paired by arbitrary row order.
    for i in range(n_o):
        if o_match[i] != -1:
            continue
        cand = -1
        n_cand = 0
        for c in grp_td.get(o_dkey[i], ()):
            if t_match[c] != -1:
                continue
            if not (o_date[i] == t_date[c]):
                continue
            if abs(t_amt[c] - o_amt[i]) > amount_tolerance:
                continue
            n_cand += 1
            if n_cand > 1:
                break                                     # ambiguous, stop early
            cand = c
        if n_cand != 1:
            continue  # 0 → no match; >1 → ambiguous, defer to review

        their_amt = t_amt[cand]
        # Reverse uniqueness: is this the only unmatched row on our side that
        # would claim this same counterparty row?
        rev = 0
        for k in grp_od.get(t_dkey[cand], ()):
            if o_match[k] != -1:
                continue
            if not (o_date[k] == t_date[cand]):
                continue
            if abs(o_amt[k] - their_amt) > amount_tolerance:
                continue
            rev += 1
            if rev > 1:
                break
        if rev != 1:
            continue  # multiple ours rows compete → ambiguous, defer

        _assign(i, cand, REC_CODES["L3"], "M3", "L3")

    # Write match results back onto the frames for metric/report extraction.
    ours["_match_idx"] = o_match
    ours["_rec_code"] = o_code
    ours["_rec_id"] = o_recid
    ours["_match_level"] = o_level
    theirs["_match_idx"] = t_match
    theirs["_rec_code"] = t_code
    theirs["_rec_id"] = t_recid
    theirs["_match_level"] = t_level

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
    drop_cols = ["_internal_id", "_match_idx"]
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
