"""
LedgerBridge AI — Core Reconciliation Engine

Performs a deterministic, multi-pass match between two standardized ledgers.
Strictly relies on Pandas for 100% mathematical accuracy.

Gross Amount is Debit - Credit on BOTH ledgers uniformly (standardize.py has
no buyer/seller distinction). Because the two parties are double-entry
counterparties, the SAME real transaction lands with OPPOSITE signs on the
two ledgers (one party's receivable is the other's payable). Every match
predicate and balance formula below therefore checks "ours + theirs ~ 0"
(mirror-sign, cancels out), not "ours - theirs ~ 0".

Match outcomes form three tiers:
  CLEAN   — L1 (date+ref+amount), L2 (ref+amount, timing), L3 (date+amount).
            Amount gap within `rounding_tolerance`.
  REVIEW  — probable pairs / anomalies that a human should confirm:
              MATCH_VARIANCE       paired, gap within the variance band
                                   (e.g. a bank charge / short payment)
              AMOUNT_MISMATCH      ref-matched, larger gap (within the ceiling)
              SIGN_REVERSED        same date & magnitude but SAME sign — a
                                   likely posting error (booked to wrong column)
              SUSPECTED_DUPLICATE  identical to another entry on the same side
  MISSING — genuinely no counterpart found.
"""

from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from config import (
    REC_CODES,
    REC_REASONS,
    DEFAULT_ROUNDING_TOLERANCE,
    DEFAULT_VARIANCE_BAND_PCT,
    DEFAULT_AM_CEILING_PCT,
    DEFAULT_DATE_TOLERANCE_DAYS,
)


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
    needs_review: pd.DataFrame = field(default_factory=pd.DataFrame)


_NEEDS_REVIEW_COLS = [
    "Reason", "Rec Code",
    "Date (Ours)", "Ref (Ours)", "Gross (Ours)",
    "Date (Theirs)", "Ref (Theirs)", "Gross (Theirs)",
    "Gap", "Description",
]


def _add_occurrence_index(df: pd.DataFrame, subset: list[str]) -> pd.Series:
    """
    Returns a series representing the occurrence number (0, 1, 2...) of a row
    within a group defined by `subset`. Crucial for matching duplicates (e.g.,
    two identical payments on the same day) exactly 1-to-1.
    """
    return df.groupby(subset, dropna=False).cumcount()


def _looks_tds(desc) -> bool:
    """Light guard so TDS journal rows are left for the TDS post-processor
    rather than being swept into the sign-reversed / duplicate passes."""
    return "tds" in str(desc).lower()


def reconcile(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    rounding_tolerance: float = DEFAULT_ROUNDING_TOLERANCE,
    variance_band_pct: float = DEFAULT_VARIANCE_BAND_PCT,
    am_ceiling_pct: float = DEFAULT_AM_CEILING_PCT,
    date_tolerance_days: int = DEFAULT_DATE_TOLERANCE_DAYS,
    detect_anomalies: bool = True,
    opening_balance_ours: float = 0.0,
    opening_balance_theirs: float = 0.0,
) -> ReconciliationResult:
    """
    Core matching logic. Cascading passes (see module docstring for tiers):
      L1: Date + Ref + Amount        L2: Ref + Amount (timing)
      L3: Date + Amount (no ref)     AM: Ref + amount differs (within ceiling)
      + sign-reversed & suspected-duplicate anomaly passes (if detect_anomalies)

    Amount bands (gap = abs(ours + theirs), mirror-sign):
      gap <= rounding_tolerance                    → clean match
      rounding < gap <= variance_band(amount)      → MATCH_VARIANCE
      variance_band < gap <= am_ceiling(amount)    → AMOUNT_MISMATCH (ref only)
      gap > am_ceiling(amount)                     → not paired
    variance_band / am_ceiling are percentages of the row magnitude, floored at
    rounding_tolerance. Set variance_band_pct=0 + a huge am_ceiling_pct +
    detect_anomalies=False to reproduce a plain single-tolerance engine.
    """
    # Working copies
    ours = ours.copy()
    theirs = theirs.copy()

    # We need an internal ID to track rows across passes
    ours["_internal_id"] = range(len(ours))
    theirs["_internal_id"] = range(len(theirs))

    # Initialize tracking columns
    for df in (ours, theirs):
        df["_match_idx"] = -1
        df["_rec_code"] = ""
        df["_rec_id"] = ""
        df["_match_level"] = ""

    # Occurrence indices for duplicate handling — keyed on magnitude so identical
    # duplicates still pair 1-to-1; distinct amounts each form their own group
    # (occ 0), so variance-band pairs are unaffected.
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
    # Bucket `theirs` once by each pass's natural key so every `ours` row only
    # scans its small candidate group (O(n+m) instead of O(n·m)).
    o_date = ours["Date"].to_numpy()                       # datetime64[ns]; NaT != NaT
    t_date = theirs["Date"].to_numpy()
    o_amt = ours["Gross Amount"].to_numpy(dtype=float)
    t_amt = theirs["Gross Amount"].to_numpy(dtype=float)
    o_ref = ours["Invoice Ref"].to_numpy(dtype=object)
    t_ref = theirs["Invoice Ref"].to_numpy(dtype=object)
    o_desc = ours["Description"].to_numpy(dtype=object)
    t_desc = theirs["Description"].to_numpy(dtype=object)
    o_dkey = o_date.view("int64")                          # stable hashable date key
    t_dkey = t_date.view("int64")

    n_o, n_t = len(ours), len(theirs)
    o_match = np.full(n_o, -1, dtype=np.int64)
    t_match = np.full(n_t, -1, dtype=np.int64)
    o_code = [""] * n_o; o_recid = [""] * n_o; o_level = [""] * n_o
    t_code = [""] * n_t; t_recid = [""] * n_t; t_level = [""] * n_t

    day = np.timedelta64(1, "D")

    def _has_ref(ref) -> bool:
        return not (ref is None or (isinstance(ref, float) and np.isnan(ref)) or ref == "")

    def _variance_thr(o, t) -> float:
        return max(rounding_tolerance, variance_band_pct * max(abs(o), abs(t)))

    def _ceiling_thr(o, t) -> float:
        return max(_variance_thr(o, t), am_ceiling_pct * max(abs(o), abs(t)))

    def _band_code(gap, o, t, clean_code):
        """Clean if within rounding, else flag as a variance."""
        return clean_code if gap <= rounding_tolerance else REC_CODES["VARIANCE"]

    # Bucket theirs rows into candidate groups (ascending row order).
    grp_dr: dict = {}   # (date_key, ref) -> [theirs positions]   (L1)
    grp_r: dict = {}    # ref            -> [theirs positions]   (L2, AM)
    grp_td: dict = {}   # date_key       -> [theirs positions]   (L3, sign-reversed)
    for c in range(n_t):
        grp_td.setdefault(t_dkey[c], []).append(c)
        r = t_ref[c]
        if _has_ref(r):
            grp_dr.setdefault((t_dkey[c], r), []).append(c)
            grp_r.setdefault(r, []).append(c)
    grp_od: dict = {}   # date_key -> [ours positions]  (reverse-uniqueness)
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
            gap = abs(t_amt[c] + o_amt[i])                # mirror-sign
            if gap > _variance_thr(o_amt[i], t_amt[c]):
                continue
            if occ_l1_t[c] != occ_l1_o[i]:
                continue
            _assign(i, c, _band_code(gap, o_amt[i], t_amt[c], REC_CODES["L1"]), "M1", "L1")
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
            gap = abs(t_amt[c] + o_amt[i])
            if gap > _variance_thr(o_amt[i], t_amt[c]):
                continue
            if occ_l2_t[c] != occ_l2_o[i]:
                continue
            dd = (t_date[c] - o_date[i]) / day            # NaN if either NaT
            if not (abs(dd) <= date_tolerance_days):
                continue
            _assign(i, c, _band_code(gap, o_amt[i], t_amt[c], REC_CODES["L2"]), "M2", "L2")
            break

    # ---------------- AMOUNT MISMATCHES (Ref match, amount differs) ----------------
    # Pair the same-ref leftover with the CLOSEST-to-cancelling candidate, but
    # only if within the ceiling — prevents an invoice binding to an unrelated
    # same-ref journal of wildly different value.
    for i in range(n_o):
        if o_match[i] != -1:
            continue
        r = o_ref[i]
        if not _has_ref(r):
            continue
        best_c, best_gap = -1, None
        for c in grp_r.get(r, ()):
            if t_match[c] != -1:
                continue
            gap = abs(t_amt[c] + o_amt[i])
            if gap > _ceiling_thr(o_amt[i], t_amt[c]):
                continue
            if best_gap is None or gap < best_gap:
                best_gap, best_c = gap, c
        if best_c != -1:
            code = _band_code(best_gap, o_amt[i], t_amt[best_c], REC_CODES["AMOUNT_MISMATCH"])
            if best_gap > _variance_thr(o_amt[i], t_amt[best_c]):
                code = REC_CODES["AMOUNT_MISMATCH"]
            _assign(i, best_c, code, "AM", "AM")

    # ---------------- LEVEL 3: Date + Amount (Missing/Mismatched Ref) ----------------
    # Fuzzy fallback on Date + Amount alone. Only accept an UNAMBIGUOUS 1:1 match
    # (exactly one candidate each way) so unrelated same-date/amount rows aren't
    # paired by row order. The variance band here is what rescues bank-charge
    # pairs that share a date but have different refs.
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
            if abs(t_amt[c] + o_amt[i]) > _variance_thr(o_amt[i], t_amt[c]):
                continue
            n_cand += 1
            if n_cand > 1:
                break
            cand = c
        if n_cand != 1:
            continue

        their_amt = t_amt[cand]
        rev = 0
        for k in grp_od.get(t_dkey[cand], ()):
            if o_match[k] != -1:
                continue
            if not (o_date[k] == t_date[cand]):
                continue
            if abs(o_amt[k] + their_amt) > _variance_thr(o_amt[k], their_amt):
                continue
            rev += 1
            if rev > 1:
                break
        if rev != 1:
            continue

        gap = abs(their_amt + o_amt[i])
        _assign(i, cand, _band_code(gap, o_amt[i], their_amt, REC_CODES["L3"]), "M3", "L3")

    # ---------------- SIGN-REVERSED (possible posting error) ----------------
    # Same date, same-signed, near-equal MAGNITUDE (|o| ≈ |t|). A true mirror
    # pair has opposite signs, so same-signed equal magnitudes indicate one side
    # booked the entry to the wrong column. Unambiguous 1:1 only.
    def _sign_candidate(o_i, t_c) -> bool:
        if o_amt[o_i] == 0 or t_amt[t_c] == 0:
            return False
        if (o_amt[o_i] > 0) != (t_amt[t_c] > 0):          # must be SAME sign
            return False
        return abs(abs(o_amt[o_i]) - abs(t_amt[t_c])) <= rounding_tolerance

    if detect_anomalies:
        for i in range(n_o):
            if o_match[i] != -1 or _looks_tds(o_desc[i]):
                continue
            cand, n_cand = -1, 0
            for c in grp_td.get(o_dkey[i], ()):
                if t_match[c] != -1 or _looks_tds(t_desc[c]):
                    continue
                if not (o_date[i] == t_date[c]):
                    continue
                if not _sign_candidate(i, c):
                    continue
                n_cand += 1
                if n_cand > 1:
                    break
                cand = c
            if n_cand != 1:
                continue
            rev = 0
            for k in grp_od.get(t_dkey[cand], ()):
                if o_match[k] != -1 or _looks_tds(o_desc[k]):
                    continue
                if not (o_date[k] == t_date[cand]):
                    continue
                if not _sign_candidate(k, cand):
                    continue
                rev += 1
                if rev > 1:
                    break
            if rev != 1:
                continue
            _assign(i, cand, REC_CODES["SIGN_REVERSED"], "SR", "SR")

    # ---------------- SUSPECTED DUPLICATES (one-sided) ----------------
    # An unmatched row explicitly marked as a duplicate — description contains
    # "duplicate" or the ref carries a DUP marker. We deliberately do NOT flag
    # on identical (date, amount) alone: two genuinely identical transactions on
    # one side (that each pair with a counterpart) are not duplicates, so a
    # pure value-sibling rule would false-positive. Value-only duplicate
    # detection can be a future opt-in.
    if detect_anomalies:
        def _flag_dups(n, desc, ref, match, code):
            for j in range(n):
                if match[j] != -1 or code[j] != "" or _looks_tds(desc[j]):
                    continue
                if ("duplicate" in str(desc[j]).lower()) or ("DUP" in str(ref[j]).upper()):
                    code[j] = REC_CODES["SUSPECTED_DUP"]

        _flag_dups(n_o, o_desc, o_ref, o_match, o_code)
        _flag_dups(n_t, t_desc, t_ref, t_match, t_code)

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
    # Only rows not already classified (matched or flagged as a review anomaly).
    ours.loc[(ours["_match_idx"] == -1) & (ours["_rec_code"] == ""), "_rec_code"] = REC_CODES["MISSING_THEIRS"]
    theirs.loc[(theirs["_match_idx"] == -1) & (theirs["_rec_code"] == ""), "_rec_code"] = REC_CODES["MISSING_OURS"]

    # Publish internal codes into the canonical "Rec Code" column consumed by the
    # standardized-ledger export and the TDS post-processor.
    ours["Rec Code"] = ours["_rec_code"]
    theirs["Rec Code"] = theirs["_rec_code"]

    # ---------------- CALCULATE METRICS ----------------
    # Mirror-sign: a clean pair satisfies ours + theirs ~ 0, so the balance
    # check is a SUM (requires opening balances entered mirror-signed too).
    sum_ours = ours["Gross Amount"].sum()
    sum_theirs = theirs["Gross Amount"].sum()
    cb_ours = opening_balance_ours + sum_ours
    cb_theirs = opening_balance_theirs + sum_theirs
    diff = cb_ours + cb_theirs

    # Reconciling items = every row NOT in a clean match, on both sides. Clean
    # pairs cancel (ours + theirs ~ 0) and are excluded, so the residual — what
    # the categorised items fail to explain — stays ~ 0 regardless of how the
    # non-clean rows are sub-categorised (variance / mismatch / sign / dup /
    # missing all count fully toward the gap).
    clean_codes = {REC_CODES["L1"], REC_CODES["L2"], REC_CODES["L3"]}
    noncl_ours = ours[~ours["_rec_code"].isin(clean_codes)]["Gross Amount"].sum()
    noncl_theirs = theirs[~theirs["_rec_code"].isin(clean_codes)]["Gross Amount"].sum()
    reconciling_item = noncl_ours + noncl_theirs
    residual = diff - reconciling_item

    tds_ours = float(ours["TDS Amount"].sum()) if "TDS Amount" in ours.columns else 0.0
    tds_theirs = float(theirs["TDS Amount"].sum()) if "TDS Amount" in theirs.columns else 0.0

    def _count(df, key):
        return int((df["_rec_code"] == REC_CODES[key]).sum())

    summary = {
        "total_our_records": len(ours),
        "total_their_records": len(theirs),
        "matched_l1": _count(ours, "L1"),
        "matched_l2": _count(ours, "L2"),
        "matched_l3": _count(ours, "L3"),
        "amount_mismatches": _count(ours, "AMOUNT_MISMATCH"),
        "variance": _count(ours, "VARIANCE"),
        "sign_reversed": _count(ours, "SIGN_REVERSED"),
        "suspected_duplicates": _count(ours, "SUSPECTED_DUP") + _count(theirs, "SUSPECTED_DUP"),
        "missing_in_theirs": _count(ours, "MISSING_THEIRS"),
        "missing_in_ours": _count(theirs, "MISSING_OURS"),

        "opening_balance_ours": opening_balance_ours,
        "opening_balance_theirs": opening_balance_theirs,
        "sum_our_transactions": sum_ours,
        "sum_their_transactions": sum_theirs,
        "closing_balance_ours": cb_ours,
        "closing_balance_theirs": cb_theirs,
        "difference": diff,
        "reconciling_item": reconciling_item,
        "residual": residual,
        "reconciled": abs(residual) <= max(1.0, rounding_tolerance * 2),
        "amount_tolerance": rounding_tolerance,
        "rounding_tolerance": rounding_tolerance,
        "variance_band_pct": variance_band_pct,
        "am_ceiling_pct": am_ceiling_pct,
        "tds_ours": tds_ours,
        "tds_theirs": tds_theirs,
        "tds_difference": tds_ours - tds_theirs,
    }

    # ---------------- Build display tables ----------------
    display_cols = ["Date", "Voucher Type", "Voucher No", "Invoice Ref", "Description", "Gross Amount", "TDS Amount", "_rec_code", "_rec_id"]

    def _merge_pairs(code_key):
        o_sel = ours[ours["_rec_code"] == REC_CODES[code_key]].copy()
        if o_sel.empty:
            return pd.DataFrame()
        return o_sel.merge(theirs, left_on="_match_idx", right_on="_internal_id", suffixes=("_ours", "_theirs"))

    # Amount Mismatches (kept for back-compat consumers)
    am_merged = pd.DataFrame()
    _am = _merge_pairs("AMOUNT_MISMATCH")
    if not _am.empty:
        am_merged = _am[[
            "Date_ours", "Invoice Ref_ours", "Gross Amount_ours", "Description_ours",
            "Date_theirs", "Invoice Ref_theirs", "Gross Amount_theirs", "Description_theirs",
        ]].copy()
        am_merged["Difference"] = am_merged["Gross Amount_ours"] + am_merged["Gross Amount_theirs"]

    # Timing Differences (L2)
    l2_merged = pd.DataFrame()
    _l2 = _merge_pairs("L2")
    if not _l2.empty:
        l2_merged = _l2[[
            "Date_ours", "Invoice Ref_ours", "Gross Amount_ours",
            "Date_theirs", "Invoice Ref_theirs", "Gross Amount_theirs",
        ]].copy()
        l2_merged["Days Difference"] = (l2_merged["Date_theirs"] - l2_merged["Date_ours"]).dt.days

    # Combined Matched table (clean L1/L2/L3 only)
    matched_cols = [
        "Match Level", "Invoice Ref", "Date (Ours)", "Date (Theirs)",
        "Gross Amount (Ours)", "Gross Amount (Theirs)", "Description",
    ]
    matched_frames = []
    for lvl in ("L1", "L2", "L3"):
        m = _merge_pairs(lvl)
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

    # Needs Review — one table, reason per row (paired + one-sided duplicates)
    review_rows = []
    for code_key in ("AMOUNT_MISMATCH", "VARIANCE", "SIGN_REVERSED"):
        code = REC_CODES[code_key]
        m = _merge_pairs(code_key)
        for _, row in m.iterrows():
            review_rows.append({
                "Reason":          REC_REASONS[code],
                "Rec Code":        code,
                "Date (Ours)":     row["Date_ours"],
                "Ref (Ours)":      row["Invoice Ref_ours"],
                "Gross (Ours)":    row["Gross Amount_ours"],
                "Date (Theirs)":   row["Date_theirs"],
                "Ref (Theirs)":    row["Invoice Ref_theirs"],
                "Gross (Theirs)":  row["Gross Amount_theirs"],
                "Gap":             row["Gross Amount_ours"] + row["Gross Amount_theirs"],
                "Description":     row["Description_ours"],
            })
    dup_code = REC_CODES["SUSPECTED_DUP"]
    for side_df, is_ours in ((ours, True), (theirs, False)):
        for _, row in side_df[side_df["_rec_code"] == dup_code].iterrows():
            review_rows.append({
                "Reason":         REC_REASONS[dup_code],
                "Rec Code":       dup_code,
                "Date (Ours)":    row["Date"] if is_ours else pd.NaT,
                "Ref (Ours)":     row["Invoice Ref"] if is_ours else "",
                "Gross (Ours)":   row["Gross Amount"] if is_ours else np.nan,
                "Date (Theirs)":  pd.NaT if is_ours else row["Date"],
                "Ref (Theirs)":   "" if is_ours else row["Invoice Ref"],
                "Gross (Theirs)": np.nan if is_ours else row["Gross Amount"],
                "Gap":            np.nan,
                "Description":    row["Description"],
            })
    needs_review_df = (
        pd.DataFrame(review_rows, columns=_NEEDS_REVIEW_COLS)
        if review_rows else pd.DataFrame(columns=_NEEDS_REVIEW_COLS)
    )

    # Missing (genuinely no counterpart)
    missing_theirs_df = ours[ours["_rec_code"] == REC_CODES["MISSING_THEIRS"]][display_cols].copy()
    missing_ours_df = theirs[theirs["_rec_code"] == REC_CODES["MISSING_OURS"]][display_cols].copy()

    # Cleanup temp columns
    drop_cols = ["_internal_id", "_match_idx", "_rec_code", "_rec_id", "_match_level"]
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
        timing_differences=l2_merged,
        needs_review=needs_review_df,
    )
