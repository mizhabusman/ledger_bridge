"""
LedgerBridge AI — Core Reconciliation Engine

Performs a deterministic, multi-pass match between two standardized ledgers.
Strictly relies on Pandas for 100% mathematical accuracy — no percentage guesses.

Gross Amount is Debit - Credit on BOTH ledgers uniformly (standardize.py has no
buyer/seller distinction). Because the two parties are double-entry counterparties,
the SAME real transaction lands with OPPOSITE signs on the two ledgers (one party's
receivable is the other's payable). Every match predicate and balance formula below
therefore checks "ours + theirs ~ 0" (mirror-sign, cancels out).

Match outcomes:
  CONFIRMED (exact, self-proving):
    MATCHED_L1        date + ref + amount (gross cancels within rounding_tolerance)
    MATCHED_L2_TIMING ref + amount, dates differ within the timing window
    MATCHED_INCL_TDS  ref-matched; cancels EXACTLY once the counterparty's own
                      booked per-invoice TDS is added back (buyer books net, seller
                      books gross). Confirmed for identity, but NOT balance-neutral —
                      the +TDS delta is a real reconciling item (unbooked TDS credit).
    MATCHED_L3_REVIEW no shared ref; exact mirror on date+amount; unambiguous 1:1;
                      voucher-class compatible (never a credit note vs an invoice).
  REVIEW (never auto-cleared):
    AMOUNT_MISMATCH        same ref, gap not explained by TDS, within am_ceiling
    SUGGESTED_TDS_UNVERIFIED both sides carry TDS — cannot self-prove; verify
    SIGN_REVERSED_REVIEW   same date & magnitude but SAME sign (posting error)
    SUSPECTED_DUPLICATE    row explicitly marked a duplicate
  MISSING — genuinely no counterpart found.

There is NO fuzzy / percentage "variance" auto-match: a pair is only ever confirmed
on an exact (with or without the counterparty's own TDS) mirror cancellation.
"""

import re
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from config import (
    REC_CODES,
    BALANCE_NEUTRAL_CODES,
    DEFAULT_ROUNDING_TOLERANCE,
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

_MATCHED_COLS = [
    "Match Level", "Invoice Ref", "Date (Ours)", "Date (Theirs)",
    "Gross Amount (Ours)", "Gross Amount (Theirs)", "Note", "Description",
]

# Voucher-number code tokens → class. Derived from the voucher NUMBER token (most
# reliable) plus Voucher Type / Description keywords, NOT from Voucher Type alone
# (on many files both purchases and credit notes are booked as "G.Journal").
_CN_CODES   = {"CN", "RCRV", "SR", "DN", "CRN"}
_BANK_CODES = {"BR", "BP", "BRV", "BPV"}
_INV_CODES  = {"SV", "TCV", "PV", "SI", "PI", "INV"}
_JRNL_CODES = {"JV", "JNL"}


def _add_occurrence_index(df: pd.DataFrame, subset: list[str]) -> pd.Series:
    """Occurrence number (0,1,2...) of a row within a group — pairs duplicates 1:1."""
    return df.groupby(subset, dropna=False).cumcount()


def _looks_tds(desc) -> bool:
    """Guard so TDS journal rows are left for the TDS post-processor rather than
    swept into the sign-reversed / duplicate passes."""
    return "tds" in str(desc).lower()


def _voucher_class(vno, vtype, desc) -> str:
    """Classify a row so the no-ref pass never pairs incompatible kinds (e.g. a
    credit note against an invoice).

    The voucher-NUMBER code token (SV / TCV / BP / RCRV / CN …) is the reliable
    signal and takes PRIORITY over description keywords — a Tally purchase voucher
    is labelled 'GST Vr. Payments/ Purchase', so matching bare description words
    like 'payment' would wrongly call it BANK. Description keywords are only a
    fallback when the voucher number carries no recognised code."""
    text = f"{vtype} {desc}".lower()
    codes = set(re.findall(r"[A-Z]+", str(vno).upper()))

    # TDS overrides everything (a TDS journal must never be matched as anything else)
    if "tds" in text or any("TDS" in c for c in codes):
        return "TDS_JOURNAL"

    # Primary: the voucher-number code token
    if codes & _CN_CODES:
        return "CREDIT_NOTE"
    if codes & _BANK_CODES:
        return "BANK"
    if codes & _INV_CODES:
        return "INVOICE"
    if codes & _JRNL_CODES:
        return "JOURNAL"

    # Fallback: precise description keywords (NOT bare 'payment', which appears in
    # Tally purchase-voucher class labels)
    if any(k in text for k in (
        "credit note", "c.note", "cr note", "debit note",
        "p.return", "p. return", "purchase return", "sales return",
    )):
        return "CREDIT_NOTE"
    if any(k in text for k in ("bank", "neft", "rtgs", "imps", "cheque")):
        return "BANK"
    if any(k in text for k in ("sales", "purchase", "invoice", "bill")):
        return "INVOICE"
    if any(k in text for k in ("receipt", "payment")):
        return "BANK"
    if "journal" in text:
        return "JOURNAL"
    return "OTHER"


def reconcile(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    rounding_tolerance: float = DEFAULT_ROUNDING_TOLERANCE,
    am_ceiling_pct: float = DEFAULT_AM_CEILING_PCT,
    date_tolerance_days: int = DEFAULT_DATE_TOLERANCE_DAYS,
    detect_anomalies: bool = True,
    match_tds: bool = True,
    opening_balance_ours: float = 0.0,
    opening_balance_theirs: float = 0.0,
) -> ReconciliationResult:
    """
    Cascading passes (see module docstring for tiers):
      L1 date+ref+amount → L2 ref+amount(timing) → AM ref/amount-differs
      → L3 date+amount (no ref, class-gated) → sign-reversed & duplicate anomalies.
    The TDS-inclusive predicate rides inside the ref-matched L1/L2 passes.

    Set match_tds=False + detect_anomalies=False to reproduce a plain exact-only
    engine (used by the differential oracle).
    """
    ours = ours.copy()
    theirs = theirs.copy()

    ours["_internal_id"] = range(len(ours))
    theirs["_internal_id"] = range(len(theirs))
    for df in (ours, theirs):
        df["_match_idx"] = -1
        df["_rec_code"] = ""
        df["_rec_id"] = ""
        df["_match_level"] = ""
        df["_note"] = ""

    # Occurrence indices (magnitude-keyed) so identical duplicates pair 1:1; distinct
    # amounts each form their own group (occ 0), so TDS-inclusive pairs are unaffected.
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

    o_date = ours["Date"].to_numpy()
    t_date = theirs["Date"].to_numpy()
    o_amt = ours["Gross Amount"].to_numpy(dtype=float)
    t_amt = theirs["Gross Amount"].to_numpy(dtype=float)
    o_tds = ours["TDS Amount"].to_numpy(dtype=float) if "TDS Amount" in ours.columns else np.zeros(len(ours))
    t_tds = theirs["TDS Amount"].to_numpy(dtype=float) if "TDS Amount" in theirs.columns else np.zeros(len(theirs))
    o_ref = ours["Invoice Ref"].to_numpy(dtype=object)
    t_ref = theirs["Invoice Ref"].to_numpy(dtype=object)
    o_desc = ours["Description"].to_numpy(dtype=object)
    t_desc = theirs["Description"].to_numpy(dtype=object)
    o_class = [_voucher_class(v, vt, d) for v, vt, d in zip(
        ours["Voucher No"], ours["Voucher Type"], ours["Description"])]
    t_class = [_voucher_class(v, vt, d) for v, vt, d in zip(
        theirs["Voucher No"], theirs["Voucher Type"], theirs["Description"])]
    o_dkey = o_date.view("int64")
    t_dkey = t_date.view("int64")

    n_o, n_t = len(ours), len(theirs)
    o_match = np.full(n_o, -1, dtype=np.int64)
    t_match = np.full(n_t, -1, dtype=np.int64)
    o_code = [""] * n_o; o_recid = [""] * n_o; o_level = [""] * n_o; o_note = [""] * n_o
    t_code = [""] * n_t; t_recid = [""] * n_t; t_level = [""] * n_t; t_note = [""] * n_t

    day = np.timedelta64(1, "D")
    tol = rounding_tolerance

    def _has_ref(ref) -> bool:
        return not (ref is None or (isinstance(ref, float) and np.isnan(ref)) or ref == "")

    def _ceil(o, t) -> float:
        return max(tol, am_ceiling_pct * max(abs(o), abs(t)))

    def _class_ok(a, b) -> bool:
        # No-ref pass may only pair the SAME class, and never a TDS journal.
        return a == b and a != "TDS_JOURNAL"

    def _tds_note(i, c):
        """Return a note string if the ref-matched pair (i,c) cancels EXACTLY once
        the counterparty's own booked per-invoice TDS is added back in the gross's
        sign direction; else None. One canonical direction only; never on bank rows;
        both-sides-TDS abstains (routed to review by the AM pass)."""
        if not match_tds:
            return None
        if o_class[i] in ("BANK", "TDS_JOURNAL") or t_class[c] in ("BANK", "TDS_JOURNAL"):
            return None
        to, tt = abs(o_tds[i]), abs(t_tds[c])
        if to > tol and tt > tol:
            return None  # both sides carry TDS → cannot self-prove; AM handles as review
        if tt > tol:     # counterparty (theirs) withheld TDS; seller booked gross
            if abs(o_amt[i] + t_amt[c] + np.sign(t_amt[c]) * tt) <= tol:
                return f"matched incl. counterparty TDS Rs {tt:,.2f}"
        elif to > tol:   # our side carries the TDS
            if abs(o_amt[i] + t_amt[c] + np.sign(o_amt[i]) * to) <= tol:
                return f"matched incl. our TDS Rs {to:,.2f}"
        return None

    # Candidate buckets
    grp_dr: dict = {}   # (date_key, ref) -> theirs positions   (L1)
    grp_r: dict = {}    # ref            -> theirs positions   (L2, AM)
    grp_td: dict = {}   # date_key       -> theirs positions   (L3)
    for c in range(n_t):
        grp_td.setdefault(t_dkey[c], []).append(c)
        r = t_ref[c]
        if _has_ref(r):
            grp_dr.setdefault((t_dkey[c], r), []).append(c)
            grp_r.setdefault(r, []).append(c)
    grp_od: dict = {}   # date_key -> ours positions  (L3 reverse-uniqueness)
    for k in range(n_o):
        grp_od.setdefault(o_dkey[k], []).append(k)

    def _assign(i, c, code, basis, note=""):
        rid = f"{basis}-{i}"
        o_match[i] = c; o_code[i] = code; o_level[i] = basis; o_recid[i] = rid; o_note[i] = note
        t_match[c] = i; t_code[c] = code; t_level[c] = basis; t_recid[c] = rid; t_note[c] = note

    # ---------------- LEVEL 1: Date + Invoice Ref + Amount (exact, then TDS) ----------------
    for i in range(n_o):
        r = o_ref[i]
        if not _has_ref(r):
            continue
        for c in grp_dr.get((o_dkey[i], r), ()):
            if t_match[c] != -1:
                continue
            if not (o_date[i] == t_date[c]):
                continue
            if occ_l1_t[c] != occ_l1_o[i]:
                continue
            if abs(o_amt[i] + t_amt[c]) <= tol:
                _assign(i, c, REC_CODES["L1"], "L1")
                break
            note = _tds_note(i, c)
            if note is not None:
                _assign(i, c, REC_CODES["TDS_MATCH"], "TDS", note)
                break

    # ---------------- LEVEL 2: Invoice Ref + Amount, dates differ (exact, then TDS) ----------------
    # No date cap here: a shared, exact invoice reference with an agreeing amount
    # (with or without the counterparty's TDS) is a confirmed match regardless of
    # how far apart the two books posted it — the buyer routinely books a purchase
    # months after the seller's invoice date (e.g. 2025-10-31 vs 2026-03-01). The
    # day gap is reported in the Timing Differences view. Refs carry the FY prefix
    # (25-26 vs 26-27), so a genuinely reused number across years does not collide.
    for i in range(n_o):
        if o_match[i] != -1:
            continue
        r = o_ref[i]
        if not _has_ref(r):
            continue
        for c in grp_r.get(r, ()):
            if t_match[c] != -1:
                continue
            if occ_l2_t[c] != occ_l2_o[i]:
                continue
            if abs(o_amt[i] + t_amt[c]) <= tol:
                _assign(i, c, REC_CODES["L2"], "L2")
                break
            note = _tds_note(i, c)
            if note is not None:
                _assign(i, c, REC_CODES["TDS_MATCH"], "TDS", note)
                break

    # ---------------- AMOUNT MISMATCH / TDS-UNVERIFIED (same ref, not exact) ----------------
    # Remaining ref-matched pairs failed exact AND single-side-TDS. Pair the
    # closest-to-cancelling candidate within the ceiling; classify as a genuine
    # mismatch, or (if both sides carry TDS) as an unverifiable TDS candidate.
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
            gap = abs(o_amt[i] + t_amt[c])
            if gap > _ceil(o_amt[i], t_amt[c]):
                continue
            if best_gap is None or gap < best_gap:
                best_gap, best_c = gap, c
        if best_c != -1:
            if abs(o_tds[i]) > tol and abs(t_tds[best_c]) > tol:
                _assign(i, best_c, REC_CODES["TDS_UNVERIFIED"], "TDSU",
                        "both sides carry TDS — verify allocation")
            else:
                _assign(i, best_c, REC_CODES["AMOUNT_MISMATCH"], "AM")

    # ---------------- LEVEL 3: Date + Amount, NO ref (exact, class-gated, unambiguous) ----------------
    for i in range(n_o):
        if o_match[i] != -1:
            continue
        cand, n_cand = -1, 0
        for c in grp_td.get(o_dkey[i], ()):
            if t_match[c] != -1:
                continue
            if not (o_date[i] == t_date[c]):
                continue
            if not _class_ok(o_class[i], t_class[c]):
                continue
            if abs(o_amt[i] + t_amt[c]) > tol:
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
            if not _class_ok(o_class[k], t_class[cand]):
                continue
            if abs(o_amt[k] + their_amt) > tol:
                continue
            rev += 1
            if rev > 1:
                break
        if rev != 1:
            continue
        _assign(i, cand, REC_CODES["L3"], "L3")

    # ---------------- SIGN-REVERSED (possible posting error) ----------------
    def _sign_candidate(o_i, t_c) -> bool:
        if o_amt[o_i] == 0 or t_amt[t_c] == 0:
            return False
        if (o_amt[o_i] > 0) != (t_amt[t_c] > 0):     # must be SAME sign
            return False
        return abs(abs(o_amt[o_i]) - abs(t_amt[t_c])) <= tol

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
            _assign(i, cand, REC_CODES["SIGN_REVERSED"], "SR")

    # ---------------- SUSPECTED DUPLICATES (one-sided, marker-based) ----------------
    if detect_anomalies:
        def _flag_dups(n, desc, ref, match, code):
            for j in range(n):
                if match[j] != -1 or code[j] != "" or _looks_tds(desc[j]):
                    continue
                if ("duplicate" in str(desc[j]).lower()) or ("DUP" in str(ref[j]).upper()):
                    code[j] = REC_CODES["SUSPECTED_DUP"]

        _flag_dups(n_o, o_desc, o_ref, o_match, o_code)
        _flag_dups(n_t, t_desc, t_ref, t_match, t_code)

    # Write results back
    ours["_match_idx"] = o_match
    ours["_rec_code"] = o_code
    ours["_rec_id"] = o_recid
    ours["_match_level"] = o_level
    ours["_note"] = o_note
    theirs["_match_idx"] = t_match
    theirs["_rec_code"] = t_code
    theirs["_rec_id"] = t_recid
    theirs["_match_level"] = t_level
    theirs["_note"] = t_note

    # Leftovers → MISSING (only rows not already classified)
    ours.loc[(ours["_match_idx"] == -1) & (ours["_rec_code"] == ""), "_rec_code"] = REC_CODES["MISSING_THEIRS"]
    theirs.loc[(theirs["_match_idx"] == -1) & (theirs["_rec_code"] == ""), "_rec_code"] = REC_CODES["MISSING_OURS"]

    # Publish canonical Rec Code + carry the match note into Notes
    ours["Rec Code"] = ours["_rec_code"]
    theirs["Rec Code"] = theirs["_rec_code"]
    ours.loc[ours["_note"] != "", "Notes"] = ours.loc[ours["_note"] != "", "_note"]
    theirs.loc[theirs["_note"] != "", "Notes"] = theirs.loc[theirs["_note"] != "", "_note"]

    # ---------------- METRICS ----------------
    sum_ours = ours["Gross Amount"].sum()
    sum_theirs = theirs["Gross Amount"].sum()
    cb_ours = opening_balance_ours + sum_ours
    cb_theirs = opening_balance_theirs + sum_theirs
    diff = cb_ours + cb_theirs

    # Reconciling items = every row whose gross does NOT cancel with a partner.
    # BALANCE_NEUTRAL_CODES (exact L1/L2/L3 only) cancel and drop out. Crucially,
    # MATCHED_INCL_TDS is a CONFIRMED match but NOT balance-neutral: the pair nets
    # to the withheld TDS (a real reconciling item), so it stays counted here.
    neutral = set(BALANCE_NEUTRAL_CODES)
    noncl_ours = ours[~ours["_rec_code"].isin(neutral)]["Gross Amount"].sum()
    noncl_theirs = theirs[~theirs["_rec_code"].isin(neutral)]["Gross Amount"].sum()
    reconciling_item = noncl_ours + noncl_theirs
    residual = diff - reconciling_item

    # Named TDS-match delta (the withheld TDS surfaced by confirmed TDS matches)
    tds_match_delta = (
        ours[ours["_rec_code"] == REC_CODES["TDS_MATCH"]]["Gross Amount"].sum()
        + theirs[theirs["_rec_code"] == REC_CODES["TDS_MATCH"]]["Gross Amount"].sum()
    )
    tds_ours = float(ours["TDS Amount"].sum()) if "TDS Amount" in ours.columns else 0.0
    tds_theirs = float(theirs["TDS Amount"].sum()) if "TDS Amount" in theirs.columns else 0.0

    def _co(df, key):
        return int((df["_rec_code"] == REC_CODES[key]).sum())

    summary = {
        "total_our_records": len(ours),
        "total_their_records": len(theirs),
        "matched_l1": _co(ours, "L1"),
        "matched_l2": _co(ours, "L2"),
        "matched_l3": _co(ours, "L3"),
        "matched_tds": _co(ours, "TDS_MATCH"),
        "amount_mismatches": _co(ours, "AMOUNT_MISMATCH"),
        "suggested_tds_unverified": _co(ours, "TDS_UNVERIFIED"),
        "sign_reversed": _co(ours, "SIGN_REVERSED"),
        "suspected_duplicates": _co(ours, "SUSPECTED_DUP") + _co(theirs, "SUSPECTED_DUP"),
        "missing_in_theirs": _co(ours, "MISSING_THEIRS"),
        "missing_in_ours": _co(theirs, "MISSING_OURS"),

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
        "am_ceiling_pct": am_ceiling_pct,
        "tds_match_delta": tds_match_delta,
        "tds_ours": tds_ours,
        "tds_theirs": tds_theirs,
        "tds_difference": tds_ours - tds_theirs,
    }

    # ---------------- Display tables ----------------
    display_cols = ["Date", "Voucher Type", "Voucher No", "Invoice Ref", "Description",
                    "Gross Amount", "TDS Amount", "_rec_code", "_rec_id"]

    def _merge_pairs(code_key):
        o_sel = ours[ours["_rec_code"] == REC_CODES[code_key]].copy()
        if o_sel.empty:
            return pd.DataFrame()
        return o_sel.merge(theirs, left_on="_match_idx", right_on="_internal_id", suffixes=("_ours", "_theirs"))

    # Confirmed Matches (L1/L2/L3 + TDS), each with a Note (TDS basis where relevant)
    matched_frames = []
    for lvl_key, basis in (("L1", "L1"), ("L2", "L2"), ("L3", "L3"), ("TDS_MATCH", "TDS")):
        m = _merge_pairs(lvl_key)
        if m.empty:
            continue
        matched_frames.append(pd.DataFrame({
            "Match Level":           basis,
            "Invoice Ref":           m["Invoice Ref_ours"],
            "Date (Ours)":           m["Date_ours"],
            "Date (Theirs)":         m["Date_theirs"],
            "Gross Amount (Ours)":   m["Gross Amount_ours"],
            "Gross Amount (Theirs)": m["Gross Amount_theirs"],
            "Note":                  m["_note_ours"],
            "Description":           m["Description_ours"],
        }))
    matched_df = (pd.concat(matched_frames, ignore_index=True)
                  if matched_frames else pd.DataFrame(columns=_MATCHED_COLS))

    # Amount Mismatches (back-compat table = AMOUNT_MISMATCH pairs only)
    am_merged = pd.DataFrame()
    _am = _merge_pairs("AMOUNT_MISMATCH")
    if not _am.empty:
        am_merged = _am[[
            "Date_ours", "Invoice Ref_ours", "Gross Amount_ours", "Description_ours",
            "Date_theirs", "Invoice Ref_theirs", "Gross Amount_theirs", "Description_theirs",
        ]].copy()
        am_merged["Difference"] = am_merged["Gross Amount_ours"] + am_merged["Gross Amount_theirs"]

    # Timing differences (L2)
    l2_merged = pd.DataFrame()
    _l2 = _merge_pairs("L2")
    if not _l2.empty:
        l2_merged = _l2[[
            "Date_ours", "Invoice Ref_ours", "Gross Amount_ours",
            "Date_theirs", "Invoice Ref_theirs", "Gross Amount_theirs",
        ]].copy()
        l2_merged["Days Difference"] = (l2_merged["Date_theirs"] - l2_merged["Date_ours"]).dt.days

    # Needs Review — paired review codes + one-sided suspected duplicates
    from config import REC_REASONS
    review_rows = []
    for code_key in ("AMOUNT_MISMATCH", "TDS_UNVERIFIED", "SIGN_REVERSED"):
        code = REC_CODES[code_key]
        m = _merge_pairs(code_key)
        for _, row in m.iterrows():
            review_rows.append({
                "Reason":         REC_REASONS.get(code, code),
                "Rec Code":       code,
                "Date (Ours)":    row["Date_ours"],
                "Ref (Ours)":     row["Invoice Ref_ours"],
                "Gross (Ours)":   row["Gross Amount_ours"],
                "Date (Theirs)":  row["Date_theirs"],
                "Ref (Theirs)":   row["Invoice Ref_theirs"],
                "Gross (Theirs)": row["Gross Amount_theirs"],
                "Gap":            row["Gross Amount_ours"] + row["Gross Amount_theirs"],
                "Description":    row["Description_ours"],
            })
    dup_code = REC_CODES["SUSPECTED_DUP"]
    for side_df, is_ours in ((ours, True), (theirs, False)):
        for _, row in side_df[side_df["_rec_code"] == dup_code].iterrows():
            review_rows.append({
                "Reason":         REC_REASONS.get(dup_code, dup_code),
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
    needs_review_df = (pd.DataFrame(review_rows, columns=_NEEDS_REVIEW_COLS)
                       if review_rows else pd.DataFrame(columns=_NEEDS_REVIEW_COLS))

    # Missing — ranked by absolute value so the largest reconciling item surfaces first
    missing_theirs_df = (ours[ours["_rec_code"] == REC_CODES["MISSING_THEIRS"]][display_cols]
                         .reindex(ours[ours["_rec_code"] == REC_CODES["MISSING_THEIRS"]]["Gross Amount"]
                                  .abs().sort_values(ascending=False).index)
                         .reset_index(drop=True))
    missing_ours_df = (theirs[theirs["_rec_code"] == REC_CODES["MISSING_OURS"]][display_cols]
                       .reindex(theirs[theirs["_rec_code"] == REC_CODES["MISSING_OURS"]]["Gross Amount"]
                                .abs().sort_values(ascending=False).index)
                       .reset_index(drop=True))

    # Cleanup temp columns
    drop_cols = ["_internal_id", "_match_idx", "_rec_code", "_rec_id", "_match_level", "_note"]
    ours = ours.drop(columns=[c for c in drop_cols if c in ours.columns])
    theirs = theirs.drop(columns=[c for c in drop_cols if c in theirs.columns])

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
