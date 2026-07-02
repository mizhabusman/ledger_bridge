"""
Self-contained end-to-end test of the deterministic core (no Claude, no sample files).

Builds synthetic raw ledgers in memory and runs the real pipeline:

    standardize -> reconcile -> write_report

Locks the reconcile.py <-> report.py contract and the finance-grade matching
rules: exact mirror matching, TDS-inclusive matching (seller books gross, buyer
books net + withholds TDS), the balance-walk integrity rule (a TDS match is
CONFIRMED but its withheld-TDS delta stays counted as a reconciling item), the
voucher-class gate (never match a credit note to an invoice), and the review
tiers. There is NO fuzzy percentage variance auto-match.

Gross Amount = Debit - Credit uniformly on both ledgers (no buyer/seller role);
the same transaction is opposite-signed on the two sides, so a match satisfies
ours + theirs ~ 0.

Run:  python test_core.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

from standardize import standardize
from reconcile import reconcile
from report import write_report, write_standardized_ledger
from tds_reconciliation import classify_tds_entries, apply_tds_reclassification
from config import CANONICAL_FIELDS, REC_CODES


# ── Synthetic raw ledgers (main contract test) ───────────────────────────────
OUR_RAW = pd.DataFrame([
    # Date,        VchType,  VchNo,  Ref,        Narration,     Dr,      Cr,   TDS
    ["2025-01-05", "Sales",  "SV-1", "INV-1001", "Sales bill", "10000", "0",  "0"],  # L1
    ["2025-01-10", "Sales",  "SV-2", "INV-1015", "Sales bill", "5000",  "0",  "0"],  # L2 (timing)
    ["2025-01-12", "Sales",  "SV-3", "INV-1012", "Sales bill", "8000",  "0",  "0"],  # amount mismatch
    ["2025-01-15", "Sales",  "SV-4", "INV-1010", "Sales bill", "3000",  "0",  "0"],  # missing in theirs
], columns=["Date", "Vch Type", "Vch No", "Ref No", "Narration", "Dr", "Cr", "TDS"])

THEIR_RAW = pd.DataFrame([
    # Date,        VchType,     VchNo,  Ref,        Narration,        Debit, Credit,  TDS
    ["2025-01-05", "Purchase",  "PV-1", "INV-1001", "Purchase bill",  "0",   "10000", "0"],  # L1
    ["2025-01-20", "Purchase",  "PV-2", "INV-1015", "Purchase bill",  "0",   "5000",  "0"],  # L2 (10d)
    ["2025-01-12", "Purchase",  "PV-3", "INV-1012", "Purchase bill",  "0",   "7000",  "0"],  # mismatch (7000 vs 8000)
    ["2025-01-18", "Purchase",  "PV-4", "INV-1099", "Purchase bill",  "0",   "4000",  "0"],  # missing in ours
], columns=["Date", "Vch Type", "Vch No", "Ref No", "Narration", "Debit", "Credit", "TDS"])

OUR_MAPPING = {
    "Date": {"source": "Date"}, "Voucher Type": {"source": "Vch Type"},
    "Voucher No": {"source": "Vch No"}, "Invoice Ref": {"source": "Ref No"},
    "Description": {"source": "Narration"}, "Debit": {"source": "Dr"},
    "Credit": {"source": "Cr"}, "TDS Amount": {"source": "TDS"},
}
THEIR_MAPPING = {
    "Date": {"source": "Date"}, "Voucher Type": {"source": "Vch Type"},
    "Voucher No": {"source": "Vch No"}, "Invoice Ref": {"source": "Ref No"},
    "Description": {"source": "Narration"}, "Debit": {"source": "Debit"},
    "Credit": {"source": "Credit"}, "TDS Amount": {"source": "TDS"},
}

_COLS = ["Date", "Vch Type", "Vch No", "Ref No", "Narration", "Dr", "Cr", "TDS"]
_MAP = {
    "Date": {"source": "Date"}, "Voucher Type": {"source": "Vch Type"},
    "Voucher No": {"source": "Vch No"}, "Invoice Ref": {"source": "Ref No"},
    "Description": {"source": "Narration"}, "Debit": {"source": "Dr"},
    "Credit": {"source": "Cr"}, "TDS Amount": {"source": "TDS"},
}


# ── Assertion harness ─────────────────────────────────────────────────────────
_failures = []

def check(name: str, ok: bool, detail: str = ""):
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"  {mark}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        _failures.append(name)


def main():
    print("=" * 70)
    print("LedgerBridge AI - Self-contained Core Test")
    print("=" * 70)

    # 1) Standardize
    print("\n[1/8] Standardizing...")
    ours = standardize(OUR_RAW, OUR_MAPPING)
    theirs = standardize(THEIR_RAW, THEIR_MAPPING)
    check("both ledgers standardized to 4 rows", len(ours) == 4 and len(theirs) == 4,
          f"got {len(ours)}, {len(theirs)}")

    # 2) Reconcile
    print("\n[2/8] Reconciling (contract)...")
    result = reconcile(ours, theirs)
    s = result.summary
    check("L1 matches == 1 (INV-1001)", s["matched_l1"] == 1, f"{s['matched_l1']}")
    check("L2 matches == 1 (INV-1015 timing)", s["matched_l2"] == 1, f"{s['matched_l2']}")
    check("amount mismatches == 1 (INV-1012)", s["amount_mismatches"] == 1, f"{s['amount_mismatches']}")
    check("missing in theirs == 1 (INV-1010)", s["missing_in_theirs"] == 1, f"{s['missing_in_theirs']}")
    check("missing in ours == 1 (INV-1099)", s["missing_in_ours"] == 1, f"{s['missing_in_ours']}")
    for key in ("amount_tolerance", "matched_tds", "suggested_tds_unverified",
                "tds_match_delta", "tds_ours", "tds_theirs", "tds_difference", "reconciled"):
        check(f"summary has '{key}'", key in s)
    check("residual ~ 0 (books self-consistent)", abs(s["residual"]) < 1.0, f"{s['residual']}")

    # matched-table contract
    check("result.matched exists with rows", not result.matched.empty, f"{len(result.matched)} rows")
    if not result.matched.empty:
        for col in ("Match Level", "Invoice Ref", "Note"):
            check(f"matched has '{col}' column", col in result.matched.columns)
        inv1015 = result.matched[result.matched["Invoice Ref"] == "INV1015"]
        check("INV-1015 present in matched as L2",
              len(inv1015) == 1 and inv1015.iloc[0]["Match Level"] == "L2")
    check("INV-1010 in missing_in_theirs",
          "INV1010" in set(result.missing_in_theirs.get("Invoice Ref", pd.Series(dtype=str))))
    check("INV-1099 in missing_in_ours",
          "INV1099" in set(result.missing_in_ours.get("Invoice Ref", pd.Series(dtype=str))))

    # 3) Report generation must not raise
    print("\n[3/8] Writing Excel report...")
    out_dir = Path(tempfile.mkdtemp(prefix="lb_test_"))
    try:
        rec = write_report(result, out_dir / "reconciliation_report.xlsx")
        write_standardized_ledger(result.our_ledger, out_dir / "our.xlsx", "Our Ledger")
        write_standardized_ledger(result.their_ledger, out_dir / "their.xlsx", "Their Ledger")
        check("write_report succeeded", Path(rec).exists())
    except Exception as e:
        check("write_report succeeded", False, repr(e))

    check("our_ledger 'Rec Code' populated (not blank)",
          (result.our_ledger["Rec Code"].astype(str) != "").all(),
          f"blank: {(result.our_ledger['Rec Code'].astype(str) == '').sum()}")
    check("our_ledger columns == canonical schema",
          list(result.our_ledger.columns) == CANONICAL_FIELDS,
          f"{list(result.our_ledger.columns)}")

    test_tds_inclusive()
    test_voucher_class_gate()
    test_no_ref_and_anomalies()
    test_tds_reclassification()
    test_invoice_ref_fallback()
    test_summary_row_filter()

    print("\n" + "=" * 70)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED - {', '.join(_failures)}")
        print("=" * 70)
        sys.exit(1)
    print("RESULT: ALL PASSED")
    print("=" * 70)


def test_tds_inclusive():
    print("\n[4/8] TDS-inclusive matching + balance integrity...")
    # Seller books full gross (10000); buyer books net (9000) + withholds TDS (1000).
    # Ref-matched -> confirmed MATCHED_INCL_TDS; the +1000 stays a reconciling item.
    seller = pd.DataFrame([
        ["2025-05-01", "Sales", "SV-7", "INV-7", "Sales bill", "10000", "0", "0"],
    ], columns=_COLS)
    buyer = pd.DataFrame([
        ["2025-08-01", "Purchase", "PV-7", "INV-7", "Purchase bill", "0", "9000", "1000"],  # posted 3 months later
    ], columns=_COLS)
    o = standardize(seller, _MAP); t = standardize(buyer, _MAP)
    r = reconcile(o, t)
    s = r.summary
    check("TDS: net-of-TDS invoice is MATCHED_INCL_TDS (not L1, not mismatch)",
          s["matched_tds"] == 1 and s["matched_l1"] == 0 and s["amount_mismatches"] == 0,
          f"tds={s['matched_tds']} l1={s['matched_l1']} am={s['amount_mismatches']}")
    check("TDS: big posting-date gap does not block a ref match",
          s["matched_l2"] == 0 and s["matched_tds"] == 1)  # matched as TDS regardless of 92-day gap
    check("TDS: match carries an explanatory note",
          (not r.matched.empty) and "TDS" in str(r.matched.iloc[0]["Note"]).upper(),
          f"{r.matched['Note'].tolist() if not r.matched.empty else 'empty'}")
    # Balance integrity: the withheld TDS is COUNTED, not erased, and residual ~ 0
    check("TDS: withheld TDS surfaced as reconciling delta (=1000)",
          abs(s["tds_match_delta"] - 1000.0) < 1e-6, f"{s['tds_match_delta']}")
    check("TDS: reconciling_item includes the TDS delta",
          abs(s["reconciling_item"] - 1000.0) < 1e-6, f"{s['reconciling_item']}")
    check("TDS: residual ~ 0 (balance still self-consistent)",
          abs(s["residual"]) < 1e-6, f"{s['residual']}")

    # Both sides carry TDS -> cannot self-prove -> SUGGESTED_TDS_UNVERIFIED (review)
    seller2 = pd.DataFrame([
        ["2025-05-01", "Sales", "SV-8", "INV-8", "Sales bill", "10000", "0", "500"],
    ], columns=_COLS)
    buyer2 = pd.DataFrame([
        ["2025-05-01", "Purchase", "PV-8", "INV-8", "Purchase bill", "0", "9000", "1000"],
    ], columns=_COLS)
    r2 = reconcile(standardize(seller2, _MAP), standardize(buyer2, _MAP))
    check("TDS: both-sides-TDS routed to SUGGESTED_TDS_UNVERIFIED (not auto-confirmed)",
          r2.summary["suggested_tds_unverified"] == 1 and r2.summary["matched_tds"] == 0,
          f"sug={r2.summary['suggested_tds_unverified']} tds={r2.summary['matched_tds']}")

    # A gap that is NOT explained by the booked TDS stays an amount mismatch
    seller3 = pd.DataFrame([
        ["2025-05-01", "Sales", "SV-9", "INV-9", "Sales bill", "10000", "0", "0"],
    ], columns=_COLS)
    buyer3 = pd.DataFrame([
        ["2025-05-01", "Purchase", "PV-9", "INV-9", "Purchase bill", "0", "8600", "1000"],  # gap 1400, TDS wouldn't zero it
    ], columns=_COLS)
    r3 = reconcile(standardize(seller3, _MAP), standardize(buyer3, _MAP))
    check("TDS: unexplained gap stays AMOUNT_MISMATCH, not a TDS match",
          r3.summary["amount_mismatches"] == 1 and r3.summary["matched_tds"] == 0,
          f"am={r3.summary['amount_mismatches']} tds={r3.summary['matched_tds']}")


def test_voucher_class_gate():
    print("\n[5/8] Voucher-class gate (no-ref)...")
    # A credit note and an invoice of EQUAL magnitude, no ref, same date: they
    # mirror numerically but must NOT be matched (different voucher class).
    seller = pd.DataFrame([
        ["2025-06-01", "Credit Note", "CN-1", "", "Credit note", "0", "5000", "0"],   # gross -5000
    ], columns=_COLS)
    buyer = pd.DataFrame([
        ["2025-06-01", "Purchase", "PV-1", "", "Purchase bill", "5000", "0", "0"],     # gross +5000
    ], columns=_COLS)
    r = reconcile(standardize(seller, _MAP), standardize(buyer, _MAP))
    check("CLASS: credit note vs invoice (equal magnitude, no ref) NOT matched",
          r.summary["matched_l3"] == 0
          and r.summary["missing_in_theirs"] == 1 and r.summary["missing_in_ours"] == 1,
          f"l3={r.summary['matched_l3']} mt={r.summary['missing_in_theirs']} mo={r.summary['missing_in_ours']}")

    # Positive control: same class (credit note <-> credit-note reversal), no ref -> matched
    seller2 = pd.DataFrame([
        ["2025-06-02", "Credit Note", "CN-2", "", "Credit note", "0", "6000", "0"],    # -6000
    ], columns=_COLS)
    buyer2 = pd.DataFrame([
        ["2025-06-02", "Credit Note", "RCRV-2", "", "P.Return", "6000", "0", "0"],       # +6000, RCRV -> CREDIT_NOTE
    ], columns=_COLS)
    r2 = reconcile(standardize(seller2, _MAP), standardize(buyer2, _MAP))
    check("CLASS: same-class no-ref mirror IS matched (L3)",
          r2.summary["matched_l3"] == 1, f"l3={r2.summary['matched_l3']}")


def test_no_ref_and_anomalies():
    print("\n[6/8] No-ref uniqueness, sign-reversed, duplicates, AM ceiling...")
    # Ambiguous no-ref group (2x2 identical journals) -> NOT matched, both missing.
    seller = pd.DataFrame([
        ["2025-03-01", "Journal", "JV-1", "", "adj", "500", "0", "0"],
        ["2025-03-01", "Journal", "JV-2", "", "adj", "500", "0", "0"],
    ], columns=_COLS)
    buyer = pd.DataFrame([
        ["2025-03-01", "Journal", "JV-3", "", "adj", "0", "500", "0"],
        ["2025-03-01", "Journal", "JV-4", "", "adj", "0", "500", "0"],
    ], columns=_COLS)
    r = reconcile(standardize(seller, _MAP), standardize(buyer, _MAP))
    check("NOREF: ambiguous 2x2 group not matched (0 L3, all missing)",
          r.summary["matched_l3"] == 0 and r.summary["missing_in_theirs"] == 2
          and r.summary["missing_in_ours"] == 2, f"l3={r.summary['matched_l3']}")

    # Unambiguous 1:1 no-ref mirror -> matched
    s2 = pd.DataFrame([["2025-04-01", "Journal", "JV-5", "", "adj", "750", "0", "0"]], columns=_COLS)
    b2 = pd.DataFrame([["2025-04-01", "Journal", "JV-6", "", "adj", "0", "750", "0"]], columns=_COLS)
    r2 = reconcile(standardize(s2, _MAP), standardize(b2, _MAP))
    check("NOREF: unambiguous 1:1 mirror matched (L3)", r2.summary["matched_l3"] == 1,
          f"l3={r2.summary['matched_l3']}")

    # Sign-reversed: same date + magnitude, SAME sign (posting error, e.g. payment
    # booked to the credit column) -> flagged, not matched, not missing.
    s3 = pd.DataFrame([["2025-06-01", "Receipt", "BR-12", "", "Receipt NEFT", "0", "500392.05", "0"]], columns=_COLS)
    b3 = pd.DataFrame([["2025-06-01", "Payment", "BP-12", "", "Payment NEFT", "0", "500392.05", "0"]], columns=_COLS)
    o3, t3 = standardize(s3, _MAP), standardize(b3, _MAP)
    check("SIGN: both rows standardize to the SAME sign (-500392.05)",
          float(o3.iloc[0]["Gross Amount"]) < 0 and float(t3.iloc[0]["Gross Amount"]) < 0)
    r3 = reconcile(o3, t3)
    check("SIGN: same-sign same-amount flagged SIGN_REVERSED (not missing)",
          r3.summary["sign_reversed"] == 1 and r3.summary["missing_in_theirs"] == 0,
          f"sr={r3.summary['sign_reversed']}")

    # Suspected duplicate (marker-based): extra -DUP row with no partner.
    s4 = pd.DataFrame([
        ["2025-08-10", "Receipt", "BR-37",     "", "Receipt NEFT",             "0", "257205.27", "0"],
        ["2025-08-12", "Receipt", "BR-37-DUP", "", "Receipt NEFT - duplicate", "0", "257205.27", "0"],
    ], columns=_COLS)
    b4 = pd.DataFrame([
        ["2025-08-10", "Payment", "BP-37", "", "Payment NEFT", "257205.27", "0", "0"],
    ], columns=_COLS)
    r4 = reconcile(standardize(s4, _MAP), standardize(b4, _MAP))
    check("DUP: marked-duplicate extra flagged SUSPECTED_DUPLICATE",
          r4.summary["suspected_duplicates"] == 1 and r4.summary["missing_in_theirs"] == 0,
          f"dup={r4.summary['suspected_duplicates']} mt={r4.summary['missing_in_theirs']}")

    # AM ceiling: same ref, wildly different amounts -> NOT paired, stays missing.
    s5 = pd.DataFrame([["2025-09-01", "Sales", "SV-X", "REF-X", "Sales bill", "60000", "0", "0"]], columns=_COLS)
    b5 = pd.DataFrame([["2025-09-01", "Journal", "JV-X", "REF-X", "Some journal", "0", "1000", "0"]], columns=_COLS)
    r5 = reconcile(standardize(s5, _MAP), standardize(b5, _MAP))
    check("CEILING: 60000-vs-1000 same-ref does NOT bind (0 mismatches, both missing)",
          r5.summary["amount_mismatches"] == 0 and r5.summary["missing_in_theirs"] == 1
          and r5.summary["missing_in_ours"] == 1, f"am={r5.summary['amount_mismatches']}")


def test_tds_reclassification():
    print("\n[7/8] TDS journal reclassification + journal-vs-journal status...")
    seller = pd.DataFrame([
        ["2025-07-01", "Sales",   "SV-100", "INV-100", "Sales bill",              "10000", "0", "0"],
        ["2025-07-01", "Journal", "JV-100", "TDSJV1",  "TDS receivable u/s 194Q", "0", "1000", "0"],
    ], columns=_COLS)
    buyer = pd.DataFrame([
        ["2025-07-01", "Purchase", "PV-100", "INV-100", "Purchase bill", "0", "10000", "0"],
    ], columns=_COLS)
    o = standardize(seller, _MAP); t = standardize(buyer, _MAP)
    r = reconcile(o, t)
    check("TDS-J: invoice matched at L1", r.summary["matched_l1"] == 1, f"{r.summary['matched_l1']}")
    check("TDS-J: TDS journal is missing in theirs", r.summary["missing_in_theirs"] == 1,
          f"{r.summary['missing_in_theirs']}")
    tds = classify_tds_entries(
        missing_in_theirs=r.missing_in_theirs, missing_in_ours=r.missing_in_ours,
        our_full_ledger=r.our_ledger, their_full_ledger=r.their_ledger)
    check("TDS-J: journal row flagged", len(tds.flagged_entries) == 1, f"{len(tds.flagged_entries)}")
    our_final, _ = apply_tds_reclassification(r.our_ledger, r.their_ledger, tds)
    tds_code = REC_CODES["TDS_ENTRY"]
    check("TDS-J: row reclassified MISSING -> TDS_ENTRY",
          int((our_final["Rec Code"] == tds_code).sum()) == 1)

    # journal-vs-journal aggregate status
    sj = pd.DataFrame([["2025-07-05", "Journal", "JV-1", "TDSR1", "TDS receivable u/s 194Q", "0", "1000", "0"]], columns=_COLS)
    bj = pd.DataFrame([["2025-07-07", "Journal", "JV-2", "TDSC1", "TDS remitted to portal (challan)", "900", "0", "0"]], columns=_COLS)
    rj = reconcile(standardize(sj, _MAP), standardize(bj, _MAP))
    tj = classify_tds_entries(missing_in_theirs=rj.missing_in_theirs, missing_in_ours=rj.missing_in_ours,
                              our_full_ledger=rj.our_ledger, their_full_ledger=rj.their_ledger)
    check("TDS-J: journal-vs-journal status == PARTIAL",
          tj.overall_status == "PARTIAL", f"{tj.overall_status}")


def test_invoice_ref_fallback():
    print("\n[8/8] Invoice Ref fallback to Voucher No when unmapped...")
    mapping_no_ref = {
        "Date": {"source": "Date"}, "Voucher Type": {"source": "Vch Type"},
        "Voucher No": {"source": "Vch No"}, "Description": {"source": "Narration"},
        "Debit": {"source": "Dr"}, "Credit": {"source": "Cr"}, "TDS Amount": {"source": "TDS"},
    }
    seller = pd.DataFrame([["2025-09-05", "Sales", "ALP/25-26/010", "", "Sales bill", "12000", "0", "0"]], columns=_COLS)
    buyer = pd.DataFrame([["2025-09-05", "Purchase", "ALP/25-26/010", "", "Purchase bill", "0", "12000", "0"]], columns=_COLS)
    o = standardize(seller, mapping_no_ref); t = standardize(buyer, mapping_no_ref)
    check("REF: Invoice Ref falls back to Voucher No value",
          o.iloc[0]["Invoice Ref"] == o.iloc[0]["Voucher No"] == "ALP2526010")
    r = reconcile(o, t)
    check("REF: shared voucher no (no explicit ref) still L1-matches", r.summary["matched_l1"] == 1,
          f"{r.summary['matched_l1']}")


def test_summary_row_filter():
    print("\n[9/9] Summary-row filter must NOT drop real transactions...")
    # Regression: a genuine credit note whose narration merely MENTIONS "total"
    # ("...MACHINES TOTAL 33 REQUIRED...") was being silently dropped because
    # 'total' was matched as a substring anywhere in a cell. It must be KEPT;
    # only true summary lines (exactly "TOTALS"/"Grand Total"/"Opening Balance")
    # and a vendor name containing "Total" must behave correctly.
    raw = pd.DataFrame([
        ["2026-03-05", "Credit Note", "CN-1", "25-26/CN-014",
         "TP CSMS ADDL MACHINES TOTAL 33 REQUIRED GST Vr. Receipts/ P.Return", "0", "804465", "0"],  # KEEP
        ["2026-03-06", "Sales", "SV-2", "INV-2", "Total Solutions Pvt Ltd - annual fee", "5000", "0", "0"],  # KEEP (vendor name)
        ["2026-03-07", "Sales", "SV-3", "INV-3", "Sub Total of prior lines", "1234", "0", "0"],  # KEEP (mentions sub total)
        ["", "", "", "", "TOTALS", "9999", "0", "0"],                       # DROP (exact summary)
        ["", "", "", "", "Grand Total", "8888", "0", "0"],                  # DROP
        ["", "", "", "", "*** OPENING BALANCE ***", "7777", "0", "0"],      # DROP
        ["", "", "", "", "Closing Balance", "6666", "0", "0"],              # DROP
    ], columns=_COLS)
    std = standardize(raw, _MAP)
    kept = set(std["Invoice Ref"])
    check("SUMMARY: credit note with 'TOTAL' in narration is KEPT",
          "2526CN014" in kept, f"kept refs={kept}")
    check("SUMMARY: 804465 amount survives standardize",
          bool((std["Gross Amount"].abs() == 804465).any()),
          f"grosses={std['Gross Amount'].tolist()}")
    check("SUMMARY: vendor name 'Total Solutions...' row KEPT", "INV2" in kept, f"{kept}")
    check("SUMMARY: 'Sub Total of prior lines' narration row KEPT", "INV3" in kept, f"{kept}")
    check("SUMMARY: genuine summary rows dropped (3 real txns kept of 7)",
          len(std) == 3, f"got {len(std)} rows: {std['Description'].tolist()}")


if __name__ == "__main__":
    main()
