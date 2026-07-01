"""
Self-contained end-to-end test of the deterministic core (no Claude, no sample files).

Builds two synthetic raw ledgers in memory (one party's book and the
counterparty's book), then runs the real pipeline:

    standardize -> reconcile -> write_report

and asserts the seeded matches / mismatches / missing rows are classified
correctly. This locks the contract between reconcile.py and report.py so the
kind of drift that broke the app (renamed summary keys, a dropped `matched`
table) fails loudly here instead of at runtime.

Gross Amount is Debit - Credit uniformly on both ledgers (no buyer/seller
role). Because the two parties are double-entry counterparties, the SAME
transaction lands with OPPOSITE signs on the two ledgers — every fixture
below reflects that: "ours" and "theirs" raw Debit/Credit values are each
party's own real books, and a genuinely matching pair has ours_amt ~=
-theirs_amt, not ours_amt == theirs_amt.

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


# ── Synthetic raw ledgers ─────────────────────────────────────────────────────
# Our books: invoices recorded in Debit, receipts in Credit.
OUR_RAW = pd.DataFrame([
    # Date,        VchType,  VchNo,  Ref,        Narration,       Dr,      Cr,   TDS
    ["2025-01-05", "Sales",  "SV-1", "INV-1001", "Sales bill",   "10000", "0",  "0"],  # L1 match
    ["2025-01-10", "Sales",  "SV-2", "INV-1015", "Sales bill",   "5000",  "0",  "0"],  # L2 (timing)
    ["2025-01-12", "Sales",  "SV-3", "INV-1012", "Sales bill",   "8000",  "0",  "0"],  # amount mismatch
    ["2025-01-15", "Sales",  "SV-4", "INV-1010", "Sales bill",   "3000",  "0",  "0"],  # missing in theirs
], columns=["Date", "Vch Type", "Vch No", "Ref No", "Narration", "Dr", "Cr", "TDS"])

# Their books: the same transactions from the counterparty's side — invoices
# land in Credit, payments in Debit (their own books, own convention).
THEIR_RAW = pd.DataFrame([
    # Date,        VchType,     VchNo,  Ref,        Narration,        Debit, Credit,  TDS
    ["2025-01-05", "Purchase",  "BP-1", "INV-1001", "Purchase bill",  "0",   "10000", "0"],  # L1 match
    ["2025-01-20", "Purchase",  "BP-2", "INV-1015", "Purchase bill",  "0",   "5000",  "0"],  # L2 (dates differ 10d)
    ["2025-01-12", "Purchase",  "BP-3", "INV-1012", "Purchase bill",  "0",   "7000",  "0"],  # amount mismatch (7000 vs 8000)
    ["2025-01-18", "Purchase",  "BP-4", "INV-1099", "Purchase bill",  "0",   "4000",  "0"],  # missing in ours
], columns=["Date", "Vch Type", "Vch No", "Ref No", "Narration", "Debit", "Credit", "TDS"])

OUR_MAPPING = {
    "Date":         {"source": "Date"},
    "Voucher Type": {"source": "Vch Type"},
    "Voucher No":   {"source": "Vch No"},
    "Invoice Ref":  {"source": "Ref No"},
    "Description":  {"source": "Narration"},
    "Debit":        {"source": "Dr"},
    "Credit":       {"source": "Cr"},
    "TDS Amount":   {"source": "TDS"},
}

THEIR_MAPPING = {
    "Date":         {"source": "Date"},
    "Voucher Type": {"source": "Vch Type"},
    "Voucher No":   {"source": "Vch No"},
    "Invoice Ref":  {"source": "Ref No"},
    "Description":  {"source": "Narration"},
    "Debit":        {"source": "Debit"},
    "Credit":       {"source": "Credit"},
    "TDS Amount":   {"source": "TDS"},
}


# ── Tiny assertion harness ────────────────────────────────────────────────────
_failures = []

def check(name: str, ok: bool, detail: str = ""):
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"  {mark}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        _failures.append(name)


def main():
    print("=" * 70)
    print("LedgerBridge AI — Self-contained Core Test")
    print("=" * 70)

    # 1) Standardize (Gross Amount = Debit - Credit, uniformly, no role)
    print("\n[1/3] Standardizing...")
    ours = standardize(OUR_RAW, OUR_MAPPING)
    theirs = standardize(THEIR_RAW, THEIR_MAPPING)
    print(f"  Our standardized:   {len(ours)} rows")
    print(f"  Their standardized: {len(theirs)} rows")
    check("both ledgers standardized to 4 rows", len(ours) == 4 and len(theirs) == 4,
          f"got {len(ours)}, {len(theirs)}")
    check("their INV-1001 is the mirror-negation of ours (10000 vs -10000)",
          abs(float(theirs.iloc[0]["Gross Amount"]) - (-10000.0)) < 1e-6,
          f"got {float(theirs.iloc[0]['Gross Amount'])}")

    # 2) Reconcile
    print("\n[2/3] Reconciling...")
    result = reconcile(ours, theirs, opening_balance_ours=0, opening_balance_theirs=0)
    s = result.summary

    check("L1 matches == 1 (INV-1001)", s["matched_l1"] == 1, f"got {s['matched_l1']}")
    check("L2 matches == 1 (INV-1015 timing)", s["matched_l2"] == 1, f"got {s['matched_l2']}")
    check("amount mismatches == 1 (INV-1012)", s["amount_mismatches"] == 1, f"got {s['amount_mismatches']}")
    check("missing in theirs == 1 (INV-1010)", s["missing_in_theirs"] == 1, f"got {s['missing_in_theirs']}")
    check("missing in ours == 1 (INV-1099)", s["missing_in_ours"] == 1, f"got {s['missing_in_ours']}")

    # summary contract keys that report.py / insights.py depend on
    for key in ("amount_tolerance", "tds_ours", "tds_theirs", "tds_difference", "reconciled"):
        check(f"summary has '{key}'", key in s)

    # matched table contract (report.py / this test read these columns)
    check("result.matched exists with rows", not result.matched.empty, f"{len(result.matched)} rows")
    if not result.matched.empty:
        for col in ("Match Level", "Invoice Ref"):
            check(f"matched has '{col}' column", col in result.matched.columns)
        inv1015 = result.matched[result.matched["Invoice Ref"] == "INV1015"]
        check("INV-1015 present in matched as L2",
              len(inv1015) == 1 and inv1015.iloc[0]["Match Level"] == "L2")

    # exception tables carry the seeded refs (clean_voucher strips '-')
    check("INV-1010 in missing_in_theirs",
          "INV1010" in set(result.missing_in_theirs.get("Invoice Ref", pd.Series(dtype=str))))
    check("INV-1099 in missing_in_ours",
          "INV1099" in set(result.missing_in_ours.get("Invoice Ref", pd.Series(dtype=str))))
    check("INV-1012 in amount_mismatches",
          not result.amount_mismatches.empty
          and "INV1012" in set(result.amount_mismatches.get("Invoice Ref_ours", pd.Series(dtype=str))))

    # 3) Report generation must not raise (this is where the app used to crash)
    print("\n[3/3] Writing Excel report...")
    out_dir = Path(tempfile.mkdtemp(prefix="lb_test_"))
    try:
        rec = write_report(result, out_dir / "reconciliation_report.xlsx")
        write_standardized_ledger(result.our_ledger, out_dir / "our.xlsx", "Our Ledger")
        write_standardized_ledger(result.their_ledger, out_dir / "their.xlsx", "Their Ledger")
        check("write_report succeeded", Path(rec).exists())
    except Exception as e:
        check("write_report succeeded", False, repr(e))

    # exported ledger must publish the canonical Rec Code + drop internal cols
    check("our_ledger 'Rec Code' populated (not blank)",
          (result.our_ledger["Rec Code"].astype(str) != "").all(),
          f"blank rows: {(result.our_ledger['Rec Code'].astype(str) == '').sum()}")
    check("our_ledger has no internal _ columns",
          not any(c.startswith("_") for c in result.our_ledger.columns),
          f"cols: {list(result.our_ledger.columns)}")
    check("our_ledger columns == canonical schema",
          list(result.our_ledger.columns) == CANONICAL_FIELDS,
          f"cols: {list(result.our_ledger.columns)}")

    # 4) Correctness-tightening edge cases (item 2) + mirror-sign redesign
    test_edge_cases()

    # 5) Rec Code sync + TDS reclassification wiring (bugs A & B)
    test_tds_reclassification()

    # 6) Invoice Ref fallback to Voucher No when unmapped (Batch 1 hygiene fix)
    test_invoice_ref_fallback()

    # 7) Needs Review tier: variance / sign-reversed / suspected duplicate / AM ceiling
    test_needs_review_tier()

    print("\n" + "=" * 70)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED — {', '.join(_failures)}")
        print("=" * 70)
        sys.exit(1)
    print("RESULT: ALL PASSED")
    print("=" * 70)


# ── Edge cases for the item-2 correctness fixes + mirror-sign redesign ────────
_EDGE_MAPPING_SELLER = {
    "Date": {"source": "Date"}, "Voucher Type": {"source": "Vch Type"},
    "Voucher No": {"source": "Vch No"}, "Invoice Ref": {"source": "Ref No"},
    "Description": {"source": "Narration"}, "Debit": {"source": "Dr"},
    "Credit": {"source": "Cr"}, "TDS Amount": {"source": "TDS"},
}
_EDGE_MAPPING_BUYER = {
    "Date": {"source": "Date"}, "Voucher Type": {"source": "Vch Type"},
    "Voucher No": {"source": "Vch No"}, "Invoice Ref": {"source": "Ref No"},
    "Description": {"source": "Narration"}, "Debit": {"source": "Dr"},
    "Credit": {"source": "Cr"}, "TDS Amount": {"source": "TDS"},
}
_EDGE_COLS = ["Date", "Vch Type", "Vch No", "Ref No", "Narration", "Dr", "Cr", "TDS"]


def test_edge_cases():
    print("\n[4/4] Edge cases (L3 uniqueness, AM closest-pair, TDS surfaced as mismatch)...")

    # ---- (A) Amount-mismatch pairing must join by CLOSEST-TO-CANCELLING
    # amount (mirror-sign), not row order. Duplicated ref INV-9 on both sides;
    # naive first-row pairing would join 100<->-205 and 200<->-105 (huge
    # diffs). Closest-to-cancelling joins 100<->-105, 200<->-205.
    seller_am = pd.DataFrame([
        ["2025-02-01", "Sales", "SV-9a", "INV-9", "bill", "100", "0", "0"],
        ["2025-02-01", "Sales", "SV-9b", "INV-9", "bill", "200", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_am = pd.DataFrame([
        ["2025-02-01", "Purchase", "BP-9a", "INV-9", "bill", "0", "205", "0"],
        ["2025-02-01", "Purchase", "BP-9b", "INV-9", "bill", "0", "105", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_am, _EDGE_MAPPING_SELLER)
    t = standardize(buyer_am, _EDGE_MAPPING_BUYER)
    r = reconcile(o, t)
    am = r.amount_mismatches
    check("AM: both duplicate-ref rows classified as mismatch", len(am) == 2, f"got {len(am)}")
    if len(am) == 2:
        pair = dict(zip(am["Gross Amount_ours"], am["Gross Amount_theirs"]))
        check("AM: 100 paired with closest-to-cancelling -105", pair.get(100.0) == -105.0, f"{pair}")
        check("AM: 200 paired with closest-to-cancelling -205", pair.get(200.0) == -205.0, f"{pair}")
        check("AM: every pairing within 5 units",
              am["Difference"].abs().max() <= 5.0, f"max diff {am['Difference'].abs().max()}")

    # ---- (B) L3 must NOT pair ambiguous same-date/amount groups (no ref).
    # 2 vs 2 identical, ref-less rows → ambiguous → left as missing, not matched.
    seller_l3 = pd.DataFrame([
        ["2025-03-01", "Journal", "JV-1", "", "adj", "500", "0", "0"],
        ["2025-03-01", "Journal", "JV-2", "", "adj", "500", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_l3 = pd.DataFrame([
        ["2025-03-01", "Journal", "JV-3", "", "adj", "0", "500", "0"],
        ["2025-03-01", "Journal", "JV-4", "", "adj", "0", "500", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_l3, _EDGE_MAPPING_SELLER)
    t = standardize(buyer_l3, _EDGE_MAPPING_BUYER)
    r = reconcile(o, t)
    check("L3: ambiguous 2x2 group not matched (0 L3)", r.summary["matched_l3"] == 0,
          f"got {r.summary['matched_l3']}")
    check("L3: ambiguous rows fall through to missing",
          r.summary["missing_in_theirs"] == 2 and r.summary["missing_in_ours"] == 2,
          f"theirs={r.summary['missing_in_theirs']} ours={r.summary['missing_in_ours']}")

    # ---- (B2) L3 SHOULD pair an unambiguous 1:1 ref-less row on date+amount
    # (mirror-sign: 750 on our side cancels with -750 on theirs).
    seller_l3b = pd.DataFrame([
        ["2025-04-01", "Journal", "JV-5", "", "adj", "750", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_l3b = pd.DataFrame([
        ["2025-04-01", "Journal", "JV-6", "", "adj", "0", "750", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_l3b, _EDGE_MAPPING_SELLER)
    t = standardize(buyer_l3b, _EDGE_MAPPING_BUYER)
    r = reconcile(o, t)
    check("L3: unambiguous 1:1 ref-less row matched", r.summary["matched_l3"] == 1,
          f"got {r.summary['matched_l3']}")

    # ---- (C) TDS gross-up heuristic REMOVED (client directive: no role,
    # no special-casing). A buyer's net-of-TDS invoice row must NOT be
    # silently inflated to match the seller's gross figure — it now surfaces
    # as a visible AMOUNT_MISMATCH whose Difference equals the TDS withheld,
    # explainable via the TDS Reconciliation sheet instead of being hidden.
    seller_tds = pd.DataFrame([
        ["2025-05-01", "Sales", "SV-7", "INV-7", "bill", "10000", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_tds = pd.DataFrame([
        ["2025-05-01", "Purchase", "BP-7", "INV-7", "bill", "0", "9000", "1000"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_tds, _EDGE_MAPPING_SELLER)
    t = standardize(buyer_tds, _EDGE_MAPPING_BUYER)
    check("TDS: buyer invoice NOT grossed up, stays net -9000",
          abs(float(t.iloc[0]["Gross Amount"]) - (-9000.0)) < 1e-6,
          f"got {float(t.iloc[0]['Gross Amount'])}")
    r = reconcile(o, t)
    check("TDS: net-of-TDS invoice surfaces as AMOUNT_MISMATCH, not a clean match",
          r.summary["matched_l1"] == 0 and r.summary["amount_mismatches"] == 1,
          f"l1={r.summary['matched_l1']} am={r.summary['amount_mismatches']}")
    check("TDS: mismatch Difference equals the TDS withheld (1000)",
          not r.amount_mismatches.empty
          and abs(float(r.amount_mismatches.iloc[0]["Difference"]) - 1000.0) < 1e-6,
          f"{r.amount_mismatches.to_dict('records') if not r.amount_mismatches.empty else 'empty'}")

    # ---- (D) Summary rows (TOTALS / sub total) must be dropped by standardize,
    # while genuine transactions survive.
    with_totals = pd.DataFrame([
        ["2025-08-01", "Sales", "SV-1", "INV-D1", "Sales bill", "1000", "0", "0"],  # keep
        ["",           "",      "",     "",       "TOTALS",     "9999", "0", "0"],  # drop
        ["2025-08-02", "Sales", "SV-2", "INV-D2", "Sub Total",  "8888", "0", "0"],  # drop
        ["2025-08-03", "Sales", "SV-3", "INV-D3", "Sales bill", "2000", "0", "0"],  # keep
    ], columns=_EDGE_COLS)
    std = standardize(with_totals, _EDGE_MAPPING_SELLER)
    check("Summary rows dropped, real rows kept", len(std) == 2,
          f"got {len(std)} rows: {std['Description'].tolist()}")

    # ---- (C2) Gross Amount is always Debit - Credit, with no special-casing
    # by row type or TDS Amount — a plain payment row proves TDS never leaks
    # into the Gross Amount computation.
    buyer_pay = pd.DataFrame([
        ["2025-06-01", "Payment", "BP-8", "INV-8", "pmt", "5000", "0", "500"],
    ], columns=_EDGE_COLS)
    t = standardize(buyer_pay, _EDGE_MAPPING_BUYER)
    check("Gross Amount = Debit - Credit regardless of TDS Amount (5000, TDS=500 ignored)",
          abs(float(t.iloc[0]["Gross Amount"]) - 5000.0) < 1e-6,
          f"got {float(t.iloc[0]['Gross Amount'])}")


def test_tds_reclassification():
    print("\n[5/5] Rec Code sync + TDS reclassification (bugs A & B)...")
    # Seller books a sale (matched) plus a separate TDS-receivable journal row
    # that has no counterpart on the buyer side (structurally different booking).
    seller = pd.DataFrame([
        ["2025-07-01", "Sales",   "SV-100", "INV-100", "Sales bill",              "10000", "0", "0"],
        ["2025-07-01", "Journal", "JV-100", "TDSJV1",  "TDS receivable u/s 194Q", "0", "1000", "0"],
    ], columns=_EDGE_COLS)
    buyer = pd.DataFrame([
        ["2025-07-01", "Purchase", "BP-100", "INV-100", "Purchase bill", "0", "10000", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller, _EDGE_MAPPING_SELLER)
    t = standardize(buyer, _EDGE_MAPPING_BUYER)
    r = reconcile(o, t)

    # The invoice matches; the TDS journal row is missing in theirs.
    check("TDS: invoice matched at L1", r.summary["matched_l1"] == 1, f"{r.summary['matched_l1']}")
    check("TDS: journal row is missing in theirs", r.summary["missing_in_theirs"] == 1,
          f"{r.summary['missing_in_theirs']}")

    tds = classify_tds_entries(
        missing_in_theirs=r.missing_in_theirs,
        missing_in_ours=r.missing_in_ours,
        our_full_ledger=r.our_ledger,
        their_full_ledger=r.their_ledger,
    )
    check("TDS: journal row flagged", len(tds.flagged_entries) == 1, f"{len(tds.flagged_entries)}")

    our_final, their_final = apply_tds_reclassification(r.our_ledger, r.their_ledger, tds)
    tds_code = REC_CODES["TDS_ENTRY"]
    n_reclassified = int((our_final["Rec Code"] == tds_code).sum())
    check("TDS: row reclassified MISSING -> TDS_ENTRY in Rec Code", n_reclassified == 1,
          f"got {n_reclassified}")
    check("TDS: reclassified row has explanatory Notes",
          our_final.loc[our_final["Rec Code"] == tds_code, "Notes"].astype(str).str.len().gt(0).all()
          if n_reclassified else False)

    # ---- journal-vs-journal overall status (both sides book TDS as journals).
    # Seller: per-invoice TDS receivable journal (1000). Buyer: monthly challan
    # journal (900). Neither has a TDS column → status must be PARTIAL, gap 100.
    seller_jj = pd.DataFrame([
        ["2025-07-05", "Journal", "JV-1", "TDSR1", "TDS receivable u/s 194Q",           "0", "1000", "0"],
    ], columns=_EDGE_COLS)
    buyer_jj = pd.DataFrame([
        ["2025-07-07", "Journal", "JV-2", "TDSC1", "TDS remitted to portal (challan)",  "900", "0", "0"],
    ], columns=_EDGE_COLS)
    o2 = standardize(seller_jj, _EDGE_MAPPING_SELLER)
    t2 = standardize(buyer_jj, _EDGE_MAPPING_BUYER)
    r2 = reconcile(o2, t2)
    tds2 = classify_tds_entries(
        missing_in_theirs=r2.missing_in_theirs, missing_in_ours=r2.missing_in_ours,
        our_full_ledger=r2.our_ledger, their_full_ledger=r2.their_ledger,
    )
    check("TDS journal-vs-journal both sides flagged",
          len(tds2.flagged_entries) == 2, f"{len(tds2.flagged_entries)}")
    check("TDS journal-vs-journal status == PARTIAL",
          tds2.overall_status == "PARTIAL", f"{tds2.overall_status}: {tds2.status_message}")


def test_invoice_ref_fallback():
    print("\n[6/6] Invoice Ref fallback to Voucher No when unmapped...")

    # No "Invoice Ref" source at all — only Voucher No, matching ledgers that
    # use the same voucher/document number as their shared cross-party ref.
    mapping_no_ref = {
        "Date": {"source": "Date"}, "Voucher Type": {"source": "Vch Type"},
        "Voucher No": {"source": "Vch No"}, "Description": {"source": "Narration"},
        "Debit": {"source": "Dr"}, "Credit": {"source": "Cr"}, "TDS Amount": {"source": "TDS"},
    }
    raw = pd.DataFrame([
        ["2025-09-01", "Sales", "ALP/25-26/001", "", "Sales bill", "5000", "0", "0"],
    ], columns=_EDGE_COLS)
    std = standardize(raw, mapping_no_ref)
    check("Invoice Ref falls back to Voucher No value",
          std.iloc[0]["Invoice Ref"] == std.iloc[0]["Voucher No"] == "ALP2526001",
          f"Invoice Ref={std.iloc[0]['Invoice Ref']!r} Voucher No={std.iloc[0]['Voucher No']!r}")

    # End-to-end: two ledgers sharing a voucher number as their only common
    # identifier, neither mapping Invoice Ref explicitly, must still L1-match.
    seller_raw = pd.DataFrame([
        ["2025-09-05", "Sales", "ALP/25-26/010", "", "Sales bill", "12000", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_raw = pd.DataFrame([
        ["2025-09-05", "Purchase", "ALP/25-26/010", "", "Purchase bill", "0", "12000", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_raw, mapping_no_ref)
    t = standardize(buyer_raw, mapping_no_ref)
    r = reconcile(o, t)
    check("Shared voucher no (no explicit Invoice Ref) still L1-matches",
          r.summary["matched_l1"] == 1, f"{r.summary['matched_l1']}")

    # When Voucher No genuinely differs between ledgers too, fallback must not
    # cause a false match — behavior stays identical to leaving Invoice Ref blank.
    seller_raw2 = pd.DataFrame([
        ["2025-09-06", "Sales", "SV-999", "", "Sales bill", "12000", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_raw2 = pd.DataFrame([
        ["2025-09-06", "Purchase", "BP-888", "", "Purchase bill", "0", "12000", "0"],
    ], columns=_EDGE_COLS)
    o2 = standardize(seller_raw2, mapping_no_ref)
    t2 = standardize(buyer_raw2, mapping_no_ref)
    r2 = reconcile(o2, t2)
    check("Differing voucher numbers: falls through to L3 (date+amount), not a false ref match",
          r2.summary["matched_l1"] == 0 and r2.summary["matched_l3"] == 1,
          f"l1={r2.summary['matched_l1']} l3={r2.summary['matched_l3']}")


def test_needs_review_tier():
    print("\n[7/7] Needs Review tier (variance / sign-reversed / duplicate / AM ceiling)...")
    from config import REC_CODES as RC

    # ---- (A) Bank charge: mirror pair, same date, DIFFERENT refs, ~1.5% gap.
    # Must be rescued from Missing into VARIANCE (a real ₹500-on-₹34k charge).
    seller = pd.DataFrame([
        ["2025-05-05", "Receipt", "RCT/005", "", "Receipt (NEFT)", "0", "34425.81", "0"],
    ], columns=_EDGE_COLS)
    buyer = pd.DataFrame([
        ["2025-05-05", "Payment", "PAY/005", "", "Payment (NEFT)", "34925.81", "0", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller, _EDGE_MAPPING_SELLER)
    t = standardize(buyer, _EDGE_MAPPING_BUYER)
    r = reconcile(o, t)
    check("VARIANCE: bank-charge pair rescued from missing (variance==1)",
          r.summary["variance"] == 1 and r.summary["missing_in_theirs"] == 0
          and r.summary["missing_in_ours"] == 0,
          f"var={r.summary['variance']} mt={r.summary['missing_in_theirs']} mo={r.summary['missing_in_ours']}")
    nr = r.needs_review
    var_rows = nr[nr["Rec Code"] == RC["VARIANCE"]]
    check("VARIANCE: gap shown ~Rs500 in needs_review",
          len(var_rows) == 1 and abs(float(var_rows.iloc[0]["Gap"]) - 500.0) < 1e-6,
          f"{var_rows[['Gap']].to_dict('records') if not var_rows.empty else 'none'}")

    # ---- (B) Sign-reversed: same date, same amount, SAME sign (posting error).
    # A payment booked to the credit column → same sign as the receipt.
    seller_sr = pd.DataFrame([
        ["2025-06-01", "Receipt", "RCT/012", "", "Receipt (NEFT)", "0", "500392.05", "0"],
    ], columns=_EDGE_COLS)
    buyer_sr = pd.DataFrame([
        ["2025-06-01", "Payment", "PAY/012", "", "Payment (NEFT)", "0", "500392.05", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_sr, _EDGE_MAPPING_SELLER)
    t = standardize(buyer_sr, _EDGE_MAPPING_BUYER)
    check("SIGN: both rows standardize to the SAME sign (-500392.05)",
          float(o.iloc[0]["Gross Amount"]) < 0 and float(t.iloc[0]["Gross Amount"]) < 0)
    r = reconcile(o, t)
    check("SIGN: same-sign same-amount pair flagged SIGN_REVERSED, not missing",
          r.summary["sign_reversed"] == 1 and r.summary["missing_in_theirs"] == 0
          and r.summary["missing_in_ours"] == 0,
          f"sr={r.summary['sign_reversed']} mt={r.summary['missing_in_theirs']} mo={r.summary['missing_in_ours']}")

    # ---- (C) Suspected duplicate: an extra marked-duplicate row with no partner.
    seller_dup = pd.DataFrame([
        ["2025-08-10", "Receipt", "RCT/037",     "", "Receipt (NEFT)",             "0", "257205.27", "0"],
        ["2025-08-12", "Receipt", "RCT/037-DUP", "", "Receipt (NEFT) - duplicate", "0", "257205.27", "0"],
    ], columns=_EDGE_COLS)
    buyer_dup = pd.DataFrame([
        ["2025-08-10", "Payment", "PAY/037", "", "Payment (NEFT)", "257205.27", "0", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_dup, _EDGE_MAPPING_SELLER)
    t = standardize(buyer_dup, _EDGE_MAPPING_BUYER)
    r = reconcile(o, t)
    # RCT/037 pairs with PAY/037 (mirror, same date); the -DUP copy is flagged.
    check("DUP: marked-duplicate extra flagged SUSPECTED_DUP (not missing)",
          r.summary["suspected_duplicates"] == 1 and r.summary["missing_in_theirs"] == 0,
          f"dup={r.summary['suspected_duplicates']} mt={r.summary['missing_in_theirs']}")

    # ---- (D) AM ceiling: same ref but wildly different amounts must NOT bind
    # (e.g. an invoice vs an unrelated same-ref journal). Stays missing.
    seller_c = pd.DataFrame([
        ["2025-09-01", "Sales", "REF-X", "REF-X", "Sales bill", "60000", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_c = pd.DataFrame([
        ["2025-09-01", "Journal", "REF-X", "REF-X", "Some journal", "0", "1000", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_c, _EDGE_MAPPING_SELLER)
    t = standardize(buyer_c, _EDGE_MAPPING_BUYER)
    r = reconcile(o, t)
    check("CEILING: 60000-vs-1000 same-ref does NOT bind (0 amount mismatches)",
          r.summary["amount_mismatches"] == 0
          and r.summary["missing_in_theirs"] == 1 and r.summary["missing_in_ours"] == 1,
          f"am={r.summary['amount_mismatches']} mt={r.summary['missing_in_theirs']} mo={r.summary['missing_in_ours']}")

    # ---- (E) Balance still reconciles across a mixed review-tier ledger.
    check("Balance reconciles with review-tier rows present", r.summary["reconciled"] is True
          or abs(r.summary["residual"]) < 1.0, f"residual={r.summary['residual']}")


if __name__ == "__main__":
    main()
