"""
Self-contained end-to-end test of the deterministic core (no Claude, no sample files).

Builds two synthetic raw ledgers in memory (a seller's book and the counterparty
buyer's book), then runs the real pipeline:

    standardize (with role) -> reconcile -> write_report

and asserts the seeded matches / mismatches / missing rows are classified
correctly. This locks the contract between reconcile.py and report.py so the
kind of drift that broke the app (renamed summary keys, a dropped `matched`
table) fails loudly here instead of at runtime.

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
# Our books = SELLER view: invoices land in Debit (AR increases), receipts in Credit.
OUR_RAW = pd.DataFrame([
    # Date,        VchType,  VchNo,  Ref,        Narration,       Dr,      Cr,   TDS
    ["2025-01-05", "Sales",  "SV-1", "INV-1001", "Sales bill",   "10000", "0",  "0"],  # L1 match
    ["2025-01-10", "Sales",  "SV-2", "INV-1015", "Sales bill",   "5000",  "0",  "0"],  # L2 (timing)
    ["2025-01-12", "Sales",  "SV-3", "INV-1012", "Sales bill",   "8000",  "0",  "0"],  # amount mismatch
    ["2025-01-15", "Sales",  "SV-4", "INV-1010", "Sales bill",   "3000",  "0",  "0"],  # missing in theirs
], columns=["Date", "Vch Type", "Vch No", "Ref No", "Narration", "Dr", "Cr", "TDS"])

# Their books = BUYER view: invoices land in Credit (AP increases), payments in Debit.
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

    # 1) Standardize with roles (Gross Amount is computed from Debit/Credit + role)
    print("\n[1/3] Standardizing...")
    ours = standardize(OUR_RAW, OUR_MAPPING, role="seller")
    theirs = standardize(THEIR_RAW, THEIR_MAPPING, role="buyer")
    print(f"  Our standardized:   {len(ours)} rows")
    print(f"  Their standardized: {len(theirs)} rows")
    check("both ledgers standardized to 4 rows", len(ours) == 4 and len(theirs) == 4,
          f"got {len(ours)}, {len(theirs)}")

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

    # 4) Correctness-tightening edge cases (item 2)
    test_edge_cases()

    # 5) Rec Code sync + TDS reclassification wiring (bugs A & B)
    test_tds_reclassification()

    print("\n" + "=" * 70)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED — {', '.join(_failures)}")
        print("=" * 70)
        sys.exit(1)
    print("RESULT: ALL PASSED")
    print("=" * 70)


# ── Edge cases for the item-2 correctness fixes ───────────────────────────────
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
    print("\n[4/4] Edge cases (L3 uniqueness, AM closest-pair, TDS gross-up)...")

    # ---- (A) Amount-mismatch pairing must join by CLOSEST amount, not row order.
    # Duplicated ref INV-9 on both sides; naive first-row pairing would join
    # 100<->205 and 200<->105 (huge diffs). Closest-pair joins 100<->105, 200<->205.
    seller_am = pd.DataFrame([
        ["2025-02-01", "Sales", "SV-9a", "INV-9", "bill", "100", "0", "0"],
        ["2025-02-01", "Sales", "SV-9b", "INV-9", "bill", "200", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_am = pd.DataFrame([
        ["2025-02-01", "Purchase", "BP-9a", "INV-9", "bill", "0", "205", "0"],
        ["2025-02-01", "Purchase", "BP-9b", "INV-9", "bill", "0", "105", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_am, _EDGE_MAPPING_SELLER, role="seller")
    t = standardize(buyer_am, _EDGE_MAPPING_BUYER, role="buyer")
    r = reconcile(o, t)
    am = r.amount_mismatches
    check("AM: both duplicate-ref rows classified as mismatch", len(am) == 2, f"got {len(am)}")
    if len(am) == 2:
        pair = dict(zip(am["Gross Amount_ours"], am["Gross Amount_theirs"]))
        check("AM: 100 paired with closest 105", pair.get(100.0) == 105.0, f"{pair}")
        check("AM: 200 paired with closest 205", pair.get(200.0) == 205.0, f"{pair}")
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
    o = standardize(seller_l3, _EDGE_MAPPING_SELLER, role="seller")
    t = standardize(buyer_l3, _EDGE_MAPPING_BUYER, role="buyer")
    r = reconcile(o, t)
    check("L3: ambiguous 2x2 group not matched (0 L3)", r.summary["matched_l3"] == 0,
          f"got {r.summary['matched_l3']}")
    check("L3: ambiguous rows fall through to missing",
          r.summary["missing_in_theirs"] == 2 and r.summary["missing_in_ours"] == 2,
          f"theirs={r.summary['missing_in_theirs']} ours={r.summary['missing_in_ours']}")

    # ---- (B2) L3 SHOULD pair an unambiguous 1:1 ref-less row on date+amount.
    seller_l3b = pd.DataFrame([
        ["2025-04-01", "Journal", "JV-5", "", "adj", "750", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_l3b = pd.DataFrame([
        ["2025-04-01", "Journal", "JV-6", "", "adj", "0", "750", "0"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_l3b, _EDGE_MAPPING_SELLER, role="seller")
    t = standardize(buyer_l3b, _EDGE_MAPPING_BUYER, role="buyer")
    r = reconcile(o, t)
    check("L3: unambiguous 1:1 ref-less row matched", r.summary["matched_l3"] == 1,
          f"got {r.summary['matched_l3']}")

    # ---- (C) Buyer TDS gross-up: net credit + withheld TDS must equal seller gross.
    # Seller books gross 10000; buyer books net 9000 credit + 1000 TDS → gross-up 10000 → L1.
    seller_tds = pd.DataFrame([
        ["2025-05-01", "Sales", "SV-7", "INV-7", "bill", "10000", "0", "0"],
    ], columns=_EDGE_COLS)
    buyer_tds = pd.DataFrame([
        ["2025-05-01", "Purchase", "BP-7", "INV-7", "bill", "0", "9000", "1000"],
    ], columns=_EDGE_COLS)
    o = standardize(seller_tds, _EDGE_MAPPING_SELLER, role="seller")
    t = standardize(buyer_tds, _EDGE_MAPPING_BUYER, role="buyer")
    check("TDS: buyer invoice grossed up to 10000",
          abs(float(t.iloc[0]["Gross Amount"]) - 10000.0) < 1e-6,
          f"got {float(t.iloc[0]['Gross Amount'])}")
    r = reconcile(o, t)
    check("TDS: grossed-up invoice matches seller at L1", r.summary["matched_l1"] == 1,
          f"got {r.summary['matched_l1']}")

    # ---- (C2) A payment row (debit only) must NOT be grossed up by its TDS.
    buyer_pay = pd.DataFrame([
        ["2025-06-01", "Payment", "BP-8", "INV-8", "pmt", "5000", "0", "500"],
    ], columns=_EDGE_COLS)
    t = standardize(buyer_pay, _EDGE_MAPPING_BUYER, role="buyer")
    check("TDS: payment row not grossed up (stays -5000)",
          abs(float(t.iloc[0]["Gross Amount"]) - (-5000.0)) < 1e-6,
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
    o = standardize(seller, _EDGE_MAPPING_SELLER, role="seller")
    t = standardize(buyer, _EDGE_MAPPING_BUYER, role="buyer")
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


if __name__ == "__main__":
    main()
