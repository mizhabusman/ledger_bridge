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

    print("\n" + "=" * 70)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED — {', '.join(_failures)}")
        print("=" * 70)
        sys.exit(1)
    print("RESULT: ALL PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
