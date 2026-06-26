"""
End-to-end test of the deterministic core (no Claude).

Uses hardcoded mappings that simulate what Claude would return, then runs:
ingest -> standardize -> reconcile

Verifies that the seeded mismatches are correctly identified.
"""

import pandas as pd
from ingest import load_ledger
from standardize import standardize
from reconcile import reconcile


# Hardcoded mappings (simulating what Claude would return for these test files)
OUR_MAPPING = {
    "Date":         {"source": "Posting Date",    "confidence": "high"},
    "Voucher Type": {"source": "Document Type",   "confidence": "high"},
    "Voucher No":   {"source": "Document Number", "confidence": "high"},
    "Invoice Ref":  {"source": "Reference",       "confidence": "high"},
    "Description":  {"source": "Narration",       "confidence": "high"},
    "Gross Amount": {"source": "Amount",          "confidence": "high"},
    "TDS Amount":   {"source": "TDS",             "confidence": "high"},
}

THEIR_MAPPING = {
    "Date":         {"source": "Date",         "confidence": "high"},
    "Voucher Type": {"source": "Vch Type",     "confidence": "high"},
    "Voucher No":   {"source": "Vch No.",      "confidence": "high"},
    "Invoice Ref":  {"source": "Ref. No.",     "confidence": "high"},
    "Description":  {"source": "Particulars",  "confidence": "high"},
    # Their ledger has Debit/Credit split — special combined mapping
    "Gross Amount": {"source": ["Debit", "Credit"], "combine": "debit_credit", "confidence": "high"},
    "TDS Amount":   {"source": "TDS Amt",      "confidence": "high"},
}


def main():
    print("=" * 70)
    print("LedgerBridge AI — End-to-end Core Test")
    print("=" * 70)

    # 1) Ingest
    print("\n[1/3] Ingesting files...")
    ours_raw = load_ledger("samples/our_ledger_sap.xlsx", sheet_name="Vendor Ledger")
    theirs_raw = load_ledger("samples/their_ledger_tally.xlsx", sheet_name="Ledger")
    print(f"  Our raw:   {len(ours_raw)} rows, columns: {list(ours_raw.columns)}")
    print(f"  Their raw: {len(theirs_raw)} rows, columns: {list(theirs_raw.columns)}")

    # 2) Standardize
    print("\n[2/3] Standardizing to canonical schema...")
    ours_std = standardize(ours_raw, OUR_MAPPING)
    theirs_std = standardize(theirs_raw, THEIR_MAPPING)
    print(f"  Our standardized:   {len(ours_std)} rows")
    print(f"  Their standardized: {len(theirs_std)} rows")
    print("\n  Sample of our standardized ledger:")
    print(ours_std.head(3).to_string())

    # Note: their ledger uses Debit/Credit semantics — Debit increases their AR (which is our AP).
    # We need their amounts to have the same sign as ours for matching.
    # In their book: Debit positive = invoice they raised (we owe them) -> we record this as positive too.
    # In their book: Credit positive = receipt from us -> we record as negative (payment).
    # So we need to FLIP the sign of their Gross Amount because (Debit - Credit) gives the wrong sign.
    # Actually: Debit - Credit:
    #   - Invoice row: Debit=15000, Credit=0 -> Gross = +15000  ✓ matches our +15000
    #   - Receipt row: Debit=0, Credit=50000 -> Gross = -50000  ✓ matches our -50000
    # So actually the signs already align! Good.

    # 3) Reconcile
    print("\n[3/3] Running reconciliation engine...")
    result = reconcile(
        ours_std,
        theirs_std,
        opening_balance_ours=0,
        opening_balance_theirs=0,
    )

    s = result.summary
    print(f"\n  --- Match Summary ---")
    print(f"  L1 (Date+Ref+Amount):  {s['matched_l1']}")
    print(f"  L2 (Ref+Amount):       {s['matched_l2']}  (timing differences)")
    print(f"  L3 (Date+Amount):      {s['matched_l3']}  (review recommended)")
    print(f"  Amount mismatches:     {s['amount_mismatches']}")
    print(f"  Missing in theirs:     {s['missing_in_theirs']}  (in our books only)")
    print(f"  Missing in ours:       {s['missing_in_ours']}    (in their books only)")

    print(f"\n  --- Balance Walk ---")
    print(f"  Sum of our transactions:    ₹{s['sum_our_transactions']:>15,.2f}")
    print(f"  Sum of their transactions:  ₹{s['sum_their_transactions']:>15,.2f}")
    print(f"  Difference:                 ₹{s['difference']:>15,.2f}")
    print(f"  Reconciling items (sum of one-sided): ₹{s['reconciling_item']:>15,.2f}")
    print(f"  Residual:                   ₹{s['residual']:>15,.2f}")
    print(f"  Reconciled (within tolerance): {'✓ YES' if s['reconciled'] else '✗ NO'}")

    print(f"\n  --- Validation against seeded mismatches ---")
    # Verify INV-1010 is in missing_in_theirs
    if "Invoice Ref" in result.missing_in_theirs.columns:
        inv_1010 = result.missing_in_theirs[result.missing_in_theirs["Invoice Ref"] == "INV1010"]
        print(f"  INV-1010 found in 'missing in theirs': {'✓' if len(inv_1010) == 1 else '✗ FAILED'}")

    # Verify INV-1099 is in missing_in_ours
    if "Invoice Ref" in result.missing_in_ours.columns:
        inv_1099 = result.missing_in_ours[result.missing_in_ours["Invoice Ref"] == "INV1099"]
        print(f"  INV-1099 found in 'missing in ours':   {'✓' if len(inv_1099) == 1 else '✗ FAILED'}")

    # Verify INV-1012 is in amount mismatches
    if not result.amount_mismatches.empty and "Invoice Ref" in result.amount_mismatches.columns:
        inv_1012 = result.amount_mismatches[result.amount_mismatches["Invoice Ref"] == "INV1012"]
        print(f"  INV-1012 found in amount mismatches:   {'✓' if len(inv_1012) == 1 else '✗ FAILED'}")
    else:
        print(f"  INV-1012 found in amount mismatches:   ✗ FAILED (amount_mismatches table empty)")

    # Verify INV-1015 is matched as L2 (timing)
    if not result.matched.empty and "Invoice Ref" in result.matched.columns:
        inv_1015 = result.matched[result.matched["Invoice Ref"] == "INV1015"]
        is_l2 = len(inv_1015) == 1 and inv_1015.iloc[0]["Match Level"] == "L2"
        print(f"  INV-1015 matched as L2 (timing):        {'✓' if is_l2 else '✗ FAILED'}")
    else:
        print(f"  INV-1015 matched as L2 (timing):        ✗ FAILED (matched table empty)")

    # Verify both PAY-004 and PAY-005 matched (duplicate handling)
    if not result.matched.empty and "Invoice Ref" in result.matched.columns:
        pays = result.matched[result.matched["Invoice Ref"].isin(["PAY004", "PAY005"])]
        print(f"  PAY-004 + PAY-005 both matched (dupes): {'✓' if len(pays) == 2 else '✗ FAILED (got ' + str(len(pays)) + ')'}")

    # Print full matched table for inspection
    print(f"\n  --- All Matched Records ---")
    if not result.matched.empty:
        print(result.matched.to_string())

    if not result.amount_mismatches.empty:
        print(f"\n  --- Amount Mismatches ---")
        print(result.amount_mismatches.to_string())

    if not result.missing_in_theirs.empty:
        print(f"\n  --- Missing in Theirs (rows in our books only) ---")
        print(result.missing_in_theirs[["Date", "Invoice Ref", "Description", "Gross Amount"]].to_string())

    if not result.missing_in_ours.empty:
        print(f"\n  --- Missing in Ours (rows in their books only) ---")
        print(result.missing_in_ours[["Date", "Invoice Ref", "Description", "Gross Amount"]].to_string())

    print("\n" + "=" * 70)
    print("Test complete.")
    print("=" * 70)

    # ---------- Generate Excel reports ----------
    print("\n[4/3] Generating Excel reports...")
    from report import write_report, write_standardized_ledger

    out_dir = "outputs"
    rec_path = write_report(result, f"{out_dir}/reconciliation_report.xlsx",
                            ai_insights="(AI insights would appear here in production — generated by Claude on the final reconciliation summary.)")
    our_path = write_standardized_ledger(result.our_ledger, f"{out_dir}/standardized_our_books.xlsx", "Our Standardized Ledger")
    their_path = write_standardized_ledger(result.their_ledger, f"{out_dir}/standardized_their_books.xlsx", "Their Standardized Ledger")

    print(f"  ✓ {rec_path}")
    print(f"  ✓ {our_path}")
    print(f"  ✓ {their_path}")


if __name__ == "__main__":
    main()
