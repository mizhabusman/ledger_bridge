# LedgerBridge AI

AI-powered ledger reconciliation tool. Upload two ledger files from any
accounting system, get back a reconciled report with exceptions flagged.

## What it does

Two finance teams have ledgers from different accounting systems (SAP, Tally,
Excel exports, etc.). Column names differ, dates are formatted differently,
amounts have currency symbols, voucher numbers have inconsistent spacing — but
the underlying transactions are the same.

LedgerBridge:

1. Reads both files (CSV or Excel)
2. Uses Claude to auto-detect which column is which
3. Lets the user confirm or edit the mapping (the accuracy gate)
4. Standardizes both files to a single 9-column canonical schema
5. Runs a 3-tier reconciliation:
   - **L1**: Date + Invoice Ref + Amount match
   - **L2**: Invoice Ref + Amount match (date differs → timing)
   - **L3**: Date + Amount match (no ref → review)
6. Flags exceptions: amount mismatches, missing entries, TDS differences
7. Generates a formatted Excel report with a closing-balance walk

## Architecture

```
┌─ Upload (Streamlit) ──────────────┐
│  ingest.py     File reading       │
│                Header detection   │
└─────────────┬─────────────────────┘
              ↓
┌─ AI Mapping (Claude API) ─────────┐
│  mapper.py     One call per file  │
│                Cached by schema   │
└─────────────┬─────────────────────┘
              ↓
┌─ Confirmation Screen ─────────────┐
│  app.py        User reviews,      │
│                edits, confirms    │
└─────────────┬─────────────────────┘
              ↓
┌─ Standardize (pure pandas) ───────┐
│  standardize.py   Clean dates,    │
│                   amounts, refs   │
└─────────────┬─────────────────────┘
              ↓
┌─ Reconcile (pure pandas) ─────────┐
│  reconcile.py     3-tier matching │
│                   Duplicate-safe  │
└─────────────┬─────────────────────┘
              ↓
┌─ Insights (Claude API, optional) ─┐
│  insights.py   One call total     │
└─────────────┬─────────────────────┘
              ↓
┌─ Excel Report ────────────────────┐
│  report.py     6 sheets, formatted│
└───────────────────────────────────┘
```

Claude is called only **2-3 times per reconciliation** (map ledger A, map
ledger B, optional insights). Everything else is deterministic pandas code,
so the actual reconciliation math is reproducible and exact.

## Setup

### 1. Install Python (3.10 or newer)

### 2. Clone or unzip this folder, then install dependencies

```bash
cd ledgerbridge
pip install -r requirements.txt
```

### 3. Add your Anthropic API key

Create a file called `.env` in this folder:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Get a key from https://console.anthropic.com/

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501) in your browser.

## How to use

1. **Upload page** — drop in both ledger files (CSV / XLSX).
2. **Mapping page** — Claude proposes which source column means what.
   Review the proposed mapping. The confidence badge (🟢🟡🔴) shows how sure
   Claude is. Change any dropdown if it's wrong. **This is the most
   important step — the reconciliation is only as accurate as the mapping.**
3. **Opening balances** — enter the opening balances (or leave at 0).
   Set the amount tolerance (default ₹1.00 covers rounding).
4. **Results** — view summary stats, AI insights, exception tables.
   Download 3 Excel files: the reconciliation report and the two
   standardized ledgers.

## The canonical 9-field schema

Every standardized ledger has exactly these columns:

| Field | Description |
|-------|-------------|
| Date | Transaction or posting date |
| Voucher Type | Sales/Purchase/Payment/Receipt/Journal (mainly Tally) |
| Voucher No | Internal voucher number (BP00001, JV001) |
| Invoice Ref | External invoice/PO number (the key match field) |
| Description | Free-text narration |
| Gross Amount | Signed: +ve for invoices/debits, -ve for payments/credits |
| TDS Amount | Tax withheld (used for netting, never matching) |
| Notes | Auto-filled with match explanation |
| Rec Code | Auto-filled: MATCHED_L1/L2/L3, AMOUNT_MISMATCH, MISSING_IN_OURS, MISSING_IN_THEIRS |

## How the matching engine works

For each row in Our ledger, in order:

- **Level 1**: Find a Their-row with same `Invoice Ref` + same `Date` +
  matching `Gross Amount` (within tolerance). Highest confidence.
- **Level 2**: Same `Invoice Ref` + matching `Gross Amount` but dates differ
  (within configurable window). Flagged as timing difference.
- **Amount Mismatch detection**: Before L3, if a Their-row has same
  `Invoice Ref` + same `Date` but different amount, it's flagged as
  `AMOUNT_MISMATCH` (the entry exists, only the value is wrong).
- **Level 3**: Match on `Date` + `Gross Amount` only (for rows with no
  invoice ref, e.g. some payment entries). Marked for review.

**Duplicate safety:** if both sides have two ₹5,000 entries on the same date,
they pair 1st↔1st and 2nd↔2nd. Each Their-row can only be claimed once.

## Project structure

```
ledgerbridge/
├── app.py              # Streamlit UI (the only file with UI code)
├── config.py           # Canonical schema, settings, Rec Codes
├── ingest.py           # File reading, header detection, encoding
├── mapper.py           # Claude column mapping + caching
├── standardize.py      # Cleaning + canonical conversion (pure pandas)
├── reconcile.py        # 3-tier matching engine (pure pandas)
├── insights.py         # AI insights for Summary sheet
├── report.py           # Excel report generation with formatting
├── make_samples.py     # Generates synthetic test ledgers
├── test_core.py        # End-to-end test (no Streamlit, no Claude)
├── requirements.txt
├── README.md
├── cache/              # Cached column mappings (auto-created)
├── outputs/            # Generated reports (auto-created)
└── samples/            # Synthetic test files (run make_samples.py)
```

## Testing

Run the deterministic core test (no API calls, no UI):

```bash
python make_samples.py     # Generates test ledgers
python test_core.py        # Runs full pipeline end-to-end
```

This proves the reconciliation engine works correctly on a known-good
test case with deliberate seeded mismatches (1 amount mismatch, 1 missing
each side, 1 timing difference, 2 duplicates).

## Adding support for a new ledger format

**No code changes needed.** Just run the tool — Claude will figure out the
mapping on first use, and the confirmed mapping is cached automatically so
the next time the same format is uploaded, no API call is needed.

## Costs

- Mapping: ~2000 input tokens + ~500 output per file. Two files = ~$0.05.
- Insights: ~1500 tokens total. ~$0.02.
- Repeat formats: $0 (mapping is cached).

Per reconciliation: well under ₹10 in API costs.

## Limitations / future work

- PDF and Tally XML imports not yet supported (Excel and CSV only).
- No multi-currency support yet — all amounts treated as a single currency.
- "To Be Entered" tab (draft journal entries for missing items) not yet built.
- No persistent database of reconciliation history.
- Single-user (no auth, no role-based access).
