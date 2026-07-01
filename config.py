"""
LedgerBridge AI — Configuration

Defines the canonical schema, settings, and constants used across the app.
"""

# The 9-field canonical schema for OUTPUT (final standardized ledger format).
# Note: Gross Amount is COMPUTED as Debit - Credit, not directly mapped.
CANONICAL_FIELDS = [
    "Date",
    "Voucher Type",
    "Voucher No",
    "Invoice Ref",
    "Description",
    "Gross Amount",     # computed = Debit - Credit
    "TDS Amount",
    "Notes",
    "Rec Code",
]

# Fields the USER MAPS (what Claude proposes and the user confirms).
# Debit and Credit are mapped as separate columns; Gross is derived.
MAPPABLE_FIELDS = [
    "Date",
    "Voucher Type",
    "Voucher No",
    "Invoice Ref",
    "Description",
    "Debit",            # raw debit column
    "Credit",           # raw credit column
    "TDS Amount",
]

# Required fields — without these, reconciliation cannot run
REQUIRED_FIELDS = ["Date"]

# Optional fields — may legitimately be missing
OPTIONAL_FIELDS = [
    "Voucher Type",
    "Voucher No",
    "Invoice Ref",
    "Description",
    "TDS Amount",
]

# Reconciliation codes written into the Rec Code column
REC_CODES = {
    "L1": "MATCHED_L1",
    "L2": "MATCHED_L2_TIMING",
    "L3": "MATCHED_L3_REVIEW",
    "AMOUNT_MISMATCH": "AMOUNT_MISMATCH",
    "VARIANCE": "MATCH_VARIANCE",             # paired, small variance (e.g. bank charge)
    "SIGN_REVERSED": "SIGN_REVERSED_REVIEW",  # same amount+date, same sign (posting error)
    "SUSPECTED_DUP": "SUSPECTED_DUPLICATE",   # duplicate of another same-side row
    "MISSING_OURS": "MISSING_IN_OURS",
    "MISSING_THEIRS": "MISSING_IN_THEIRS",
    "TDS_DIFF": "TDS_DIFFERENCE",
    "TDS_ENTRY": "TDS_ENTRY_OTHER_SIDE",   # journal entry reconciled via TDS sheet
}

# Human-readable reasons for the "Needs Review" report section, keyed by rec code.
REC_REASONS = {
    "AMOUNT_MISMATCH":       "Amount mismatch (same invoice ref, amounts differ)",
    "MATCH_VARIANCE":        "Matched with variance (e.g. bank charge / short payment)",
    "SIGN_REVERSED_REVIEW":  "Possible posting error (same amount & date, same sign)",
    "SUSPECTED_DUPLICATE":   "Suspected duplicate (identical entry on the same side)",
}

# The set of rec codes that make up the "Needs Review" tier (between a clean
# match and a genuine "missing"). Order controls display order.
NEEDS_REVIEW_CODES = [
    "AMOUNT_MISMATCH", "MATCH_VARIANCE", "SIGN_REVERSED_REVIEW", "SUSPECTED_DUPLICATE",
]

# ─────────────────────────── matching tolerances ────────────────────────────
# Two-band amount model (all comparisons are mirror-sign: a true pair satisfies
# ours + theirs ≈ 0, so the "gap" is abs(ours + theirs)):
#   - gap <= rounding_tolerance                → CLEAN match (L1/L2/L3)
#   - rounding < gap <= variance_band          → paired but flagged MATCH_VARIANCE
#   - variance_band < gap <= am_ceiling        → AMOUNT_MISMATCH (ref-matched only)
#   - gap > am_ceiling                          → not paired (left for review/missing)
# variance_band and am_ceiling are PERCENTAGES of the row magnitude, so a ₹500
# charge on a ₹34k receipt (1.5%) is caught while a ₹500 gap on a ₹600 row is not.

# Absolute rounding tolerance (currency units) for a clean match. Covers paise /
# GST rounding; anything larger is surfaced rather than silently absorbed.
DEFAULT_ROUNDING_TOLERANCE = 1.00

# Variance band as a fraction of the row magnitude (2%). Beyond rounding but
# within this → MATCH_VARIANCE. User-adjustable via the UI.
DEFAULT_VARIANCE_BAND_PCT = 0.02

# Amount-mismatch ceiling as a fraction of the row magnitude (15%). A ref-matched
# pair whose gap exceeds this is NOT paired (prevents an invoice binding to an
# unrelated same-ref journal of wildly different value).
DEFAULT_AM_CEILING_PCT = 0.15

# Back-compat alias: the old single "amount tolerance" now maps to the variance
# band's absolute floor (used where a percentage of a tiny amount would be < this).
DEFAULT_AMOUNT_TOLERANCE = 1.00

# Default date tolerance (days) for L2 timing-difference matching
DEFAULT_DATE_TOLERANCE_DAYS = 45

# Claude model.
# Column mapping is a small structured task (8 sample rows in, a small
# JSON out), so Haiku is the right cost/quality fit. Bump to claude-sonnet-4-6
# if you want richer AI insights prose.
CLAUDE_MODEL = "claude-haiku-4-5"

# Sample rows sent to Claude during mapping
MAPPING_SAMPLE_ROWS = 8

# Cache directory for confirmed mappings (keyed by column fingerprint)
CACHE_DIR = "cache"

# ─────────────────────────── API pricing ────────────────────────────────────
# Claude API pricing (USD per 1 million tokens). Update if Anthropic changes rates.
# Source: https://www.anthropic.com/pricing  (check periodically)
PRICING = {
    "claude-opus-4-8": {
        "input":   5.00,
        "output": 25.00,
    },
    "claude-sonnet-4-6": {
        "input":   3.00,
        "output":  15.00,
    },
    "claude-haiku-4-5": {
        "input":   1.00,
        "output":   5.00,
    },
}

# Model whose rates are used when a call reports a model that isn't in PRICING.
# We fall back to the priciest known model so we never under-report cost.
PRICING_FALLBACK_MODEL = "claude-opus-4-8"

# USD → INR conversion rate (approximate; update periodically)
# Used only for display in the UI; not for billing.
USD_TO_INR = 84.0
