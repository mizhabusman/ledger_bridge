"""
LedgerBridge AI — Configuration

Defines the canonical schema, settings, and constants used across the app.
"""

# The 9-field canonical schema for OUTPUT (final standardized ledger format).
# Note: Gross Amount is COMPUTED from Debit/Credit + role, not directly mapped.
CANONICAL_FIELDS = [
    "Date",
    "Voucher Type",
    "Voucher No",
    "Invoice Ref",
    "Description",
    "Gross Amount",     # computed = signed amount based on role
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

# Role of a ledger: are we looking at the buyer's books or the seller's books?
ROLES = ["buyer", "seller"]

# Reconciliation codes written into the Rec Code column
REC_CODES = {
    "L1": "MATCHED_L1",
    "L2": "MATCHED_L2_TIMING",
    "L3": "MATCHED_L3_REVIEW",
    "AMOUNT_MISMATCH": "AMOUNT_MISMATCH",
    "MISSING_OURS": "MISSING_IN_OURS",
    "MISSING_THEIRS": "MISSING_IN_THEIRS",
    "TDS_DIFF": "TDS_DIFFERENCE",
    "TDS_ENTRY": "TDS_ENTRY_OTHER_SIDE",   # journal entry reconciled via TDS sheet
}

# Default amount tolerance (currency units)
DEFAULT_AMOUNT_TOLERANCE = 1.00

# Default date tolerance (days) for L2 timing-difference matching
DEFAULT_DATE_TOLERANCE_DAYS = 45

# Claude model.
# Mapping + role detection is a small structured task (8 sample rows in, a small
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
