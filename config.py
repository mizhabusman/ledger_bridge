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
    "TDS_MATCH": "MATCHED_INCL_TDS",          # exact once counterparty's booked TDS added back
    "AMOUNT_MISMATCH": "AMOUNT_MISMATCH",
    "TDS_UNVERIFIED": "SUGGESTED_TDS_UNVERIFIED",  # TDS-related but not exactly confirmable → review
    "SIGN_REVERSED": "SIGN_REVERSED_REVIEW",  # same amount+date, same sign (posting error)
    "SUSPECTED_DUP": "SUSPECTED_DUPLICATE",   # duplicate of another same-side row
    "MISSING_OURS": "MISSING_IN_OURS",
    "MISSING_THEIRS": "MISSING_IN_THEIRS",
    "TDS_DIFF": "TDS_DIFFERENCE",
    "TDS_ENTRY": "TDS_ENTRY_OTHER_SIDE",   # journal entry reconciled via TDS sheet
}

# Codes that are CONFIRMED matches (shown in the Matched sheet). Note: not all of
# these are "balance-neutral" — see BALANCE_NEUTRAL_CODES below.
CONFIRMED_MATCH_CODES = ["MATCHED_L1", "MATCHED_L2_TIMING", "MATCHED_L3_REVIEW", "MATCHED_INCL_TDS"]

# Codes whose gross ACTUALLY cancels with its partner (ours + theirs ≈ 0), so the
# pair drops out of the balance walk. A TDS-inclusive match confirms the two rows
# are the same invoice, but the pair nets to +TDS (the withheld tax the seller has
# not booked per-invoice) — a REAL reconciling item — so MATCHED_INCL_TDS is
# CONFIRMED but deliberately NOT balance-neutral.
BALANCE_NEUTRAL_CODES = ["MATCHED_L1", "MATCHED_L2_TIMING", "MATCHED_L3_REVIEW"]

# Human-readable reasons / notes, keyed by rec code.
REC_REASONS = {
    "MATCHED_INCL_TDS":         "Matched after adding counterparty's withheld TDS",
    "AMOUNT_MISMATCH":          "Amount mismatch (same invoice ref, amounts differ beyond TDS)",
    "SUGGESTED_TDS_UNVERIFIED": "Possible TDS-related match — verify (both sides carry TDS)",
    "SIGN_REVERSED_REVIEW":     "Possible posting error (same amount & date, same sign)",
    "SUSPECTED_DUPLICATE":      "Suspected duplicate (identical entry on the same side)",
}

# The set of rec codes that make up the "Needs Review" tier (between a confirmed
# match and a genuine "missing"). Order controls display order.
NEEDS_REVIEW_CODES = [
    "AMOUNT_MISMATCH", "SUGGESTED_TDS_UNVERIFIED", "SIGN_REVERSED_REVIEW", "SUSPECTED_DUPLICATE",
]

# ─────────────────────────── matching tolerances ────────────────────────────
# Deterministic, self-proving matching (mirror-sign: a true pair satisfies
# ours + theirs ≈ 0, so "gap" = abs(ours + theirs)). NO percentage variance band:
#   - gap <= rounding_tolerance                          → CONFIRMED (L1/L2/L3)
#   - exact only after adding the counterparty's own booked per-invoice TDS
#     (ref-matched, one side carries TDS)                → CONFIRMED (MATCHED_INCL_TDS)
#   - same ref, gap not TDS-explained, within am_ceiling → AMOUNT_MISMATCH (review)
#   - everything else                                    → review / missing
# A pair is NEVER auto-confirmed on a percentage guess.

# Absolute rounding tolerance (currency units) for a clean/exact match. Covers
# paise / GST rounding; anything larger is surfaced, never silently absorbed.
DEFAULT_ROUNDING_TOLERANCE = 1.00

# Amount-mismatch ceiling as a fraction of the row magnitude (15%): only bounds
# which same-ref pairs are worth DISPLAYING as AMOUNT_MISMATCH — it never promotes
# a pair to a confirmed match. A ref-matched pair whose gap exceeds this is left
# for review/missing (prevents binding an invoice to an unrelated same-ref journal).
DEFAULT_AM_CEILING_PCT = 0.15

# Back-compat alias for any external caller still referencing a single tolerance.
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

# Cached mappings are stamped with this version. BUMP IT whenever the mapping
# prompt or the canonical schema changes, so a ledger layout seen under an older
# prompt is re-analysed instead of silently reusing a stale mapping. (v2 drops
# the removed buyer/seller "role" schema; v1/unstamped caches are invalidated.)
MAPPING_CACHE_VERSION = 2

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
