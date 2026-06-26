"""
LedgerBridge AI — Claude Mapping Engine

Sends column names + sample rows to Claude, gets back:
1. A mapping from source columns to canonical mappable fields (incl. Debit and Credit)
2. A detected role: "buyer" or "seller"

The user reviews/edits both on the confirmation screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

from config import (
    MAPPABLE_FIELDS,
    CLAUDE_MODEL,
    MAPPING_SAMPLE_ROWS,
    CACHE_DIR,
)
from ingest import get_column_fingerprint


SYSTEM_PROMPT = """You are a finance data expert specialising in Indian accounting systems.
Your task is to analyze a raw accounting ledger export and return TWO things:

1. A mapping from source columns to canonical fields
2. The detected ROLE of this ledger: "buyer" or "seller"

═════════════════════════════════════════════════════════
PART 1: COLUMN MAPPING
═════════════════════════════════════════════════════════

Map source columns to these canonical fields:

- Date: Transaction or posting date.
- Voucher Type: Classification (Sales, Purchase, Payment, Receipt, Journal, Bank Receipt, Bank Payment, Credit Note, etc.)
- Voucher No: INTERNAL voucher/document number assigned by the source system (e.g. 1000/25/BP-712, 2025-26/SV-140).
- Invoice Ref: EXTERNAL invoice reference shared between both parties (e.g. 25-26/GS-140, INV-1023). This is the KEY match field. Common headers: Invoice-No, Invoice No, Inv No, Our Inv No, Bill No, Ref No, Reference.
- Description: Free-text narration. Common headers: Narration, Particulars, Description, Remarks.
- Debit: The Debit column. Common headers: Debit, Dr, Dr Amount, Debit Amount.
- Credit: The Credit column. Common headers: Credit, Cr, Cr Amount, Credit Amount.
- TDS Amount: Tax Deducted at Source. Optional. Only if there is an explicit TDS column.

CRITICAL RULES for Debit/Credit:
- ALWAYS map Debit and Credit as SEPARATE columns. Never combine them.
- Most ledgers have explicit "Debit" and "Credit" columns. Find them by name.
- If the ledger has only a single signed "Amount" column with positives and negatives, map it to Debit and leave Credit null. The sign convention will be handled by the role.
- NEVER map a Net-Amount, Transaction-Amount, or Balance column to Debit or Credit. Those are derived values.

═════════════════════════════════════════════════════════
PART 2: ROLE DETECTION
═════════════════════════════════════════════════════════

Determine whether this ledger is from:

- "buyer" — the BUYER's view of the relationship (the vendor/supplier is the counterparty).
  Signals:
    * Voucher Type contains "Purchase", "Bank-Payment", "BP", "Payment Voucher", "G.Journal" (in a vendor ledger)
    * Description has "Payment to vendor", "GST Vr. Payments / Purchase"
    * Invoice amounts sit in the Credit column (AP increases)
    * Payment amounts sit in the Debit column (AP decreases)

- "seller" — the SELLER's view of the relationship (the customer is the counterparty).
  Signals:
    * Voucher Type contains "Sales", "SV" (Sales Voucher), "Bank Receipt", "BR", "Receipt", "Credit Note"
    * Description has "Sales", "Bill raised", "Invoice raised"
    * Invoice amounts sit in the Debit column (AR increases)
    * Payment/receipt amounts sit in the Credit column (AR decreases)

Use whatever signals are clearest. If genuinely uncertain, prefer "buyer" and mark confidence "low".

═════════════════════════════════════════════════════════
OUTPUT FORMAT
═════════════════════════════════════════════════════════

Return ONLY valid JSON, no markdown fences, no commentary:

{
  "role": "buyer" | "seller",
  "role_confidence": "high" | "medium" | "low",
  "role_reasoning": "one short sentence explaining the signals you used",
  "mapping": {
    "Date":         {"source": "<column or null>", "confidence": "high|medium|low|n/a"},
    "Voucher Type": {"source": "<column or null>", "confidence": "..."},
    "Voucher No":   {"source": "<column or null>", "confidence": "..."},
    "Invoice Ref":  {"source": "<column or null>", "confidence": "..."},
    "Description":  {"source": "<column or null>", "confidence": "..."},
    "Debit":        {"source": "<column or null>", "confidence": "..."},
    "Credit":       {"source": "<column or null>", "confidence": "..."},
    "TDS Amount":   {"source": "<column or null>", "confidence": "..."}
  }
}
"""


def _build_user_prompt(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    sample = df.head(MAPPING_SAMPLE_ROWS).fillna("").astype(str)

    rows = [" | ".join(columns), " | ".join(["---"] * len(columns))]
    for _, r in sample.iterrows():
        rows.append(" | ".join(str(v)[:60] for v in r))

    return f"""Source columns: {columns}

Sample rows ({len(sample)} shown):
{chr(10).join(rows)}

Analyze this ledger. Return JSON only with role + mapping."""


def analyze_ledger(
    df: pd.DataFrame,
    client: Anthropic,
    use_cache: bool = True,
    cost_tracker=None,
    step_name: str = "Map Ledger",
) -> dict:
    """
    Use Claude to detect the role + map columns.

    Returns:
        {
          "role": "buyer" | "seller",
          "role_confidence": "...",
          "role_reasoning": "...",
          "mapping": { canonical_field: {source, confidence}, ... }
        }

    Cached by column fingerprint.

    Optional args:
        cost_tracker: a CostTracker instance to record token usage. If None,
                      no cost tracking happens (backward-compatible).
        step_name:    label for this step in the cost breakdown.
    """
    fingerprint = get_column_fingerprint(df)

    if use_cache:
        cached = _load_cached(fingerprint)
        if cached:
            if cost_tracker is not None:
                cost_tracker.add(step_name, 0, 0, cached=True)
            return cached

    user_prompt = _build_user_prompt(df)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Record token usage if a tracker was provided
    if cost_tracker is not None:
        from cost_tracker import extract_usage
        in_tok, out_tok = extract_usage(response)
        cost_tracker.add(step_name, in_tok, out_tok, model=CLAUDE_MODEL)

    raw_text = response.content[0].text.strip()

    # Strip markdown fences if Claude wrapped the JSON
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON:\n{raw_text}\n\nError: {e}")

    # Backward-compat normalization: ensure every mappable field exists
    mapping = result.get("mapping", {})
    for field in MAPPABLE_FIELDS:
        if field not in mapping:
            mapping[field] = {"source": None, "confidence": "n/a"}
    result["mapping"] = mapping

    # Ensure role exists
    if result.get("role") not in ("buyer", "seller"):
        result["role"] = "buyer"
        result["role_confidence"] = "low"
        result["role_reasoning"] = "default (could not determine from data)"

    return result


def save_confirmed_analysis(df: pd.DataFrame, analysis: dict) -> None:
    """Persist user-confirmed role + mapping so the same format skips the API next time."""
    fingerprint = get_column_fingerprint(df)
    cache_path = Path(CACHE_DIR) / f"{_safe_filename(fingerprint)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(analysis, indent=2))


def _load_cached(fingerprint: str) -> dict | None:
    cache_path = Path(CACHE_DIR) / f"{_safe_filename(fingerprint)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return None


def _safe_filename(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:16]


# Backward-compat aliases — old code that imports map_columns / save_confirmed_mapping
# still works but operates on the new analysis shape.
def map_columns(df, client, use_cache=True):
    """Deprecated: use analyze_ledger. Returns just the mapping dict."""
    return analyze_ledger(df, client, use_cache).get("mapping", {})


def save_confirmed_mapping(df, mapping):
    """Deprecated: use save_confirmed_analysis."""
    save_confirmed_analysis(df, {"role": "buyer", "mapping": mapping})