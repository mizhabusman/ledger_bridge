"""
LedgerBridge AI — Claude Mapping Engine

Sends column names + sample rows to Claude, gets back a mapping from source
columns to canonical mappable fields (incl. Debit and Credit).

The user reviews/edits the mapping on the confirmation screen. There is no
buyer/seller role: Gross Amount is Debit - Credit uniformly for every ledger
(see standardize.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

from config import (
    MAPPABLE_FIELDS,
    CLAUDE_MODEL,
    MAPPING_SAMPLE_ROWS,
    CACHE_DIR,
    MAPPING_CACHE_VERSION,
)
from ingest import get_column_fingerprint


SYSTEM_PROMPT = """You are a finance data expert specialising in Indian accounting systems.
Your task is to analyze a raw accounting ledger export and return a mapping
from source columns to canonical fields.

═════════════════════════════════════════════════════════
COLUMN MAPPING
═════════════════════════════════════════════════════════

Map source columns to these canonical fields:

- Date: The INVOICE date — the date printed on the actual invoice/bill, not when it was posted/entered into the books. If a ledger has both an explicit "Invoice Date" column AND a separate "Posting Date"/"Entry Date"/"Value Date" column, ALWAYS prefer the invoice date column. Only fall back to a generic posting/entry date if no distinct invoice date column exists.
- Voucher Type: Classification (Sales, Purchase, Payment, Receipt, Journal, Bank Receipt, Bank Payment, Credit Note, etc.)
- Voucher No: INTERNAL voucher/document number assigned by the source system (e.g. 1000/25/BP-712, 2025-26/SV-140).
- Invoice Ref: EXTERNAL invoice reference shared between both parties (e.g. 25-26/GS-140, INV-1023). This is the KEY match field. Common headers: Invoice-No, Invoice No, Inv No, Our Inv No, Bill No, Ref No, Reference. If there is NO separate external-reference column and the Voucher No is the identifier both parties would plausibly share (e.g. both books cite the same document number when recording the transaction), map that SAME source column to both Voucher No and Invoice Ref rather than leaving Invoice Ref null.
- Description: Free-text narration. Common headers: Narration, Particulars, Description, Remarks.
- Debit: The Debit column. Common headers: Debit, Dr, Dr Amount, Debit Amount.
- Credit: The Credit column. Common headers: Credit, Cr, Cr Amount, Credit Amount.
- TDS Amount: Tax Deducted at Source. Optional. Only if there is an explicit TDS column.

CRITICAL RULES for Debit/Credit:
- ALWAYS map Debit and Credit as SEPARATE columns. Never combine them.
- Most ledgers have explicit "Debit" and "Credit" columns. Find them by name.
- If the ledger has only a single signed "Amount" column with positives and negatives, map it to Debit and leave Credit null.
- NEVER map a Net-Amount, Transaction-Amount, or Balance column to Debit or Credit. Those are derived values.

═════════════════════════════════════════════════════════
OUTPUT FORMAT
═════════════════════════════════════════════════════════

Return ONLY valid JSON, no markdown fences, no commentary:

{
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

Analyze this ledger. Return JSON only with the mapping."""


def analyze_ledger(
    df: pd.DataFrame,
    client: Anthropic,
    use_cache: bool = True,
    cost_tracker=None,
    step_name: str = "Map Ledger",
) -> dict:
    """
    Use Claude to map source columns to canonical fields.

    Returns:
        {
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

    result = _parse_analysis_json(raw_text)

    # Normalize: ensure every mappable field exists in the mapping
    mapping = result.get("mapping", {})
    for field in MAPPABLE_FIELDS:
        if field not in mapping or not isinstance(mapping[field], dict):
            mapping[field] = {"source": None, "confidence": "n/a"}
    result["mapping"] = mapping

    # Cache the freshly computed analysis so identical formats skip the API
    if use_cache:
        _write_cache(fingerprint, result)

    return result


def _parse_analysis_json(raw_text: str) -> dict:
    """
    Extract the JSON object from Claude's reply.

    Handles three cases robustly:
      1. Clean JSON.
      2. JSON wrapped in ```json ... ``` markdown fences.
      3. JSON embedded in a conversational preamble/suffix.
    """
    text = raw_text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
            text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the outermost {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Claude returned invalid JSON:\n{raw_text}\n\nError: {e}")

    raise ValueError(f"Claude returned no JSON object:\n{raw_text}")


def save_confirmed_analysis(df: pd.DataFrame, analysis: dict) -> None:
    """Persist the user-confirmed mapping so the same format skips the API next time."""
    _write_cache(get_column_fingerprint(df), analysis)


def _write_cache(fingerprint: str, analysis: dict) -> None:
    cache_path = Path(CACHE_DIR) / f"{_safe_filename(fingerprint)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp the version (on a copy) so stale caches are invalidated on read.
    payload = dict(analysis)
    payload["_cache_version"] = MAPPING_CACHE_VERSION
    cache_path.write_text(json.dumps(payload, indent=2))


def _load_cached(fingerprint: str) -> dict | None:
    cache_path = Path(CACHE_DIR) / f"{_safe_filename(fingerprint)}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    # Ignore caches written under an older prompt/schema version — re-analyse
    # (and overwrite the same file) rather than silently reuse a stale mapping.
    if data.get("_cache_version") != MAPPING_CACHE_VERSION:
        return None
    data.pop("_cache_version", None)
    return data


def _safe_filename(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:16]
