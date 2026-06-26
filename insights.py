"""
LedgerBridge AI — AI Insights Engine

After reconciliation, send a structured summary of the exceptions to Claude
and get back a plain-English explanation suitable for the Summary sheet.

One API call per reconciliation. Failure here is non-fatal — insights are
additive, not load-bearing.
"""

from __future__ import annotations

from anthropic import Anthropic

from config import CLAUDE_MODEL
from reconcile import ReconciliationResult


INSIGHTS_SYSTEM_PROMPT = """You are a senior accountant explaining a ledger
reconciliation to a finance team. You receive a structured summary of matches
and exceptions and you write a brief, professional explanation.

Rules:
- Be specific. Cite exact invoice references, amounts, and dates from the data.
- Be brief: 3-6 short paragraphs, no bullet lists, no markdown formatting.
- Suggest likely reasons for each category of exception (e.g. timing difference,
  TDS not yet booked, data entry error, missing invoice).
- Suggest next actions only when obvious.
- Never invent facts not in the data.
- If everything reconciles cleanly, say so plainly.

Write as if briefing a controller. Direct, useful, no fluff.
"""


def generate_insights(
    result: ReconciliationResult,
    client: Anthropic,
    cost_tracker=None,
) -> str:
    """
    Generate human-readable insights for the reconciliation summary.

    Returns plain text. On API failure, returns a safe fallback string.

    Optional args:
        cost_tracker: a CostTracker instance to record token usage.
                      If None, no cost tracking happens.
    """
    summary = _build_summary_prompt(result)

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            system=INSIGHTS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": summary}],
        )
        # Record token usage if requested
        if cost_tracker is not None:
            from cost_tracker import extract_usage
            in_tok, out_tok = extract_usage(response)
            cost_tracker.add("Generate AI Insights", in_tok, out_tok, model=CLAUDE_MODEL)
        return response.content[0].text.strip()
    except Exception as e:
        return f"(AI insights unavailable: {e})"


def _build_summary_prompt(result: ReconciliationResult) -> str:
    """Build a compact, structured prompt for Claude."""
    s = result.summary

    parts = []
    parts.append("RECONCILIATION RESULTS")
    parts.append("=" * 50)
    parts.append(f"Total records: ours={s['total_our_records']}, theirs={s['total_their_records']}")
    parts.append(f"Matches: L1={s['matched_l1']}, L2={s['matched_l2']}, L3={s['matched_l3']}")
    parts.append(f"Exceptions: {s['amount_mismatches']} amount mismatch, "
                 f"{s['missing_in_theirs']} missing-in-theirs, "
                 f"{s['missing_in_ours']} missing-in-ours")
    parts.append("")
    parts.append(f"Closing balances: ours=₹{s['closing_balance_ours']:,.2f}, theirs=₹{s['closing_balance_theirs']:,.2f}")
    parts.append(f"Difference: ₹{s['difference']:,.2f}")
    parts.append(f"Reconciling items (one-sided): ₹{s['reconciling_item']:,.2f}")
    parts.append(f"Residual after explained items: ₹{s['residual']:,.2f}")
    parts.append(f"TDS: ours=₹{s['tds_ours']:,.2f}, theirs=₹{s['tds_theirs']:,.2f}, diff=₹{s['tds_difference']:,.2f}")
    parts.append(f"Reconciled within tolerance: {'YES' if s['reconciled'] else 'NO'}")
    parts.append("")

    if not result.amount_mismatches.empty:
        parts.append("AMOUNT MISMATCHES:")
        for _, r in result.amount_mismatches.head(20).iterrows():
            parts.append(
                f"  - Invoice {r['Invoice Ref']} on {r['Date']}: "
                f"ours=₹{r['Our Amount']:,.2f}, theirs=₹{r['Their Amount']:,.2f}, "
                f"diff=₹{r['Difference']:,.2f}. Desc: {r['Description']}"
            )
        parts.append("")

    if not result.missing_in_theirs.empty:
        parts.append("MISSING IN THEIR BOOKS (only in ours):")
        for _, r in result.missing_in_theirs.head(20).iterrows():
            parts.append(
                f"  - {r['Invoice Ref']} on {r['Date']}: ₹{r['Gross Amount']:,.2f} — {r['Description']}"
            )
        parts.append("")

    if not result.missing_in_ours.empty:
        parts.append("MISSING IN OUR BOOKS (only in theirs):")
        for _, r in result.missing_in_ours.head(20).iterrows():
            parts.append(
                f"  - {r['Invoice Ref']} on {r['Date']}: ₹{r['Gross Amount']:,.2f} — {r['Description']}"
            )
        parts.append("")

    if not result.timing_differences.empty:
        parts.append("TIMING DIFFERENCES (matched on Ref+Amount but dates differ):")
        for _, r in result.timing_differences.head(20).iterrows():
            parts.append(
                f"  - {r['Invoice Ref']}: ours={r['Our Date']}, theirs={r['Their Date']}, "
                f"amount=₹{r['Our Amount']:,.2f}"
            )
        parts.append("")

    parts.append("Write a brief professional analysis of these results.")
    return "\n".join(parts)