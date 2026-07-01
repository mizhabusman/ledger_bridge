"""
LedgerBridge AI — API Cost Tracker

Captures token usage from every Claude API call and computes the cost
per step + cumulative cost. Used purely for UI display and reporting;
does NOT influence any reconciliation logic.

Token counts come from the Anthropic SDK response (`response.usage`),
which is included in every successful API call at no extra charge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import PRICING, USD_TO_INR, CLAUDE_MODEL, PRICING_FALLBACK_MODEL


@dataclass
class UsageRecord:
    """One API call's resource use + computed cost."""
    step_name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_inr: float
    cached: bool = False   # True if this step skipped the API (cache hit)


@dataclass
class CostTracker:
    """
    Accumulates token usage and cost across all Claude calls in a single
    reconciliation run. One instance per reconciliation.
    """
    records: list[UsageRecord] = field(default_factory=list)

    def add(
        self,
        step_name: str,
        input_tokens: int,
        output_tokens: int,
        model: str = CLAUDE_MODEL,
        cached: bool = False,
    ) -> UsageRecord:
        """Record one API call. Returns the UsageRecord."""
        if cached:
            # Cache hits cost nothing
            rec = UsageRecord(
                step_name=step_name, model=model,
                input_tokens=0, output_tokens=0,
                cost_usd=0.0, cost_inr=0.0, cached=True,
            )
        else:
            usd = _calculate_cost(model, input_tokens, output_tokens)
            rec = UsageRecord(
                step_name=step_name, model=model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost_usd=usd,
                cost_inr=usd * USD_TO_INR,
                cached=False,
            )
        self.records.append(rec)
        return rec

    def total(self) -> dict:
        """Sum across all recorded calls."""
        return {
            "input_tokens":  sum(r.input_tokens  for r in self.records),
            "output_tokens": sum(r.output_tokens for r in self.records),
            "cost_usd":      sum(r.cost_usd      for r in self.records),
            "cost_inr":      sum(r.cost_inr      for r in self.records),
            "calls":         sum(1 for r in self.records if not r.cached),
            "cache_hits":    sum(1 for r in self.records if r.cached),
        }

    def summary_rows(self) -> list[dict]:
        """Per-step breakdown suitable for tables/UI."""
        return [
            {
                "Step":        r.step_name,
                "Input":       r.input_tokens,
                "Output":      r.output_tokens,
                "Cost (USD)":  f"${r.cost_usd:.4f}",
                "Cost (INR)":  f"₹{r.cost_inr:.2f}",
                "Cached":      "Yes" if r.cached else "No",
            }
            for r in self.records
        ]


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Compute USD cost given token counts and model.

    Falls back to Opus pricing if the model isn't in our table — we'd rather
    over-estimate than show the user a cost lower than reality.
    """
    rates = PRICING.get(model) or PRICING[PRICING_FALLBACK_MODEL]
    in_cost  = (input_tokens  / 1_000_000) * rates["input"]
    out_cost = (output_tokens / 1_000_000) * rates["output"]
    return in_cost + out_cost


def extract_usage(response) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from an Anthropic API response.

    Safe against missing fields — returns (0, 0) if usage data isn't present,
    so a malformed response never breaks the pipeline.
    """
    try:
        usage = response.usage
        return (
            int(getattr(usage, "input_tokens", 0)),
            int(getattr(usage, "output_tokens", 0)),
        )
    except Exception:
        return (0, 0)