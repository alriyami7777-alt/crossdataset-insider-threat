"""Evaluation helpers (CPU-only distribution-shift / transfer diagnostics)."""

from .domain_gap import pairwise_domain_gap, summarize_domain_gap

__all__ = ["pairwise_domain_gap", "summarize_domain_gap"]
