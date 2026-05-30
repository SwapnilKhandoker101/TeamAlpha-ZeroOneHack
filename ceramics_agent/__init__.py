"""Ceramics supply-chain optimizer — a second deterministic decision agent.

This package lives *alongside* :mod:`gas_agent` (which it never modifies) and
reuses its proven spine: the deterministic hedge-policy engine
(:mod:`gas_agent.hedge_policy`), the Featherless→template explanation ladder,
the Sybilion REST client + disk cache, and the "the LLM only explains, never
decides" rule.

It answers three procurement questions for a German ceramics manufacturer —
*what to buy* (supplier), *what to pay* (negotiation room), and *where to sell*
(sales channel) — for a chosen product / quantity / timeline under user-set
cost-factor weights. Like the gas agent, every number is deterministic
arithmetic that traces back to data or a rule; the only model call narrates the
finished recommendation.
"""
