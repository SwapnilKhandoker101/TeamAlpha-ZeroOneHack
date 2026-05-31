# Forecasting AI — supply & hedging decision agent

**Hackathon track:** Forecasting AI. **Built on:** the Sybilion probabilistic forecasting API.

---

## TL;DR

One page, one company description, **two auditable decisions** for a mid-size German
glass & ceramics manufacturer:

1. **Gas hedge** — what share of next quarter's natural gas (Dutch **TTF** benchmark) to
   **lock forward now** vs. buy later on the spot market.
2. **Ceramics run** — for one production run: how much **blended input cost to lock**,
   **which supplier** to buy from, **which channel** to sell through, plus a negotiated deal.

The headline principle: **the LLM never computes a decision number.** Deterministic policy
code computes every hedge ratio, lock %, score, quote and backtest result; the LLM only
*prepares inputs* (picks Sybilion filters, reads a shock's severity) and *explains outputs*.
So identical inputs always reproduce the identical decision, and every number is traceable.
It runs **fully offline with no API keys** (committed cached forecast + template narration),
a **live supply-shock** lets a judge change an assumption mid-run and watch **both** decisions
adapt instantly, and **reproducible backtests** show the policy beats naive baselines.

---

## Problem & domain rationale

A German glass/ceramics maker's largest volatile cost is **natural gas** (high-temperature
firing). Gas prices are famously hard to point-forecast — on this very series Sybilion's point
forecast runs **~28% MAPE**. So a responsible agent does **not** bet on the price line. It bets
on the **shape of uncertainty**:

- The **gas hedge** is sized from the forecast's **confidence band** (a tighter band → lock more;
  a wider band → stay flexible) plus a curated mix of credible supply/demand drivers — never the
  point estimate.
- The **ceramics lock** reuses the *same* hedge-policy engine on a **blended 4-factor cost band**
  (gas / clay / power / freight), then routes the supplier choice by the lock stance and books a
  channel + a negotiated margin.

This is a *decision* agent, not a price predictor — which is exactly the honest way to use a
probabilistic forecast whose mean is weak but whose bands are informative.

---

## Approach

**A deterministic spine; the LLM only narrates.**

```
Featherless LLM (prepares inputs)  →  Sybilion API (forecast + driver importances)
   →  deterministic curation + hedge/cost policy (DECIDES)  →  Featherless LLM (explains)
   →  Streamlit dashboard (+ live shock, 3D globe, voice)
```

- **THE RULE.** `gas_agent/hedge_policy.py` and `ceramics_agent/cost_policy.py` compute the
  numbers, deterministically, from the band + drift (+ an optional shock risk-premium). The LLM
  is forbidden — in its prompts — from inventing drivers/numbers or re-deciding a ratio. It picks
  Sybilion filters (`keyword_agent`), reads a shock's severity (`scenario`), and explains the
  decided numbers (`explanation_agent`, the globe brief, the chat). The chat can even **explain
  the app itself** ("what is this app / why this design?") — still explanation-only.
- **Bands, not the point forecast.** The decision reads q10 / q50 / q90; the median's role is only
  to set a small drift tilt, never the price you pay.
- **Reused engines.** The ceramics agent imports `gas_agent.hedge_policy` unchanged — the lock %
  is literally a hedge ratio on a blended cost band. The globe reuses the gas geo engine
  (`CountryAggregate`, coordinates) for two new layers.
- **Offline-first.** Every external call has a deterministic fallback; the dashboard reads cached
  Sybilion artifacts and a committed 4-factor mock. **No keys → full demo.** A first-class
  Live⟷Cached toggle runs the real submit→poll on demand and caches it; toggling back instantly
  restores the reproducible numbers (it never repoints the pinned demo job).

---

## How to run

```bash
uv sync                              # install deps (Python ≥ 3.11; or: pip install -r requirements.txt)
uv run streamlit run app.py          # the dashboard — runs with NO keys (cached/mock)
uv run pytest -q                     # 270 offline tests, all green
uv run python -m gas_agent.decision_backtest    # gas backtest verdict (offline)
uv run python -m ceramics_agent.backtest        # ceramics backtest verdict (offline)
```

Optional live mode + voice: copy `.env.example` → `.env`, add `SYBILION_API_KEY` (live forecast),
`FEATHERLESS_API_KEY` (LLM narration), and NVIDIA/HF keys (voice in/out). All optional.

**What's live vs mocked (honest):**

| Piece | Default (no keys) | With keys |
|-------|-------------------|-----------|
| Gas TTF forecast | committed cached Sybilion job | live submit→poll (toggle) |
| Ceramics 4-factor cost forecast | committed mock (`cache/mock_ceramics_forecast.json`) | live 4-factor Sybilion (toggle) |
| Narration / chat / briefs | deterministic templates | Featherless LLM |
| Voice in/out | hidden (text works) | NVIDIA Riva → local `say` / HF Whisper |
| **Every decision number** | **deterministic — identical either way** | **identical** |

---

## Results — real numbers

### Gas hedge backtest (replays Sybilion's own backtest windows)

In a *rising* window, lowest mean cost isn't the goal — **variance reduction + adaptivity** is
(the reason a manufacturer hedges). Replayed over **63 months**:

| Strategy | Mean cost (EUR/MWh) | Cost swing (±) |
|----------|--------------------:|---------------:|
| **Agent policy** | **35.71** | **4.47** |
| Always spot (0%) | 36.96 | 6.81 |
| Always lock 50% | 35.25 | 4.35 |
| Always lock 100% | 33.54 | 4.16 |
| Random hedge ratio (seeded) | 35.48 | 4.98 |

- **+€1.25/MWh cheaper** and **€2.34/MWh steadier** than do-nothing spot; **matches** a static 50%.
- Beats the **coin-flip** random ratio on swing (±4.47 vs ±4.98).
- Always-100% is cheapest here **only because this window rose** — a maximal directional bet that
  is *dearest in a falling market*. The policy is the only strategy that sizes each month's lock to
  the forecast band. *(Stated honestly in the UI — we don't claim a win we didn't earn.)*
- **Under a live supply shock** (prices spike, the policy lifts its hedge): policy realizes
  **€35.93/MWh — €5.46/MWh cheaper** than buying at the shocked spot (€41.40). The decision logic
  still wins when the assumption shifts mid-run.
- Calm next-quarter hedge ratio ≈ **30.5%** (incl. a standing supply-risk premium of +6.9%).

*Stated assumption:* the lock price is proxied by the decision-time spot (the artifact carries no
historical forward curve). Surfaced in the UI, not hidden.

### Ceramics optimizer backtest (replays committed sales history)

Ceramics **maximizes margin**. Replayed over **12 months** (default demo run):

| Strategy | Realized margin/month |
|----------|----------------------:|
| **Agent policy** | **€34,290** |
| Random supplier/channel (seeded) | €20,792 → **+65%** |
| Cheapest + best-margin (static) | €33,819 → **+1%** |
| Always top-ranked supplier (ignores lock routing) | €33,819 → **+1%** |

- **+65% vs a random pick**; edges the strong static heuristics by routing the lock stance to the
  **balanced, more-reliable** supplier whose margin survives the reliability haircut.
- **24-month robustness replay** (prior year + recent year): agent **€32,918/month, +60% vs
  random** — the edge is not an artefact of one 12-month window.
- Default decision: lock **40%** of input cost, buy from **Alpine Clay Works** (€0.86/unit), sell
  via **Online Retail Export** (€1.51/unit) → unit margin **€0.65**, **€3,253** on the run.

*Stated assumption:* realized margin discounts each supplier's nominal margin by its reliability
(the backtest's expected-value model). The "random" baseline is seeded so it reproduces run-to-run.

---

## What worked

- **The decision/LLM split held end-to-end.** Every number is deterministic and reproducible; the
  LLM genuinely only narrates. This is the project's spine and it never bent.
- **Reuse paid off.** The ceramics lock *is* the gas hedge engine on a blended band; the second
  globe *is* the first globe's engine with new data. One shock re-decides both agents.
- **Offline floor.** A clean checkout with no keys runs the entire flow — both decisions, the
  globe, the chat, the backtests — which is exactly what a jury can clone and trust.

## What didn't / honest limitations

- **No historical forward curve**, so the gas backtest proxies the lock price with decision-time
  spot. A real forward curve would sharpen it. (Stated in-app.)
- **Curation is name-rule based** (energy/markets whitelist vs demographic blacklist). It's
  transparent and auditable, but not semantic.
- **Lottie pipeline animation deferred** — to protect the offline/on-stage floor we ship its
  documented graceful fallback (`st.status`/`st.progress` with rotating LLM blurbs) rather than add
  a network-fetched animation dependency.

## Next 36h

- Wire a live forward curve into the gas backtest to drop the spot-proxy assumption.
- Unify the two curation paths (name rules + the catalog credibility helpers).
- A short demo video + slides from these numbers.

---

## Credits

- **Sybilion** — probabilistic forecasting API (forecast bands + driver importances).
- **Featherless** — OpenAI-compatible LLM inference (input prep + narration; text only).
- **NVIDIA Riva** — voice TTS/ASR over NVCF gRPC (with a local `say` / HF Whisper fallback).
- **Libraries:** httpx, pandas, numpy, plotly, streamlit, openai, yfinance, openpyxl, pytest.
- Built with **Claude Code**.
