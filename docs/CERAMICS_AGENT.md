# The ceramics supply-chain optimizer (second agent)

A second decision agent, in a second Streamlit tab, sharing the gas agent's spine:
**deterministic & auditable** (the LLM only explains, never computes a number),
**offline-first** (runs with no keys on a committed mock forecast + template
narrative), and **non-regressive** (the gas tab and `gas_agent/` package are
untouched). It lives in its own `ceramics_agent/` package.

> This is the design note. `CLAUDE.md` is the operational handoff (§5 has the live
> numbers and git state); `docs/APP_GUIDE.md` / `docs/ARCHITECTURE.md` cover the gas
> agent.

---

## What it decides

For **one production run** — a product, a quantity, a timeline, four cost-factor
weights (gas / clay / power / transport) and a competition level — the agent decides
three things and then negotiates the deal:

1. **How much input cost to lock now** (the "lock %", the headline).
2. **Which supplier to buy from** (kept, scored, ranked; the pick is routed by the lock %).
3. **Which channel to sell through** (kept, scored, ranked).

Then a **two-round negotiation** sets the buy/sell prices and the margin, and a
**3-strategy backtest** checks the policy beats a random pick.

---

## The pipeline (module by module)

| Step | Module | What it does |
|------|--------|--------------|
| Forecast | `forecast.py` | Loads a **4-factor** cost forecast (gas / clay / power / shipping), each a per-month `MonthForecast` band. Mock by default (`cache/mock_ceramics_forecast.json`); live Sybilion path is opt-in behind `SYBILION_API_KEY` and **never** repoints `cache/latest_job.txt`. |
| Lock % | `cost_policy.py` | Blends the four factor bands into one synthetic cost band and **reuses `gas_agent.hedge_policy.decide_month`** to turn it into a lock %. |
| Curation | `curation.py` | Keep/reject suppliers & channels (substring credibility), then score + rank them. |
| Negotiation | `negotiation.py` | Two rounds → buy price, sell price, unit & total margin, with a full trace. |
| Backtest | `backtest.py` | Replays 12 historical months under 3 strategies. |
| Orchestration | `recommend.py` | Assembles all of the above into one deterministic `Recommendation`. |
| Explanation | `explanation.py` | Featherless narrates the **already-decided** recommendation; template fallback. |
| UI | `dashboard.py` | `render_ceramics_tab(render_voiceover)` — the second tab. |
| Data | `catalog.py` | Products (bill of materials), suppliers, channels, historical sales, credibility whitelists. |

---

## The key design win: lock % **is** a hedge ratio

The ceramics "lock %" — *what share of next quarter's input cost to fix forward now*
— is structurally identical to the gas hedge ratio. So `cost_policy.py` **reuses the
gas band→ratio engine unchanged**, with ceramics-tuned thresholds:

```python
CERAMICS_POLICY_PARAMS = HedgePolicyParams(low_band=0.25, high_band=0.60)
```

A **tight** blended band (confident outlook) → a **high** lock; a **wide** band
(volatile) → a **low** lock; the [0.10, 0.90] clamp and the drift tilt come for free.

### Two clearly-separated aggregations

These are deliberately kept apart (and documented inline) so neither contaminates the
other:

1. **Decision volatility** drives the lock %: a weighted blend of each factor's
   relative band width `(q90 − q10) / q50` and median level, packed into a synthetic
   `MonthForecast`. **Weights matter here** — they reshape the band, hence the lock %.
2. **Physical per-unit cost** drives the chart + the negotiation: real EUR/unit from
   the product's bill of materials × the factor quantiles. This is what the cost-band
   chart shows.

---

## Curation & lock-routed selection

Each kept supplier/channel gets transparent sub-scores (0–100) and a weighted total:

- **Supplier:** `0.5·cost + 0.3·reliability + 0.2·lead-time-fit` (cheapest avg price
  factor → cost 100; exact timeline match → lead 100; reliability = on-time %).
- **Channel:** `0.4·margin + 0.3·seasonality + 0.3·order-fit` (richest target margin →
  100; target-quarter seasonality; quantity ≥ min-order → order-fit 100).

The **supplier pick is routed by the lock %** (`select_supplier`), so the buy decision
and the lock decision stay coupled:

- lock ≥ `LOCK_HIGH` (0.60) → commit to the **best-ranked** supplier.
- lock ≤ `LOCK_LOW` (0.30) → minimise exposure, take the **cheapest**.
- in between → the balanced **mid-ranked** supplier.

The channel pick is simply the top-ranked kept channel.

---

## Two-round, market-anchored negotiation

`negotiate(...)` is pure arithmetic and fully traced (`QuoteRow` per move):

- **Round 1 — openings.** Supplier quote = its repriced materials × `(1 + 15%
  supplier margin)`. Channel offer = a **market reference cost** (a typical maker at
  1.0 baseline factors) marked up by the channel's target margin. Because the sell
  side is anchored to the *market*, not to this supplier, a **cheaper supplier widens
  the margin** rather than dragging the sell price down.
- **Round 2 — counters.** Supplier adds `+5%` if the target quarter is hot
  (seasonality ≥ 1.2) and `+10%` for a rush order (timeline < 7 days). The channel
  discounts the sell price by the user's competition read: `−7%` high / `−3%` medium /
  `0%` low.
- **Close.** `unit_margin = sell − buy`; `total = unit_margin × quantity`. Identical
  inputs → identical rows.

---

## Backtest — does the policy beat the baselines?

`run_ceramics_backtest` replays 12 historical months under three strategies:

- **Agent** — the lock %, the lock-routed supplier, the top channel, negotiated.
- **Random** — supplier + channel drawn from `random.Random(CERAMICS_BACKTEST_SEED=7)`,
  **seeded so it reproduces** run-to-run (the determinism invariant holds even for the
  baseline).
- **Cheap + best-margin** — static cheapest supplier + highest-margin channel.

Realized margin **discounts each supplier's nominal margin by its reliability %** (a
stated assumption — an unreliable supplier costs you in practice). Ceramics
**maximizes** margin, so *higher is better* (the opposite sense to the gas agent,
which minimises cost variance) — the verdict leads with the margin delta.

---

## Invariants (same spine as gas)

- **THE RULE:** every number (lock %, scores, quotes, margins, backtest) is
  deterministic arithmetic. The **only** LLM call explains the finished
  recommendation and is forbidden, in its system prompt, from re-deciding any of it.
- **Offline-first:** the committed `cache/mock_ceramics_forecast.json` is the demo's
  source of truth; the explanation falls back to a template; voice is optional. The
  live Sybilion 4-factor refresh is opt-in and never repoints `latest_job.txt`.
- **Gas frozen:** no `gas_agent/` module is touched; the only `app.py` change wraps
  the original dashboard body (renamed `render_gas_tab()`) in `st.tabs([...])`.

---

## Live numbers (default landing view)

**Handmade Bowl / 5,000 units / 14-day timeline / medium competition**, default
weights (gas 40% · clay 35% · energy 15% · transport 10%), mock forecast:

- **Lock 40%** of next-quarter input cost — blended band **43% (moderate)** → mid lock.
- Chosen supplier **Alpine Clay Works** (mid-ranked, score 72); channel **Online Retail
  Export** (score 87).
- Negotiated: buy **€0.86/unit**, sell **€1.51/unit** → unit margin **€0.65**, **€3,253
  total**.
- Backtest (12 months): agent **€34,290/mo**, **+65%** vs random (€20,792), **+1%** vs
  cheap+best-margin (€33,819).

The mid-ranked supplier (not the top-scored one) is correct: a 40% lock falls in the
"mid lock" band. Change the weights and the lock % — and the whole recommendation —
moves; the same inputs always reproduce the same numbers.

---

## Run / regenerate

```bash
uv run streamlit run app.py                        # Ceramics optimizer tab
uv run pytest -q tests/test_ceramics_*.py          # the 68 ceramics tests (offline)
uv run python scripts/build_ceramics_forecast.py   # regenerate the committed mock forecast
```
