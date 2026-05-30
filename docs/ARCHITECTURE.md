# Architecture & Design Rationale

*How the TTF gas-hedging agent was built, and **why each decision was made**.*

Where `APP_GUIDE.md` answers "what does this do and how do I use it," this document
answers "**why is it built this way?**" — the choices, the trade-offs, and the
alternatives we deliberately rejected. It's written so a judge (or a future
maintainer) can see that nothing here is accidental.

---

## Table of contents

1. [The framing decisions](#1-the-framing-decisions)
2. [THE RULE, and why it's the whole point](#2-the-rule-and-why-its-the-whole-point)
3. [Pipeline, stage by stage — what & why](#3-pipeline-stage-by-stage--what--why)
4. [Deep dives on the load-bearing choices](#4-deep-dives-on-the-load-bearing-choices)
5. [Technology choices](#5-technology-choices)
6. [How the three judging axes are covered](#6-how-the-three-judging-axes-are-covered)
7. [What we descoped, and why](#7-what-we-descoped-and-why)
8. [Build order (how it actually came together)](#8-build-order-how-it-actually-came-together)

---

## 1. The framing decisions

Before any code, four product choices shaped everything else.

### 1.1 Why a *decision* agent, not a *forecasting* model

The hackathon is a **forecasting** track owned by Sybilion, so the temptation is to
try to forecast the gas price better. We deliberately did **not**. We measured
Sybilion's point forecast on this series and it's weak — **~28% MAPE** in the cached
backtest. Competing on point accuracy would be a losing game and, worse, *boring*: a
slightly-better number isn't a decision a business can act on.

So we moved one layer up: take the forecast as given (band and all) and build the
thing a buyer actually needs — **a decision**. This reframes Sybilion's weak point
forecast from a liability into the *premise* of the design: *because* the midpoint is
unreliable, you must decide on the **uncertainty band**, not the point. That's a more
defensible and more interesting product.

### 1.2 Why European gas (TTF)

We needed a domain where (a) uncertainty genuinely matters to a real buyer, (b) there
are rich external drivers with a clear causal story, and (c) there's an obvious,
relatable "assumption shifts mid-run" event for the adaptivity axis. European gas
fits all three: it's volatile, it has well-understood supply drivers (Norway, Russia,
LNG, FX), and **geopolitical supply shocks** (Strait of Hormuz, Nord Stream,
sanctions) are exactly the mid-run shift judges want to see handled.

### 1.3 Why the "German glass manufacturer" persona

The decision needs an owner with skin in the game. A **mid-size German glass &
ceramics manufacturer** is ideal: gas is its single largest volatile cost, it buys
forward each quarter, and it is risk-averse (a factory can't absorb a price spike the
way a trader can). That risk-aversion is what makes **variance reduction**, not
expected-cost minimisation, the right objective — which in turn justifies the whole
hedging policy (see §4.3).

### 1.4 Why "hedge ratio," specifically

The output had to be a single, actionable number a procurement lead can sign off on.
"What % of next quarter do I lock forward now vs. leave to spot?" is that number. It's
bounded [0,1], it's intuitive, and it maps cleanly onto the forecast's uncertainty
(more confidence → lock more). A price target would have been less actionable and
would have re-introduced the point-forecast problem we just escaped.

---

## 2. THE RULE, and why it's the whole point

> **The LLMs prepare inputs and explain outputs. They never compute the hedge ratio.
> `hedge_policy.py` does, deterministically.**

This isn't a stylistic preference — it's the design's credibility. An LLM that both
decides *and* explains can rationalise anything; you can never tell whether the number
came from analysis or from vibes, and you can't reproduce it. By forcing the decision
through deterministic arithmetic and confining the LLMs to (a) *preparing* the data
request and (b) *narrating* a number that's already fixed, we get three things judges
care about:

- **Reproducibility** — same inputs, same decision, every time.
- **Traceability** — every component of the ratio (band term, drift tilt, risk
  premium) is exposed in the UI's decision table; you can audit the math by hand.
- **Faithful explanations** — the explainer is given the final number and the real
  driver names and is told, in its prompt, that the decision is final and it must not
  invent figures or propose a different ratio.

Concretely, the rule is enforced at every LLM boundary:

| LLM touchpoint | What it's allowed to do | What it must never do |
|----------------|-------------------------|------------------------|
| `keyword_agent` | Pick Sybilion category/region/keyword filters from the catalog | Invent IDs; decide anything |
| `scenario.parse_shock_request` | Classify a message → `{is_shock, magnitude, label}` | Touch the ratio |
| `explanation_agent` | Narrate *why* the fixed ratio is what it is | Invent drivers/numbers; re-decide |
| `geo.country_brief` | Explain why a country moves gas | Give hedging advice / a ratio |

Every one of these also has a **deterministic fallback**, so the rule holds even when
the model is offline.

---

## 3. Pipeline, stage by stage — what & why

```
Featherless picks Sybilion filters
  → Sybilion returns forecast + ranked drivers
  → curation drops the spurious drivers
  → deterministic policy sets the hedge ratio
  → Featherless explains it
  → (live) a supply shock can re-run the whole thing on shocked inputs
```

### Stage 1 — Keyword agent (`keyword_agent.py`) · *prepares input*

**What:** A small LLM (Mistral-Small) reads the buyer persona and selects the Sybilion
**category IDs, region codes, and search keywords** for the forecast call.

**Why this exists:** Sybilion's value is its driver discovery, but it needs to be
pointed at the right slice of the catalog. Hand-coding filters would be brittle and
wouldn't showcase the agentic loop. So the LLM configures the request.

**Why it "selects, never invents":** The real catalog is embedded in the prompt and
*every* returned ID is validated against it (`_validate_selection`); anything
off-catalogue is dropped. An LLM that hallucinates a category ID would silently break
the forecast — validation makes that impossible.

**Why a closed loop:** If curation later keeps too few credible drivers
(`needs_refine`, < 8), `refine_filters` widens the selection (union with the full
credibility whitelist) and re-pulls. The widening is deterministic so the loop makes
progress even with no LLM.

### Stage 2 — Sybilion forecast + drivers · *the data*

**What:** Sybilion returns a per-month **quantile forecast** (q05/q10/q50/q90/q95) and
a ranked list of **external driver** series it found correlated with TTF.

**Why we cache it:** The artifacts for the demo job are fetched once (via API/MCP
during dev) and stored under `cache/<job_id>/`. The dashboard reads the cache so the
demo is **instant and network-independent** — no live polling on stage. The REST
client (`sybilion_client.py`) is a faithful documented client for the live path, but
the main path is cache-first by design.

### Stage 3 — Driver curation (`driver_curation.py`) · *THE EDGE, deterministic*

**What:** Splits Sybilion's ranked drivers into **kept** (credible for European gas)
vs **rejected** (spurious), with a per-driver reason.

**Why it's the edge:** Sybilion ranks by *in-sample correlation*, which inevitably
surfaces **spurious correlates** — "Population — Sri Lanka," "Population — Serbia" —
that track the gas price by coincidence, not cause. Acting on those would be the
classic correlation-≠-causation trap. Curation is the agent's domain judgment made
explicit and visible: it keeps "Exports of Natural gas in Europe" and throws out
demographic proxies, and *shows you both piles*. On the current job that's **25 kept /
6 rejected**.

**Why rule-based, not an LLM:** Credibility is a stable, auditable judgment — an
energy/markets whitelist (gas, LNG, oil, power, coal/carbon, FX, rates, commodities,
trade flows) vs a demographic blacklist (population, fertility, mortality, tourism).
Rules are reproducible and fast; an LLM would be neither, and this is too important to
the decision's integrity to leave to a model. (Order matters: spurious keywords are
checked *first* so a demographic series can't sneak in on a stray "energy" substring.)

### Stage 4 — Hedge policy (`hedge_policy.py`) · *THE CORE, deterministic*

**What:** Turns each month's `{median, q10, q90}` + today's spot into a hedge ratio.
Detailed in §4.1.

**Why here and only here:** This is the one place the decision is made, by arithmetic,
so it's reproducible and traceable (THE RULE, §2).

### Stage 5 — Explanation (`explanation_agent.py`) · *explains output*

**What:** Mistral-Large narrates *why* the already-fixed ratio is what it is, citing
the real kept-driver names and the real band/drift numbers.

**Why a bigger model here:** Filter-picking needs reliable *structured* output (small
model, temperature 0). Explanation needs *fluent, faithful prose* a procurement lead
will read — worth the larger model. The prompt hard-codes the rules ("the band sets
the base size, drift is only a secondary tilt, the ratio is final, never invent") so
the narrative can't drift from the math.

### Stage 6 — Adaptive shock (`scenario.py`) · *the mid-run shift*

**What:** A free-text headline ("Iran closes the Strait of Hormuz") re-runs the entire
decision on **shocked inputs**: median bumped up, band widened, a risk premium added
to the lock floor, and the driver mix re-ranked so the geopolitical supplier leads.
Detailed in §4.4.

**Why it stays inside THE RULE:** The LLM only classifies the message into
`{is_shock, magnitude, label}` — *severity*, never the ratio. Everything that moves
the number is deterministic arithmetic fed into the *same* `hedge_policy`. So the
adaptivity is reproducible: the same shock always produces the same new decision.

---

## 4. Deep dives on the load-bearing choices

### 4.1 The hedge-policy formula — why this shape

For each month:

```
band_width      = (q90 − q10) / median                      # relative width of the 80% band
ratio_from_band = clamp(1 − (band_width − low)/(high − low), 0, 1)   # confidence → size
drift_pct       = (median − spot) / spot                    # forward vs today's spot
direction_tilt  = clamp(drift_pct × tilt_gain, −max_tilt, +max_tilt) # secondary tilt
hedge_ratio     = clamp(ratio_from_band + direction_tilt + risk_premium, min_hedge, max_hedge)
```

- **Band sets the base size (not the midpoint).** This is the core thesis: a *narrow*
  band means the model is confident → little reason to keep optionality → lock more; a
  *wide* band means the future is uncertain → keep optionality → lock less. The
  decision is driven by *how sure* the forecast is, which is exactly the part of a
  weak forecast that's still informative.
- **Drift is only a secondary tilt.** If the forward sits above spot, waiting costs
  more → tilt toward locking; below spot → tilt away. We **cap it** (`max_tilt` ±0.30)
  so the unreliable point estimate can nudge but never dominate the decision. This is
  the formula encoding "don't trust the midpoint too much."
- **Floor and cap (`min_hedge` 0.10, `max_hedge` 0.90).** Never lock nothing (you'd
  carry full price risk) and never lock everything (you'd lose all flexibility and
  any chance to benefit from a fall). A real procurement desk always keeps some of
  both.
- **Why those band thresholds (`low_band` 0.30, `high_band` 1.10).** They're
  **calibrated to this series' own backtest band widths** — `low_band` near the 15th
  percentile (unusually confident), `high_band` near the observed max (almost no
  confidence). So "tight" and "wide" mean tight/wide *relative to what this model
  actually produces*, not arbitrary constants. That's why on the current job the bands
  read as "moderate" and the calm quarter ratio lands at a sensible ~26%.

### 4.2 Why a *clamped linear* policy and not something fancier

We considered a learned/optimised policy. Rejected it: with ~28% MAPE and a short
series, any optimiser would overfit, and — fatally — it would be **unexplainable**,
which kills the traceability axis. A transparent linear map from band+drift to a
clamped ratio is something a judge can verify by hand and a procurement lead can trust.
The decision table in the UI shows each term separately for exactly this reason.

### 4.3 The backtest's lock-price proxy — the most important honest choice

The decision backtest (`decision_backtest.py`) replays the policy over Sybilion's
historical windows and compares realized cost against **always-spot (0%)** and
**always-lock-50%**. The subtle decision is *what price you pay when you "lock."*

- **We proxy the lock price with the decision-time spot**, not the forecast median.
- **Why:** the artifacts don't carry a historical forward curve; a forward sits near
  today's spot (ignoring small carry/seasonality), and a buyer can actually transact
  there. Crucially, **locking at the model's median would be betting on the weak point
  forecast — the exact thing this agent refuses to do.** The forecast's only job is to
  *size* the ratio (via band + drift); the price you pay is the market's.
- **Why we say it out loud:** it's a *stated* assumption, surfaced in the UI caption,
  not a hidden one that flatters the result.
- **The honest verdict it produces:** in a *falling* market nothing beats pure spot on
  average cost, so the win we claim is **variance reduction** (the whole reason a
  factory hedges) plus beating a blind 50% lock. On the cached job, across **63
  replayed months**: the policy is **€1.25/MWh cheaper** than always-spot **and**
  **€2.34/MWh tighter** in cost swing, and **matches** a static 50% lock (policy
  €35.71 ± 4.47; spot €36.96 ± 6.81; lock-50% €35.25 ± 4.35). We lead with the
  variance win because it's the defensible one — and we don't overclaim the cost win.

### 4.4 The shock — why it makes the ratio *rise*

A naive intuition says "more uncertainty → wider band → lock *less*." A supply shock
is the opposite, and encoding that correctly is the point:

- `apply_shock` lifts the **median** (a shock pushes the forward above spot) and widens
  the **band** (the future got murkier).
- The wider band pulls `ratio_from_band` *down*, but the lifted median pushes
  `drift_pct` (and the tilt) *up*, and a **`risk_premium`** is added straight onto the
  lock floor (tail insurance).
- Net: the **hedge ratio rises**. Economically correct — a supply shock skews risk to
  the upside, so a buyer locks *more* now even though the point forecast is murkier.
- The same shock re-ranks the drivers (`shocked_curation`): risk suppliers
  (Iran/Qatar/Russia/Algeria) get a magnitude-scaled boost and a synthetic "Global
  supply-risk premium" driver is injected at the top, so the chart, the globe, and the
  explanation all tell the new story. This ties the *driver mix* to the *decision*.

### 4.5 Why the globe is Plotly orthographic (not pydeck)

The first globe used pydeck's `_GlobeView` + a `ColumnLayer`. It rendered **flat** and
relied on fetching external Natural-Earth GeoJSON from a CDN — a live network
dependency that could die on stage. We replaced it with a **Plotly `Scattergeo`
orthographic projection**: it's genuinely drag-spinnable, Plotly ships the country
geometry so there's **no external basemap fetch**, and Streamlit's
`st.plotly_chart(..., on_select="rerun")` gives click-to-drill-down for free. Same
data (green columns = kept importance, red markers = spurious-only), zero stage risk.

### 4.6 Voice — why NVIDIA + local, never Featherless

Narration is a "wow" stretch. The plan was NVIDIA primary + Featherless fallback —
until we hit a **404: Featherless serves text only, it has no `/v1/audio/speech`
endpoint.** So the architecture is **NVIDIA Riva (gRPC) primary → local OS `say`
fallback**:

- A **60-second sliding-window rate limiter** guards NVIDIA's free tier (~40 req/min):
  a deque of call timestamps; over the limit (or on a 429) it skips NVIDIA and uses the
  local engine. This is the "rate-limited fallback" behaviour, just to a different
  secondary provider than first planned.
- **Pre-synthesised clips** (`scripts/build_voiceover.py` → `cache/audio/`) make the
  on-stage path instant and the live limit essentially unreachable during a demo.
- If no provider is reachable, `synthesize()` returns `None` and the player simply
  hides — the dashboard is unaffected. Voice is strictly additive.

---

## 5. Technology choices

| Choice | Why |
|--------|-----|
| **Sybilion API** | The track's forecasting engine; gives probabilistic bands + driver discovery, which is exactly what a band-based decision needs. |
| **Featherless (OpenAI-compatible)** | Hosted inference for the European **Mistral** models — fits the "European AI sovereignty" framing for a European-gas product, and the OpenAI-compatible API keeps the client trivial. |
| **Mistral-Small** for filter-picking | Needs reliable, cheap *structured* JSON output at temperature 0 — small model is plenty. |
| **Mistral-Large** for explanation | Needs *fluent, faithful prose* a human will read — worth the larger model. |
| **Streamlit** | Fastest path to an interactive, chart-heavy dashboard with chat + session state; single-process, which is why the in-process voice rate limiter works. |
| **Plotly** | Drag-spinnable orthographic globe + price-band fills + click events, all with bundled geometry (no CDN). |
| **NVIDIA Riva (gRPC)** | The only reachable real TTS; needs the `nvidia-riva-client` (lazy-imported so it's optional). |
| **uv** | Fast, reproducible Python env. |
| **Deterministic everything in the decision path** | Reproducibility + traceability — the two axes that separate this from "an LLM said so." |

Models are swappable strings in `.env` (`KEYWORD_MODEL`, `EXPLANATION_MODEL`), so the
provider/model can change without touching code.

---

## 6. How the three judging axes are covered

| Axis | Where it lives | The argument |
|------|----------------|--------------|
| **1. Beats a naive baseline** | `decision_backtest.py` + backtest panel | 63-month replay: cheaper *and* steadier than always-spot, matches a 50% lock — with the lock-price assumption stated, not hidden. |
| **2. Traceable reasoning** | `hedge_policy` exposes every term; decision table; curation kept-vs-rejected; faithful LLM narrative; per-country briefs | You can audit the ratio by hand; the LLM only narrates a number it didn't choose. |
| **3. Adaptive mid-run** | `scenario.py` + sidebar chat/button | A typed shock re-runs the *same* deterministic policy on shocked inputs; the ratio rises, charts/globe/explanation update live, and it's reproducible. |

---

## 7. What we descoped, and why

- **Beating Sybilion's point forecast** — explicitly *not* the goal (§1.1); we build
  on the band instead.
- **A learned/optimised policy** — would overfit a short, noisy series and, worse, be
  unexplainable (§4.2).
- **Live on-stage re-forecasting** — the REST client supports submit→poll→fetch, but
  the main path is cache-first for speed and reliability; only a "re-pick filters"
  toggle re-runs the keyword agent live. (`validate_forecast_data` is the cheap
  pre-check that would guard any live submit.)
- **AR / 3D headset output** — no judging-axis payoff, high integration risk in 36h.
- **A real historical forward curve in the backtest** — not in the artifacts; the
  stated decision-time-spot proxy is the honest stand-in (§4.3).

---

## 8. Build order (how it actually came together)

The deterministic core was built and tested **first**, so the LLMs could be layered on
top of something trustworthy rather than the other way around:

1. Scaffold + TTF series (`scripts/build_ttf_series.py` → `data/ttf_series.json`).
2. Run the real Sybilion forecast, cache the artifacts (`cache/<job>/`).
3. **`hedge_policy.py` + tests** — the deterministic decision, proven before anything
   could depend on it.
4. `sybilion_client.py` (REST + cache loaders) and a minimal dashboard.
5. **`driver_curation.py`** (the edge) + the keyword closed loop.
6. `explanation_agent.py` — the faithful narrator.
7. **`decision_backtest.py`** — axis #1, the proof it beats baselines.
8. **`scenario.py`** + chat — axis #3, the live adaptivity.
9. `geo.py` + the globe — axis #2, made spatial.
10. `voice.py` — the stretch narration.
11. Docs (this file, `APP_GUIDE.md`, `CLAUDE.md`).

Each step was chosen to either (a) cover a judging axis or (b) make an existing axis
more legible — never decoration for its own sake. The flashy surfaces (globe, voice,
live chat) are the *delivery vehicles* for the substance underneath, not a substitute
for it.
