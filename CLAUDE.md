# CLAUDE.md — project state & handoff

This file is the briefing for any Claude session (or human) that picks up this repo.
Read it first. It tells you what the project is, the rules you must not break, where
everything lives, how to run it, and what state the work is in right now.

> **Companion docs:** `docs/APP_GUIDE.md` (how the app works, for a user/judge) and
> `docs/ARCHITECTURE.md` (how it was built and *why* each choice was made). This file
> is the operational handoff; those two are the explanatory references.

---

## 1. What this project is

A **gas-hedging decision agent** built on top of the **Sybilion** probabilistic
forecasting API, for the hackathon "Forecasting AI" track.

- **Domain:** European natural gas, the Dutch **TTF** benchmark (EUR/MWh).
- **Persona:** a mid-size **German glass & ceramics manufacturer** that buys gas
  forward each quarter (gas is its largest volatile cost).
- **The decision it makes:** a **per-month hedge ratio** — *what share of next
  quarter's gas to lock in forward now vs. leave to buy on the spot market.* It is
  **not** a price prediction. Sybilion's point forecast is weak here (~28% MAPE),
  so the decision is built on the forecast's **confidence band + curated driver
  mix**, never the point estimate.
- **Shape of the pipeline:** Featherless LLM (prepares inputs) → Sybilion API
  (forecast + drivers) → deterministic curation + hedge policy (decides) →
  Featherless LLM (explains) → Streamlit dashboard (+ live shock scenario, globe,
  voice narration).

The three hackathon judging axes the build targets:
1. **Decision beats a naive baseline** — `decision_backtest.py` + the backtest panel.
2. **Traceable reasoning** — every number in the decision is exposed; LLMs only narrate.
3. **Adaptive when an assumption shifts** — the live supply-shock scenario.

### A second decision agent — ceramics supply-chain optimizer

The same spine also powers a **second** agent, for the same manufacturer's
**ceramics** line. For one production run (product / quantity / timeline) under
user-set cost-factor weights (gas / clay / power / transport) it decides three
things — **how much input cost to lock now**, **which supplier to buy from**, and
**which channel to sell through** — then runs a two-round negotiation and a
3-strategy backtest. The "lock %" is literally a hedge ratio on a *blended* 4-factor
cost band, so it **reuses `gas_agent.hedge_policy` unchanged**
(`HedgePolicyParams(low_band=0.25, high_band=0.60)`). Same invariants: deterministic
& auditable (the LLM only explains, never computes a number) and offline-first
(committed mock forecast + template narrative). Lives in its own `ceramics_agent/`
package (mirrors `gas_agent/`). See §5 for live numbers and `docs/CERAMICS_AGENT.md`.

### Now unified into one chat-centric single page (the `edges.md` "unified app" wave)

The two agents are no longer separate tabs — they are **one professional single
page** driven by one company description (W1–W12, all built & verified, 270 tests):

- **One intake** (`ceramics_agent/intake.py`): describe the business in free text →
  the LLM *extracts stated facts only* (never invents a number) → it asks 1–3
  follow-ups for missing fields → that one description is the shared persona for
  **both** decisions. A staged **pipeline animation** (`pipeline.py`) plays while it
  forecasts (rotating LLM blurbs, template fallback).
- **Stacked decisions** (`app.py`, tabs dropped): a cross-decision **driver-impact
  panel** (`impact.py` — "what moves what"), then the gas hedge section, then the
  ceramics section, under one **Live⟷Cached** toggle + a gas/ceramics focus control.
- **3D globe, both layers** (`ceramics_agent/geo.py`, reuses the gas geo engine): a
  toggle between **where to sell** (demand markets, sized by demand potential) and
  **where to buy** (supplier sourcing, sized by curation score), with click-to-brief.
- **One shock, both agents** (`ceramics_agent/scenario.py`): a single supply-shock
  headline in the bottom chat re-decides the gas hedge **and** the ceramics lock at
  once (factor-routed), all deterministic — THE RULE holds.
- **Wide bottom chat + voice** (W7): one message routes to both agents; "why"
  questions explain either decision; and the agent can **explain itself** —
  *"what is this app / why this design?"* answers from `voice_chat.APP_OVERVIEW`
  (explanation-only, W11).
- **Backtests as a headline** (W12): extra baselines (gas always-100% + seeded
  random ratio; ceramics always-top-ranked), a **24-month** ceramics robustness
  replay, and a **shocked-scenario** gas replay — all *additive*, with the original
  verdicts **byte-identical** (re-verified, see §5).
- **Professional restyle** (W8): `.streamlit/config.toml` dark theme + a CSS layer
  (`app._inject_global_css`) — hero, card metrics, accent headers.

`gas_agent/` decision modules stay frozen; the only gas edits are **additive,
behaviour-preserving** ones (`voice.synthesize(providers=…)`, a `GAS_BACKTEST_SEED`
config constant, the two extra backtest baselines + a `shock_magnitude` param on
`decision_backtest` that defaults to the byte-identical calm replay, and the
`voice_chat` about-intent). Submission docs (`REPORT.md`, expanded `README.md`,
`requirements.txt`) are in place; the demo runs keyless from a clean checkout.

---

## 2. Hard invariants — DO NOT BREAK THESE

These are the spine of the project. Violating any of them breaks the demo's
credibility, not just a test.

1. **THE RULE: the LLM never computes the hedge ratio.** `hedge_policy.py` does,
   deterministically, from the band + drift (+ an optional shock risk-premium).
   LLMs *prepare inputs* (`keyword_agent`, the shock classifier) and *explain
   outputs* (`explanation_agent`, `geo.country_brief`). They are explicitly
   forbidden, in their prompts, from inventing drivers/numbers or re-deciding the
   ratio. If you add an LLM call, it must not produce or alter the ratio.
2. **Plain-English naming throughout.** `band_width`, `hedge_ratio`, `keep_driver`,
   `drift_pct` — no cryptic abbreviations. Match the existing style.
3. **The demo must run with no keys.** Every external call has a deterministic
   fallback and the dashboard reads cached Sybilion artifacts. Never introduce a
   hard dependency on a live API for the main path.
4. **Secrets stay out of git.** `.env` holds the **real** `SYBILION_API_KEY` and
   `FEATHERLESS_API_KEY` and is gitignored. Never commit it or echo its contents.
5. **Never commit unless the user explicitly asks.** (And never `git add .` blindly —
   stage named files so `.env` can't slip in.)
6. **Determinism is the selling point.** The same inputs must always yield the same
   decision. Keep randomness and model output out of the decision path.

---

## 3. Repo map

```
hackathon/
├── app.py                       # Streamlit dashboard — wires everything together
├── CLAUDE.md                    # this file
├── pyproject.toml               # Python ≥3.11, deps; pytest config
├── .env / .env.example          # secrets + model/voice config (.env is gitignored)
├── data/
│   └── ttf_series.json          # committed monthly TTF history (the spot anchor)
├── cache/
│   ├── latest_job.txt           # → f445eec1-62e2-43e1-9995-18ba7ee668c3
│   ├── mock_ceramics_forecast.json  # committed 4-factor ceramics mock (the offline demo source)
│   ├── <job_id>/                # cached real Sybilion artifacts (gitignored *.json)
│   │   ├── forecast.json
│   │   ├── external_signals.json
│   │   ├── backtest_metrics.json
│   │   └── backtest_trajectories.json
│   └── audio/                   # pre-synthesised narration (gitignored, regenerable)
│       ├── golden.wav + golden.json   (sidecar: provider/voice/mime)
│       └── shock.wav  + shock.json
├── sybilion_forecast/           # reference copies of the artifacts + input.json payload
├── gas_agent/
│   ├── config.py                # env vars, paths, have_*_key() helpers
│   ├── llm.py                   # Featherless (OpenAI-compatible) wrapper; graceful fallback
│   ├── catalog.py               # Sybilion category/region IDs + credibility whitelist
│   ├── keyword_agent.py         # LLM → Sybilion filter selection (selects, never invents)
│   ├── driver_curation.py       # THE EDGE: deterministic keep/reject of drivers
│   ├── hedge_policy.py          # THE CORE: deterministic hedge-ratio engine
│   ├── scenario.py              # adaptive supply-shock (deterministic math + LLM classifier)
│   ├── explanation_agent.py     # LLM narrates the already-decided ratio
│   ├── decision_backtest.py     # replay vs 0/50/100% + seeded random ratio + shocked (W12)
│   ├── geo.py                   # country coords + aggregation + per-country brief
│   ├── sybilion_client.py       # REST client + disk cache + artifact parsers
│   ├── transcribe.py            # push-to-talk ASR ladder (NVIDIA Riva → HF Whisper → None)
│   ├── voice_chat.py            # grounded Q&A + APP_OVERVIEW self-knowledge (explain-only, W11)
│   └── voice.py                 # TTS: NVIDIA Riva + local `say`; additive synthesize(providers=…)
├── ceramics_agent/              # SECOND AGENT — mirrors gas_agent/ (imports gas, never edits it)
│   ├── catalog.py               # products (BOM), suppliers, channels, history (12) + EXTENDED (24)
│   ├── forecast.py              # 4-factor cost-forecast loader (mock default; live Sybilion opt-in)
│   ├── cost_policy.py           # blended cost band → lock % (REUSES gas_agent.hedge_policy)
│   ├── curation.py              # deterministic keep/reject + score/rank of suppliers & channels
│   ├── negotiation.py           # two-round, market-anchored negotiation (deterministic)
│   ├── backtest.py              # replay: agent vs random / cheap / always-top-ranked (W12)
│   ├── recommend.py             # orchestrates forecast→lock→curation→negotiation→backtest (pure)
│   ├── explanation.py           # LLM narrates the decided recommendation (explain-only; template fallback)
│   ├── intake.py                # W1: company-description → CompanyProfile (LLM extracts, never invents)
│   ├── pipeline.py              # W2: staged pipeline-animation stage list + LLM blurbs (template fallback)
│   ├── impact.py                # W4: cross-decision "what moves what" driver-impact rows (pure)
│   ├── scenario.py              # W6: one shock re-decides ceramics too (factor-routed, deterministic)
│   ├── geo.py                   # W5: globe both layers — where to sell / where to buy (reuses gas geo)
│   └── dashboard.py             # the ceramics SECTION (tabs dropped — render_ceramics_tab + globe)
├── .streamlit/
│   └── config.toml              # W8: professional dark theme (palette behind the CSS layer)
├── scripts/
│   ├── build_ttf_series.py      # regenerate data/ttf_series.json from Yahoo TTF=F
│   ├── save_cached_artifact.py  # import an MCP-exported artifact into cache/<job>/
│   ├── build_voiceover.py       # pre-synthesise cache/audio/{golden,shock}.wav
│   └── build_ceramics_forecast.py   # generate & commit cache/mock_ceramics_forecast.json
├── REPORT.md                    # submission write-up (TL;DR, approach, real numbers, credits)
├── README.md                    # clean-checkout setup/run + no-keys demo + what's live vs mocked
├── requirements.txt             # exported from uv.lock so `pip install -r` works on a clean checkout
├── tests/                       # 270 tests, all green (offline, deterministic)
└── docs/
    ├── APP_GUIDE.md             # user guide (how it works, how we know it's good)
    ├── ARCHITECTURE.md          # build & design-rationale (why each choice)
    └── CERAMICS_AGENT.md        # the second agent — design note + live numbers
```

---

## 4. How to run

```bash
uv sync                              # install deps
cp .env.example .env                 # optional: add real keys (demo runs without them)
uv run streamlit run app.py          # the dashboard — one page, both decisions, runs keyless
uv run pytest -q                     # 270 tests, all pass (offline, deterministic)
uv run python -m gas_agent.decision_backtest   # gas backtest verdict (+ extended baselines + shocked)
uv run python -m ceramics_agent.backtest       # ceramics backtest verdict (+ top-ranked + 24-month)
uv run python scripts/build_voiceover.py       # regenerate narration into cache/audio/
uv run python scripts/build_ceramics_forecast.py  # regenerate the committed ceramics mock
```

- **uv** is the package manager. Python ≥ 3.11 (dev on 3.13).
- The dashboard reads `cache/latest_job.txt` → loads that job's cached artifacts.
  If it's missing/mispointed you get "No cached forecast found."

---

## 5. Current state (as of 2026-05-31)

**Everything is built and verified, including the full unified-app wave (W1–W12).**
All three judging axes are covered. **270 tests pass** (offline, deterministic). The
app was re-verified end-to-end in the browser: intake → pipeline animation → both
decisions, the globe toggles sell/buy, the gas backtest shows five baselines, the
ceramics backtest shows four + a 24-month robustness line, the chat explains itself,
and **zero Streamlit exceptions**. The **gas decision math is byte-identical** (calm
quarter ratio 30.5%, standing premium +6.9%, curation 25 kept / 6 rejected, backtest
verdict unchanged — see the tables below). New live numbers from W12 are in the
sub-tables further down. The unified-app design is summarised in §1 ("Now unified
into one chat-centric single page"); `docs/` (APP_GUIDE / ARCHITECTURE) still describe
the pre-unification tabbed layout and are the next doc to refresh if time allows.

A second wave (the `edges.md` follow-up) is also built and verified:
1. **Future-proof whitelist keywords** — `driver_curation` now recognises Brent,
   a `global risk & volatility` theme (VIX / geopolitical / risk index), and a
   `macro indicators` theme (PMI / inflation / CPI). No-ops on the current cached
   job (no such series yet); they classify as *keep* the day Sybilion surfaces them.
2. **Standing supply-risk premium** — deterministic "dynamic weighting":
   `scenario.standing_risk_premium(curation)` turns the kept-driver risk-importance
   share into a premium fed through the **existing** `hedge_policy.risk_premium`
   (THE RULE holds — pure arithmetic, no LLM). It lifts the calm lock and composes
   with the shock. The backtest stays **premium-free** (a transparent live overlay).
3. **Optional live Sybilion refresh** — `sybilion_client.build_forecast_payload` +
   `run_live_forecast` (submit→poll→cache 4 artifacts, never repoints
   `latest_job.txt`), behind a sidebar toggle (default OFF, disabled with no key).
   Cached demo is restored instantly on toggle-off.
4. **Push-to-talk voice assistant** — `transcribe.py` (NVIDIA Riva ASR → HuggingFace
   Whisper → None, sharing `voice.py`'s rate-limit window) + `voice_chat.answer_question`
   (grounded, explanation-only, Featherless→template). `st.audio_input` mic in the
   sidebar; hidden when no ASR key (text chat + no-keys demo unchanged).

Live numbers from the cached job `f445eec1-…` (so future sessions can sanity-check
they haven't regressed the decision):

| Quantity | Value |
|----------|-------|
| Today's spot (last actual TTF) | **€47.28/MWh** |
| Forecast horizon | 2026-06 … 2026-11 (6 months) |
| **Calm next-quarter hedge ratio** | **~30.5%** (months: 53% / 10% / 29%) — includes the standing premium below; the premium-free core is **25.8%** |
| **Standing supply-risk premium** | **+6.9%** on the lock floor — risk-region/global-risk drivers are **34.7%** of kept-driver importance (deterministic, via `hedge_policy.risk_premium`) |
| Curation | **25 kept / 6 rejected** (needs_refine = False) |
| Top kept driver | "Exports of Oil and petroleum products in Germany" |
| Rejected sample | Population — Sri Lanka / Serbia / Bangladesh / Europe |
| Backtest | **63 replayed months** (premium-free): policy €1.25/MWh **cheaper** than spot, €2.34/MWh **tighter** swing, **matches** a 50% lock (policy €35.71±4.47; spot €36.96±6.81; lock-50% €35.25±4.35) |
| Backtest extra baselines (W12) | always-lock-100% **€33.54±4.16**, seeded random ratio **€35.48±4.98** — policy's ±4.47 swing beats the coin flip's ±4.98; always-100% is lowest here *only because this window rose* (a directional bet, dearest in a falling market). Headline trio above stays **byte-identical**. |
| Backtest shocked replay (W12, mag 1.0) | prices spike + policy hedges more → policy **€35.93/MWh**, **€5.46 cheaper** than buying at the shocked spot (€41.40): the logic still beats the baselines under a mid-run shift |
| Shock (magnitude 1.0) | next-quarter ratio **rises** to **~40.1%** vs the new calm 30.5% (applied premium +32%); Iran/risk supplier lights up green on the globe; "Global supply-risk premium" leads the drivers |

### Ceramics optimizer (second agent) — built & verified

The full C1–C8 build is done: `catalog` (BOM products, suppliers, channels,
historical sales) → `forecast` (4-factor loader, mock default + live Sybilion
opt-in) → `cost_policy` (blended band → lock %, reusing `gas_agent.hedge_policy`) →
`curation` (score/rank suppliers & channels) → `negotiation` (two rounds) →
`backtest` (3 strategies + always-top-ranked) → `recommend` (orchestrator) →
`explanation` (explain-only LLM + template) → `dashboard` (the ceramics **section**,
now stacked below the gas one — tabs dropped in W3 — plus the W5 globe). It runs fully
offline on the committed `cache/mock_ceramics_forecast.json`; the live 4-factor path is
behind the `SYBILION_API_KEY` guard and **never** repoints `latest_job.txt`. Verified
end-to-end in the browser: both sections render, **the gas decision is byte-identical**,
zero Streamlit exceptions, and identical inputs yield identical numbers.

Default landing view — **Handmade Bowl / 5,000 units / 14-day timeline / medium
competition** / default weights (gas 40% · clay 35% · energy 15% · transport 10%),
mock forecast, so a future session can sanity-check the decision hasn't regressed:

| Quantity | Value |
|----------|-------|
| Forecast | 4 factors (gas / clay / power / shipping) × 6 months (2026-06 … 2026-11), `source = "mock"` |
| **Lock now (next-quarter input cost)** | **40%** — blended cost band **43% (moderate)** → "mid lock, balanced mid-ranked supplier" |
| Physical unit cost | ≈ **€0.75/unit** (median, from BOM × the 4 factor bands) |
| Suppliers (kept / ranked) | Eastern European Materials **98** (cost 100 · rel 92 · lead 100) · **Alpine Clay Works 72** ✅ chosen (cost 67 · rel 96 · lead 50) · Premium Ceramics Supply **44** |
| Channels (kept / ranked) | **Online Retail Export 87** ✅ chosen (margin 100 · season 56 · order-fit 100) · Hospitality & Restaurant Supply **73** |
| **Negotiated deal** | buy from Alpine at **€0.86/unit**, sell via Online at **€1.51/unit** → unit margin **€0.65**, **€3,253 total** (× 5,000) |
| **Backtest (12 replayed months)** | agent **€34,290/mo** realized margin — **+65%** vs a seeded-random pick (€20,792), **+1%** vs static cheap+best-margin (€33,819), **+1%** vs always-top-ranked (€33,819, ignores the lock routing — W12) |
| **Backtest robustness (24-month replay, W12)** | prior year + recent year: agent **€32,918/mo**, **+60%** vs random — the edge isn't an artefact of one 12-month window. Headline 12-month verdict stays **byte-identical**. |

Note: the chosen supplier is the **mid-ranked** Alpine (not the top-scored Eastern)
**by design** — a 40% lock falls in the "mid lock" band, which routes to the balanced
mid-ranked supplier (`select_supplier`). High lock → top-ranked; low lock → cheapest.
The backtest discounts each supplier's nominal margin by its reliability (a stated
assumption); the random baseline is seeded (`config.CERAMICS_BACKTEST_SEED = 7`) so it
reproduces run-to-run. Ceramics **maximizes** margin (higher = better), unlike the gas
agent which minimizes cost variance.

### Git state — IMPORTANT
On branch `wrapper`, HEAD = `148ac58 The app version where we left it over on saturday`
(the ceramics base + the first edges wave were committed earlier). The working tree
holds the **entire unified-app wave (W1–W12), uncommitted**:
- **Modified (tracked):** `app.py` (single-page layout: intake, CSS, drivers panel,
  globe wiring, dual-route chat, extended gas backtest), `ceramics_agent/{backtest,
  catalog,cost_policy,dashboard,forecast,recommend}.py`, `gas_agent/{config,
  decision_backtest,voice,voice_chat}.py`, `README.md`, `CLAUDE.md`, and the tests
  `test_{ceramics_backtest,ceramics_forecast,ceramics_recommend,decision_backtest,
  voice,voice_chat}.py`.
- **Untracked (new this wave):** `.streamlit/config.toml`, `REPORT.md`,
  `requirements.txt`, `ceramics_agent/{geo,impact,intake,pipeline,scenario}.py`, and
  the tests `test_{ceramics_geo,ceramics_intake,ceramics_scenario,combined_chat_routing,
  drivers_impact,pipeline_blurbs}.py`. Also a session-only `.claude/launch.json` (the
  preview-server config used for browser verification — safe to keep or drop).
- **Pre-existing scratch, NOT part of this work:** `scripts/get_drivers.py` /
  `get_drivers.md`.

The four touched `gas_agent/` files are **additive & behaviour-preserving only** —
`voice.synthesize(providers=…)`, `config.GAS_BACKTEST_SEED`, `decision_backtest`'s
extra baselines + a `shock_magnitude` param (default 0 = byte-identical calm replay),
and `voice_chat`'s about-intent. The gas **decision math** (`hedge_policy`,
`driver_curation`, `scenario` transforms, cost math) is untouched, and the headline
gas numbers are re-verified byte-identical. Do not commit unless asked; if asked,
stage named files (never `.env`, never the scratch above).

---

## 6. Environment & tooling quirks (these bit us — save yourself the time)

- **macOS host.** The local voice fallback is the `say` command; there is **no
  `timeout` command** on macOS — use the Bash tool's own timeout param instead.
- **Featherless has no TTS** (`/v1/audio/speech` 404s). It is **text-only**. Voice
  therefore routes **NVIDIA Riva (gRPC) → local `say`**, never Featherless. Don't
  re-add a Featherless TTS path.
- **`st.cache_data` persists across reruns** within a running Streamlit process and
  *will cache a stale `None`*. If a cached value looks wrong after a code change,
  **restart the Streamlit server** to clear it (a rerun is not enough).
- **`st.audio` renders a wavesurfer widget**, not a native `<audio>` tag — look for
  `[data-testid="stAudio"]` if you're inspecting the DOM.
- **Long leading `sleep` (≳25s) is blocked** by the harness; use `sleep ≤ 20` or the
  background/notify mechanisms.
- First Featherless call can be slow (cold model; the wrapper allows ~90s).

---

## 7. Conventions for future work

- **Tests are deterministic and offline.** No test hits a network or a live model;
  LLM/TTS backends are monkeypatched. Keep new tests this way.
- **Every external call needs a fallback** (`llm.featherless_available()`,
  `voice.synthesize() -> None`, template narratives, keyword shock parser). Preserve
  the "runs with no keys" guarantee.
- **The cache is the demo's source of truth.** Live re-forecasting exists in
  `sybilion_client` but is intentionally *off* on the main path (only the free-form
  "re-pick filters" toggle re-runs the keyword agent, no submit→poll). Keep on-stage
  paths cache-fast.
- When you touch the decision math, re-run `uv run python -m gas_agent.decision_backtest`
  and the calm-ratio check in §5 to confirm you didn't move the numbers unintentionally.

---

## 8. Known limitations / possible next steps (not bugs — honest scope)

- **Lock-price proxy = decision-time spot.** The backtest assumes you lock at the
  spot at decision time (a forward sits near spot, ignoring carry/seasonality),
  because the artifacts don't carry a historical forward curve. This is a *stated*
  assumption, surfaced in the UI. A real forward curve would sharpen it.
- **No live Sybilion poll on the main path** (by design, for demo speed). The REST
  client supports submit→poll→fetch; wiring a guarded live re-forecast behind the
  existing toggle is the natural extension.
- **Curation is name-rule based.** `driver_curation.classify_driver` keys off driver
  names (energy/markets whitelist vs demographic blacklist). `catalog.py` also has
  category/region credibility helpers that the curation step does not currently use —
  unifying them is a possible cleanup.
- **Voice is best-effort.** With no NVIDIA key it uses local `say` (macOS only); on a
  non-macOS host with no NVIDIA key the player simply hides. Pre-synthesised clips in
  `cache/audio/` make the on-stage path instant regardless.
