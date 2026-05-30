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

The same spine now powers a **second** agent, in a second Streamlit tab, for the
same manufacturer's **ceramics** line. For one production run (product / quantity /
timeline) under user-set cost-factor weights (gas / clay / power / transport) it
decides three things — **how much input cost to lock now**, **which supplier to buy
from**, and **which channel to sell through** — then runs a two-round negotiation
and a 3-strategy backtest. The "lock %" is literally a hedge ratio on a *blended*
4-factor cost band, so it **reuses `gas_agent.hedge_policy` unchanged**
(`HedgePolicyParams(low_band=0.25, high_band=0.60)`). Same invariants: deterministic
& auditable (the LLM only explains, never computes a number), offline-first
(committed mock forecast + template narrative), and **the gas tab stays
byte-identical**. Lives in its own `ceramics_agent/` package (mirrors `gas_agent/`);
the only `app.py` change wraps the original dashboard body in tabs. See §5 for live
numbers and `docs/CERAMICS_AGENT.md` for the design note.

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
│   ├── decision_backtest.py     # replay vs always-spot / always-50% baselines
│   ├── geo.py                   # country coords + aggregation + per-country brief
│   ├── sybilion_client.py       # REST client + disk cache + artifact parsers
│   └── voice.py                 # TTS: NVIDIA Riva primary + rate limiter + local `say` fallback
├── ceramics_agent/              # SECOND AGENT — mirrors gas_agent/ (gas stays untouched)
│   ├── catalog.py               # products (BOM), suppliers, channels, historical sales + whitelists
│   ├── forecast.py              # 4-factor cost-forecast loader (mock default; live Sybilion opt-in)
│   ├── cost_policy.py           # blended cost band → lock % (REUSES gas_agent.hedge_policy)
│   ├── curation.py              # deterministic keep/reject + score/rank of suppliers & channels
│   ├── negotiation.py           # two-round, market-anchored negotiation (deterministic)
│   ├── backtest.py              # 3-strategy replay (agent vs seeded-random vs cheap+best-margin)
│   ├── recommend.py             # orchestrates forecast→lock→curation→negotiation→backtest (pure)
│   ├── explanation.py           # LLM narrates the decided recommendation (explain-only; template fallback)
│   └── dashboard.py             # render_ceramics_tab(render_voiceover) — the second tab UI
├── scripts/
│   ├── build_ttf_series.py      # regenerate data/ttf_series.json from Yahoo TTF=F
│   ├── save_cached_artifact.py  # import an MCP-exported artifact into cache/<job>/
│   ├── build_voiceover.py       # pre-synthesise cache/audio/{golden,shock}.wav
│   └── build_ceramics_forecast.py   # generate & commit cache/mock_ceramics_forecast.json
├── tests/                       # 179 tests, all green (111 gas + 68 ceramics)
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
uv run streamlit run app.py          # the dashboard (Gas hedging + Ceramics optimizer tabs)
uv run pytest -q                     # 179 tests, all pass (111 gas + 68 ceramics)
uv run python -m gas_agent.decision_backtest   # offline backtest verdict (gas)
uv run python scripts/build_voiceover.py       # regenerate narration into cache/audio/
uv run python scripts/build_ceramics_forecast.py  # regenerate the committed ceramics mock
```

- **uv** is the package manager. Python ≥ 3.11 (dev on 3.13).
- The dashboard reads `cache/latest_job.txt` → loads that job's cached artifacts.
  If it's missing/mispointed you get "No cached forecast found."

---

## 5. Current state (as of 2026-05-30)

**Everything in the original 13-task plan is built and verified.** All three judging
axes are covered. **179 tests pass** (111 gas + 68 ceramics).

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
| Shock (magnitude 1.0) | next-quarter ratio **rises** to **~40.1%** vs the new calm 30.5% (applied premium +32%); Iran/risk supplier lights up green on the globe; "Global supply-risk premium" leads the drivers |

### Ceramics optimizer (second agent) — built & verified

The full C1–C8 build is done: `catalog` (BOM products, suppliers, channels,
historical sales) → `forecast` (4-factor loader, mock default + live Sybilion
opt-in) → `cost_policy` (blended band → lock %, reusing `gas_agent.hedge_policy`) →
`curation` (score/rank suppliers & channels) → `negotiation` (two rounds) →
`backtest` (3 strategies) → `recommend` (orchestrator) → `explanation` (explain-only
LLM + template) → `dashboard` (the second tab). It runs fully offline on the
committed `cache/mock_ceramics_forecast.json`; the live 4-factor path is behind the
`SYBILION_API_KEY` guard and **never** repoints `latest_job.txt`. Verified end-to-end
in the browser: both tabs render, **the gas tab is byte-identical**, zero Streamlit
exceptions, and identical inputs yield identical numbers.

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
| **Backtest (12 replayed months)** | agent **€34,290/mo** realized margin — **+65%** vs a seeded-random pick (€20,792), **+1%** vs static cheap+best-margin (€33,819) |

Note: the chosen supplier is the **mid-ranked** Alpine (not the top-scored Eastern)
**by design** — a 40% lock falls in the "mid lock" band, which routes to the balanced
mid-ranked supplier (`select_supplier`). High lock → top-ranked; low lock → cheapest.
The backtest discounts each supplier's nominal margin by its reliability (a stated
assumption); the random baseline is seeded (`config.CERAMICS_BACKTEST_SEED = 7`) so it
reproduces run-to-run. Ceramics **maximizes** margin (higher = better), unlike the gas
agent which minimizes cost variance.

### Git state — IMPORTANT
The `edges.md` follow-up wave is now **committed** (HEAD = `16dc100 voice chat`).
The working tree is **ahead of HEAD** with the **entire ceramics agent uncommitted**:
- **Modified:** `app.py` (wrap the original `main()` body — renamed `render_gas_tab()` —
  in `st.tabs([...])`; the gas tab content is unchanged), `gas_agent/config.py`
  (additive ceramics constants: `CERAMICS_DEFAULT_WEIGHTS`, `CERAMICS_BACKTEST_SEED`,
  `CERAMICS_MOCK_FORECAST`), `.gitignore` (negation so the committed mock forecast is
  trackable), `CLAUDE.md`.
- **Untracked (this build):** the whole `ceramics_agent/` package (9 modules),
  `cache/mock_ceramics_forecast.json` (the committed offline demo source),
  `scripts/build_ceramics_forecast.py`, `docs/CERAMICS_AGENT.md`, and the 8
  `tests/test_ceramics_*.py` files.
- **Untracked (pre-existing scratch, NOT part of this work):** `scripts/get_drivers.py`,
  `scripts/get_drivers.md`.

No `gas_agent/` module was touched (gas stays frozen). Do not commit unless asked; if
asked, stage named files (never `.env`, never the scratch above).

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
