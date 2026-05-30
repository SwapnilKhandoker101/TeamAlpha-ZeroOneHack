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
├── scripts/
│   ├── build_ttf_series.py      # regenerate data/ttf_series.json from Yahoo TTF=F
│   ├── save_cached_artifact.py  # import an MCP-exported artifact into cache/<job>/
│   └── build_voiceover.py       # pre-synthesise cache/audio/{golden,shock}.wav
├── tests/                       # 82 tests, all green (pytest)
└── docs/
    ├── APP_GUIDE.md             # user guide (how it works, how we know it's good)
    └── ARCHITECTURE.md          # build & design-rationale (why each choice)
```

---

## 4. How to run

```bash
uv sync                              # install deps
cp .env.example .env                 # optional: add real keys (demo runs without them)
uv run streamlit run app.py          # the dashboard
uv run pytest -q                     # 82 tests, all pass
uv run python -m gas_agent.decision_backtest   # offline backtest verdict
uv run python scripts/build_voiceover.py       # regenerate narration into cache/audio/
```

- **uv** is the package manager. Python ≥ 3.11 (dev on 3.13).
- The dashboard reads `cache/latest_job.txt` → loads that job's cached artifacts.
  If it's missing/mispointed you get "No cached forecast found."

---

## 5. Current state (as of 2026-05-30)

**Everything in the original 13-task plan is built and verified.** All three judging
axes are covered. 82 tests pass.

Live numbers from the cached job `f445eec1-…` (so future sessions can sanity-check
they haven't regressed the decision):

| Quantity | Value |
|----------|-------|
| Today's spot (last actual TTF) | **€47.28/MWh** |
| Forecast horizon | 2026-06 … 2026-11 (6 months) |
| **Calm next-quarter hedge ratio** | **~26%** (months: 46% / 10% / 22%) |
| Curation | **25 kept / 6 rejected** (needs_refine = False) |
| Top kept driver | "Exports of Oil and petroleum products in Germany" |
| Rejected sample | Population — Sri Lanka / Serbia / Bangladesh / Europe |
| Backtest | **63 replayed months**: policy €1.25/MWh **cheaper** than spot, €2.34/MWh **tighter** swing, **matches** a 50% lock (policy €35.71±4.47; spot €36.96±6.81; lock-50% €35.25±4.35) |
| Shock (magnitude 1.0) | next-quarter ratio **rises** vs calm; Iran/risk supplier lights up green on the globe; "Global supply-risk premium" leads the drivers |

### Git state — IMPORTANT
The working tree is **ahead of the last commit** and these changes are **uncommitted**:
- **Modified:** `app.py` (Plotly globe replacing pydeck + voice wiring), `gas_agent/config.py`
  (voice/NVIDIA config), `.env.example`, `.gitignore`.
- **Untracked:** `gas_agent/voice.py`, `scripts/build_voiceover.py`, `tests/test_voice.py`,
  the whole `docs/` folder, `CLAUDE.md`.

The last commit (`bd23457`) still contains the **old pydeck globe**; the working tree
has the **working Plotly orthographic globe**. Do not commit unless asked; if asked,
stage named files (never `.env`).

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
