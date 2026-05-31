# Forecasting AI — supply & hedging decision agent

One chat-centric page that turns one **company description** into **two auditable
decisions** for a mid-size German **glass & ceramics** manufacturer, built on the
**Sybilion** probabilistic forecasting API (hackathon "Forecasting AI" track):

1. **Gas hedge** — what share of next quarter's natural gas (Dutch **TTF** benchmark) to
   **lock in forward now** vs. buy later on the spot market.
2. **Ceramics run** — for one production run: how much **input cost to lock**, **which
   supplier** to buy from, **which channel** to sell through, plus a negotiated deal.

**The headline principle: the LLM never computes a decision number.** Deterministic policy
code computes every hedge ratio, lock %, score, quote and backtest result — so the same
inputs always reproduce the same decision and every number is traceable. The LLM only
*prepares inputs* (picks Sybilion filters, reads a shock's severity) and *explains outputs*.

> **Why this domain:** gas (high-temperature firing) is this manufacturer's largest volatile
> cost, and gas is hard to point-forecast (~28% MAPE on this series). So the agent does **not**
> bet on the price line — it sizes its decisions from the forecast's **confidence band**. That
> is the honest way to use a probabilistic forecast whose mean is weak but whose bands inform.

> **Docs:** `REPORT.md` (submission write-up with real numbers) · `docs/APP_GUIDE.md` (how it
> works) · `docs/ARCHITECTURE.md` (why each choice) · `docs/CERAMICS_AGENT.md` (the 2nd agent)
> · `CLAUDE.md` (project state & handoff).

---

## Quick start (runs with NO keys)

```bash
uv sync                       # install dependencies (one time) — Python ≥ 3.11 + uv
uv run streamlit run app.py   # start the dashboard
```

Streamlit prints a local URL (default **http://localhost:8501**) and usually opens your
browser. **No API keys are required** — the app runs entirely off a committed cached Sybilion
forecast and a committed 4-factor mock, with template narration and the voice player hidden.
Describe a business (or submit blank for the demo base case) → watch it forecast → both
decisions render, with a 3D globe and a bottom chat.

Prefer `pip`? A clean checkout also works with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional: live forecasts, LLM narration, voice

```bash
cp .env.example .env          # then add the keys you have (all optional)
```

| Add this key | Unlocks |
|--------------|---------|
| `SYBILION_API_KEY` | the **Live Sybilion** toggle — forecast against today's market (caches the result; toggling off restores the reproducible demo) |
| `FEATHERLESS_API_KEY` | LLM narration / chat / country briefs (else deterministic templates) |
| `NVIDIA_API_KEY` (+ `--extra voice`) or `HF_API_KEY` | spoken narration / push-to-talk voice (else hidden; text works) |

**Every decision number is identical with or without keys** — keys only change how live the
forecast is and whether the prose/voice are model-written. Run the NVIDIA voice path with:
`uv run --extra voice streamlit run app.py`.

## What's live vs. mocked (honest)

| Piece | Default (no keys) | With keys |
|-------|-------------------|-----------|
| Gas TTF forecast | committed cached Sybilion job | live submit→poll (toggle) |
| Ceramics 4-factor cost forecast | committed mock | live 4-factor Sybilion (toggle) |
| Narration / chat / briefs | deterministic templates | Featherless LLM |
| Voice in / out | hidden (text works) | NVIDIA Riva → local `say` / HF Whisper |
| **Every decision number** | **deterministic — identical** | **identical** |

---

## Other commands

```bash
uv run pytest -q                              # 270 offline tests, all green
uv run python -m gas_agent.decision_backtest  # gas backtest verdict (policy vs 0/50/100% + random + shocked)
uv run python -m ceramics_agent.backtest      # ceramics backtest verdict (agent vs random/cheap/top-ranked + 24-mo)
uv run python scripts/build_voiceover.py      # regenerate narration clips
```

## Stop / background / port

```bash
# stop a foreground server: Ctrl-C   |   by port: lsof -ti :8501 | xargs kill
uv run streamlit run app.py --server.headless true &   # background
uv run streamlit run app.py --server.port 8600         # different port
```

If the app says *"No cached forecast found,"* check that `cache/latest_job.txt` points to a
folder under `cache/` containing `forecast.json`.

**Secrets:** `.env` holds your real keys and is gitignored — only `.env.example` is tracked.
