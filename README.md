# TTF Gas Hedging Agent

A gas-hedging decision agent on the **Sybilion** forecasting API: it turns a
probabilistic TTF (European natural gas) forecast into a **per-month hedge ratio**
for a German glass manufacturer. Built for the hackathon "Forecasting AI" track.

> **Docs:** `docs/APP_GUIDE.md` (how it works), `docs/ARCHITECTURE.md` (why each
> choice), `CLAUDE.md` (project state & handoff).

---

## Quick start

```bash
uv sync                       # install dependencies (one time)
cp .env.example .env          # optional — the demo runs with no keys
uv run streamlit run app.py   # start the dashboard
```

Streamlit prints a local URL (default **http://localhost:8501**) and usually opens
your browser automatically. The app runs entirely off cached data, so no API keys are
required.

## Stop the server

- **Foreground (normal case):** press **`Ctrl-C`** in the terminal running it.
- **Won't stop / lost the terminal:** kill it by port or process name:

  ```bash
  lsof -ti :8501 | xargs kill        # kill whatever holds port 8501
  # or:
  pkill -f "streamlit run app.py"    # kill by command
  ```

## Run in the background

```bash
uv run streamlit run app.py --server.headless true &   # start detached
# ... later ...
pkill -f "streamlit run app.py"                        # stop it
```

## Run on a different port

```bash
uv run streamlit run app.py --server.port 8600
```

---

## Other commands

```bash
uv run pytest -q                              # 82 tests
uv run python -m gas_agent.decision_backtest  # offline backtest verdict
uv run python scripts/build_voiceover.py      # regenerate narration clips
```

**Requirements:** Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv).
If the app says *"No cached forecast found,"* check that `cache/latest_job.txt` points
to a folder under `cache/` containing `forecast.json`.
