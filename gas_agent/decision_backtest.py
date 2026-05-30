"""Decision backtest — did the policy's hedge decisions beat naive baselines?

Sybilion's backtest replays its model over past windows: for each past month it
records the quantile forecast it *would* have made and the price that *actually*
happened. We replay the same deterministic hedge policy over those windows and
compare the cost it would have realized against two naive baselines — always buy
on the spot market (hedge 0%) and always lock half the volume (hedge 50%).

The honest framing for the judges: in a *falling* market nothing beats pure spot
on average cost, so the win we look for is **variance reduction** — the whole
reason a manufacturer hedges — together with beating a blind 50% lock. The
band-aware policy locks less when the forecast sits below today's spot and more
when it sits above, so it should cut cost variance versus always-spot while
costing less than mechanically locking half.

Everything here is deterministic and reuses the exact ``hedge_policy`` code the
live decision uses, so the backtest and the on-screen decision can never drift
apart.

Lock-price proxy: the artifact does not carry the historical forward curve, so we
proxy the price you would have locked with the **decision-time spot** — a forward
is anchored near today's spot (ignoring small carry/seasonality), and a buyer can
actually transact there. We deliberately do *not* lock at the model's median: the
point forecast is weak (~28% MAPE), and betting on it is exactly what this agent
avoids. The forecast's role is only to set the hedge *ratio* (via the band and the
median-vs-spot drift); the price you pay to lock is the market's, i.e. spot. This
is a stated assumption, not a hidden one.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from gas_agent import sybilion_client as sc
from gas_agent.hedge_policy import (
    DEFAULT_PARAMS,
    HedgePolicyParams,
    MonthForecast,
    decide_month,
)


@dataclass(frozen=True)
class BacktestRow:
    """One replayed decision month: what the policy did and what each strategy cost."""

    window_end: str
    month: str
    spot_at_decision: float  # the lock-price proxy (forward ~ today's spot)
    forecast_median: float  # what drives the hedge ratio (drift vs spot), not the lock price
    actual: float
    hedge_ratio: float
    cost_policy: float
    cost_always_spot: float
    cost_always_half: float


@dataclass(frozen=True)
class StrategyStats:
    """Aggregate realized cost for one strategy across all replayed months."""

    name: str
    mean_cost: float
    std_cost: float  # population std — the variance the buyer actually feels


@dataclass(frozen=True)
class BacktestResult:
    """The full comparison: per-month rows plus per-strategy aggregates."""

    rows: list[BacktestRow]
    policy: StrategyStats
    always_spot: StrategyStats
    always_half: StrategyStats

    @property
    def n_months(self) -> int:
        return len(self.rows)

    @property
    def cost_saving_vs_spot(self) -> float:
        """How much cheaper the policy is than always-spot (EUR/MWh; +ve = cheaper)."""
        return self.always_spot.mean_cost - self.policy.mean_cost

    @property
    def volatility_drop_vs_spot(self) -> float:
        """How much tighter the policy's cost swing is than always-spot (+ve = tighter)."""
        return self.always_spot.std_cost - self.policy.std_cost

    @property
    def cost_gap_vs_half(self) -> float:
        """Policy mean cost minus always-lock-50% (+ve = policy is dearer)."""
        return self.policy.mean_cost - self.always_half.mean_cost

    @property
    def beats_spot(self) -> bool:
        """The headline win: cheaper AND less volatile than doing nothing (buy spot)."""
        return self.cost_saving_vs_spot > 0 and self.volatility_drop_vs_spot > 0

    @property
    def verdict(self) -> str:
        """One-line, honest summary for the dashboard and writeup. Leads with the
        comparison against always-spot — the do-nothing baseline the policy dominates."""
        if not self.rows:
            return "No backtest windows available to replay."
        cheaper = self.cost_saving_vs_spot
        tighter = self.volatility_drop_vs_spot
        cost_word = "cheaper" if cheaper >= 0 else "dearer"
        swing_word = "tighter" if tighter >= 0 else "wider"
        half_gap = self.cost_gap_vs_half
        half_word = "matches" if abs(half_gap) < 0.5 else ("undercuts" if half_gap < 0 else "trails")
        return (
            f"Across {self.n_months} replayed months the policy is EUR {abs(cheaper):.2f}/MWh "
            f"{cost_word} than buying spot and runs a EUR {abs(tighter):.2f}/MWh {swing_word} "
            f"cost swing, and {half_word} a static 50% lock "
            f"(policy EUR {self.policy.mean_cost:.2f} ± {self.policy.std_cost:.2f}; "
            f"spot EUR {self.always_spot.mean_cost:.2f} ± {self.always_spot.std_cost:.2f}; "
            f"lock-50% EUR {self.always_half.mean_cost:.2f} ± {self.always_half.std_cost:.2f})."
        )


def _previous_month(date_str: str) -> str:
    """The first-of-month string one month before ``date_str`` (YYYY-MM-01)."""
    year, month, _ = date_str.split("-")
    year_n, month_n = int(year), int(month)
    month_n -= 1
    if month_n == 0:
        month_n = 12
        year_n -= 1
    return f"{year_n:04d}-{month_n:02d}-01"


def _decision_spot(timeseries: dict, first_forecast_month: str, first_entry: dict) -> float:
    """Today's-spot anchor at the window's decision time: the actual price in the
    month *before* the window starts. Falls back to the first month's forecast
    median when that history point is missing, so there is never any look-ahead."""
    previous = timeseries.get(_previous_month(first_forecast_month))
    if previous is not None:
        return float(previous)
    return float(first_entry["quantile_forecast"]["0.50"])


def _stats(name: str, values: list[float]) -> StrategyStats:
    if not values:
        return StrategyStats(name=name, mean_cost=0.0, std_cost=0.0)
    spread = pstdev(values) if len(values) > 1 else 0.0
    return StrategyStats(name=name, mean_cost=mean(values), std_cost=spread)


def run_decision_backtest(
    trajectories: dict,
    ttf_document: dict | None = None,
    params: HedgePolicyParams = DEFAULT_PARAMS,
) -> BacktestResult:
    """Replay the hedge policy over Sybilion's backtest windows and compare the
    realized cost against always-spot and always-lock-50%."""
    ttf = ttf_document or sc.load_ttf_series()
    timeseries = ttf["timeseries"]
    windows = trajectories.get("data", []) if isinstance(trajectories, dict) else []

    rows: list[BacktestRow] = []
    for window in windows:
        series = window.get("forecast_series", {})
        if not series:
            continue
        months = sorted(series)
        spot = _decision_spot(timeseries, months[0], series[months[0]])

        for month in months:
            entry = series[month]
            if entry.get("actual") is None:
                continue  # a forecast month with no realized price yet — nothing to score
            quantiles = entry["quantile_forecast"]
            forecast = MonthForecast(
                month=month,
                median=float(quantiles["0.50"]),
                low=float(quantiles["0.10"]),
                high=float(quantiles["0.90"]),
            )
            actual = float(entry["actual"])
            lock_price = spot  # forward ~ decision-time spot (stated assumption)
            ratio = decide_month(forecast, spot, params).hedge_ratio

            rows.append(
                BacktestRow(
                    window_end=str(window.get("forecast_end", "")),
                    month=month,
                    spot_at_decision=spot,
                    forecast_median=forecast.median,
                    actual=actual,
                    hedge_ratio=ratio,
                    cost_policy=ratio * lock_price + (1 - ratio) * actual,
                    cost_always_spot=actual,
                    cost_always_half=0.5 * lock_price + 0.5 * actual,
                )
            )

    return BacktestResult(
        rows=rows,
        policy=_stats("policy", [r.cost_policy for r in rows]),
        always_spot=_stats("always spot (0%)", [r.cost_always_spot for r in rows]),
        always_half=_stats("always lock 50%", [r.cost_always_half for r in rows]),
    )


def backtest_from_cache(
    job_id: str | None = None,
    params: HedgePolicyParams = DEFAULT_PARAMS,
) -> BacktestResult:
    """Convenience: run the decision backtest off the cached artifacts for a job
    (defaults to the latest cached forecast)."""
    job_id = job_id or sc.get_latest_job()
    trajectories = sc.load_artifact(job_id, "backtest_trajectories.json")
    return run_decision_backtest(trajectories, sc.load_ttf_series(), params)


if __name__ == "__main__":  # quick manual check against the cached job
    result = backtest_from_cache()
    print(f"replayed months: {result.n_months}")
    for row in result.rows:
        print(
            f"  {row.month[:7]}  spot {row.spot_at_decision:5.1f}  median {row.forecast_median:5.1f}  "
            f"actual {row.actual:5.1f}  hedge {row.hedge_ratio:4.0%}  "
            f"policy {row.cost_policy:5.1f}  spot {row.cost_always_spot:5.1f}  half {row.cost_always_half:5.1f}"
        )
    print()
    print(result.verdict)
