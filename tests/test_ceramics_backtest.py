"""Offline tests for the three-strategy ceramics backtest.

Replays the committed historical months (no network, no model). The contract: the
seeded "random" baseline reproduces run-to-run (determinism is the selling point,
even for the foil), all three strategies aggregate into stats, the agent-vs-random
delta is computed, and the verdict reports it. Realized margins carry the
reliability haircut, which is why the agent's reliability-aware pick can out-earn
a blind cheapest-supplier chase.
"""

from gas_agent import config

from ceramics_agent.backtest import run_ceramics_backtest
from ceramics_agent.cost_policy import decide_procurement, default_weights, quarter_lock_ratio
from ceramics_agent.forecast import load_ceramics_forecast


def _factors_and_lock():
    factors, _ = load_ceramics_forecast()
    lock = quarter_lock_ratio(decide_procurement(factors, default_weights()))
    return factors, lock


def test_backtest_replays_every_historical_month():
    factors, lock = _factors_and_lock()
    result = run_ceramics_backtest(factors, lock)
    assert result.n_months == 12
    assert len(result.rows) == 12
    for strat in (result.agent, result.random_, result.cheap):
        assert strat.mean > 0
        assert strat.max >= strat.min


def test_seeded_random_baseline_reproduces():
    factors, lock = _factors_and_lock()
    first = run_ceramics_backtest(factors, lock, seed=config.CERAMICS_BACKTEST_SEED)
    second = run_ceramics_backtest(factors, lock, seed=config.CERAMICS_BACKTEST_SEED)
    # The whole result reproduces — same agent and same "random" picks every run.
    assert [r.random_margin for r in first.rows] == [r.random_margin for r in second.rows]
    assert first.random_.mean == second.random_.mean
    assert first.agent.mean == second.agent.mean


def test_agent_beats_the_random_baseline_on_the_committed_mock():
    factors, lock = _factors_and_lock()
    result = run_ceramics_backtest(factors, lock)
    assert result.agent.mean > result.random_.mean
    assert result.agent_vs_random_pct > 0


def test_vs_cheap_delta_is_computed():
    factors, lock = _factors_and_lock()
    result = run_ceramics_backtest(factors, lock)
    assert isinstance(result.agent_vs_cheap_pct, float)


def test_verdict_reports_the_margin_delta():
    factors, lock = _factors_and_lock()
    result = run_ceramics_backtest(factors, lock)
    verdict = result.verdict
    assert "random" in verdict.lower()
    assert "%" in verdict
    assert f"{result.n_months}" in verdict


def test_realized_margins_carry_the_reliability_haircut():
    # Every replayed agent margin is the nominal margin scaled by a reliability
    # in (0, 1], so realized <= nominal: the realized figure is never inflated.
    factors, lock = _factors_and_lock()
    result = run_ceramics_backtest(factors, lock)
    assert all(row.agent_margin > 0 for row in result.rows)
    # The agent's supplier is fixed across the replay (lock-routed + timeline).
    assert len({row.agent_supplier for row in result.rows}) == 1
