"""Regression tests for the training markets (run-1 GARCH bug, see PLAN.md change 13).

In run 1, one Student-t draw of ~150 sd gave a +12 daily log return in the final-stage GARCH world. Because the
martingale correction is log E[exp(r)], that single value set the drift shift to 0.115 per day and every path fell
~97% in 35 days. These tests check, over several seeds, that no generator can be dominated by one draw again.
"""

import numpy as np
import pandas as pd
import pytest

from dh import config, market
from dh.pipeline import _policy_seed
from dh.sims import Bootstrap, Garch, martingale

# seeds 0-4 plus the exact seed the pipeline uses for the GARCH world. With the shock cap removed, that seed
# reproduces the run-1 shift of 0.1151/day and seed 0 gives 0.016/day, so both make this test fail (checked).
SEEDS = [*range(5), _policy_seed("garch", 0.0)]
N_PATHS = 20_000


def _garch_t_history(n=1500, omega=0.2, alpha=0.09, beta=0.909, nu=3.0, seed=0) -> pd.Series:
    """Synthetic BTC-like history: near-integrated GARCH(1,1) with Student-t(3) shocks, in log-return units."""
    rng = np.random.default_rng(seed)
    var, out = omega / (1 - alpha - beta), np.empty(n)
    scale = np.sqrt((nu - 2) / nu)
    for i in range(n):
        out[i] = np.sqrt(var) * rng.standard_t(nu) * scale
        var = omega + alpha * out[i] ** 2 + beta * var
    return pd.Series(out / 100.0, index=pd.date_range("2018-01-01", periods=n, freq="D"))


def _check_paths(R: np.ndarray, label: str) -> None:
    assert np.isfinite(R).all(), label
    _, shift = martingale(R)
    assert abs(shift) < 0.01, f"{label}: martingale shift {shift:.4f}/day (run-1 bug gave 0.115)"
    end = np.exp(np.cumsum(R - shift, axis=1)[:, config.HIST + config.MAX_DAYS - 1] - np.cumsum(R - shift, axis=1)[:, config.HIST - 1])
    assert np.mean(end < 0.1) < 0.01, f"{label}: {np.mean(end < 0.1):.1%} of paths lose >90% over one option life"


def test_garch_shocks_are_capped_on_heavy_tailed_history():
    g = Garch().fit(_garch_t_history())
    for seed in SEEDS:
        R = g.sample(N_PATHS, np.random.default_rng(seed))
        bound = np.sqrt(g.var_cap) * g.z_cap / 100.0
        assert np.abs(R).max() <= bound + 1e-12, "a shock escaped the cap"
        _check_paths(R, f"synthetic garch seed {seed}")


@pytest.mark.parametrize("fit_end", [config.VAL_FIT_END, config.FINAL_FIT_END], ids=["validation", "final"])
def test_generators_on_real_btc_history(fit_end):
    """Same check on the real 2018-2020 and 2018-2021 fits (skipped when the data cache is absent)."""
    try:
        r = market.spot_daily_returns(end=fit_end)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"no cached spot data: {e}")
    g, b = Garch().fit(r), Bootstrap().fit(r)
    for seed in SEEDS:
        _check_paths(g.sample(N_PATHS, np.random.default_rng(seed)), f"garch {fit_end} seed {seed}")
        _check_paths(b.sample(N_PATHS, np.random.default_rng(seed)), f"bootstrap {fit_end} seed {seed}")
