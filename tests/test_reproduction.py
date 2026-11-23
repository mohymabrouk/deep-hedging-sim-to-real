"""Reproduction check of test run 4 (PLAN.md change 17): the 11 hedgers of run 3 were not changed, so run 4 must
reproduce run 3 (archived in results/run3_official/) up to floating-point noise, option by option.

Reads the committed result files only; skipped when either run's files are absent."""

import numpy as np
import pandas as pd
import pytest

from dh import config

R = config.RESULTS
RUN3 = R / "run3_official"
OLD = ("no_hedge", "bs_iv", "bs_dvol") + tuple(f"dh_{g}" for g in config.GENERATORS) + tuple(f"dh_resid_{g}" for g in config.GENERATORS)


def _load(d):
    p, t = d / "performance_test.csv", d / "test_pnl_per_option.csv"
    if not (p.exists() and t.exists()):
        pytest.skip(f"no test results in {d}")
    return pd.read_csv(p).set_index("hedger"), pd.read_csv(t, usecols=["hedger", "instrument", "pnl", "loss", "turnover"])


def test_run4_reproduces_run3_for_the_eleven_existing_hedgers():
    perf3, t3 = _load(RUN3)
    perf4, t4 = _load(R)
    if not set(config.CLASSICAL) & set(perf4.index):
        pytest.skip("results/ do not hold run 4 yet")
    assert set(OLD) <= set(perf3.index) and set(OLD) <= set(perf4.index)
    num = [c for c in perf3.columns if c in perf4.columns and np.issubdtype(perf3[c].dtype, np.number)]
    gap = np.abs(perf3.loc[list(OLD), num].to_numpy(float) - perf4.loc[list(OLD), num].to_numpy(float))
    assert np.nanmax(gap) < 1e-6, "a summary statistic (including a bootstrap bound) moved"
    for h in OLD:
        a = t3[t3["hedger"] == h].set_index("instrument")
        b = t4[t4["hedger"] == h].set_index("instrument")
        assert a.index.sort_values().equals(b.index.sort_values()), f"{h}: different options"
        b = b.loc[a.index]
        assert np.max(np.abs(a["pnl"].to_numpy() - b["pnl"].to_numpy())) < 1e-5, f"{h}: an option's P&L moved"
        assert np.max(np.abs(a["turnover"].to_numpy() - b["turnover"].to_numpy())) < 1e-6, f"{h}: an option's turnover moved"
