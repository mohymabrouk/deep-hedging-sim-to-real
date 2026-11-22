"""Future truncation: the action at step t must not change when anything after t is replaced."""

import numpy as np
import pandas as pd
import pytest
import torch

from dh import config, market
from dh.hedging import BSDelta, Contracts, DeepHedger, gbm_paths, returns_to_prices, run_hedger

H, TMAX = config.HIST, config.MAX_DAYS


def _setup(n=64, seed=0):
    torch.manual_seed(seed)
    P = returns_to_prices(gbm_paths(n, 0.7, seed))[:, : H + TMAX + 1]
    dvol = 0.6 + 0.05 * torch.randn(P.shape, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    c = Contracts(torch.exp(0.3 * torch.rand(n, generator=g, dtype=torch.float64) - 0.15),
                  torch.randint(config.MIN_DAYS, TMAX + 1, (n,), generator=g), torch.rand(n, generator=g) < 0.5,
                  iv0=torch.full((n,), 0.6, dtype=torch.float64), dvol0=dvol[:, H].clone())
    return {"P": P, "dvol": dvol}, c


@pytest.mark.parametrize("hedger", [DeepHedger(), BSDelta(use_dvol=False), BSDelta(use_dvol=True)], ids=["deep", "bs_iv", "bs_dvol"])
def test_actions_do_not_depend_on_the_future(hedger):
    mkt, c = _setup()
    with torch.no_grad():
        full = run_hedger(hedger, mkt, c)
        for t in (0, 3, 10, 20, TMAX - 2):
            cut = {k: v.clone() for k, v in mkt.items()}
            g = torch.Generator().manual_seed(t)
            for k in cut:  # scramble everything strictly after time t
                cut[k][:, H + t + 1:] = cut[k][:, H + t + 1:] * torch.exp(torch.randn(cut[k][:, H + t + 1:].shape, generator=g, dtype=torch.float64))
            q = run_hedger(hedger, cut, c)
            assert torch.equal(q[:, : t + 1], full[:, : t + 1]), f"action at <= {t} changed"
            assert not torch.equal(q[:, t + 1:], full[:, t + 1:]), "scramble had no effect: test would be vacuous"


def test_trained_policy_also_causal():
    """Same check for a briefly trained policy (weights that actually use the inputs)."""
    from dh.hedging import train_policy
    P_all = returns_to_prices(gbm_paths(2000, 0.7, 5))
    pol, _ = train_policy(P_all, 0.5, iters=20, batch=512, log_every=0)
    mkt, c = _setup(seed=1)
    with torch.no_grad():
        full = run_hedger(pol, mkt, c)
        cut = {k: v.clone() for k, v in mkt.items()}
        cut["P"][:, H + 6:] *= 1.3
        q = run_hedger(pol, cut, c)
    assert torch.equal(q[:, :6], full[:, :6])
    assert not torch.equal(q[:, 6:], full[:, 6:])  # and the future does matter once it is the present


def test_daily_price_is_the_price_at_08_utc():
    """daily_at_hour takes the open of the 08:00 bar, i.e. the price known at 08:00, never a later close."""
    idx = pd.date_range("2022-01-01 06:00", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0, 4.0, 5.0], "close": [1.5, 2.5, 3.5, 4.5, 5.5]}, index=idx)
    s = market.daily_at_hour(df, 8)
    assert s.iloc[0] == 3.0


def test_dataset_market_arrays_use_entry_aligned_dates(monkeypatch):
    """market_arrays: column H is the entry day; history columns are strictly earlier days."""
    from datetime import date, timedelta

    from dh import dataset
    days = [date(2022, 1, 1) + timedelta(days=k) for k in range(120)]
    perp = pd.Series(np.arange(120, dtype=float) + 100.0, index=days)
    monkeypatch.setattr(market, "perp_08", lambda: perp)
    monkeypatch.setattr(market, "funding_cash_daily", lambda: pd.Series(0.0, index=days))
    monkeypatch.setattr(market, "dvol_08", lambda: pd.Series(50.0, index=days))
    df = pd.DataFrame({"instrument": ["X"], "entry_day": [days[40]], "T": [10], "strike": [140.0], "is_call": [True],
                       "iv": [50.0], "index_price": [140.0]})
    out, mkt, F, c = dataset.market_arrays(df)
    P = mkt["P"][0].numpy() * 140.0
    assert P[H] == perp[days[40]] and P[H - 1] == perp[days[39]]
    assert P[H + 10] == perp[days[50]] and P[-1] == perp[days[50]]  # padded with the expiry price
