"""Residual deep hedger (test run 3): q = BS delta(sigma_entry) + bound * tanh(MLP(x)).

1. At initialisation the policy *is* the "BS delta, entry IV" benchmark, and training can move it by at most `bound`.
2. The implied-vol input uses only information available at entry: on real options it is the IV of the entry trade
   (07:00-08:00 UTC on the entry day, before the first hedge at 08:00), it is static over the option's life, and
   nothing after the rebalancing time changes an action. The 'trailing' stand-in uses returns up to entry only; the
   'fwd_noisy' stand-in looks ahead by design (it imitates a forecast) and only exists on simulated training paths.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from dh import config, dataset, deribit, market
from dh.hedging import (BSDelta, Contracts, ResidualHedger, entry_vol_stand_in, gbm_paths, residual_features,
                        returns_to_prices, run_hedger, train_policy)

H, TMAX = config.HIST, config.MAX_DAYS


def _setup(n=96, seed=0):
    torch.manual_seed(seed)
    P = returns_to_prices(gbm_paths(n, 0.7, seed))[:, : H + TMAX + 1]
    dvol = 0.6 + 0.05 * torch.randn(P.shape, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    c = Contracts(torch.exp(0.4 * torch.rand(n, generator=g, dtype=torch.float64) - 0.2),
                  torch.randint(config.MIN_DAYS, TMAX + 1, (n,), generator=g), torch.rand(n, generator=g) < 0.5,
                  iv0=0.3 + torch.rand(n, generator=g, dtype=torch.float64), dvol0=dvol[:, H].clone())
    return {"P": P, "dvol": dvol}, c


def _randomised(bound=0.3, seed=0):
    """A residual policy whose last layer is NOT zero, so that every input matters."""
    torch.manual_seed(seed)
    pol = ResidualHedger(bound)
    torch.nn.init.normal_(pol.net[-1].weight, std=0.5)
    torch.nn.init.normal_(pol.net[-1].bias, std=0.5)
    return pol.eval()


# ---- 1. equals BS delta at initialisation -----------------------------------------------------------------------
@pytest.mark.parametrize("bound", [0.1, 0.3])
def test_residual_policy_equals_bs_delta_at_init(bound):
    mkt, c = _setup()
    pol = ResidualHedger(bound)
    assert torch.count_nonzero(pol.net[-1].weight) == 0 and torch.count_nonzero(pol.net[-1].bias) == 0
    with torch.no_grad():
        q = run_hedger(pol, mkt, c)
        q_bs = run_hedger(BSDelta(use_dvol=False), mkt, c)
    assert torch.equal(q, q_bs)  # bit for bit, at every rebalancing time
    assert float(q.abs().sum()) > 0


def test_trained_residual_stays_within_its_bound_of_bs_delta():
    P_all = returns_to_prices(gbm_paths(2000, 0.7, 5))
    pol, hist = train_policy(P_all, 0.9, iters=15, batch=512, log_every=0, residual_bound=0.1, stand_in="fwd_noisy")
    mkt, c = _setup(seed=2)
    with torch.no_grad():
        q = run_hedger(pol, mkt, c)
        q_bs = run_hedger(BSDelta(use_dvol=False), mkt, c)
    gap = (q - q_bs).abs()
    assert float(gap.max()) <= 0.1 + 1e-12
    assert float(gap.max()) > 0  # training did move the policy away from BS delta


def test_saved_residual_policy_reloads_with_its_bound(tmp_path):
    pol = _randomised(bound=0.3)
    torch.save(pol.state_dict(), tmp_path / "p.pt")
    sd = torch.load(tmp_path / "p.pt")
    again = ResidualHedger(float(sd["bound"]))
    again.load_state_dict(sd)
    mkt, c = _setup(seed=3)
    with torch.no_grad():
        assert torch.equal(run_hedger(pol, mkt, c), run_hedger(again.eval(), mkt, c))


# ---- 2. the IV input uses only information available at entry ------------------------------------------------------
def test_iv_feature_is_the_static_entry_iv_and_ignores_everything_after_entry():
    mkt, c = _setup()
    cut = {k: v.clone() for k, v in mkt.items()}
    g = torch.Generator().manual_seed(9)
    for k in cut:  # scramble every price and DVOL value strictly after entry
        cut[k][:, H + 1:] *= torch.exp(torch.randn(cut[k][:, H + 1:].shape, generator=g, dtype=torch.float64))
    q_prev = torch.zeros(len(c.K), dtype=torch.float64)
    for t in (0, 5, 20):
        for m in (mkt, cut):
            view = {k: v[:, : H + t + 1] for k, v in m.items()}
            x = residual_features(view, t, q_prev, c, BSDelta().act(view, t, q_prev, c))
            assert torch.equal(x[:, 6], c.iv0.to(torch.float32))  # sigma_entry = entry IV at every t
    # and the input is used: another entry IV gives other actions
    pol = _randomised()
    c2 = c.subset(torch.arange(len(c.K)))
    c2.iv0 = c.iv0 * 1.5
    with torch.no_grad():
        assert not torch.equal(run_hedger(pol, mkt, c)[:, 0], run_hedger(pol, mkt, c2)[:, 0])


def test_residual_actions_do_not_depend_on_the_future():
    mkt, c = _setup()
    pol = _randomised()
    with torch.no_grad():
        full = run_hedger(pol, mkt, c)
        for t in (0, 3, 10, 20, TMAX - 2):
            cut = {k: v.clone() for k, v in mkt.items()}
            g = torch.Generator().manual_seed(t)
            for k in cut:
                cut[k][:, H + t + 1:] *= torch.exp(torch.randn(cut[k][:, H + t + 1:].shape, generator=g, dtype=torch.float64))
            q = run_hedger(pol, cut, c)
            assert torch.equal(q[:, : t + 1], full[:, : t + 1]), f"action at <= {t} changed"
            assert not torch.equal(q[:, t + 1:], full[:, t + 1:])


def test_real_entry_iv_is_the_iv_of_the_entry_trade(monkeypatch):
    """build_entries: the IV comes from the same trade as the entry (the last trade in 07:00-08:00 UTC of the entry
    day); trades on later days do not change it. market_arrays passes it on unchanged as Contracts.iv0."""
    d0 = date(2022, 3, 1)
    inst = "BTC-25MAR22-45000-C"  # 24 days to expiry on d0

    def trade(tid, ts, iv, price=0.02):
        return {"trade_id": tid, "timestamp": ts, "trade_seq": tid, "instrument_name": inst, "price": price,
                "mark_price": price, "iv": iv, "index_price": 44000.0, "direction": "buy", "amount": 1.0}

    day_ms = {d0: 1646118000000, d0 + timedelta(days=1): 1646204400000}  # 07:00 UTC
    trades = {d0: [trade(1, day_ms[d0] + 60_000, 70.0), trade(2, day_ms[d0] + 3_000_000, 72.5)],
              d0 + timedelta(days=1): [trade(3, day_ms[d0 + timedelta(days=1)] + 60_000, 99.0)]}
    monkeypatch.setattr(deribit, "trades_cache_path", lambda day: type("P", (), {"exists": lambda self: day in trades})())
    monkeypatch.setattr(deribit, "day_trades", lambda day: trades[day])
    e = dataset.build_entries(d0, d0 + timedelta(days=1))
    assert len(e) == 1 and e["entry_day"].iloc[0] == d0
    assert e["iv"].iloc[0] == 72.5 and e["trade_id"].iloc[0] == 2  # last trade of the entry window, not day 2

    days = [d0 + timedelta(days=k) for k in range(-40, 80)]
    monkeypatch.setattr(market, "perp_08", lambda: pd.Series(np.linspace(40000, 50000, len(days)), index=days))
    monkeypatch.setattr(market, "funding_cash_daily", lambda: pd.Series(0.0, index=days))
    monkeypatch.setattr(market, "dvol_08", lambda: pd.Series(60.0, index=days))
    df = e.assign(index_price=44000.0)
    _, _, _, c = dataset.market_arrays(df)
    assert float(c.iv0[0]) == pytest.approx(0.725, abs=1e-15)


def test_trailing_stand_in_uses_returns_up_to_entry_only():
    P = returns_to_prices(gbm_paths(64, 0.7, 4))
    T = torch.randint(config.MIN_DAYS, TMAX + 1, (64,), generator=torch.Generator().manual_seed(0))
    s = entry_vol_stand_in(P, T, "trailing")
    P2 = P.clone()
    P2[:, H + 1:] *= 1.7  # change everything after entry
    assert torch.equal(s, entry_vol_stand_in(P2, T, "trailing"))
    P3 = P.clone()
    P3[:, H - 5] *= 1.1  # a price inside the trailing window
    assert not torch.equal(s, entry_vol_stand_in(P3, T, "trailing"))


def test_forward_stand_in_is_the_realised_vol_over_the_option_life(monkeypatch):
    """Training only: with no noise it is exactly the realised vol from entry to expiry; prices after expiry
    and before entry do not enter it."""
    monkeypatch.setattr(config, "RESID_IV_NOISE", 0.0)
    P = returns_to_prices(gbm_paths(32, 0.7, 6))
    T = torch.full((32,), 14)
    s = entry_vol_stand_in(P, T, "fwd_noisy", torch.Generator().manual_seed(0))
    lr = torch.log(P[:, H + 1: H + 15] / P[:, H: H + 14])
    assert torch.allclose(s, torch.sqrt((lr**2).mean(1) * 365.0), rtol=0, atol=1e-12)
    P2 = P.clone()
    P2[:, H + 15:] *= 2.0
    P2[:, : H] *= 0.5
    assert torch.allclose(s, entry_vol_stand_in(P2, T, "fwd_noisy", torch.Generator().manual_seed(0)), rtol=0, atol=1e-12)
