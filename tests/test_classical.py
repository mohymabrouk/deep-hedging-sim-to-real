"""Classical hedgers of test run 4 (PLAN.md change 17): Whalley-Wilmott band, Hull-White minimum-variance delta,
smile-implied delta, and the estimators behind them (Hull-White regression, smile slope at entry).

1. Each one collapses to BS delta at the entry IV, bit for bit: zero band, a = b = c = 0, flat smile.
2. None of them looks ahead: actions up to day t do not change when every price after t is replaced.
3. The band rule holds the position inside the band and trades to its nearest edge otherwise.
4. The estimators recover known coefficients, apply the declared filters, and use only the entry window.
"""

import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
import torch

from dh import classical, config, deribit
from dh.hedging import BSDelta, Contracts, MVDelta, SmileDelta, WWBand, bs_greeks, gbm_paths, returns_to_prices, run_hedger

H, TMAX = config.HIST, config.MAX_DAYS


def _setup(n=96, seed=0, slope=None):
    torch.manual_seed(seed)
    P = returns_to_prices(gbm_paths(n, 0.7, seed))[:, : H + TMAX + 1]
    dvol = 0.6 + 0.05 * torch.randn(P.shape, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    c = Contracts(torch.exp(0.4 * torch.rand(n, generator=g, dtype=torch.float64) - 0.2),
                  torch.randint(config.MIN_DAYS, TMAX + 1, (n,), generator=g), torch.rand(n, generator=g) < 0.5,
                  iv0=0.3 + torch.rand(n, generator=g, dtype=torch.float64), dvol0=dvol[:, H].clone(),
                  smile_slope=slope if slope is None else slope(n, g))
    return {"P": P, "dvol": dvol}, c


def _random_slopes(n, g):
    return 0.8 * torch.rand(n, generator=g, dtype=torch.float64) - 0.4


# ---- 1. each collapses to BS delta -----------------------------------------------------------------------------
@pytest.mark.parametrize("hedger", [WWBand(math.inf), MVDelta(0.0, 0.0, 0.0), SmileDelta()], ids=["ww_zero_band", "mv_abc0", "smile_flat"])
def test_classical_hedger_with_no_adjustment_equals_bs_delta(hedger):
    mkt, c = _setup()  # smile_slope defaults to 0
    with torch.no_grad():
        q = run_hedger(hedger, mkt, c)
        q_bs = run_hedger(BSDelta(use_dvol=False), mkt, c)
    assert torch.equal(q, q_bs)
    assert float(q.abs().sum()) > 0


def test_bs_greeks_delta_is_bit_identical_to_the_benchmark_and_gamma_is_right():
    mkt, c = _setup()
    view = {k: v[:, : H + 4] for k, v in mkt.items()}
    q_prev = torch.zeros(len(c.K), dtype=torch.float64)
    S = view["P"][:, -1]
    tau = torch.clamp((c.T - 3).to(torch.float64), min=1e-9) / 365.0
    g = bs_greeks(S, c.K, tau, c.iv0, c.is_call)
    assert torch.equal(g["delta"], BSDelta().act(view, 3, q_prev, c))
    eps = 1e-5  # gamma = d delta / dS (finite difference)
    up = bs_greeks(S * (1 + eps), c.K, tau, c.iv0, c.is_call)["delta"]
    dn = bs_greeks(S * (1 - eps), c.K, tau, c.iv0, c.is_call)["delta"]
    assert torch.allclose(g["gamma"], (up - dn) / (2 * eps * S), rtol=1e-5, atol=1e-8)


# ---- 2. no look-ahead -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hedger", [WWBand(10.0), MVDelta(0.1, -0.3, 0.2), SmileDelta()], ids=["ww_band", "mv_delta", "smile_delta"])
def test_classical_actions_do_not_depend_on_the_future(hedger):
    mkt, c = _setup(slope=_random_slopes)
    with torch.no_grad():
        full = run_hedger(hedger, mkt, c)
        for t in (0, 3, 10, 20, TMAX - 2):
            cut = {k: v.clone() for k, v in mkt.items()}
            g = torch.Generator().manual_seed(t)
            for k in cut:
                cut[k][:, H + t + 1:] *= torch.exp(torch.randn(cut[k][:, H + t + 1:].shape, generator=g, dtype=torch.float64))
            q = run_hedger(hedger, cut, c)
            assert torch.equal(q[:, : t + 1], full[:, : t + 1]), f"action at <= {t} changed"
            assert not torch.equal(q[:, t + 1:], full[:, t + 1:])


# ---- 3. hedger mechanics ---------------------------------------------------------------------------------------------
def test_ww_band_holds_inside_the_band_and_trades_to_the_nearest_edge():
    mkt, c = _setup(n=200, seed=1)
    ww = WWBand(10.0)
    q_prev = torch.zeros(len(c.K), dtype=torch.float64)
    with torch.no_grad():
        for t in range(TMAX):
            view = {k: v[:, : H + t + 1] for k, v in mkt.items()}
            alive = c.T > t
            q = ww.act(view, t, q_prev, c)
            S = view["P"][:, -1]
            tau = torch.clamp((c.T - t).to(torch.float64), min=1e-9) / 365.0
            g = bs_greeks(S, c.K, tau, c.iv0, c.is_call)
            h = ww.half_width(S, g["gamma"])
            inside = (q_prev >= g["delta"] - h) & (q_prev <= g["delta"] + h)
            assert torch.equal(q[inside & alive], q_prev[inside & alive])  # no trade inside the band
            edge = torch.where(q_prev > g["delta"] + h, g["delta"] + h, g["delta"] - h)
            assert torch.allclose(q[~inside & alive], edge[~inside & alive], rtol=0, atol=1e-15)
            q_prev = q * alive.to(q.dtype)
    turnover = lambda q: (torch.cat([q, torch.zeros_like(q[:, :1])], 1) - torch.cat([torch.zeros_like(q[:, :1]), q], 1)).abs().sum()
    with torch.no_grad():
        assert turnover(run_hedger(ww, mkt, c)) < turnover(run_hedger(BSDelta(), mkt, c))


def test_wider_band_for_lower_risk_aversion():
    S, gam = torch.tensor([1.0]), torch.tensor([3.4])
    widths = [float(WWBand(g).half_width(S, gam)) for g in config.WW_RISK_AVERSIONS]
    assert all(a > b for a, b in zip(widths, widths[1:]))
    assert widths[0] == pytest.approx((1.5 * 0.0005 * 3.4**2 / 1.0) ** (1 / 3))


def test_mv_delta_obeys_put_call_parity():
    """One (a, b, c) on the call delta: MV call delta - MV put delta = 1 at the same strike, as for BS delta."""
    mkt, c = _setup()
    cp, cc = c.subset(torch.arange(len(c.K))), c.subset(torch.arange(len(c.K)))
    cp.is_call = torch.zeros_like(c.is_call)
    cc.is_call = torch.ones_like(c.is_call)
    mv = MVDelta(0.2, -0.5, 0.4)
    with torch.no_grad():
        diff = run_hedger(mv, mkt, cc) - run_hedger(mv, mkt, cp)
    alive = (torch.arange(TMAX).unsqueeze(0) < c.T.unsqueeze(1))
    assert torch.allclose(diff[alive], torch.ones_like(diff[alive]), rtol=0, atol=1e-12)


def test_smile_delta_lowers_delta_when_iv_rises_with_strike():
    """theta > 0 (IV up with strike, a call skew): sticky moneyness lowers the delta of calls and puts alike."""
    mkt, c = _setup()
    c.smile_slope = torch.full((len(c.K),), 0.5, dtype=torch.float64)
    with torch.no_grad():
        q = run_hedger(SmileDelta(), mkt, c)[:, 0]
        q_bs = run_hedger(BSDelta(), mkt, c)[:, 0]
    assert bool((q < q_bs).all())


# ---- 4. estimators ---------------------------------------------------------------------------------------------------
def test_hw_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 20_000
    dc = rng.uniform(0.05, 0.95, n)
    x = rng.normal(0, 0.03, n) * rng.uniform(0.1, 0.4, n)
    a, b, c = 0.05, -0.4, 0.3
    y = x * (a + b * dc + c * dc**2) + rng.normal(0, 1e-4, n)
    fit = classical.hw_fit(pd.DataFrame({"call_delta": dc, "x": x, "y": y, "is_call": dc > 0.5}))
    assert fit["a"] == pytest.approx(a, abs=0.01) and fit["b"] == pytest.approx(b, abs=0.05) and fit["c"] == pytest.approx(c, abs=0.05)
    assert 0 < fit["gain_in_sample"] < 1


def _trade(name, ts, iv, index=50_000.0, mark=0.02):
    return {"instrument_name": name, "timestamp": ts, "trade_seq": ts, "trade_id": str(ts), "iv": iv, "index_price": index,
            "mark_price": mark, "price": mark}


def test_hw_pairs_apply_the_declared_filters_and_use_mark_prices():
    d = date(2021, 6, 1)
    snap0 = {"BTC-25JUN21-50000-C": _trade("BTC-25JUN21-50000-C", 1, 80.0, 50_000.0, 0.03),  # kept
             "BTC-2JUN21-50000-C": _trade("BTC-2JUN21-50000-C", 2, 80.0),  # 1 day to expiry: dropped
             "BTC-25JUN21-70000-C": _trade("BTC-25JUN21-70000-C", 3, 80.0),  # |log(K/S)| = 0.34: dropped
             "BTC-25JUN21-52000-C": _trade("BTC-25JUN21-52000-C", 4, 80.0, mark=0.0001),  # below one tick: dropped
             "BTC-25JUN21-48000-P": _trade("BTC-25JUN21-48000-P", 5, 80.0)}  # not traded on d + 1: dropped
    snap1 = {k: _trade(k, 10, 81.0, 51_000.0, 0.035) for k in snap0 if not k.endswith("48000-P")}
    rows = classical.hw_pairs(snap0, snap1, d)
    assert [r["instrument"] for r in rows] == ["BTC-25JUN21-50000-C"]
    r = rows[0]
    assert r["days"] == 24 and r["iv"] == pytest.approx(0.8)
    assert r["f0"] == pytest.approx(0.03 * 50_000.0) and r["f1"] == pytest.approx(0.035 * 51_000.0)


def _smile_trades(day, expiry_code, smile, strikes, index=50_000.0, minute=10):
    t0 = int(datetime(day.year, day.month, day.day, 7, tzinfo=timezone.utc).timestamp() * 1000)
    out = []
    for i, k in enumerate(strikes):
        m = math.log(k / index)
        out.append(_trade(f"BTC-{expiry_code}-{k}-{'C' if k >= index else 'P'}", t0 + (minute + i) * 60_000, 100 * smile(m), index))
    return out


def test_smile_slope_recovers_a_quadratic_smile_and_only_interpolates():
    day, expiry = date(2021, 6, 1), date(2021, 6, 25)
    smile = lambda m: 0.6 + 0.3 * m + 0.5 * m**2
    strikes = [40_000, 44_000, 48_000, 50_000, 53_000, 57_000, 62_000]
    trades = _smile_trades(day, "25JUN21", smile, strikes)
    trades += _smile_trades(day, "18JUN21", lambda m: 0.9 - 2.0 * m, strikes)  # another expiry must not matter
    fit = classical.fit_smile(classical.smile_points(trades, expiry))
    for m in (-0.15, 0.0, 0.1):
        assert classical.slope_at(fit, m) == pytest.approx(0.3 + 1.0 * m, abs=1e-9)
    assert math.isnan(classical.slope_at(fit, 0.3))  # beyond the highest traded strike: no extrapolation
    few = classical.smile_points(_smile_trades(day, "25JUN21", smile, strikes[:3]), expiry)
    assert classical.fit_smile(few) is None  # fewer than 4 strikes


def test_smile_half_windows_split_the_trades_at_0730():
    day, expiry = date(2021, 6, 1), date(2021, 6, 25)
    strikes = [40_000, 44_000, 48_000, 50_000, 53_000, 57_000]
    early = _smile_trades(day, "25JUN21", lambda m: 0.6 + 0.3 * m, strikes, minute=5)
    late = _smile_trades(day, "25JUN21", lambda m: 0.6 - 0.3 * m, strikes, minute=40)
    s1 = classical.slope_at(classical.fit_smile(classical.smile_points(early + late, expiry, half=1, day=day)), 0.0)
    s2 = classical.slope_at(classical.fit_smile(classical.smile_points(early + late, expiry, half=2, day=day)), 0.0)
    assert s1 == pytest.approx(0.3, abs=1e-9) and s2 == pytest.approx(-0.3, abs=1e-9)


def test_entry_slopes_read_only_the_entry_day(monkeypatch):
    """The slope of an option comes from its entry day's 07:00-08:00 trades; other days are never read."""
    d0 = date(2021, 6, 1)
    strikes = [40_000, 44_000, 48_000, 50_000, 53_000, 57_000]
    trades = {d0: _smile_trades(d0, "25JUN21", lambda m: 0.6 + 0.3 * m, strikes)}
    read = []
    monkeypatch.setattr(deribit, "trades_cache_path", lambda day: type("P", (), {"exists": lambda self: True})())
    monkeypatch.setattr(deribit, "day_trades", lambda day: read.append(day) or trades.get(day, []))
    df = pd.DataFrame({"entry_day": [d0, d0], "expiry": [date(2021, 6, 25), date(2021, 6, 25)],
                       "log_moneyness": [0.0, math.log(70_000 / 50_000)]})
    theta = classical.entry_slopes(df)
    assert theta[0] == pytest.approx(0.3, abs=1e-9) and math.isnan(theta[1])
    assert read == [d0]


def test_slope_reliability_keep_rule():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 0.3, 1000)
    noisy = lambda s: true + rng.normal(0, s, 1000)
    assert classical.slope_reliability(noisy(0.05), noisy(0.1), noisy(0.1))["kept"]
    assert not classical.slope_reliability(noisy(0.05), noisy(2.0), noisy(2.0))["kept"]  # halves disagree
    sparse = np.where(rng.random(1000) < 0.3, true, np.nan)
    assert not classical.slope_reliability(sparse, true, true)["kept"]  # coverage 30% < 50%
