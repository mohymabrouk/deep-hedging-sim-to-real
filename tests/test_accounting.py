"""The P&L accounting shared by training and the real backtest."""

import math

import numpy as np
import torch

from dh.accounting import hedge_pnl
from dh.hedging import bs_delta, bs_price


def _gbm(n, steps, T_days, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    dt = T_days / 365.0 / steps
    z = torch.randn((n, steps), generator=g, dtype=torch.float64)
    lr = sigma * math.sqrt(dt) * z - 0.5 * sigma**2 * dt
    S = torch.exp(torch.cat([torch.zeros(n, 1, dtype=torch.float64), torch.cumsum(lr, 1)], 1))
    return S, dt


def _bs_hedge_error(steps, n=4000, sigma=0.6, T_days=30, K=1.05, is_call=True, seed=0):
    S, dt = _gbm(n, steps, T_days, sigma, seed)
    calls = torch.full((n,), is_call)
    Kt = torch.full((n,), K, dtype=torch.float64)
    tau = torch.tensor([(steps - i) * dt for i in range(steps)], dtype=torch.float64)
    q = torch.stack([bs_delta(S[:, i], Kt, tau[i].expand(n), torch.full((n,), sigma, dtype=torch.float64), calls) for i in range(steps)], 1)
    premium = bs_price(torch.tensor(1.0, dtype=torch.float64), torch.tensor(K, dtype=torch.float64), T_days / 365.0,
                       torch.tensor(sigma, dtype=torch.float64), torch.tensor(is_call))
    payoff = torch.relu(S[:, -1] - K) if is_call else torch.relu(K - S[:, -1])
    return hedge_pnl(S, q, payoff, premium, cost_rate=0.0)["pnl"], float(premium)


def test_bs_delta_on_gbm_zero_cost_fine_rebalancing_is_near_exact():
    """Selling at the BS price and delta-hedging finely under GBM must give ~zero P&L (checks the accounting)."""
    pnl_fine, premium = _bs_hedge_error(steps=30 * 96)  # every 15 minutes
    pnl_daily, _ = _bs_hedge_error(steps=30)
    assert abs(float(pnl_fine.mean())) < 0.0005  # < 0.05% of S0
    assert float(pnl_fine.std()) < 0.07 * premium  # < 7% of the premium
    # discrete-hedging error shrinks like 1/sqrt(steps): 96x more steps -> about 10x smaller
    ratio = float(pnl_daily.std() / pnl_fine.std())
    assert 6.0 < ratio < 14.0


def test_bs_delta_put_side_too():
    pnl, premium = _bs_hedge_error(steps=30 * 96, K=0.95, is_call=False, seed=3)
    assert abs(float(pnl.mean())) < 0.0005 and float(pnl.std()) < 0.1 * premium


def test_hand_computed_fees_gains_funding():
    P = np.array([[100.0, 110.0, 105.0]])
    q = np.array([[0.5, 1.0]])
    res = hedge_pnl(P, q, payoff=np.array([3.0]), premium=np.array([4.0]), funding_cash=np.array([[1.0, 2.0]]),
                    option_fee=np.array([0.1]), cost_rate=0.001)
    # trades: +0.5 at 100, +0.5 at 110, -1.0 at 105 (unwind)
    assert np.isclose(res["perp_fees"][0], 0.001 * (50 + 55 + 105))
    assert np.isclose(res["gains"][0], 0.5 * 10 + 1.0 * (-5))
    assert np.isclose(res["funding"][0], 0.5 * 1.0 + 1.0 * 2.0)
    assert np.isclose(res["turnover"][0], 2.0)
    assert np.isclose(res["pnl"][0], 4.0 - 3.0 + 0.0 - 0.21 - 0.1 - 2.5)


def test_padding_after_expiry_changes_nothing():
    """An option expiring at t=1 inside a 3-step array: zero positions after expiry, price padded."""
    P = np.array([[100.0, 104.0, 104.0, 104.0]])
    q = np.array([[0.6, 0.0, 0.0]])
    short = hedge_pnl(P[:, :2], q[:, :1], payoff=np.array([0.0]), cost_rate=0.001)
    padded = hedge_pnl(P, q, payoff=np.array([0.0]), cost_rate=0.001)
    for k in ("pnl", "gains", "perp_fees", "turnover"):
        assert np.isclose(short[k][0], padded[k][0])


def test_numpy_and_torch_give_identical_numbers():
    rng = np.random.default_rng(1)
    P = 1 + 0.05 * rng.standard_normal((50, 11)).cumsum(1)
    q = rng.uniform(-1, 1, (50, 10))
    F = 0.001 * rng.standard_normal((50, 10))
    pay, prem, fee = rng.uniform(0, 0.1, 50), rng.uniform(0, 0.1, 50), rng.uniform(0, 0.001, 50)
    a = hedge_pnl(P, q, pay, prem, F, fee, 0.0005)
    b = hedge_pnl(*(torch.as_tensor(x) for x in (P, q, pay, prem, F, fee)), 0.0005)
    for k in a:
        assert np.allclose(a[k], b[k].numpy())
