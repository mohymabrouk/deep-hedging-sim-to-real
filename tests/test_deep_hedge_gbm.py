"""A deep hedger trained on GBM without costs should approximately recover Black-Scholes delta."""

import math

import numpy as np
import torch

from dh import config
from dh.hedging import Contracts, bs_delta, gbm_paths, returns_to_prices, train_policy

SIGMA = 0.6


def test_deep_hedger_recovers_bs_delta_on_gbm():
    P_all = returns_to_prices(gbm_paths(20_000, SIGMA, seed=11))
    pol, hist = train_policy(P_all, alpha=0.5, iters=500, batch=2048, cost_rate=0.0, seed=7, log_every=0)
    assert hist[-1] < hist[0]
    H = config.HIST
    errs = []
    for T_left in (7, 14, 28):
        for call in (True, False):
            logm = np.linspace(-0.15, 0.15, 13)  # log(S/K)
            n = len(logm)
            # price history whose last 10 daily returns have realised vol exactly SIGMA
            r = SIGMA / math.sqrt(365.0) * np.tile([1.0, -1.0], (n, (H + 1) // 2 + 1))[:, :H]
            lp = np.concatenate([np.zeros((n, 1)), np.cumsum(r, 1)], 1)
            P = torch.as_tensor(np.exp(lp - lp[:, [-1]]))  # S_t = 1
            K = torch.as_tensor(np.exp(-logm))
            c = Contracts(K, torch.full((n,), T_left), torch.full((n,), call))
            tau = torch.full((n,), T_left / 365.0, dtype=torch.float64)
            delta = bs_delta(torch.ones(n, dtype=torch.float64), K, tau, torch.full((n,), SIGMA, dtype=torch.float64), c.is_call)
            with torch.no_grad():
                q = pol.act({"P": P}, 0, delta, c)  # t = 0, T = T_left days left, previous hedge = delta
            errs.append((q - delta).abs().numpy())
    errs = np.concatenate(errs)
    assert errs.mean() < 0.05, f"mean |q - delta| = {errs.mean():.3f}"
    assert errs.max() < 0.15, f"max |q - delta| = {errs.max():.3f}"
