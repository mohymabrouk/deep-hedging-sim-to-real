"""Hedgers, the causal hedging loop, and deep-hedging training.

Everything runs in torch. Prices are in units of S0 (index at entry), so P&L comes out in fractions of S0.

The loop is causal by construction: at step t a hedger only receives `market[k][:, :H+t+1]`, i.e. data up to
and including the rebalancing time t (H = number of history days before entry).
"""

import logging
import math
import time

import numpy as np
import torch
from torch import nn

from dh import config
from dh.accounting import hedge_pnl

log = logging.getLogger(__name__)
TMAX = config.MAX_DAYS


# ---------------------------------------------------------------------------------------------------
# contracts and the loop
# ---------------------------------------------------------------------------------------------------
class Contracts:
    """Per-option static data, all torch tensors of shape [N]."""

    def __init__(self, K, T, is_call, iv0=None, dvol0=None, smile_slope=None):
        self.K = torch.as_tensor(np.array(K, dtype=np.float64) if not torch.is_tensor(K) else K, dtype=torch.float64)
        self.T = torch.as_tensor(np.array(T, dtype=np.int64) if not torch.is_tensor(T) else T, dtype=torch.int64)
        self.is_call = torch.as_tensor(np.array(is_call, dtype=bool) if not torch.is_tensor(is_call) else is_call, dtype=torch.bool)
        n = len(self.K)
        self.iv0 = torch.as_tensor(iv0 if iv0 is not None else np.full(n, np.nan), dtype=torch.float64)
        self.dvol0 = torch.as_tensor(dvol0 if dvol0 is not None else np.full(n, np.nan), dtype=torch.float64)
        # d sigma / d log(K/S) at entry, for the smile-implied delta (test run 4); 0 = no smile information
        self.smile_slope = torch.as_tensor(smile_slope if smile_slope is not None else np.zeros(n), dtype=torch.float64)

    def subset(self, idx):
        c = Contracts.__new__(Contracts)
        for k in ("K", "T", "is_call", "iv0", "dvol0", "smile_slope"):
            setattr(c, k, getattr(self, k)[idx])
        return c


def run_hedger(hedger, market: dict, c: Contracts, H: int = config.HIST, tmax: int = TMAX):
    """Positions q [N, tmax]: q[:, t] is held over (t, t+1]; zero from each option's expiry on."""
    n = len(c.K)
    q_prev = torch.zeros(n, dtype=torch.float64)
    out = []
    for t in range(tmax):
        view = {k: v[:, : H + t + 1] for k, v in market.items()}
        q = hedger.act(view, t, q_prev, c).to(torch.float64)
        q = q * (c.T > t).to(q.dtype)
        out.append(q)
        q_prev = q
    return torch.stack(out, 1)


def price_window(P, T, H: int = config.HIST, tmax: int = TMAX):
    """Hedge prices at t = 0..tmax, padded with the expiry price after each option's expiry."""
    steps = torch.arange(tmax + 1).unsqueeze(0)
    idx = H + torch.minimum(steps, T.unsqueeze(1))
    return torch.gather(P, 1, idx)


def payoff_on_path(P, c: Contracts, H: int = config.HIST):
    ST = torch.gather(P, 1, (H + c.T).unsqueeze(1)).squeeze(1)
    return torch.where(c.is_call, torch.relu(ST - c.K), torch.relu(c.K - ST))


# ---------------------------------------------------------------------------------------------------
# features (only data <= t)
# ---------------------------------------------------------------------------------------------------
def realised_vol(P_hist, window: int = config.RV_WINDOW):
    """Annualised realised vol from the last `window` daily log returns of the price history up to t."""
    lp = torch.log(P_hist[:, -(window + 1):])
    r = lp[:, 1:] - lp[:, :-1]
    return torch.sqrt((r**2).mean(1) * config.DAYS_PER_YEAR)


def bs_delta(S, K, tau, sigma, is_call):
    """Black-Scholes delta, r = 0. tau in years, sigma annualised (fraction)."""
    tau = torch.clamp(tau, min=1e-6)
    sigma = torch.clamp(sigma, min=1e-4)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * torch.sqrt(tau))
    nd1 = torch.special.ndtr(d1)
    return torch.where(is_call, nd1, nd1 - 1.0)


def bs_price(S, K, tau, sigma, is_call):
    tau = torch.clamp(torch.as_tensor(tau, dtype=torch.float64), min=1e-12)
    sd = sigma * torch.sqrt(tau)
    d1 = (torch.log(S / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    call = S * torch.special.ndtr(d1) - K * torch.special.ndtr(d2)
    return torch.where(is_call, call, call - S + K)


def policy_features(view, t: int, q_prev, c: Contracts):
    P = view["P"]
    S = P[:, -1]
    logm = torch.log(S / c.K)
    tau = torch.clamp((c.T - t).to(torch.float64), min=1.0) / config.DAYS_PER_YEAR
    rv = torch.clamp(realised_vol(P), min=0.05)
    z = torch.clamp(logm / (rv * torch.sqrt(tau)), -4.0, 4.0)
    x = torch.stack([10.0 * logm, tau * config.DAYS_PER_YEAR / config.MAX_DAYS, q_prev, rv, c.is_call.to(torch.float64), 0.5 * z], 1)
    return x.to(torch.float32)


# ---------------------------------------------------------------------------------------------------
# hedgers
# ---------------------------------------------------------------------------------------------------
class NoHedge:
    def act(self, view, t, q_prev, c):
        return torch.zeros_like(q_prev)


class BSDelta:
    """Black-Scholes delta. sigma = entry IV, or (use_dvol) entry IV + (DVOL_t - DVOL_entry), floored at 5%."""

    def __init__(self, use_dvol: bool = False):
        self.use_dvol = use_dvol

    def act(self, view, t, q_prev, c):
        S = view["P"][:, -1]
        tau = torch.clamp((c.T - t).to(torch.float64), min=1e-9) / config.DAYS_PER_YEAR
        sigma = c.iv0.clone()
        if self.use_dvol:
            shift = view["dvol"][:, -1] - c.dvol0
            sigma = torch.where(torch.isfinite(shift), sigma + shift, sigma)
            sigma = torch.clamp(sigma, min=0.05)
        return bs_delta(S, c.K, tau, sigma, c.is_call)


class DeepHedger(nn.Module):
    """One MLP for every contract: features -> hedge ratio in [0, 1] (calls) or [-1, 0] (puts)."""

    def __init__(self, hidden: int = config.POLICY_HIDDEN, layers: int = config.POLICY_LAYERS, n_in: int = 6):
        super().__init__()
        mods, d = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def act(self, view, t, q_prev, c):
        x = policy_features(view, t, q_prev, c)
        call_q = torch.sigmoid(self(x)).to(torch.float64)
        return torch.where(c.is_call, call_q, call_q - 1.0)


# ---------------------------------------------------------------------------------------------------
# residual deep hedger (added for test run 3): BS delta at the entry vol + a bounded learned correction
# ---------------------------------------------------------------------------------------------------
def entry_vol(c: Contracts) -> torch.Tensor:
    """sigma_entry as a policy input: c.iv0 (fraction), clipped to the 5%-300% IV filter of the real options.
    On real options c.iv0 is the IV of the entry trade (07:00-08:00 UTC on the entry day), so it is known at entry;
    on simulated paths it is a stand-in (`entry_vol_stand_in`). It is static over the option's life."""
    return torch.clamp(c.iv0, config.MIN_IV / 100.0, config.MAX_IV / 100.0)


def residual_features(view, t: int, q_prev, c: Contracts, delta):
    """The six inputs of `policy_features` + sigma_entry + the base BS delta at sigma_entry (all data <= t)."""
    x = policy_features(view, t, q_prev, c)
    extra = torch.stack([entry_vol(c), delta], 1).to(torch.float32)
    return torch.cat([x, extra], 1)


class ResidualHedger(nn.Module):
    """q_t = BS delta(S_t, K, tau_t, sigma_entry) + bound * tanh(MLP(x_t)), clipped to [0, 1] (calls), [-1, 0] (puts).

    The base is exactly the "BS delta, entry IV" benchmark (`BSDelta`, sigma = c.iv0). The MLP has the shape of
    `DeepHedger` (3 x 64, SiLU) with 8 inputs (`residual_features`); its last layer starts at zero, so an untrained
    policy *is* the benchmark and training can only move the hedge by at most `bound` BTC per option."""

    N_IN = 8

    def __init__(self, bound: float, hidden: int = config.POLICY_HIDDEN, layers: int = config.POLICY_LAYERS):
        super().__init__()
        mods, d = [], self.N_IN
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        last = nn.Linear(d, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        mods.append(last)
        self.net = nn.Sequential(*mods)
        self.register_buffer("bound", torch.tensor(float(bound), dtype=torch.float64))
        self.base = BSDelta(use_dvol=False)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def act(self, view, t, q_prev, c):
        delta = self.base.act(view, t, q_prev, c).to(torch.float64)
        x = residual_features(view, t, q_prev, c, delta)
        q = delta + self.bound * torch.tanh(self(x)).to(torch.float64)
        return torch.where(c.is_call, torch.clamp(q, 0.0, 1.0), torch.clamp(q, -1.0, 0.0))


# ---------------------------------------------------------------------------------------------------
# classical hedgers (added for test run 4; pre-declared in PLAN.md, change 17). All use the entry IV, like BSDelta.
# ---------------------------------------------------------------------------------------------------
def bs_greeks(S, K, tau, sigma, is_call) -> dict:
    """BS delta (same formula and clamps as `bs_delta`, so it is bit-identical), call delta, gamma per unit of S and
    phi(d1), r = 0. vega / S = phi(d1) sqrt(tau)."""
    tau = torch.clamp(tau, min=1e-6)
    sigma = torch.clamp(sigma, min=1e-4)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * torch.sqrt(tau))
    nd1 = torch.special.ndtr(d1)
    pdf = torch.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    return {"delta": torch.where(is_call, nd1, nd1 - 1.0), "call_delta": nd1, "pdf_d1": pdf,
            "gamma": pdf / (S * sigma * torch.sqrt(tau)), "tau": tau}


def _entry_iv_greeks(view, t, c):
    """Greeks at the rebalancing time t with the entry IV, the way BSDelta(use_dvol=False) computes its delta."""
    S = view["P"][:, -1]
    tau = torch.clamp((c.T - t).to(torch.float64), min=1e-9) / config.DAYS_PER_YEAR
    return S, bs_greeks(S, c.K, tau, c.iv0.clone(), c.is_call)


class WWBand:
    """Whalley & Wilmott (1997) no-trade band around BS delta at the entry IV. Half-width
    H = (3 kappa S Gamma^2 / (2 gamma))^(1/3); prices are in units of S0, so gamma is a risk aversion per unit of S0.
    Hold the position while it is inside [delta - H, delta + H], else trade to the nearest edge; q_{-1} = 0.
    gamma = inf closes the band, and the hedger is then BS delta at the entry IV."""

    def __init__(self, risk_aversion: float, cost: float = config.PERP_TAKER_FEE):
        self.risk_aversion = float(risk_aversion)
        self.cost = float(cost)

    def half_width(self, S, gamma):
        return (1.5 * self.cost * S * gamma**2 / self.risk_aversion) ** (1.0 / 3.0)

    def act(self, view, t, q_prev, c):
        S, g = _entry_iv_greeks(view, t, c)
        h = self.half_width(S, g["gamma"])
        return torch.minimum(torch.maximum(q_prev, g["delta"] - h), g["delta"] + h)


class MVDelta:
    """Hull & White (2017) minimum-variance delta at the entry IV:
    delta_BS + vega / (S sqrt(T)) (a + b delta_C + c delta_C^2) = delta_BS + phi(d1) (a + b delta_C + c delta_C^2),
    delta_C = call delta (put delta + 1), so one (a, b, c) serves calls and puts (put-call parity)."""

    def __init__(self, a: float, b: float, c: float):
        self.a, self.b, self.c = float(a), float(b), float(c)

    def adjustment(self, g):
        dc = g["call_delta"]
        return g["pdf_d1"] * (self.a + self.b * dc + self.c * dc**2)

    def act(self, view, t, q_prev, c):
        _, g = _entry_iv_greeks(view, t, c)
        return g["delta"] + self.adjustment(g)


class SmileDelta:
    """Smile-implied delta (Alexander & Imeraj 2023): the model-free delta of scale-invariant models (Alexander &
    Nogueira 2007), equal to the sticky-moneyness delta: delta_BS - (vega / S) theta = delta_BS - phi(d1) sqrt(tau) theta.
    theta = d sigma / d log(K/S) is estimated at entry (c.smile_slope); it and the entry IV stay fixed."""

    def act(self, view, t, q_prev, c):
        _, g = _entry_iv_greeks(view, t, c)
        return g["delta"] - g["pdf_d1"] * torch.sqrt(g["tau"]) * c.smile_slope


def entry_vol_stand_in(P: torch.Tensor, T: torch.Tensor, kind: str, g: torch.Generator | None = None,
                       H: int = config.HIST) -> torch.Tensor:
    """sigma_entry for simulated contracts, which have no implied vol. P: [N, L+1] prices, entry at column H.

    'fwd_noisy': realised vol of the path over the option's life (daily returns from entry to expiry, annualised)
                 x exp(eta * eps - eta^2 / 2), eps ~ N(0, 1) per contract, eta = RESID_IV_NOISE: a noisy forecast
                 of the vol the option will actually see, which is what an implied vol is. Training only.
    'trailing':  realised vol of the RESID_TRAIL_WINDOW daily returns up to entry (data <= entry only).
    Both are clipped to the 5%-300% IV filter of the real options (fraction)."""
    lr = torch.log(P[:, 1:] / P[:, :-1])  # lr[:, j] = log(P[j+1] / P[j])
    if kind == "fwd_noisy":
        j = torch.arange(lr.shape[1]).unsqueeze(0)
        live = ((j >= H) & (j < H + T.unsqueeze(1))).to(lr.dtype)  # returns H .. H+T-1: entry -> expiry
        rv = torch.sqrt((lr**2 * live).sum(1) / T.to(lr.dtype) * config.DAYS_PER_YEAR)
        eta = config.RESID_IV_NOISE
        eps = torch.randn(len(T), generator=g, dtype=lr.dtype)
        sig = rv * torch.exp(eta * eps - 0.5 * eta**2)
    elif kind == "trailing":
        w = config.RESID_TRAIL_WINDOW
        sig = torch.sqrt((lr[:, H - w: H] ** 2).mean(1) * config.DAYS_PER_YEAR)
    else:
        raise ValueError(kind)
    return torch.clamp(sig, config.MIN_IV / 100.0, config.MAX_IV / 100.0)


# ---------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------
def returns_to_prices(R: np.ndarray, H: int = config.HIST) -> torch.Tensor:
    """Daily log returns [M, L] -> prices [M, L+1] normalised so that the entry price P[:, H] = 1."""
    lp = np.concatenate([np.zeros((R.shape[0], 1)), np.cumsum(R, 1)], 1)
    lp = lp - lp[:, [H]]
    return torch.as_tensor(np.exp(lp), dtype=torch.float64)


def sample_contracts(n: int, g: torch.Generator) -> Contracts:
    T = torch.randint(config.MIN_DAYS, config.MAX_DAYS + 1, (n,), generator=g)
    logk = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * config.MAX_ABS_LOG_MONEYNESS
    is_call = torch.rand(n, generator=g) < 0.5
    return Contracts(torch.exp(logk), T, is_call)


def cvar_objective(loss, alpha: float):
    """Rockafellar-Uryasev CVaR_alpha with the threshold at its optimum (the alpha-quantile)."""
    w = torch.quantile(loss.detach(), alpha)
    return w + torch.relu(loss - w).mean() / (1.0 - alpha)


def train_policy(P_all: torch.Tensor, alpha: float, iters: int = config.POLICY_ITERS, batch: int = config.POLICY_BATCH,
                 cost_rate: float = config.PERP_TAKER_FEE, seed: int = config.SEED, lr: float = config.POLICY_LR,
                 log_every: int = 250, residual_bound: float | None = None,
                 stand_in: str | None = None) -> tuple[nn.Module, list[float]]:
    """Deep hedging (Buehler et al. 2019): minimise CVaR_alpha of the loss of a short option, in % of S0.

    Default: the plain `DeepHedger`. With `residual_bound` and `stand_in`: a `ResidualHedger` whose contracts get
    sigma_entry = `entry_vol_stand_in(P, T, stand_in)`. The stand-in noise has its own random stream, so for a given
    seed every stand-in and bound sees the same paths and contracts; the default path draws exactly as before."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if residual_bound is None:
        policy = DeepHedger()
    else:
        if stand_in is None:
            raise ValueError("a residual policy needs a sigma_entry stand-in")
        policy = ResidualHedger(residual_bound)
        g_iv = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    hist, t0 = [], time.time()
    M = P_all.shape[0]
    for it in range(iters):
        idx = torch.randint(0, M, (batch,), generator=g)
        P = P_all[idx]
        c = sample_contracts(batch, g)
        if residual_bound is not None:
            c.iv0 = entry_vol_stand_in(P, c.T, stand_in, g_iv)
        q = run_hedger(policy, {"P": P}, c)
        res = hedge_pnl(price_window(P, c.T), q, payoff_on_path(P, c), cost_rate=cost_rate)
        loss = -100.0 * res["pnl"]  # % of S0
        obj = cvar_objective(loss, alpha)
        opt.zero_grad()
        obj.backward()
        opt.step()
        sched.step()
        hist.append(float(obj.detach()))
        if log_every and (it % log_every == 0 or it == iters - 1):
            log.info("  it %4d  CVaR%.2f %.4f  (%.0fs)", it, alpha, float(obj.detach()), time.time() - t0)
    policy.eval()
    return policy, hist


def evaluate_hedger(hedger, market: dict, c: Contracts) -> torch.Tensor:
    with torch.no_grad():
        return run_hedger(hedger, market, c)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def gbm_paths(n: int, sigma: float, seed: int, length: int = config.PATH_LEN, steps_per_day: int = 1) -> np.ndarray:
    """Zero-drift GBM daily log returns (martingale prices)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / config.DAYS_PER_YEAR / steps_per_day
    z = rng.standard_normal((n, length * steps_per_day))
    return sigma * math.sqrt(dt) * z - 0.5 * sigma**2 * dt
