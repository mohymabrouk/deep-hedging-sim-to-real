"""Training markets (a)-(c): block bootstrap, GARCH(1,1)-t, rough volatility. (d) diffusion is in diffusion.py.

Each generator is fitted on daily 08:00 log returns <= a cutoff date and samples paths of PATH_LEN daily returns.
`martingale` shifts every generator's returns so that E[exp(r)] = 1: the policy should learn to hedge, not to
bet on the 2018-2021 drift.
"""

import logging
import math

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from dh import config

log = logging.getLogger(__name__)


def martingale(R: np.ndarray) -> tuple[np.ndarray, float]:
    shift = float(np.log(np.mean(np.exp(R))))
    return R - shift, shift


class Bootstrap:
    """Stationary bootstrap (Politis & Romano 1994): blocks of geometric length, circular wrap."""

    name = "bootstrap"

    def __init__(self, mean_block: int = config.BOOTSTRAP_MEAN_BLOCK):
        self.mean_block = mean_block

    def fit(self, r: pd.Series, **_):
        self.r = r.to_numpy(float)
        return self

    def params(self) -> dict:
        return {"mean_block_days": self.mean_block, "n_returns": int(len(self.r))}

    def sample(self, n: int, rng: np.random.Generator, length: int = config.PATH_LEN) -> np.ndarray:
        L, p = len(self.r), 1.0 / self.mean_block
        idx = np.empty((n, length), dtype=np.int64)
        idx[:, 0] = rng.integers(L, size=n)
        for j in range(1, length):
            jump = rng.random(n) < p
            idx[:, j] = np.where(jump, rng.integers(L, size=n), (idx[:, j - 1] + 1) % L)
        return self.r[idx]


class Garch:
    """GARCH(1,1), zero mean, standardised Student-t innovations, fitted by maximum likelihood with `arch`.

    Each path starts from a conditional variance drawn from the in-sample filtered variances (BTC fits are close
    to integrated, alpha + beta ~ 1, so the unconditional variance is not a usable starting point). The conditional
    variance is capped at the in-sample maximum: with alpha + beta = 1 and nu ~ 3, uncapped paths explode (log
    returns above 10 in a day were seen in validation), which no BTC history supports. Standardised shocks are
    also truncated at the largest |standardised residual| of the fit: with nu ~ 3 a single draw of ~150 sd gave a
    +12 daily log return in the first final-stage run, which alone set the drift correction (0.115 per day) and broke
    that world (see README, run 1 vs run 2)."""

    name = "garch"

    def fit(self, r: pd.Series, **_):
        res = arch_model(100 * r.to_numpy(float), mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        self.omega, self.alpha, self.beta, self.nu = (float(res.params[k]) for k in ("omega", "alpha[1]", "beta[1]", "nu"))
        self.loglik = float(res.loglikelihood)
        self.var_pool = np.asarray(res.conditional_volatility, float) ** 2
        self.var_cap = float(self.var_pool.max())
        self.z_cap = float(np.nanmax(np.abs(np.asarray(res.std_resid, float))))
        return self

    def params(self) -> dict:
        return {"omega_pct2": self.omega, "alpha": self.alpha, "beta": self.beta, "nu": self.nu,
                "persistence": self.alpha + self.beta, "loglik": self.loglik,
                "start_vol_median_annual": float(np.sqrt(np.median(self.var_pool) * 365) / 100),
                "vol_cap_annual": float(np.sqrt(self.var_cap * 365) / 100), "shock_cap_sd": self.z_cap}

    def sample(self, n: int, rng: np.random.Generator, length: int = config.PATH_LEN) -> np.ndarray:
        var = rng.choice(self.var_pool, size=n)
        scale = math.sqrt((self.nu - 2) / self.nu)
        out = np.empty((n, length))
        for j in range(length):
            eps = np.sqrt(var) * np.clip(rng.standard_t(self.nu, size=n) * scale, -self.z_cap, self.z_cap)
            out[:, j] = eps
            var = np.minimum(self.omega + self.alpha * eps**2 + self.beta * var, self.var_cap)
        return out / 100.0


def fgn_cholesky(H: float, n: int) -> np.ndarray:
    """Cholesky factor of the covariance of n unit-step fractional Gaussian noise increments."""
    k = np.arange(n, dtype=float)
    gam = 0.5 * (np.abs(k + 1) ** (2 * H) + np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H))
    cov = gam[np.abs(k[:, None] - k[None, :]).astype(int)]
    return np.linalg.cholesky(cov + 1e-12 * np.eye(n))


class RoughVol:
    """Rough volatility under P: the RFSV model of Gatheral, Jaisson & Rosenbaum (2018) on the daily grid (the
    P-dynamics behind rough Bergomi).

    Fit: log sigma_d = 0.5 log RV_d (RV from 5-minute returns, 08:00-08:00 days);
    m(2, D) = mean (log sigma_{d+D} - log sigma_d)^2 = nu^2 D^{2H}: OLS of log m on log D (D = 1..30) gives H and nu.
    Model: X = log sigma is a fractional OU process, X_{t+1} = X_t + nu g_t - lam (X_t - m), g = unit fGn (exact,
    Cholesky). m = mean of log sigma; the small mean reversion lam is set so the stationary std of X matches the
    data (without it, log-vol dispersion grows without bound over a 64-day path). Paths start in the stationary
    regime (500-day burn-in). Returns: r_t = sigma_t z_t - sigma_t^2 / 2, z_t = rho g_t + sqrt(1 - rho^2) e_t
    (leverage: the return shock is correlated with the shock to log-vol over the same day).
    """

    name = "rough"
    burn = 500

    def __init__(self, max_lag: int = config.ROUGH_MAX_LAG):
        self.max_lag = max_lag

    def fit(self, r: pd.Series, rv: pd.Series = None, **_):
        full = pd.date_range(min(rv.index), max(rv.index), freq="D").date
        ls = np.log(rv.reindex(full).to_numpy(float)) * 0.5
        lags = np.arange(1, self.max_lag + 1)
        self.zeta = {}
        for qq in (0.5, 1.0, 1.5, 2.0, 3.0):
            m = np.array([np.nanmean(np.abs(ls[d:] - ls[:-d]) ** qq) for d in lags])
            slope, icpt = np.polyfit(np.log(lags), np.log(m), 1)
            self.zeta[qq] = (float(slope), float(icpt))
        slope2, icpt2 = self.zeta[2.0]
        self.H = slope2 / 2.0
        self.nu = math.exp(icpt2 / 2.0)
        qs = np.array(list(self.zeta))
        self.H_from_zeta = float(np.polyfit(qs, [self.zeta[q][0] for q in qs], 1)[0])
        self.m = float(np.nanmean(ls))
        self.s = float(np.nanstd(ls))
        rvs = rv.reindex(full)
        z = (r.reindex(full) / np.sqrt(rvs)).to_numpy()
        dl = np.log(rvs).diff().shift(-1).to_numpy()
        ok = np.isfinite(z) & np.isfinite(dl)
        self.rho = float(np.clip(np.corrcoef(z[ok], dl[ok])[0, 1], -0.9, 0.9))
        self.chol = fgn_cholesky(self.H, self.burn + config.PATH_LEN)
        # mean reversion: match the stationary std of log sigma (simulated, common random numbers over the grid)
        rng = np.random.default_rng(0)
        g = rng.standard_normal((2000, self.burn)) @ fgn_cholesky(self.H, self.burn).T
        grid = np.geomspace(1e-3, 0.5, 40)
        sds = [self._ou(g, lam)[:, -1].std() for lam in grid]
        self.lam = float(grid[int(np.argmin(np.abs(np.array(sds) - self.s)))])
        return self

    def _ou(self, g: np.ndarray, lam: float) -> np.ndarray:
        X = np.empty((g.shape[0], g.shape[1] + 1))
        X[:, 0] = self.m
        for t in range(g.shape[1]):
            X[:, t + 1] = X[:, t] + self.nu * g[:, t] - lam * (X[:, t] - self.m)
        return X

    def params(self) -> dict:
        return {"H": self.H, "nu_daily": self.nu, "rho": self.rho, "H_from_zeta_slope": self.H_from_zeta,
                "zeta_q": {str(k): v[0] for k, v in self.zeta.items()}, "mean_log_sigma": self.m, "std_log_sigma": self.s,
                "mean_reversion_per_day": self.lam, "median_vol_annual": float(math.exp(self.m) * math.sqrt(365))}

    def sample(self, n: int, rng: np.random.Generator, length: int = config.PATH_LEN) -> np.ndarray:
        g = rng.standard_normal((n, self.burn + length)) @ self.chol.T
        X = self._ou(g, self.lam)[:, self.burn: self.burn + length]  # X_t for the path days
        gp = g[:, self.burn: self.burn + length]  # shock from t to t+1
        sig = np.exp(X)
        z = self.rho * gp + math.sqrt(1 - self.rho**2) * rng.standard_normal((n, length))
        return sig * z - 0.5 * sig**2


# ---------------------------------------------------------------------------------------------------
# realism statistics
# ---------------------------------------------------------------------------------------------------
def _pooled_lag_corr(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """corr(x_t, y_{t+k}) pooled over rows (paths); x, y: [n, L]."""
    a = x[:, :-k].ravel() if k else x.ravel()
    b = y[:, k:].ravel() if k else y.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def realism_stats(R: np.ndarray, real: np.ndarray | None = None) -> dict:
    """R: [n, L] daily log returns (a single real series is passed as [1, L])."""
    flat = R.ravel()
    lags = range(1, 11)
    out = {
        "ann_vol": float(flat.std() * math.sqrt(365)),
        "excess_kurtosis": float(stats.kurtosis(flat)),
        "skew": float(stats.skew(flat)),
        "acf_abs_r_lag1_10": float(np.mean([_pooled_lag_corr(np.abs(R), np.abs(R), k) for k in lags])),
        "leverage_corr_r_r2_lag1_10": float(np.mean([_pooled_lag_corr(R, R**2, k) for k in lags])),
        "q01": float(np.quantile(flat, 0.01)),
        "q99": float(np.quantile(flat, 0.99)),
    }
    if real is not None:
        out["W1"] = float(stats.wasserstein_distance(flat, real))
        out["W1_standardised"] = float(stats.wasserstein_distance(flat / flat.std(), real / real.std()))
        out["q01_ratio_to_real"] = out["q01"] / float(np.quantile(real, 0.01))
        out["q99_ratio_to_real"] = out["q99"] / float(np.quantile(real, 0.99))
    return out
