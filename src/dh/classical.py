"""Estimation for the classical hedgers of test run 4 (PLAN.md, change 17), from P1's cached Deribit option trades
(07:00-08:00 UTC every day):

* the Hull & White (2017) regression of daily option price changes, which gives (a, b, c) of the minimum-variance
  delta (`hedging.MVDelta`);
* the smile slope theta = d sigma / d log(K/S) at entry, from the same-expiry trades of the entry window, for the
  smile-implied delta (`hedging.SmileDelta`), with its split-half reliability check.

Only data up to the first hedge is used: the slope of an option comes from trades in 07:00-08:00 UTC of its entry day,
and the first hedge is placed at 08:00.
"""

import logging
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.special import ndtr

from dh import config, dataset, deribit

log = logging.getLogger(__name__)


def _snapshot(day: date) -> dict[str, dict]:
    """Last trade of each instrument in 07:00-08:00 UTC of `day`, or {} if the day is not cached."""
    if not deribit.trades_cache_path(day).exists():
        return {}
    return dataset.last_trades(deribit.day_trades(day))


def _pdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------------------------------
# Hull & White (2017) minimum-variance delta
# ---------------------------------------------------------------------------------------------------
def hw_pairs(snap0: dict[str, dict], snap1: dict[str, dict], day: date) -> list[dict]:
    """One observation per instrument traded in the window on `day` and on the next day (filters on `day`)."""
    lo, hi = config.HW_DAYS_TO_EXPIRY
    rows = []
    for name, t0 in snap0.items():
        t1 = snap1.get(name)
        if t1 is None:
            continue
        try:
            expiry, strike, is_call = dataset.parse_instrument(name)
        except ValueError:
            continue
        days = (expiry - day).days
        iv = t0.get("iv")
        s0, s1 = float(t0["index_price"]), float(t1["index_price"])
        if not (lo <= days <= hi) or iv is None or not (config.MIN_IV <= iv <= config.MAX_IV):
            continue
        if abs(math.log(strike / s0)) >= config.HW_MAX_ABS_LOG_MONEYNESS or float(t0["mark_price"]) < config.MIN_PRICE_BTC:
            continue
        rows.append({"day": day, "instrument": name, "strike": strike, "is_call": is_call, "days": days, "iv": iv / 100.0,
                     "S0": s0, "S1": s1, "f0": float(t0["mark_price"]) * s0, "f1": float(t1["mark_price"]) * s1})
    return rows


def hw_observations(first: date, last: date) -> pd.DataFrame:
    """Daily changes for the Hull-White regression: both days in [first, last]. Adds BS quantities on day d."""
    rows, d = [], first
    snap0 = _snapshot(d)
    while d < last:
        snap1 = _snapshot(d + timedelta(days=1))
        rows += hw_pairs(snap0, snap1, d)
        snap0, d = snap1, d + timedelta(days=1)
    df = pd.DataFrame(rows)
    return add_bs_terms(df)


def add_bs_terms(df: pd.DataFrame) -> pd.DataFrame:
    """delta_BS, call delta and phi(d1) on day d (r = 0, tau = days / 365), and the regression variables."""
    tau = df["days"].to_numpy(float) / config.DAYS_PER_YEAR
    sd = df["iv"].to_numpy(float) * np.sqrt(tau)
    d1 = (np.log(df["S0"].to_numpy(float) / df["strike"].to_numpy(float)) + 0.5 * sd**2) / sd
    dc = ndtr(d1)
    df = df.assign(call_delta=dc, pdf_d1=_pdf(d1), delta=np.where(df["is_call"], dc, dc - 1.0))
    lo, hi = config.HW_CALL_DELTA_RANGE
    df = df[(df["call_delta"] >= lo) & (df["call_delta"] <= hi)].reset_index(drop=True)
    dS = df["S1"] - df["S0"]
    df["y"] = (df["f1"] - df["f0"] - df["delta"] * dS) / df["S0"]  # hedging residual of BS delta, units of S
    df["x"] = df["pdf_d1"] * dS / df["S0"]  # vega / (S sqrt(T)) x dS / S
    return df


def hw_fit(obs: pd.DataFrame) -> dict:
    """OLS without intercept of y on x [1, delta_C, delta_C^2] (Hull & White 2017, eq. for delta_MV)."""
    dc = obs["call_delta"].to_numpy(float)
    X = obs["x"].to_numpy(float)[:, None] * np.stack([np.ones_like(dc), dc, dc**2], 1)
    y = obs["y"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = float(resid @ resid) / max(1, len(y) - 3)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]), "se_a": float(se[0]), "se_b": float(se[1]),
            "se_c": float(se[2]), "n": int(len(y)), "n_days": int(obs["day"].nunique()) if "day" in obs else None,
            "n_calls": int(obs["is_call"].sum()), "gain_in_sample": float(1.0 - (resid @ resid) / (y @ y)),
            "first_day": str(obs["day"].min()) if "day" in obs else None, "last_day_d": str(obs["day"].max()) if "day" in obs else None}


# ---------------------------------------------------------------------------------------------------
# smile slope at entry (Alexander & Imeraj 2023, smile-implied delta)
# ---------------------------------------------------------------------------------------------------
def _window_ms(day: date) -> tuple[int, int, int]:
    """[07:00, 07:30, 08:00) UTC of `day` in ms."""
    t0 = datetime(day.year, day.month, day.day, config.SNAPSHOT_HOUR_START, tzinfo=timezone.utc)
    ms = int(t0.timestamp() * 1000)
    return ms, ms + 30 * 60 * 1000, ms + 60 * 60 * 1000


def smile_points(trades: list[dict], expiry: date, half: int | None = None, day: date | None = None) -> pd.DataFrame:
    """One point per strike of `expiry` traded in the window: median trade IV (fraction) and m = log(K / median index).
    half = 1 / 2 keeps only the trades in 07:00-07:30 / 07:30-08:00 UTC of `day`."""
    if half is not None:
        a, mid, b = _window_ms(day)
        lo, hi = (a, mid) if half == 1 else (mid, b)
    by_strike: dict[float, tuple[list, list]] = {}
    for t in trades:
        if half is not None and not (lo <= t["timestamp"] < hi):
            continue
        try:
            e, strike, _ = dataset.parse_instrument(t["instrument_name"])
        except ValueError:
            continue
        iv = t.get("iv")
        if e != expiry or iv is None or not (config.MIN_IV <= iv <= config.MAX_IV):
            continue
        ivs, idx = by_strike.setdefault(strike, ([], []))
        ivs.append(iv / 100.0)
        idx.append(float(t["index_price"]))
    rows = [{"strike": k, "iv": float(np.median(v[0])), "m": math.log(k / float(np.median(v[1])))} for k, v in by_strike.items()]
    pts = pd.DataFrame(rows, columns=["strike", "iv", "m"])
    return pts[pts["m"].abs() <= config.SMILE_MAX_ABS_M].reset_index(drop=True)


def fit_smile(pts: pd.DataFrame) -> dict | None:
    """OLS quadratic sigma = alpha + beta m + kappa m^2, or None with fewer than SMILE_MIN_STRIKES strikes."""
    if len(pts) < config.SMILE_MIN_STRIKES:
        return None
    kappa, beta, alpha = np.polyfit(pts["m"].to_numpy(float), pts["iv"].to_numpy(float), 2)
    return {"alpha": float(alpha), "beta": float(beta), "kappa": float(kappa), "m_min": float(pts["m"].min()),
            "m_max": float(pts["m"].max()), "n_strikes": int(len(pts))}


def slope_at(fit: dict | None, m: float) -> float:
    """theta = d sigma / d m at m (clipped), NaN if there is no fit or m is outside the fitted strikes."""
    if fit is None or not (fit["m_min"] <= m <= fit["m_max"]):
        return float("nan")
    return float(np.clip(fit["beta"] + 2.0 * fit["kappa"] * m, -config.SMILE_SLOPE_CLIP, config.SMILE_SLOPE_CLIP))


def entry_slopes(df: pd.DataFrame, half: int | None = None) -> np.ndarray:
    """theta for every option row of df (entry_day, expiry, log_moneyness); NaN where it cannot be estimated."""
    out = np.full(len(df), np.nan)
    pos = {k: i for i, k in enumerate(df.index)}
    for day, part in df.groupby("entry_day"):
        if not deribit.trades_cache_path(day).exists():
            continue
        trades = deribit.day_trades(day)
        fits = {}
        for k, r in part.iterrows():
            e = r["expiry"]
            if e not in fits:
                fits[e] = fit_smile(smile_points(trades, e, half=half, day=day))
            out[pos[k]] = slope_at(fits[e], float(r["log_moneyness"]))
    return out


def slope_reliability(full: np.ndarray, h1: np.ndarray, h2: np.ndarray) -> dict:
    """Coverage of the full-window slope and the split-half (07:00-07:30 vs 07:30-08:00) agreement; keep rule of
    PLAN.md change 17."""
    both = np.isfinite(h1) & np.isfinite(h2)
    n_pairs = int(both.sum())
    corr = float(np.corrcoef(h1[both], h2[both])[0, 1]) if n_pairs >= 3 else float("nan")
    spear = float(pd.Series(h1[both]).corr(pd.Series(h2[both]), method="spearman")) if n_pairs >= 3 else float("nan")
    cov = float(np.isfinite(full).mean())
    keep = bool(cov >= config.SMILE_KEEP_MIN_COVERAGE and n_pairs >= config.SMILE_KEEP_MIN_PAIRS
                and np.isfinite(corr) and corr >= config.SMILE_KEEP_MIN_SPLIT_CORR)
    f = full[np.isfinite(full)]
    return {"n_options": int(len(full)), "coverage": cov, "n_split_pairs": n_pairs, "split_half_pearson": corr,
            "split_half_spearman": spear, "theta_median": float(np.median(f)) if len(f) else None,
            "theta_p05": float(np.percentile(f, 5)) if len(f) else None, "theta_p95": float(np.percentile(f, 95)) if len(f) else None,
            "share_at_clip": float(np.mean(np.abs(f) >= config.SMILE_SLOPE_CLIP)) if len(f) else None,
            "rule": f"keep if coverage >= {config.SMILE_KEEP_MIN_COVERAGE}, split-half Pearson >= {config.SMILE_KEEP_MIN_SPLIT_CORR} "
                    f"on >= {config.SMILE_KEEP_MIN_PAIRS} options", "kept": keep}
