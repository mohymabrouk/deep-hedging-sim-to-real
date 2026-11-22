"""Run hedgers on real options and compute the metrics (loss in % of S0), with paired block-bootstrap CIs."""

import math

import numpy as np
import pandas as pd
import torch

from dh import config
from dh.accounting import hedge_pnl
from dh.hedging import evaluate_hedger, price_window


def positions(hedgers: dict, mkt: dict, c) -> dict[str, torch.Tensor]:
    return {name: evaluate_hedger(h, mkt, c) for name, h in hedgers.items()}


def pnl_table(df: pd.DataFrame, mkt: dict, F: torch.Tensor, c, pos: dict, fee_mult: float = 1.0) -> pd.DataFrame:
    """One row per (option, hedger). Money columns in % of S0 (index at entry)."""
    s0 = df["index_price"].to_numpy()
    prem = torch.as_tensor(df["premium_usd"].to_numpy() / s0)
    pay = torch.as_tensor(df["payoff_usd"].to_numpy() / s0)
    ofee = torch.as_tensor(df["option_fee_usd"].to_numpy() / s0) * fee_mult
    Pw = price_window(mkt["P"], c.T)
    rows = []
    for name, q in pos.items():
        res = hedge_pnl(Pw, q, pay, prem, F, ofee, config.PERP_TAKER_FEE * fee_mult)
        out = pd.DataFrame({k: 100.0 * v.numpy() for k, v in res.items() if k != "turnover"})
        out["turnover"] = res["turnover"].numpy()
        out["loss"] = -out["pnl"]
        out["hedger"] = name
        out["instrument"] = df["instrument"].to_numpy()
        out["entry_day"] = df["entry_day"].to_numpy()
        out["expiry"] = df["expiry"].to_numpy()
        out["is_call"] = df["is_call"].to_numpy()
        out["q0"] = q[:, 0].numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def cvar(x: np.ndarray, level: float = config.CVAR_LEVEL) -> float:
    """Mean of the worst (1 - level) fraction of losses."""
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(math.ceil((1 - level) * len(x))))
    return float(x[:k].mean())


def summarize(t: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    keys = ["hedger"] + (by or [])
    g = t.groupby(keys, sort=False)
    out = g.agg(n=("loss", "size"), mean_pnl=("pnl", "mean"), mean_loss=("loss", "mean"), std_loss=("loss", "std"),
                turnover=("turnover", "mean"), perp_fees=("perp_fees", "mean"), option_fee=("option_fee", "mean"),
                funding=("funding", "mean"), hedge_gains=("gains", "mean"), premium=("premium", "mean"), payoff=("payoff", "mean"))
    out["cvar95_loss"] = g["loss"].apply(cvar)
    cols = ["n", "mean_pnl", "mean_loss", "std_loss", "cvar95_loss", "turnover", "perp_fees", "option_fee", "funding", "hedge_gains", "premium", "payoff"]
    return out[cols].reset_index()


def block_bootstrap_indices(expiries: np.ndarray, block: int, reps: int, seed: int) -> list[np.ndarray]:
    """Moving-block bootstrap over sorted expiry dates; every option of a drawn expiry comes with it."""
    uniq = np.array(sorted(set(expiries)))
    groups = [np.flatnonzero(expiries == e) for e in uniq]
    E = len(uniq)
    b = min(block, E)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        starts = rng.integers(0, E - b + 1, size=int(math.ceil(E / b)))
        pos = np.concatenate([np.arange(s, s + b) for s in starts])[:E]
        out.append(np.concatenate([groups[p] for p in pos]))
    return out


def paired_vs(t: pd.DataFrame, base: str = "bs_iv", reps: int = config.BOOT_REPS, block: int = config.BOOT_BLOCK,
              seed: int = config.SEED) -> pd.DataFrame:
    """Improvement of each hedger over `base` on the same options: mean(loss_base - loss_h) and
    CVaR95(loss_base) - CVaR95(loss_h); positive = better than base. 95% percentile CIs."""
    wide = t.pivot_table(index="instrument", columns="hedger", values="loss", sort=False)
    meta = t[t["hedger"] == base].set_index("instrument").loc[wide.index]
    exp = meta["expiry"].astype(str).to_numpy()
    idxs = block_bootstrap_indices(exp, block, reps, seed)
    L = wide.to_numpy()
    names = list(wide.columns)
    j0 = names.index(base)
    rows = []
    for j, h in enumerate(names):
        d = L[:, j0] - L[:, j]
        dm = [d[i].mean() for i in idxs]
        dc = [cvar(L[i, j0]) - cvar(L[i, j]) for i in idxs]
        ds = [L[i, j0].std() - L[i, j].std() for i in idxs]
        rows.append({"hedger": h, "n": len(d), "n_expiries": len(set(exp)),
                     "d_mean_loss": d.mean(), "d_mean_lo": np.percentile(dm, 2.5), "d_mean_hi": np.percentile(dm, 97.5),
                     "d_cvar95": cvar(L[:, j0]) - cvar(L[:, j]), "d_cvar_lo": np.percentile(dc, 2.5), "d_cvar_hi": np.percentile(dc, 97.5),
                     "d_std": L[:, j0].std() - L[:, j].std(), "d_std_lo": np.percentile(ds, 2.5), "d_std_hi": np.percentile(ds, 97.5),
                     "share_options_better": float((d > 0).mean())})
    return pd.DataFrame(rows)
