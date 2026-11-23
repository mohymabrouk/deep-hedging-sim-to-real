"""Two descriptive checks added with test run 4 (PLAN.md, change 17). Neither trains, tunes or selects anything.

compat:  does the compatibility gap of arXiv 2608.20842 (difference in hedging risk of the same frozen strategies
         between a simulated world and real 2021 options) rank the worlds' deep hedgers by their run-3 test CVaR95
         better than the realism statistics of results/realism.csv?
haircut: arXiv 2605.21696 says deep hedgers learn a delta haircut explained by spot-IV co-movement. Compare the
         residual hedgers' corrections at entry with the Hull-White minimum-variance adjustment, on the test entries.
"""

import itertools
import json
import logging

import numpy as np
import pandas as pd
import torch

from dh import config, dataset, evaluate
from dh.accounting import hedge_pnl
from dh.hedging import (BSDelta, Contracts, DeepHedger, entry_vol_stand_in, payoff_on_path, price_window, returns_to_prices,
                        run_hedger)
from dh.pipeline import load_resid

log = logging.getLogger(__name__)
STRATEGIES = ("bs_trailing", "dh", "dh_resid")


# ---------------------------------------------------------------------------------------------------
# A. compatibility gap vs realism
# ---------------------------------------------------------------------------------------------------
def _loss(P_window, q, payoff) -> np.ndarray:
    """C = payoff - hedge gains + 0.05% perp costs, % of S0 (no premium, option fee or funding)."""
    res = hedge_pnl(P_window, q, payoff, cost_rate=config.PERP_TAKER_FEE)
    return (-100.0 * res["pnl"]).numpy()


def _risk(x: np.ndarray) -> dict:
    return {"cvar95": evaluate.cvar(x), "std": float(np.std(x, ddof=1)), "mean": float(np.mean(x))}


def _policies(world: str) -> dict:
    pol = DeepHedger()
    pol.load_state_dict(torch.load(config.MODELS / "final" / f"dh_{world}.pt"))
    pol.eval()
    return {"bs_trailing": BSDelta(use_dvol=False), "dh": pol, "dh_resid": load_resid(world)}


def synthetic_contracts(c_real: Contracts, n: int) -> Contracts:
    """Path j hedges real contract j mod len(c_real): the same contract mix in both environments."""
    return c_real.subset(torch.arange(n) % len(c_real.K))


def gap_table() -> pd.DataFrame:
    frozen = json.loads((config.RESULTS / "frozen_config.json").read_text())
    stand_in = {g: v["stand_in"] for g, v in frozen["selected_residual"].items()}
    df, mkt, _, c = dataset.market_arrays(dataset.load_options("validation"))
    P_real = mkt["P"]
    pay_real = torch.as_tensor(df["payoff_usd"].to_numpy() / df["index_price"].to_numpy())
    Pw_real = price_window(P_real, c.T)
    trailing_real = entry_vol_stand_in(P_real, c.T, "trailing")
    rows = []
    for world in config.GENERATORS:
        pols = _policies(world)
        P_syn = returns_to_prices(np.load(config.MODELS / "final" / f"paths_{world}.npy"))
        cs = synthetic_contracts(c, P_syn.shape[0])
        pay_syn, Pw_syn = payoff_on_path(P_syn, cs), price_window(P_syn, cs.T)
        g = torch.Generator().manual_seed(config.SEED)
        iv_syn = {"bs_trailing": entry_vol_stand_in(P_syn, cs.T, "trailing"), "dh": cs.iv0,
                  "dh_resid": entry_vol_stand_in(P_syn, cs.T, stand_in[world], g)}
        iv_real = {"bs_trailing": trailing_real, "dh": c.iv0, "dh_resid": c.iv0}  # residual: entry IV, as in test
        for s in STRATEGIES:
            cs_s, cr_s = cs.subset(torch.arange(len(cs.K))), c.subset(torch.arange(len(c.K)))
            cs_s.iv0, cr_s.iv0 = iv_syn[s], iv_real[s]
            with torch.no_grad():
                l_syn = _loss(Pw_syn, run_hedger(pols[s], {"P": P_syn}, cs_s), pay_syn)
                l_real = _loss(Pw_real, run_hedger(pols[s], {"P": P_real}, cr_s), pay_real)
            a, b = _risk(l_syn), _risk(l_real)
            rows.append({"world": world, "strategy": s, "hedger": {"bs_trailing": "bs_trailing", "dh": f"dh_{world}",
                                                                    "dh_resid": f"dh_resid_{world}"}[s],
                         "n_synthetic": len(l_syn), "n_real": len(l_real),
                         **{f"{k}_synthetic": v for k, v in a.items()}, **{f"{k}_real_2021": v for k, v in b.items()},
                         "gap_cvar95": a["cvar95"] - b["cvar95"], "gap_std": a["std"] - b["std"], "gap_mean": a["mean"] - b["mean"]})
            log.info("compat %s %s: CVaR95 synthetic %.3f, real %.3f, gap %+.3f; std gap %+.3f", world, s, a["cvar95"],
                     b["cvar95"], rows[-1]["gap_cvar95"], rows[-1]["gap_std"])
    return pd.DataFrame(rows)


def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def spearman_perm(x, y) -> tuple[float, float]:
    """Spearman correlation and its one-sided exact permutation p-value (H1: rho > 0), all n! orderings of y."""
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perms = np.array(list(itertools.permutations(ry)))
    a = rx - rx.mean()
    b = perms - perms.mean(1, keepdims=True)
    r = (b @ a) / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum(1)))
    return rho, float(np.mean(r >= rho - 1e-12))


def predictors(gaps: pd.DataFrame, realism: pd.DataFrame, validation: dict) -> pd.DataFrame:
    """One row per deep hedger with every pre-declared predictor (smaller = predicted better)."""
    real = realism.set_index("series").loc["real BTC 2022->"]
    rs = realism.set_index("series")
    dist = pd.DataFrame({
        "i-a W1 standardised": rs["W1_standardised"],
        "i-b W1 raw": rs["W1"],
        "i-c |excess kurtosis - real|": (rs["excess_kurtosis"] - real["excess_kurtosis"]).abs(),
        "i-d |ACF |r| - real|": (rs["acf_abs_r_lag1_10"] - real["acf_abs_r_lag1_10"]).abs(),
        "i-e |leverage - real|": (rs["leverage_corr_r_r2_lag1_10"] - real["leverage_corr_r_r2_lag1_10"]).abs(),
        "i-f |annual vol - real|": (rs["ann_vol"] - real["ann_vol"]).abs(),
        "i-g quantile ratio distance": (rs["q01_ratio_to_real"] - 1).abs() + (rs["q99_ratio_to_real"] - 1).abs(),
    }).loc[list(config.GENERATORS)]
    dist["i-h mean realism rank"] = dist.rank().mean(1)
    G = gaps.set_index(["world", "strategy"])
    rows = []
    for world in config.GENERATORS:
        for fam, s in (("original", "dh"), ("residual", "dh_resid")):
            h = f"dh_{world}" if s == "dh" else f"dh_resid_{world}"
            r = {"hedger": h, "world": world, "family": fam,
                 "ii-a |gap CVaR95|, own policy": abs(G.loc[(world, s), "gap_cvar95"]),
                 "ii-b |gap std|, own policy": abs(G.loc[(world, s), "gap_std"]),
                 "ii-c |gap CVaR95|, BS delta trailing vol": abs(G.loc[(world, "bs_trailing"), "gap_cvar95"]),
                 "ii-d max |gap CVaR95| over 3 strategies": max(abs(G.loc[(world, x), "gap_cvar95"]) for x in STRATEGIES),
                 **dist.loc[world].to_dict(), "iii validation CVaR95 (2021 options)": validation[h]}
            rows.append(r)
    return pd.DataFrame(rows)


def _validation_scores() -> dict:
    out = {}
    frozen = json.loads((config.RESULTS / "frozen_config.json").read_text())
    for g in config.GENERATORS:
        v = pd.read_csv(config.RESULTS / "validation" / f"dh_{g}.csv")
        out[f"dh_{g}"] = float(v.loc[np.isclose(v["alpha"], frozen["selected_alpha"][g]), "cvar95_loss"].iloc[0])
        out[f"dh_resid_{g}"] = float(pd.read_csv(config.RESULTS / "validation" / f"dh_resid_{g}.csv")["cvar95_loss"].min())
    return out


def compat(target_file=None) -> None:
    R = config.RESULTS
    target_file = target_file or (R / "run3_official" / "performance_test.csv")
    gaps = gap_table()
    gaps.to_csv(R / "compatibility_gaps.csv", index=False)
    pred = predictors(gaps, pd.read_csv(R / "realism.csv"), _validation_scores())
    perf = pd.read_csv(target_file).set_index("hedger")
    pred["test_cvar95_run3"] = pred["hedger"].map(perf["cvar95_loss"])
    pred["test_std_run3"] = pred["hedger"].map(perf["std_loss"])
    pred.to_csv(R / "compatibility_predictors.csv", index=False)
    cols = [c for c in pred.columns if c.startswith(("i-", "ii-", "iii"))]
    rows = []
    for fam, part in (("original (n=4)", pred[pred["family"] == "original"]), ("residual (n=4)", pred[pred["family"] == "residual"]),
                      ("both (n=8)", pred)):
        best = part.loc[part["test_cvar95_run3"].idxmin(), "hedger"]
        for col in cols:
            rho, p = spearman_perm(part[col], part["test_cvar95_run3"])
            rho_s, p_s = spearman_perm(part[col], part["test_std_run3"])
            top = part.loc[part[col] == part[col].min(), "hedger"].tolist()
            rows.append({"group": fam, "predictor": col, "kind": "compatibility" if col.startswith("ii-") else
                         ("realism" if col.startswith("i-") else "validation score"), "n": len(part),
                         "spearman_vs_test_cvar95": rho, "p_one_sided": p, "spearman_vs_test_std": rho_s, "p_one_sided_std": p_s,
                         "predicted_best": "/".join(top), "actual_best": best, "picks_best": best in top})
    out = pd.DataFrame(rows)
    out.to_csv(R / "compatibility_ranks.csv", index=False)
    log.info("compatibility ranks:\n%s", out.to_string())


# ---------------------------------------------------------------------------------------------------
# C. delta haircut vs the Hull-White minimum-variance adjustment (arXiv 2605.21696)
# ---------------------------------------------------------------------------------------------------
def _describe(x: np.ndarray, adj: np.ndarray, q: np.ndarray, bs: np.ndarray, is_call: np.ndarray) -> dict:
    nz = (np.abs(x) > 1e-12) & (np.abs(adj) > 1e-12)
    X = np.stack([np.ones_like(adj), adj], 1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    resid = x - X @ coef
    return {"n": len(x), "mean_correction_calls": float(x[is_call].mean()), "mean_correction_puts": float(x[~is_call].mean()),
            "share_haircut": float(np.mean(np.abs(q) < np.abs(bs))),
            "share_haircut_calls": float(np.mean(np.abs(q[is_call]) < np.abs(bs[is_call]))),
            "share_haircut_puts": float(np.mean(np.abs(q[~is_call]) < np.abs(bs[~is_call]))),
            "pearson_with_mv_adjustment": float(np.corrcoef(x, adj)[0, 1]),
            "spearman_with_mv_adjustment": float(pd.Series(x).corr(pd.Series(adj), method="spearman")),
            "share_same_sign": float(np.mean(np.sign(x[nz]) == np.sign(adj[nz]))), "n_sign": int(nz.sum()),
            "ols_intercept": float(coef[0]), "ols_slope": float(coef[1]),
            "ols_r2": float(1 - (resid @ resid) / ((x - x.mean()) @ (x - x.mean())))}


def haircut() -> None:
    R = config.RESULTS
    e = pd.read_csv(R / "test_entry_hedge_ratios.csv")
    bs = e["bs_iv"].to_numpy()
    adj = e["mv_delta"].to_numpy() - bs
    is_call = e["is_call"].to_numpy(bool)
    rows = [{"hedger": "mv_delta", "family": "Hull-White MV adjustment (reference)",
             **_describe(adj, adj, e["mv_delta"].to_numpy(), bs, is_call)}]
    for fam, prefix in (("residual", "dh_resid_"), ("original (supplement)", "dh_")):
        for g in config.GENERATORS:
            h = f"{prefix}{g}"
            q = e[h].to_numpy()
            rows.append({"hedger": h, "family": fam, **_describe(q - bs, adj, q, bs, is_call)})
    out = pd.DataFrame(rows)
    out.to_csv(R / "haircut_check.csv", index=False)
    log.info("haircut check:\n%s", out.to_string())
