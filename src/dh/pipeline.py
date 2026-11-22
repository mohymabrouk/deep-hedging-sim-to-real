"""Protocol stages: validate (fit <= 2020, choose alpha on 2021 options), final (refit <= 2021, retrain),
test (score 2022 -> latest once, logged)."""

import hashlib
import json
import logging
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import torch

from dh import config, dataset, evaluate, market
from dh.diffusion import Diffusion, make_windows
from dh.hedging import BSDelta, DeepHedger, NoHedge, returns_to_prices, train_policy
from dh.sims import Bootstrap, Garch, RoughVol, martingale, realism_stats

log = logging.getLogger(__name__)
BENCHMARKS = {"no_hedge": NoHedge(), "bs_iv": BSDelta(use_dvol=False), "bs_dvol": BSDelta(use_dvol=True)}


def _stage_dir(stage: str):
    d = config.MODELS / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------------------------------
def training_data(fit_end: date) -> dict:
    r = market.spot_daily_returns(end=fit_end)
    out = {"r": r, "first_day": str(r.index[0]), "last_day": str(r.index[-1]), "n_days": len(r)}
    return out


def generator_paths(name: str, stage: str, fit_end: date) -> np.ndarray:
    """Fit generator `name` on data <= fit_end and sample N_SIM_PATHS martingale paths (cached per stage)."""
    d = _stage_dir(stage)
    path = d / f"paths_{name}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    data = training_data(fit_end)
    r = data["r"]
    rng = np.random.default_rng(_policy_seed(name, 0.0))
    if name == "bootstrap":
        gen = Bootstrap().fit(r)
    elif name == "garch":
        gen = Garch().fit(r)
    elif name == "rough":
        gen = RoughVol().fit(r, rv=market.spot_rv_daily(end=fit_end))
    elif name == "diffusion":
        if (d / "diffusion.pt").exists():  # trained model on disk: only sample
            gen = Diffusion.load(d / "diffusion.pt")
        else:
            windows = make_windows([market.spot_daily_returns(hour=h, end=fit_end).to_numpy() for h in range(24)])
            gen = Diffusion().fit(windows=windows)
            torch.save(gen.state(), d / "diffusion.pt")
    else:
        raise ValueError(name)
    R = gen.sample(config.N_SIM_PATHS, rng)
    if not np.isfinite(R).all():
        raise RuntimeError(f"{name}: non-finite returns sampled")
    R, shift = martingale(R)
    np.save(path, R)
    params = {"generator": name, "stage": stage, "fit_end": str(fit_end), "train_first_day": data["first_day"],
              "train_last_day": data["last_day"], "n_train_days": data["n_days"], "martingale_shift_per_day": shift,
              "fit_and_sample_seconds": round(time.time() - t0), **gen.params()}
    _json(config.RESULTS / "simulators" / f"{stage}_{name}.json", params)
    log.info("generator %s (%s): %s", name, stage, params)
    return R


# ---------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------
def _policy_seed(name: str, alpha: float) -> int:
    return config.SEED + int(hashlib.sha256(f"{name}{alpha}".encode()).hexdigest()[:6], 16) % 10_000


def validate(only: list[str] | None = None) -> None:
    """Thin slice + hyperparameter choice on the 2021 validation options."""
    opts = dataset.load_options("validation")
    df, mkt, F, c = dataset.market_arrays(opts)
    log.info("validation options: %d (%s -> %s)", len(df), df["entry_day"].min(), df["expiry"].max())
    vdir = config.RESULTS / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    if not only or "bench" in only:
        t = evaluate.pnl_table(df, mkt, F, c, evaluate.positions(BENCHMARKS, mkt, c))
        evaluate.summarize(t).to_csv(vdir / "benchmarks.csv", index=False)
        log.info("benchmarks on validation:\n%s", evaluate.summarize(t).to_string())
    for name in [g for g in (only or config.GENERATORS) if g in config.GENERATORS]:
        R = generator_paths(name, "validation", config.VAL_FIT_END)
        P_all = returns_to_prices(R)
        rows = []
        for alpha in config.ALPHAS:
            t0 = time.time()
            pol, hist = train_policy(P_all, alpha, seed=_policy_seed(name, alpha))
            torch.save(pol.state_dict(), _stage_dir("validation") / f"dh_{name}_a{alpha}.pt")
            t = evaluate.pnl_table(df, mkt, F, c, evaluate.positions({f"dh_{name}": pol}, mkt, c))
            s = evaluate.summarize(t).iloc[0].to_dict()
            s.update({"generator": name, "alpha": alpha, "train_seconds": round(time.time() - t0), "final_train_objective": hist[-1]})
            rows.append(s)
            log.info("validation %s alpha=%.2f: mean loss %.3f, std %.3f, CVaR95 %.3f (%.0fs)", name, alpha,
                     s["mean_loss"], s["std_loss"], s["cvar95_loss"], s["train_seconds"])
        pd.DataFrame(rows).to_csv(vdir / f"dh_{name}.csv", index=False)


def _alpha_for(name: str, wait: bool = False) -> float:
    """CVaR level with the lowest validation CVaR95 loss for generator `name` (optionally wait for its file)."""
    path = config.RESULTS / "validation" / f"dh_{name}.csv"
    while wait and not path.exists():
        time.sleep(30)
    v = pd.read_csv(path)
    return float(v.loc[v["cvar95_loss"].idxmin(), "alpha"])


def selected_alphas() -> dict[str, float]:
    out = {name: _alpha_for(name) for name in config.GENERATORS}
    _json(config.RESULTS / "selected_hyperparameters.json", {"criterion": "lowest validation CVaR95 loss (1x fees)", "alpha": out})
    return out


def freeze() -> dict:
    """Frozen config: every constant + the CVaR level chosen per world on validation (results/frozen_config.json)."""
    alphas = selected_alphas()
    val = {g: pd.read_csv(config.RESULTS / "validation" / f"dh_{g}.csv")[["alpha", "mean_pnl", "std_loss", "cvar95_loss"]].to_dict("records")
           for g in config.GENERATORS}
    cfg = {**config.as_dict(), "selected_alpha": alphas}
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rule": "per world, the CVaR level alpha with the lowest CVaR95 loss on the 2021 validation options (1x fees)",
              "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16],
              "selected_alpha": alphas, "validation_scores": val, "config": cfg}
    _json(config.RESULTS / "frozen_config.json", frozen)
    log.info("frozen config %s: %s", frozen["config_hash"], alphas)
    return frozen


def final(only: list[str] | None = None) -> None:
    """Refit on data <= 2021-12-31 and retrain with the chosen alpha. The generator fit does not need the
    validation result, so it can start before validation ends; training waits for the chosen alpha."""
    held_out = market.spot_daily_returns(start=config.TEST_START, end=market.hedge_data_end()).to_numpy()
    _json(config.RESULTS / "realism" / "real_heldout.json", realism_stats(held_out[None, :]))
    for name in (only or config.GENERATORS):
        R = generator_paths(name, "final", config.FINAL_FIT_END)
        _json(config.RESULTS / "realism" / f"{name}.json", realism_stats(R[:, : config.PATH_LEN], held_out))
        alpha = _alpha_for(name, wait=True)
        t0 = time.time()
        pol, hist = train_policy(returns_to_prices(R), alpha, seed=_policy_seed(name, alpha))
        torch.save(pol.state_dict(), _stage_dir("final") / f"dh_{name}.pt")
        log.info("final policy %s alpha=%.2f trained in %.0fs", name, alpha, time.time() - t0)
    if all((config.RESULTS / "validation" / f"dh_{g}.csv").exists() for g in config.GENERATORS):
        selected_alphas()


def realism_table() -> pd.DataFrame:
    rows = [{"series": "real BTC 2022->", **json.loads((config.RESULTS / "realism" / "real_heldout.json").read_text())}]
    for name in config.GENERATORS:
        p = config.RESULTS / "realism" / f"{name}.json"
        if p.exists():
            rows.append({"series": name, **json.loads(p.read_text())})
    return pd.DataFrame(rows)


def _code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((config.ROOT / "src" / "dh").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def score(split: str, hedgers: dict, out_dir, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every hedger on the options of `split` and write all tables to out_dir (prefix_*.csv)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = dataset.load_options(split)
    df, mkt, F, c = dataset.market_arrays(opts)
    log.info("%s options: %d (%s -> %s)", split, len(df), df["entry_day"].min(), df["expiry"].max())
    pos = evaluate.positions(hedgers, mkt, c)
    tables = {m: evaluate.pnl_table(df, mkt, F, c, pos, fee_mult=m) for m in config.FEE_MULTIPLIERS}
    t = tables[1.0]
    t.to_csv(out_dir / f"{prefix}_pnl_per_option.csv", index=False)
    perf = evaluate.summarize(t).merge(evaluate.paired_vs(t), on="hedger", suffixes=("", "_pair"))
    perf.to_csv(out_dir / f"performance_{prefix}.csv", index=False)
    t = t.assign(subperiod=np.where(pd.to_datetime(t["entry_day"]) < pd.Timestamp(config.SUBPERIOD_SPLIT), "2022-23", "2024->"),
                 option_type=np.where(t["is_call"], "call", "put"))
    sub = []
    for key in ("subperiod", "option_type"):
        for v, part in t.groupby(key):
            s = evaluate.summarize(part).merge(evaluate.paired_vs(part), on="hedger", suffixes=("", "_pair"))
            s.insert(1, "split", f"{key}={v}")
            sub.append(s)
    pd.concat(sub, ignore_index=True).to_csv(out_dir / f"performance_splits{'' if prefix == 'test' else '_' + prefix}.csv", index=False)
    fees = []
    for m, tm in tables.items():
        s = evaluate.summarize(tm)
        s.insert(1, "fee_multiplier", m)
        fees.append(s)
    pd.concat(fees, ignore_index=True).to_csv(out_dir / f"fee_sensitivity{'' if prefix == 'test' else '_' + prefix}.csv", index=False)
    qs = pd.DataFrame({name: q[:, 0].numpy() for name, q in pos.items()})
    qs["log_moneyness"] = df["log_moneyness"].to_numpy()
    qs["is_call"] = df["is_call"].to_numpy()
    qs["T"] = df["T"].to_numpy()
    qs.to_csv(out_dir / f"{prefix}_entry_hedge_ratios.csv", index=False)
    return df, perf


def test(force: bool = False) -> None:
    """Score the test period once. The start is logged before scoring, so a crashed run also counts as a run."""
    log_path = config.RESULTS / "test_log.jsonl"
    previous = log_path.exists() and log_path.read_text().strip()
    if previous and not force:
        raise SystemExit("test already run (see results/test_log.jsonl); use --force to run again (it will be logged)")
    t_start = time.time()
    alphas = selected_alphas()
    hedgers = dict(BENCHMARKS)
    for name in config.GENERATORS:
        pol = DeepHedger()
        pol.load_state_dict(torch.load(config.MODELS / "final" / f"dh_{name}.pt"))
        pol.eval()
        hedgers[f"dh_{name}"] = pol
    cfg = config.as_dict()
    cfg["selected_alpha"] = alphas
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    _json(config.RESULTS / "test_config.json", cfg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "config_hash": cfg_hash, "code_hash": _code_hash(), "forced_rerun": bool(previous)}) + "\n")
    df, perf = score("test", hedgers, config.RESULTS, "test")
    entry = {"event": "done", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_hash": cfg_hash,
             "code_hash": _code_hash(), "forced_rerun": bool(previous), "n_options": int(len(df)), "n_expiries": int(df["expiry"].nunique()),
             "first_entry": str(df["entry_day"].min()), "last_expiry": str(df["expiry"].max()), "selected_alpha": alphas,
             "seconds": round(time.time() - t_start)}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("test logged: %s", entry)
    log.info("performance:\n%s", perf.to_string())
