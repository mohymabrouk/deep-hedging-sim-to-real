"""Protocol stages: validate (fit <= 2020, choose alpha on 2021 options), final (refit <= 2021, retrain),
test (score 2022 -> latest once, logged). The residual hedgers (added for test run 3) have their own
validate_resid and final_resid stages; freeze and test cover both families."""

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
from dh.hedging import BSDelta, DeepHedger, NoHedge, ResidualHedger, returns_to_prices, train_policy
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


def _resid_choice(name: str) -> dict:
    """Residual hedger of world `name`: the (stand-in, bound, alpha) cell with the lowest validation CVaR95 loss."""
    v = pd.read_csv(config.RESULTS / "validation" / f"dh_resid_{name}.csv")
    best = v.loc[v["cvar95_loss"].idxmin()]
    return {"stand_in": str(best["stand_in"]), "bound": float(best["bound"]), "alpha": float(best["alpha"])}


def selected_residual() -> dict[str, dict]:
    return {name: _resid_choice(name) for name in config.GENERATORS
            if (config.RESULTS / "validation" / f"dh_resid_{name}.csv").exists()}


def selected_alphas() -> dict[str, float]:
    out = {name: _alpha_for(name) for name in config.GENERATORS}
    _json(config.RESULTS / "selected_hyperparameters.json",
          {"criterion": "lowest validation CVaR95 loss (1x fees)", "alpha": out, "residual": selected_residual()})
    return out


def _frozen_cfg() -> tuple[dict, str]:
    """Every constant + the choices made on validation, and its hash (computed the same way in freeze and test)."""
    cfg = {**config.as_dict(), "selected_alpha": selected_alphas(), "selected_residual": selected_residual()}
    return cfg, hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def freeze() -> dict:
    """Frozen config: every constant + the choices made per world on validation (results/frozen_config.json)."""
    cfg, cfg_hash = _frozen_cfg()
    missing = [g for g in config.GENERATORS if g not in cfg["selected_residual"]]
    if missing:
        raise SystemExit(f"residual validation missing for {missing}: run validate_resid first")
    val = {g: pd.read_csv(config.RESULTS / "validation" / f"dh_{g}.csv")[["alpha", "mean_pnl", "std_loss", "cvar95_loss"]].to_dict("records")
           for g in config.GENERATORS}
    val_resid = {g: pd.read_csv(config.RESULTS / "validation" / f"dh_resid_{g}.csv")[
        ["stage", "stand_in", "bound", "alpha", "mean_pnl", "std_loss", "cvar95_loss"]].to_dict("records") for g in config.GENERATORS}
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rule": "per world, the CVaR level alpha with the lowest CVaR95 loss on the 2021 validation options (1x fees)",
              "rule_residual": "per world, the (sigma_entry stand-in, bound c, alpha) with the lowest CVaR95 loss on the 2021 "
                               "validation options (1x fees); stage 1: every stand-in x bound at alpha = RESID_STAGE1_ALPHA, "
                               "stage 2: the stage-1 winner at the other ALPHAS",
              "config_hash": cfg_hash, "selected_alpha": cfg["selected_alpha"], "selected_residual": cfg["selected_residual"],
              "validation_scores": val, "validation_scores_residual": val_resid, "config": cfg}
    _json(config.RESULTS / "frozen_config.json", frozen)
    log.info("frozen config %s: %s, residual %s", cfg_hash, cfg["selected_alpha"], cfg["selected_residual"])
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


# ---------------------------------------------------------------------------------------------------
# residual hedgers (test run 3)
# ---------------------------------------------------------------------------------------------------
def _resid_seed(name: str) -> int:
    """One seed per world for every residual cell: the cells differ only by their settings (common random numbers)."""
    return _policy_seed(f"resid_{name}", 0.0)


def _train_resid(name: str, P_all, stand_in: str, bound: float, alpha: float):
    t0 = time.time()
    pol, hist = train_policy(P_all, alpha, seed=_resid_seed(name), residual_bound=bound, stand_in=stand_in)
    return pol, hist, round(time.time() - t0)


def validate_resid(only: list[str] | None = None) -> None:
    """Residual hedgers on the 2021 validation options, simulators fitted <= 2020 (the cached validation paths).
    Stage 1: every stand-in x bound at alpha = RESID_STAGE1_ALPHA. Stage 2: the stage-1 winner (lowest CVaR95) at
    the other alphas. Resumable: cells already in results/validation/dh_resid_<world>.csv are not retrained."""
    opts = dataset.load_options("validation")
    df, mkt, F, c = dataset.market_arrays(opts)
    log.info("validation options: %d (%s -> %s)", len(df), df["entry_day"].min(), df["expiry"].max())
    vdir = config.RESULTS / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    for name in [g for g in (only or config.GENERATORS) if g in config.GENERATORS]:
        out = vdir / f"dh_resid_{name}.csv"
        rows = pd.read_csv(out).to_dict("records") if out.exists() else []
        done = {(r["stand_in"], float(r["bound"]), float(r["alpha"])) for r in rows}
        P_all = returns_to_prices(generator_paths(name, "validation", config.VAL_FIT_END))

        def run_cell(stage, stand_in, bound, alpha):
            if (stand_in, float(bound), float(alpha)) in done:
                return
            pol, hist, secs = _train_resid(name, P_all, stand_in, bound, alpha)
            torch.save(pol.state_dict(), _stage_dir("validation") / f"dh_resid_{name}_{stand_in}_c{bound}_a{alpha}.pt")
            t = evaluate.pnl_table(df, mkt, F, c, evaluate.positions({f"dh_resid_{name}": pol}, mkt, c))
            s = evaluate.summarize(t).iloc[0].to_dict()
            s.update({"generator": name, "stage": stage, "stand_in": stand_in, "bound": bound, "alpha": alpha,
                      "train_seconds": secs, "final_train_objective": hist[-1]})
            rows.append(s)
            done.add((stand_in, float(bound), float(alpha)))
            pd.DataFrame(rows).to_csv(out, index=False)
            log.info("validation resid %s %s c=%.2f alpha=%.2f: mean loss %.3f, std %.3f, CVaR95 %.3f (%ds)", name, stand_in,
                     bound, alpha, s["mean_loss"], s["std_loss"], s["cvar95_loss"], secs)

        a1 = config.RESID_STAGE1_ALPHA
        for stand_in in config.RESID_STAND_INS:
            for bound in config.RESID_BOUNDS:
                run_cell(1, stand_in, bound, a1)
        st1 = pd.DataFrame(rows)
        st1 = st1[st1["stage"] == 1].reset_index(drop=True)
        best = st1.loc[st1["cvar95_loss"].idxmin()]
        log.info("validation resid %s stage 1 winner: %s c=%.2f (CVaR95 %.3f)", name, best["stand_in"], best["bound"], best["cvar95_loss"])
        for alpha in [a for a in config.ALPHAS if a != a1]:
            run_cell(2, str(best["stand_in"]), float(best["bound"]), alpha)
        log.info("validation resid %s choice: %s", name, _resid_choice(name))


def final_resid(only: list[str] | None = None) -> None:
    """Retrain the residual hedgers on the final-stage paths (simulators fitted <= 2021) with the frozen choice."""
    frozen = json.loads((config.RESULTS / "frozen_config.json").read_text())
    if frozen.get("selected_residual") != selected_residual():
        raise SystemExit("frozen config does not match the residual validation files: run freeze first")
    for name in (only or config.GENERATORS):
        ch = frozen["selected_residual"][name]
        R = generator_paths(name, "final", config.FINAL_FIT_END)
        pol, hist, secs = _train_resid(name, returns_to_prices(R), ch["stand_in"], ch["bound"], ch["alpha"])
        torch.save(pol.state_dict(), _stage_dir("final") / f"dh_resid_{name}.pt")
        log.info("final residual policy %s %s trained in %ds (objective %.3f)", name, ch, secs, hist[-1])


def load_resid(name: str, stage: str = "final") -> ResidualHedger:
    sd = torch.load(config.MODELS / stage / f"dh_resid_{name}.pt")
    pol = ResidualHedger(float(sd["bound"]))
    pol.load_state_dict(sd)
    pol.eval()
    return pol


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
    cfg, cfg_hash = _frozen_cfg()
    frozen = json.loads((config.RESULTS / "frozen_config.json").read_text())
    if frozen["config_hash"] != cfg_hash:
        raise SystemExit(f"config hash {cfg_hash} differs from the frozen config {frozen['config_hash']}: run freeze first")
    alphas = cfg["selected_alpha"]
    hedgers = dict(BENCHMARKS)
    for name in config.GENERATORS:
        pol = DeepHedger()
        pol.load_state_dict(torch.load(config.MODELS / "final" / f"dh_{name}.pt"))
        pol.eval()
        hedgers[f"dh_{name}"] = pol
    for name in config.GENERATORS:
        hedgers[f"dh_resid_{name}"] = load_resid(name)
    _json(config.RESULTS / "test_config.json", cfg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "config_hash": cfg_hash, "code_hash": _code_hash(), "forced_rerun": bool(previous)}) + "\n")
    df, perf = score("test", hedgers, config.RESULTS, "test")
    entry = {"event": "done", "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_hash": cfg_hash,
             "code_hash": _code_hash(), "forced_rerun": bool(previous), "n_options": int(len(df)), "n_expiries": int(df["expiry"].nunique()),
             "first_entry": str(df["entry_day"].min()), "last_expiry": str(df["expiry"].max()), "selected_alpha": alphas,
             "selected_residual": cfg["selected_residual"], "hedgers": list(hedgers), "seconds": round(time.time() - t_start)}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("test logged: %s", entry)
    log.info("performance:\n%s", perf.to_string())
