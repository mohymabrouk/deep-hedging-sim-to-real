"""Figures and markdown tables from the result files (every README number comes from these files)."""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dh import config  # noqa: E402
from dh.hedging import Contracts, DeepHedger, bs_delta  # noqa: E402
from dh.pipeline import realism_table  # noqa: E402

log = logging.getLogger(__name__)

LABEL = {"no_hedge": "No hedge", "bs_iv": "BS delta, entry IV", "bs_dvol": "BS delta, IV moved by DVOL",
         "dh_bootstrap": "Deep hedge, bootstrap world", "dh_garch": "Deep hedge, GARCH-t world",
         "dh_rough": "Deep hedge, rough-vol world", "dh_diffusion": "Deep hedge, diffusion world"}
# categorical slots in fixed order (reference palette); "no hedge" is a neutral reference
COLOR = {"no_hedge": "#8a8984", "bs_iv": "#2a78d6", "bs_dvol": "#eb6834", "dh_bootstrap": "#1baf7a",
         "dh_garch": "#eda100", "dh_rough": "#e87ba4", "dh_diffusion": "#008300"}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_pnl_distributions(t: pd.DataFrame, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    lo, hi = np.percentile(t.loc[t["hedger"] != "no_hedge", "loss"], [0.5, 99.5])
    bins = np.linspace(lo, hi, 70)
    for h in config.HEDGERS:
        if h == "no_hedge":
            continue
        x = t.loc[t["hedger"] == h, "loss"]
        ax.hist(x, bins=bins, histtype="step", linewidth=2, color=COLOR[h], label=LABEL[h], density=True)
    ax.set_xlabel("loss, % of S0 (positive = money lost)", color=INK2)
    ax.set_ylabel("density", color=INK2)
    ax.set_title("Loss distribution on test options (hedged strategies)", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    _style(ax)
    p = np.linspace(0.90, 0.999, 200)
    for h in config.HEDGERS:
        x = t.loc[t["hedger"] == h, "loss"].to_numpy()
        ax.plot(100 * p, np.quantile(x, p), linewidth=2, color=COLOR[h], label=LABEL[h])
    ax.set_xlabel("loss quantile level, %", color=INK2)
    ax.set_ylabel("loss, % of S0", color=INK2)
    ax.set_title("Right tail: loss quantiles from 90% to 99.9%", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_hedge_ratio(out) -> None:
    """Hedge ratio at entry vs log-moneyness for a 14-day option, realised vol 50%, previous hedge = BS delta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE, sharey=False)
    H, sigma, T_left = config.HIST, 0.5, 14
    logm = np.linspace(-0.2, 0.2, 81)  # log(S/K)
    n = len(logm)
    r = sigma / np.sqrt(365.0) * np.tile([1.0, -1.0], (n, H // 2 + 1))[:, :H]
    lp = np.concatenate([np.zeros((n, 1)), np.cumsum(r, 1)], 1)
    P = torch.as_tensor(np.exp(lp - lp[:, [-1]]))
    K = torch.as_tensor(np.exp(-logm))
    for ax, call in zip(axes, (True, False)):
        _style(ax)
        c = Contracts(K, torch.full((n,), T_left), torch.full((n,), call))
        delta = bs_delta(torch.ones(n, dtype=torch.float64), K, torch.full((n,), T_left / 365.0, dtype=torch.float64),
                         torch.full((n,), sigma, dtype=torch.float64), c.is_call)
        ax.plot(logm, delta.numpy(), color=INK, linestyle="--", linewidth=2, label="BS delta, sigma = 50%")
        for g in config.GENERATORS:
            path = config.MODELS / "final" / f"dh_{g}.pt"
            if not path.exists():
                continue
            pol = DeepHedger()
            pol.load_state_dict(torch.load(path))
            pol.eval()
            with torch.no_grad():
                q = pol.act({"P": P}, 0, delta, c).numpy()
            ax.plot(logm, q, color=COLOR[f"dh_{g}"], linewidth=2, label=LABEL[f"dh_{g}"])
        ax.set_xlabel("log(S/K)", color=INK2)
        ax.set_ylabel("hedge ratio (BTC per option)", color=INK2)
        ax.set_title(f"{'Call' if call else 'Put'}: 14 days left, realised vol 50%", color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_paired(perf: pd.DataFrame, out) -> None:
    rows = perf[perf["hedger"] != "bs_iv"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), facecolor=SURFACE)
    for ax, (m, lo, hi, title) in zip(axes, [("d_mean_loss", "d_mean_lo", "d_mean_hi", "Mean loss reduction vs BS delta (entry IV)"),
                                             ("d_cvar95", "d_cvar_lo", "d_cvar_hi", "CVaR95 loss reduction vs BS delta (entry IV)")]):
        _style(ax)
        y = np.arange(len(rows))[::-1]
        for yi, (_, r) in zip(y, rows.iterrows()):
            ax.plot([r[lo], r[hi]], [yi, yi], color=COLOR[r["hedger"]], linewidth=2)
            ax.plot(r[m], yi, "o", markersize=8, color=COLOR[r["hedger"]], markeredgecolor=SURFACE, markeredgewidth=2)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y, [LABEL[h] for h in rows["hedger"]], fontsize=9, color=INK)
        ax.set_xlabel("% of S0, > 0 = better than BS delta (95% block-bootstrap CI)", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ci(a, lo, hi) -> str:
    return f"{a:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def perf_md(perf: pd.DataFrame) -> str:
    rows = []
    for _, r in perf.iterrows():
        rows.append({"hedger": LABEL[r["hedger"]], "mean P&L": r["mean_pnl"], "std loss": r["std_loss"], "CVaR95 loss": r["cvar95_loss"],
                     "Δ mean loss vs BS-IV [95% CI]": "—" if r["hedger"] == "bs_iv" else _ci(r["d_mean_loss"], r["d_mean_lo"], r["d_mean_hi"]),
                     "Δ CVaR95 vs BS-IV [95% CI]": "—" if r["hedger"] == "bs_iv" else _ci(r["d_cvar95"], r["d_cvar_lo"], r["d_cvar_hi"]),
                     "turnover": r["turnover"], "perp fees": r["perp_fees"], "funding": r["funding"]})
    return _md(pd.DataFrame(rows))


def run() -> None:
    R = config.RESULTS
    out = []
    perf = pd.read_csv(R / "performance_test.csv")
    t = pd.read_csv(R / "test_pnl_per_option.csv")
    runs = [e for e in (json.loads(x) for x in (R / "test_log.jsonl").read_text().splitlines() if x.strip()) if e.get("event") == "done"]
    log_line = runs[-1]  # the tables in results/ come from the latest test run
    out.append(f"# Result tables (generated by `python -m dh.cli report`)\n\nTest run {len(runs)} of {len(runs)}"
               f"{' (forced rerun)' if log_line.get('forced_rerun') else ''}: {log_line['timestamp_utc']}, config hash "
               f"`{log_line['config_hash']}`, code hash `{log_line['code_hash']}`, {log_line['n_options']} options, "
               f"{log_line['n_expiries']} expiries, entries from {log_line['first_entry']}, last expiry {log_line['last_expiry']}.\n")
    out.append("## Performance on test options (1x fees; loss and money in % of S0; Δ > 0 = better than BS-IV)\n")
    out.append(perf_md(perf))
    splits = pd.read_csv(R / "performance_splits.csv")
    for split, part in splits.groupby("split", sort=False):
        out.append(f"\n## Split: {split} (n = {int(part['n'].iloc[0])} options)\n")
        out.append(perf_md(part))
    fees = pd.read_csv(R / "fee_sensitivity.csv")
    fw = fees.pivot_table(index="hedger", columns="fee_multiplier", values=["mean_pnl", "cvar95_loss"], sort=False)
    fw.columns = [f"{a} @ {b:g}x fees" for a, b in fw.columns]
    fw = fw.reset_index()
    fw["hedger"] = fw["hedger"].map(LABEL)
    out.append("\n## Fee sensitivity (fees x0 / x1 / x2; policies not retrained; funding unchanged)\n")
    out.append(_md(fw))
    rt = realism_table()
    cols = ["series", "ann_vol", "excess_kurtosis", "acf_abs_r_lag1_10", "leverage_corr_r_r2_lag1_10", "q01", "q99", "W1", "W1_standardised"]
    out.append("\n## Simulator realism vs held-out daily returns (2022 -> end of hedge data); generators fitted on <= 2021\n")
    out.append(_md(rt[[c for c in cols if c in rt.columns]], "{:.4f}"))
    val = pd.concat([pd.read_csv(R / "validation" / f"dh_{g}.csv") for g in config.GENERATORS], ignore_index=True)
    out.append("\n## Validation (2021 options): CVaR level alpha per generator (chosen = lowest CVaR95 loss)\n")
    out.append(_md(val[["generator", "alpha", "n", "mean_pnl", "std_loss", "cvar95_loss", "turnover", "train_seconds"]]))
    vb = pd.read_csv(R / "validation" / "benchmarks.csv")
    out.append("\n## Validation benchmarks (2021 options)\n")
    out.append(_md(vb[["hedger", "n", "mean_pnl", "std_loss", "cvar95_loss", "turnover", "perp_fees", "funding"]]))
    (R / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    rt.to_csv(R / "realism.csv", index=False)
    fig_pnl_distributions(t, R / "fig_pnl_distributions.png")
    fig_hedge_ratio(R / "fig_hedge_ratio_vs_moneyness.png")
    fig_paired(perf, R / "fig_paired_improvement.png")
    log.info("report written: %s", R / "tables.md")
