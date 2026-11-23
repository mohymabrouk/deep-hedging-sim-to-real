"""Figures and markdown tables from the result files (every README number comes from these files)."""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from dh import config, evaluate  # noqa: E402
from dh.hedging import Contracts, DeepHedger, bs_delta  # noqa: E402
from dh.pipeline import load_resid, realism_table  # noqa: E402

log = logging.getLogger(__name__)

WORLD = {"bootstrap": "bootstrap", "garch": "GARCH-t", "rough": "rough-vol", "diffusion": "diffusion"}
LABEL = {"no_hedge": "No hedge", "bs_iv": "BS delta, entry IV", "bs_dvol": "BS delta, IV moved by DVOL",
         **{f"dh_{g}": f"Deep hedge, {w} world" for g, w in WORLD.items()},
         **{f"dh_resid_{g}": f"Residual deep hedge, {w} world" for g, w in WORLD.items()},
         "ww_band": "Whalley-Wilmott band", "mv_delta": "Minimum-variance delta (Hull-White)",
         "smile_delta": "Smile-implied delta (Alexander-Imeraj)"}
# categorical slots in fixed order (reference palette); "no hedge" is a neutral reference. A world keeps its hue in
# both deep-hedger families; the family is shown by the marker (hollow = existing, filled = residual) or the panel.
# The three classical hedgers (run 4) share slot 7 (violet): there are only two slots left, so their identity is
# carried by line style / marker shape and by the text label of every row, not by hue.
COLOR = {"no_hedge": "#8a8984", "bs_iv": "#2a78d6", "bs_dvol": "#eb6834", "dh_bootstrap": "#1baf7a",
         "dh_garch": "#eda100", "dh_rough": "#e87ba4", "dh_diffusion": "#008300"}
COLOR.update({f"dh_resid_{g}": COLOR[f"dh_{g}"] for g in config.GENERATORS})
COLOR.update({h: "#6250d6" for h in config.CLASSICAL})
LINESTYLE = {"ww_band": "-", "mv_delta": "--", "smile_delta": ":"}
MARKER = {"ww_band": "s", "mv_delta": "D", "smile_delta": "^"}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
PLAIN = [f"dh_{g}" for g in config.GENERATORS]
RESID = [f"dh_resid_{g}" for g in config.GENERATORS]
CLASSICAL = list(config.CLASSICAL)
DEEP = PLAIN + RESID


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _present(t: pd.DataFrame, names) -> list[str]:
    have = set(t["hedger"])
    return [h for h in names if h in have]


def fig_pnl_distributions(t: pd.DataFrame, out) -> None:
    """Row 1: existing deep hedgers; row 2 (if scored): residual deep hedgers; row 3 (if scored): classical hedgers;
    each against the BS benchmarks."""
    families = ([("existing deep hedgers", PLAIN)] + ([("residual deep hedgers", RESID)] if _present(t, RESID) else [])
                + ([("classical hedgers", CLASSICAL)] if _present(t, CLASSICAL) else []))
    fig, axes = plt.subplots(len(families), 2, figsize=(12, 4.6 * len(families)), facecolor=SURFACE, squeeze=False)
    lo, hi = np.percentile(t.loc[t["hedger"] != "no_hedge", "loss"], [0.5, 99.5])
    bins = np.linspace(lo, hi, 70)
    p = np.linspace(0.90, 0.999, 200)
    for row, (fam, deep) in enumerate(families):
        ax = axes[row, 0]
        _style(ax)
        for h in _present(t, ["bs_iv", "bs_dvol"] + deep):
            x = t.loc[t["hedger"] == h, "loss"]
            ax.hist(x, bins=bins, histtype="step", linewidth=2, color=COLOR[h], label=LABEL[h], density=True,
                    linestyle=LINESTYLE.get(h, "-"))
        ax.set_xlabel("loss, % of S0 (positive = money lost)", color=INK2)
        ax.set_ylabel("density", color=INK2)
        ax.set_title(f"Loss distribution on test options: {fam}", color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=8, frameon=False)
        ax = axes[row, 1]
        _style(ax)
        for h in _present(t, ["no_hedge", "bs_iv", "bs_dvol"] + deep):
            x = t.loc[t["hedger"] == h, "loss"].to_numpy()
            ax.plot(100 * p, np.quantile(x, p), linewidth=2, color=COLOR[h], label=LABEL[h], linestyle=LINESTYLE.get(h, "-"))
        ax.set_xlabel("loss quantile level, %", color=INK2)
        ax.set_ylabel("loss, % of S0", color=INK2)
        ax.set_title(f"Right tail, loss quantiles 90% to 99.9%: {fam}", color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_hedge_ratio(out) -> None:
    """Hedge ratio at entry vs log-moneyness for a 14-day option, realised vol 50% (and entry IV 50% for the residual
    policies), previous hedge = BS delta. Row 1: existing deep hedgers; row 2: residual deep hedgers (if trained)."""
    H, sigma, T_left = config.HIST, 0.5, 14
    logm = np.linspace(-0.2, 0.2, 81)  # log(S/K)
    n = len(logm)
    r = sigma / np.sqrt(365.0) * np.tile([1.0, -1.0], (n, H // 2 + 1))[:, :H]
    lp = np.concatenate([np.zeros((n, 1)), np.cumsum(r, 1)], 1)
    P = torch.as_tensor(np.exp(lp - lp[:, [-1]]))
    K = torch.as_tensor(np.exp(-logm))
    have_resid = all((config.MODELS / "final" / f"dh_resid_{g}.pt").exists() for g in config.GENERATORS)
    rows = [("existing deep hedgers, realised vol 50%", "plain")] + (
        [("residual deep hedgers, realised vol and entry IV 50%", "resid")] if have_resid else [])
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.4 * len(rows)), facecolor=SURFACE, squeeze=False)
    for i, (fam, kind) in enumerate(rows):
        for ax, call in zip(axes[i], (True, False)):
            _style(ax)
            c = Contracts(K, torch.full((n,), T_left), torch.full((n,), call), iv0=np.full(n, sigma))
            delta = bs_delta(torch.ones(n, dtype=torch.float64), K, torch.full((n,), T_left / 365.0, dtype=torch.float64),
                             torch.full((n,), sigma, dtype=torch.float64), c.is_call)
            ax.plot(logm, delta.numpy(), color=INK, linestyle="--", linewidth=2, label="BS delta, sigma = 50%")
            for g in config.GENERATORS:
                if kind == "plain":
                    path = config.MODELS / "final" / f"dh_{g}.pt"
                    if not path.exists():
                        continue
                    pol = DeepHedger()
                    pol.load_state_dict(torch.load(path))
                    pol.eval()
                    name = f"dh_{g}"
                else:
                    pol, name = load_resid(g), f"dh_resid_{g}"
                with torch.no_grad():
                    q = pol.act({"P": P}, 0, delta, c).numpy()
                ax.plot(logm, q, color=COLOR[name], linewidth=2, label=LABEL[name])
            ax.set_xlabel("log(S/K)", color=INK2)
            ax.set_ylabel("hedge ratio (BTC per option)", color=INK2)
            ax.set_title(f"{'Call' if call else 'Put'}, 14 days left: {fam}", color=INK, fontsize=10.5, loc="left")
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_paired(perf: pd.DataFrame, out) -> None:
    """Paired improvement vs BS delta (entry IV) for every hedger but no hedge (off scale; see the tables)."""
    rows = perf[~perf["hedger"].isin(["bs_iv", "no_hedge"])].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    for ax, (m, lo, hi, title) in zip(axes, [("d_mean_loss", "d_mean_lo", "d_mean_hi", "Mean loss reduction vs BS delta (entry IV)"),
                                             ("d_cvar95", "d_cvar_lo", "d_cvar_hi", "CVaR95 loss reduction vs BS delta (entry IV)")]):
        _style(ax)
        y = np.arange(len(rows))[::-1]
        for yi, (_, r) in zip(y, rows.iterrows()):
            filled = r["hedger"].startswith("dh_resid_") or not r["hedger"].startswith("dh_")
            ax.plot([r[lo], r[hi]], [yi, yi], color=COLOR[r["hedger"]], linewidth=2)
            ax.plot(r[m], yi, MARKER.get(r["hedger"], "o"), markersize=8, color=COLOR[r["hedger"]] if filled else SURFACE,
                    markeredgecolor=COLOR[r["hedger"]], markeredgewidth=2)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y, [LABEL[h] for h in rows["hedger"]], fontsize=9, color=INK)
        ax.set_xlabel("% of S0, > 0 = better than BS delta (95% block-bootstrap CI)", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_headline(perf: pd.DataFrame, n_options: int, run_no: int, out) -> None:
    """The one figure under the README headline: change in CVaR95 loss vs BS delta at the entry IV, with the paired
    block-bootstrap 95% CI, for the classical hedgers (run 4 on) and each world's existing (hollow) and residual
    (filled) deep hedger."""
    p = perf.set_index("hedger")
    worlds = [g for g in config.GENERATORS if f"dh_{g}" in p.index]
    classical = [h for h in CLASSICAL if h in p.index]
    n_rows = sum(int(h in p.index) for g in worlds for h in (f"dh_{g}", f"dh_resid_{g}")) + len(classical)
    n_groups = len(worlds) + int(bool(classical))
    height = 0.48 * n_rows + 0.35 * n_groups + 2.2
    fig, ax = plt.subplots(figsize=(11, height), facecolor=SURFACE)
    fig.subplots_adjust(left=0.25, right=0.78, top=1 - 1.25 / height, bottom=0.75 / height)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    yt, yl, y = [], [], 0.0
    rows = []
    short = {"ww_band": "Whalley-Wilmott band", "mv_delta": "min-variance delta (HW)", "smile_delta": "smile-implied delta"}
    for name in classical:
        rows.append((y, name, "classical"))
        yt.append(y)
        yl.append(f"classical: {short[name]}")
        y -= 1.0
    if classical:
        y -= 0.6
    for g in worlds:
        for name, fam in ((f"dh_resid_{g}", "residual"), (f"dh_{g}", "existing")):
            if name in p.index:
                rows.append((y, name, fam))
                yt.append(y)
                yl.append(f"{WORLD[g]} world, {fam}")
                y -= 1.0
        y -= 0.6
    xs = np.concatenate([p.loc[[r[1] for r in rows], "d_cvar_lo"], p.loc[[r[1] for r in rows], "d_cvar_hi"], [0.0]])
    pad = 0.08 * (xs.max() - xs.min())
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.axvspan(0, xs.max() + pad, color="#2a78d6", alpha=0.05, linewidth=0)
    ax.axvline(0, color=INK, linewidth=1.2)
    for yi, name, fam in rows:
        r = p.loc[name]
        col = COLOR[name]
        ax.plot([r["d_cvar_lo"], r["d_cvar_hi"]], [yi, yi], color=col, linewidth=2.2, solid_capstyle="round")
        ax.plot(r["d_cvar95"], yi, MARKER.get(name, "o"), markersize=9, markeredgewidth=2, markeredgecolor=col,
                color=col if fam in ("residual", "classical") else SURFACE, zorder=3)
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.01, yi, f"{r['d_cvar95']:+.2f}  [{r['d_cvar_lo']:+.2f}, {r['d_cvar_hi']:+.2f}]", transform=tr, va="center",
                ha="left", fontsize=9, color=INK2, family="monospace")
    tr = blended_transform_factory(ax.transAxes, ax.transAxes)
    ax.text(1.01, 1.0, "Δ CVaR95 [95% CI]", transform=tr, va="bottom", ha="left", fontsize=9, color=INK)
    ax.set_yticks(yt, yl, fontsize=9.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(min(yt) - 0.8, max(yt) + 0.8)
    ax.set_xlabel("Δ CVaR95 loss vs BS delta at the entry IV, % of S0    (> 0 = smaller tail loss than BS delta)",
                  color=INK2, fontsize=9.5)
    ax.text(0, max(yt) + 0.75, "BS delta, entry IV", ha="center", va="bottom", fontsize=8.5, color=INK, clip_on=False)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=9, markeredgewidth=2, color=INK2, markeredgecolor=INK2,
                      label="residual: BS delta at entry IV + bounded learned correction"),
               Line2D([], [], marker="o", linestyle="", markersize=9, markeredgewidth=2, color=SURFACE, markeredgecolor=INK2,
                      label="existing: MLP on realised-vol features only")]
    if classical:
        handles.append(Line2D([], [], marker="s", linestyle="", markersize=8, markeredgewidth=2, color=COLOR["ww_band"],
                              markeredgecolor=COLOR["ww_band"], label="classical: BS-delta variants fitted on 2021"))
    who = "Classical and deep hedgers" if classical else "Deep hedgers"
    fig.text(0.01, 1 - 0.18 / height, f"{who} vs Black-Scholes delta on {n_options:,} real Deribit BTC options, 2022-2026 "
             f"(test run {run_no})", color=INK, fontsize=12, ha="left", va="top")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.45 / height), ncol=3 if classical else 2,
               frameon=False, fontsize=8.5 if classical else 9)
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
                     "Δ std vs BS-IV [95% CI]": "—" if r["hedger"] == "bs_iv" else _ci(r["d_std"], r["d_std_lo"], r["d_std_hi"]),
                     "turnover": r["turnover"], "perp fees": r["perp_fees"], "funding": r["funding"]})
    return _md(pd.DataFrame(rows))


def run_comparison(perf: pd.DataFrame, t: pd.DataFrame, prev_dir, prev_no: int, run_no: int) -> tuple[str, pd.DataFrame | None]:
    """The previous run (archived in prev_dir) vs the current run, side by side, and the reproduction check: for every
    hedger scored in both runs, the largest absolute difference in any summary statistic and in any single option's P&L."""
    if not (prev_dir / "performance_test.csv").exists():
        return "", None
    prev = pd.read_csv(prev_dir / "performance_test.csv").set_index("hedger")
    cur = perf.set_index("hedger")
    rows = []
    for h in cur.index:
        a = prev.loc[h] if h in prev.index else None
        b = cur.loc[h]
        rows.append({"hedger": LABEL[h],
                     f"run {prev_no}: CVaR95 loss": "not run" if a is None else f"{a['cvar95_loss']:.3f}",
                     f"run {prev_no}: Δ CVaR95 vs BS-IV [95% CI]": "not run" if a is None else ("—" if h == "bs_iv" else _ci(a["d_cvar95"], a["d_cvar_lo"], a["d_cvar_hi"])),
                     f"run {run_no}: CVaR95 loss": f"{b['cvar95_loss']:.3f}",
                     f"run {run_no}: Δ CVaR95 vs BS-IV [95% CI]": "—" if h == "bs_iv" else _ci(b["d_cvar95"], b["d_cvar_lo"], b["d_cvar_hi"]),
                     f"run {run_no}: Δ mean loss vs BS-IV [95% CI]": "—" if h == "bs_iv" else _ci(b["d_mean_loss"], b["d_mean_lo"], b["d_mean_hi"])})
    side = _md(pd.DataFrame(rows))
    t_prev = pd.read_csv(prev_dir / "test_pnl_per_option.csv")
    num = [c for c in prev.columns if c in cur.columns and np.issubdtype(prev[c].dtype, np.number)]
    rep = []
    for h in [h for h in cur.index if h in prev.index]:
        a = t_prev[t_prev["hedger"] == h].set_index("instrument")["pnl"]
        b = t[t["hedger"] == h].set_index("instrument")["pnl"]
        rep.append({"hedger": h, "n_options": int(len(b)), "same_options": bool(a.index.sort_values().equals(b.index.sort_values())),
                    "max_abs_diff_summary": float(np.nanmax(np.abs(prev.loc[h, num].to_numpy(float) - cur.loc[h, num].to_numpy(float)))),
                    "max_abs_diff_option_pnl": float((a - b.reindex(a.index)).abs().max())})
    rep = pd.DataFrame(rep)
    return side, rep


def classical_vs_best_deep(perf: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame | None:
    """Each classical hedger vs BS delta (from perf) and vs the best deep hedger of the run (lowest test CVaR95 of the
    eight, a post-hoc pick), with the same paired block bootstrap over expiries."""
    classical = [h for h in CLASSICAL if h in set(perf["hedger"])]
    deep = perf[perf["hedger"].isin(DEEP)]
    if not classical or deep.empty:
        return None
    best = str(deep.loc[deep["cvar95_loss"].idxmin(), "hedger"])
    pv = evaluate.paired_vs(t[t["hedger"].isin([best] + classical)], base=best).set_index("hedger")
    p = perf.set_index("hedger")
    keys = ("d_cvar95", "d_cvar_lo", "d_cvar_hi", "d_std", "d_std_lo", "d_std_hi", "d_mean_loss", "d_mean_lo", "d_mean_hi")
    rows = []
    for h in classical:
        r, b = p.loc[h], pv.loc[h]
        rows.append({"hedger": h, "best_deep_hedger": best, "cvar95_loss": r["cvar95_loss"], "std_loss": r["std_loss"],
                     "mean_loss": r["mean_loss"], "turnover": r["turnover"], "perp_fees": r["perp_fees"],
                     "turnover_ratio_vs_bs_iv": r["turnover"] / p.loc["bs_iv", "turnover"],
                     "perp_fees_diff_vs_bs_iv": r["perp_fees"] - p.loc["bs_iv", "perp_fees"],
                     **{f"{k}_vs_bs_iv": r[k] for k in keys}, **{f"{k}_vs_best_deep": b[k] for k in keys}})
    return pd.DataFrame(rows)


def checks_md(R) -> list[str]:
    """Tables of the two descriptive checks of PLAN.md change 17, if they have been run."""
    out = []
    if (R / "compatibility_ranks.csv").exists():
        g = pd.read_csv(R / "compatibility_gaps.csv")
        out.append("\n## Compatibility gap (arXiv 2608.20842): hedging risk of frozen strategies on each world's synthetic episodes "
                   "minus on the real 2021 validation options (loss = payoff - hedge gains + 0.05% costs, % of S0; PLAN.md change 17)\n")
        out.append(_md(g[["world", "strategy", "cvar95_synthetic", "cvar95_real_2021", "gap_cvar95", "std_synthetic", "std_real_2021",
                          "gap_std"]]))
        rk = pd.read_csv(R / "compatibility_ranks.csv")
        out.append("\n## Which predictor ranks the deep hedgers by their run-3 test CVaR95? Spearman rank correlation (> 0 = ranks "
                   "correctly), one-sided exact permutation p-value; with n = 4 only a perfect ranking reaches p = 0.042\n")
        out.append(_md(rk[["group", "predictor", "kind", "n", "spearman_vs_test_cvar95", "p_one_sided", "spearman_vs_test_std",
                           "predicted_best", "actual_best"]]))
    if (R / "haircut_check.csv").exists():
        hc = pd.read_csv(R / "haircut_check.csv")
        out.append("\n## Delta haircut check (arXiv 2605.21696): correction at entry (q0 minus BS delta at the entry IV) vs the "
                   "Hull-White minimum-variance adjustment, test options\n")
        out.append(_md(hc[["hedger", "family", "n", "mean_correction_calls", "mean_correction_puts", "share_haircut",
                           "pearson_with_mv_adjustment", "spearman_with_mv_adjustment", "share_same_sign", "ols_intercept",
                           "ols_slope", "ols_r2"]]))
    return out


def classical_validation_md(vdir) -> list[str]:
    out = []
    if not (vdir / "ww_band.csv").exists():
        return out
    ww = pd.read_csv(vdir / "ww_band.csv")
    ww["chosen"] = np.where(ww["cvar95_loss"] == ww["cvar95_loss"].min(), "yes", "")
    out.append("\n## Validation (2021 options): Whalley-Wilmott risk aversion gamma, per unit of S0 (chosen = lowest CVaR95 loss)\n")
    out.append(_md(ww[["risk_aversion", "n", "mean_pnl", "std_loss", "cvar95_loss", "turnover", "perp_fees", "chosen"]]))
    fit = json.loads((vdir / "mv_delta_fit.json").read_text())
    out.append("\n## Hull-White minimum-variance delta: OLS on 2021 daily Deribit mark-price changes (no selection on P&L)\n")
    out.append(_md(pd.DataFrame([{k: fit[k] for k in ("a", "se_a", "b", "se_b", "c", "se_c", "n", "n_calls", "gain_in_sample")}]),
                   "{:.4f}"))
    rel = json.loads((vdir / "smile_slope_reliability.json").read_text())
    out.append("\n## Smile slope at entry: reliability on the 2021 validation options (keep rule declared in PLAN.md change 17)\n")
    out.append(_md(pd.DataFrame([{k: rel[k] for k in ("n_options", "coverage", "n_split_pairs", "split_half_pearson",
                                                       "split_half_spearman", "theta_median", "theta_p05", "theta_p95", "kept")}])))
    out.append("\n## Validation (2021 options): classical hedgers at the chosen settings, paired vs BS-IV\n")
    out.append(perf_md(pd.read_csv(vdir / "classical.csv")))
    return out


def run() -> None:
    R = config.RESULTS
    out = []
    perf = pd.read_csv(R / "performance_test.csv")
    t = pd.read_csv(R / "test_pnl_per_option.csv")
    runs = [e for e in (json.loads(x) for x in (R / "test_log.jsonl").read_text().splitlines() if x.strip()) if e.get("event") == "done"]
    log_line = runs[-1]  # the tables in results/ come from the latest test run
    run_no = len(runs)
    out.append(f"# Result tables (generated by `python -m dh.cli report`)\n\nTest run {run_no} of {run_no}"
               f"{' (forced rerun)' if log_line.get('forced_rerun') else ''}: {log_line['timestamp_utc']}, config hash "
               f"`{log_line['config_hash']}`, code hash `{log_line['code_hash']}`, {log_line['n_options']} options, "
               f"{log_line['n_expiries']} expiries, entries from {log_line['first_entry']}, last expiry {log_line['last_expiry']}.\n")
    out.append("## Performance on test options (1x fees; loss and money in % of S0; Δ > 0 = better than BS-IV)\n")
    out.append(perf_md(perf))
    cl = classical_vs_best_deep(perf, t)
    if cl is not None:
        cl.to_csv(R / "classical_vs_best_deep.csv", index=False)
        best = cl["best_deep_hedger"].iloc[0]
        rows = [{"hedger": LABEL[r["hedger"]], "CVaR95 loss": r["cvar95_loss"], "std loss": r["std_loss"], "turnover": r["turnover"],
                 "turnover / BS-IV": r["turnover_ratio_vs_bs_iv"],
                 "Δ CVaR95 vs BS-IV [95% CI]": _ci(r["d_cvar95_vs_bs_iv"], r["d_cvar_lo_vs_bs_iv"], r["d_cvar_hi_vs_bs_iv"]),
                 "Δ std vs BS-IV [95% CI]": _ci(r["d_std_vs_bs_iv"], r["d_std_lo_vs_bs_iv"], r["d_std_hi_vs_bs_iv"]),
                 "Δ CVaR95 vs best deep [95% CI]": _ci(r["d_cvar95_vs_best_deep"], r["d_cvar_lo_vs_best_deep"], r["d_cvar_hi_vs_best_deep"]),
                 "Δ std vs best deep [95% CI]": _ci(r["d_std_vs_best_deep"], r["d_std_lo_vs_best_deep"], r["d_std_hi_vs_best_deep"])}
                for _, r in cl.iterrows()]
        out.append(f"\n## Classical hedgers vs BS delta and vs the best deep hedger of run {run_no} ({LABEL[best]}: lowest test "
                   "CVaR95 of the eight, a post-hoc pick); Δ > 0 = better than the reference\n")
        out.append(_md(pd.DataFrame(rows)))
    prev_no = run_no - 1
    prev_dir = R / f"run{prev_no}_official"
    side, rep = run_comparison(perf, t, prev_dir, prev_no, run_no)
    if side:
        out.append(f"\n## Run {prev_no} vs run {run_no}, test options (1x fees; % of S0; run {prev_no} is archived in "
                   f"`results/run{prev_no}_official/`)\n")
        out.append(side)
        out.append("\n## Reproduction check: hedgers scored in both runs (largest absolute difference, % of S0)\n")
        out.append(_md(rep, "{:.2e}"))
        rep.to_csv(R / f"reproduction_vs_run{prev_no}.csv", index=False)
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
    out += checks_md(R)
    val = pd.concat([pd.read_csv(R / "validation" / f"dh_{g}.csv") for g in config.GENERATORS], ignore_index=True)
    out.append("\n## Validation (2021 options): CVaR level alpha per generator (chosen = lowest CVaR95 loss)\n")
    out.append(_md(val[["generator", "alpha", "n", "mean_pnl", "std_loss", "cvar95_loss", "turnover", "train_seconds"]]))
    vr = [R / "validation" / f"dh_resid_{g}.csv" for g in config.GENERATORS]
    if all(p.exists() for p in vr):
        v = pd.concat([pd.read_csv(p) for p in vr], ignore_index=True)
        best = v.groupby("generator")["cvar95_loss"].transform("min")
        v["chosen"] = np.where(v["cvar95_loss"] == best, "yes", "")
        out.append("\n## Validation (2021 options): residual hedgers, sigma_entry stand-in x bound c x alpha "
                   f"(stage 1 at alpha = {config.RESID_STAGE1_ALPHA}, stage 2 = stage-1 winner at the other alphas; "
                   "chosen = lowest CVaR95 loss)\n")
        out.append(_md(v[["generator", "stage", "stand_in", "bound", "alpha", "n", "mean_pnl", "std_loss", "cvar95_loss", "turnover",
                          "train_seconds", "chosen"]]))
    out += classical_validation_md(R / "validation")
    vb = pd.read_csv(R / "validation" / "benchmarks.csv")
    out.append("\n## Validation benchmarks (2021 options)\n")
    out.append(_md(vb[["hedger", "n", "mean_pnl", "std_loss", "cvar95_loss", "turnover", "perp_fees", "funding"]]))
    (R / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    rt.to_csv(R / "realism.csv", index=False)
    fig_headline(perf, int(log_line["n_options"]), run_no, R / "fig_headline.png")
    fig_pnl_distributions(t, R / "fig_pnl_distributions.png")
    fig_hedge_ratio(R / "fig_hedge_ratio_vs_moneyness.png")
    fig_paired(perf, R / "fig_paired_improvement.png")
    log.info("report written: %s", R / "tables.md")
