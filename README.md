# Deep hedging BTC options: which simulated market holds up on real Deribit options?

[![tests](https://github.com/mohymabrouk/deep-hedging-sim-to-real/actions/workflows/tests.yml/badge.svg)](https://github.com/mohymabrouk/deep-hedging-sim-to-real/actions/workflows/tests.yml)

**Headline.** On **9,527 real Deribit BTC options** that expired between January 2022 and October 2026, hedged once a
day with the Binance perpetual after fees and funding, **no hedger is significantly better than Black-Scholes delta at
the option's implied vol**. That covers 8 deep hedgers trained in 4 simulated markets and, since test run 4, 3
classical baselines fitted on 2021 data.
- **All three classical baselines are significantly worse than BS delta in CVaR95.** The loss is in % of spot and
  BS delta's CVaR95 is 3.64. The changes, with 95% intervals:
  - Whalley-Wilmott no-trade band: 3.78, -0.14 [-0.25, -0.01];
  - Hull-White minimum-variance delta: 3.82, -0.18 [-0.22, -0.12];
  - smile-implied delta: 3.81, -0.17 [-0.31, -0.05].
- **The Whalley-Wilmott band does save trading**, the advantage reported on Deribit by arXiv 2608.29025. It trades
  14% less than BS delta and saves 0.011% of spot in perp fees. But with one rebalance a day the fees are only 0.082%
  of spot, and the band's tracking error costs more in the tail than it saves.
- **The best deep hedger still only ties BS delta.** That is the residual hedger trained in the rough-volatility world,
  at 3.63 and +0.00 [-0.45, +0.53]. It is not significantly better than any classical baseline in CVaR95. Its loss is
  significantly more dispersed than theirs: its standard deviation is 0.21-0.27 higher.
- **Two descriptive checks** were added with run 4:
  - The "compatibility gap" of arXiv 2608.20842 does not rank this project's deep hedgers by their test results.
  - The delta haircut reported by arXiv 2605.21696 does not show up in them.
  - Both use 4 to 8 hedgers, so they are weak evidence either way.
- **Reproduction.** The 11 hedgers of run 3 were not changed and reproduce it bit for bit.

![Change in CVaR95 loss vs BS delta at the entry IV, classical and deep hedgers, test run 4](results/fig_headline.png)

## Results

Every number comes from `results/` (`tables.md` holds all tables at three decimals).
- **The latest test run is run 4 of 4.** It is a forced rerun that adds the three classical baselines.
- **When and with which config.** It was logged at 2026-11-23 11:27 UTC with config hash `22c21a25ad4237ee`. That hash
  was frozen at 11:27 UTC, after the 2021 validation of the baselines. The test stage refuses to run if the two hashes
  differ.
- **Archives.** Runs 1, 2 and 3 are in `results/run1_official/`, `results/run2_official/` and `results/run3_official/`.

Losses are in **% of S0**, the BTC index at entry, for a short option on 1 BTC. CVaR95 is the average of the worst 5%
of losses. Δ is the reference's loss minus the hedger's loss, so **Δ > 0 means better than BS delta**. The 95%
intervals come from a moving-block bootstrap over expiry dates: blocks of 4 expiries, 2,000 resamples.

### Test options, run 4, 1x fees (9,527 options, 251 expiries)

| Hedger | Mean P&L | Std of loss | CVaR95 loss | Δ CVaR95 vs BS delta [95% CI] | Δ std vs BS delta [95% CI] | Δ mean loss vs BS delta [95% CI] | Turnover | Perp fees |
|---|---|---|---|---|---|---|---|---|
| No hedge | 0.15 | 6.11 | 19.80 | -16.16 [-19.69, -12.50] | -4.69 [-5.51, -3.84] | -0.05 [-0.42, +0.31] | 0.00 | 0.000 |
| **BS delta, entry IV** | 0.20 | 1.42 | **3.64** | benchmark | benchmark | benchmark | 1.63 | 0.082 |
| BS delta, IV moved by DVOL | 0.20 | 1.41 | 3.59 | +0.04 [-0.10, +0.20] | +0.01 [-0.02, +0.04] | +0.00 [-0.01, +0.02] | 1.63 | 0.082 |
| Whalley-Wilmott band | 0.20 | 1.48 | 3.78 | -0.14 [-0.25, -0.01] | -0.06 [-0.08, -0.03] | +0.00 [-0.02, +0.02] | 1.40 | 0.070 |
| Minimum-variance delta (Hull-White) | 0.19 | 1.42 | 3.82 | -0.18 [-0.22, -0.12] | -0.00 [-0.02, +0.02] | -0.01 [-0.03, +0.01] | 1.63 | 0.082 |
| Smile-implied delta (Alexander-Imeraj) | 0.20 | 1.46 | 3.81 | -0.17 [-0.31, -0.05] | -0.04 [-0.07, -0.02] | -0.00 [-0.02, +0.02] | 1.60 | 0.080 |
| Deep hedge, bootstrap world | 0.18 | 2.13 | 5.07 | -1.43 [-2.12, -0.62] | -0.71 [-0.87, -0.55] | -0.02 [-0.15, +0.11] | 1.35 | 0.068 |
| Deep hedge, GARCH-t world | 0.22 | 1.97 | 4.51 | -0.87 [-1.76, +0.04] | -0.55 [-0.73, -0.37] | +0.02 [-0.09, +0.13] | 1.56 | 0.078 |
| Deep hedge, rough-vol world | 0.20 | 1.71 | 3.98 | -0.34 [-1.09, +0.45] | -0.29 [-0.43, -0.14] | -0.00 [-0.08, +0.08] | 1.54 | 0.078 |
| Deep hedge, diffusion world | 0.18 | 2.18 | 5.54 | -1.90 [-2.63, -1.01] | -0.76 [-0.91, -0.58] | -0.01 [-0.13, +0.11] | 1.29 | 0.065 |
| Residual deep hedge, bootstrap world | 0.25 | 1.68 | 3.97 | -0.33 [-0.82, +0.18] | -0.26 [-0.37, -0.15] | +0.05 [-0.06, +0.16] | 1.53 | 0.077 |
| Residual deep hedge, GARCH-t world | 0.19 | 1.74 | 3.69 | -0.05 [-0.51, +0.48] | -0.32 [-0.43, -0.21] | -0.01 [-0.07, +0.06] | 1.55 | 0.078 |
| **Residual deep hedge, rough-vol world** | 0.20 | 1.69 | **3.63** | +0.00 [-0.45, +0.53] | -0.27 [-0.37, -0.16] | -0.00 [-0.07, +0.06] | 1.52 | 0.076 |
| Residual deep hedge, diffusion world | 0.21 | 1.65 | 3.93 | -0.29 [-0.72, +0.20] | -0.23 [-0.33, -0.13] | +0.02 [-0.06, +0.09] | 1.42 | 0.072 |

### The three classical baselines (run 4)

Each one starts from BS delta at the entry IV and has at most three parameters, all fitted on 2021 data only
(`PLAN.md`, change 17). "Best deep hedger" means the residual rough-vol hedger, the deep hedger with the lowest test
CVaR95. It was picked after seeing the test results, so its comparison is post hoc
(`results/classical_vs_best_deep.csv`).

| Hedger | CVaR95 loss | Std of loss | Turnover (vs BS delta) | Perp fees (Δ vs BS delta) | Δ CVaR95 vs BS delta [95% CI] | Δ std vs BS delta [95% CI] | Δ mean loss vs BS delta [95% CI] | Δ CVaR95 vs best deep [95% CI] | Δ std vs best deep [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Whalley-Wilmott band | 3.78 | 1.48 | 1.40 (0.86x) | 0.070 (-0.011) | -0.14 [-0.25, -0.01] | -0.06 [-0.08, -0.03] | +0.00 [-0.02, +0.02] | -0.14 [-0.60, +0.28] | +0.21 [+0.11, +0.31] |
| Minimum-variance delta (Hull-White) | 3.82 | 1.42 | 1.63 (1.01x) | 0.082 (+0.000) | -0.18 [-0.22, -0.12] | -0.00 [-0.02, +0.02] | -0.01 [-0.03, +0.01] | -0.18 [-0.72, +0.29] | +0.27 [+0.15, +0.38] |
| Smile-implied delta (Alexander-Imeraj) | 3.81 | 1.46 | 1.60 (0.98x) | 0.080 (-0.001) | -0.17 [-0.31, -0.05] | -0.04 [-0.07, -0.02] | -0.00 [-0.02, +0.02] | -0.17 [-0.63, +0.22] | +0.23 [+0.13, +0.31] |

- **Whalley-Wilmott band.** The band is around BS delta at the entry IV. Its risk aversion, 1000 per unit of S0, was
  picked from 8 values on the 2021 options. It cuts turnover by 14% and perp fees from 0.082 to 0.070% of S0, and its
  mean loss is unchanged (+0.00 [-0.02, +0.02]). But the tail is wider: its standard deviation of loss is 0.06 higher
  and its CVaR95 0.14 higher, both significant.
  - It is worse than BS delta even with zero fees (CVaR95 3.63 against 3.48) and with double fees (3.93 against
    3.80).
  - The reason: daily rebalancing already makes trading cheap, so cost savings are small, while lagging the delta
    adds hedging error.
- **Minimum-variance delta.** The Hull & White coefficients come from their own regression on 5,293 daily changes of
  2021 Deribit mark prices: a = -0.185 (s.e. 0.043), b = 0.407 (0.176), c = -0.185 (0.170).
  - Their in-sample variance reduction is only 1.1% (`results/validation/mv_delta_fit.json`).
  - The resulting adjustment at entry is small: -0.018 BTC on average for calls and -0.005 for puts
    (`results/haircut_check.csv`).
  - It leaves the dispersion unchanged (Δ std -0.00 [-0.02, +0.02]) and raises CVaR95 by 0.18.
- **Smile-implied delta.** This is the model-free delta of scale-invariant models, the variant Alexander & Imeraj
  report gains with.
  - Its smile slope is estimated at entry from the same-expiry trades of the 07:00-08:00 UTC window.
  - **The slope passed its pre-declared reliability check.** On the 2021 options it covered 93.5% of options, and the
    estimates from the two half-hour windows correlate at 0.94 (784 options;
    `results/validation/smile_slope_reliability.json`). On test it covers 9,144 of 9,527 options; the rest fall back
    to BS delta.
  - It is significantly worse than BS delta: CVaR95 +0.17 higher and std of loss +0.04 higher.
  - It does not help on puts either, where Alexander & Imeraj found their gains (for out-of-the-money puts): on all
    puts, CVaR95 3.59 against 3.42, Δ -0.17 [-0.35, -0.04].
- **Validation had already flagged two of them.** On the 2021 options the minimum-variance and smile-implied deltas
  were significantly worse than BS delta: Δ CVaR95 -0.28 [-0.35, -0.02] and -0.37 [-0.56, -0.12]. The Whalley-Wilmott
  band was slightly better but not significantly: +0.07 [-0.05, +0.29]. All three went to the test anyway, as
  declared in `PLAN.md` before the validation run.
- **Against the best deep hedger,** none is significantly different in CVaR95 (-0.14 to -0.18, every interval
  contains 0). All three have a significantly narrower loss distribution, with a standard deviation 0.21 to 0.27
  lower.

**Reproduction check** (`results/reproduction_vs_run3.csv`, also in `tables.md`).
- The 11 hedgers of run 3 use the same policy files and code paths, and ran with the same thread count (2).
- They reproduce run 3 **bit for bit**: the largest absolute difference is 0 in every summary statistic (bootstrap
  bounds included) and in every option's P&L.
- `tests/test_reproduction.py` checks this from the committed result files.

### CVaR95 loss by period, option type and fee level (run 4)

Fee levels apply to the same positions; the policies are not retrained.

| Hedger | 2022-23 (3,049) | 2024 on (6,478) | Calls (4,840) | Puts (4,687) | Fees x0 | Fees x1 | Fees x2 |
|---|---|---|---|---|---|---|---|
| No hedge | 20.69 | 19.35 | 22.66 | 16.50 | 19.77 | 19.80 | 19.83 |
| BS delta, entry IV | 3.75 | 3.57 | 3.83 | 3.42 | 3.48 | 3.64 | 3.80 |
| BS delta, IV moved by DVOL | 3.68 | 3.52 | 3.82 | 3.35 | 3.44 | 3.59 | 3.75 |
| Whalley-Wilmott band | 3.87 | 3.71 | 3.98 | 3.53 | 3.63 | 3.78 | 3.93 |
| Minimum-variance delta (Hull-White) | 3.98 | 3.73 | 4.03 | 3.55 | 3.65 | 3.82 | 3.98 |
| Smile-implied delta (Alexander-Imeraj) | 3.93 | 3.74 | 4.00 | 3.59 | 3.65 | 3.81 | 3.97 |
| Deep hedge, bootstrap world | 5.03 | 5.09 | 5.37 | 4.75 | 4.93 | 5.07 | 5.21 |
| Deep hedge, GARCH-t world | 4.59 | 4.46 | 5.07 | 3.71 | 4.40 | 4.51 | 4.62 |
| Deep hedge, rough-vol world | 3.94 | 3.99 | 4.33 | 3.58 | 3.86 | 3.98 | 4.11 |
| Deep hedge, diffusion world | 5.56 | 5.53 | 5.88 | 5.18 | 5.40 | 5.54 | 5.68 |
| Residual deep hedge, bootstrap world | 4.04 | 3.93 | 3.56 | 4.25 | 3.83 | 3.97 | 4.11 |
| Residual deep hedge, GARCH-t world | 3.80 | 3.63 | 3.89 | 3.47 | 3.57 | 3.69 | 3.82 |
| Residual deep hedge, rough-vol world | 3.72 | 3.58 | 3.91 | 3.31 | 3.51 | 3.63 | 3.76 |
| Residual deep hedge, diffusion world | 3.91 | 3.93 | 3.93 | 3.91 | 3.79 | 3.93 | 4.07 |

- **No hedger is significantly better than BS delta in any split.** The paired intervals for each split are in
  `tables.md`.
- **Classical baselines.** Each has a higher CVaR95 than BS delta in every split and at every fee level. The gap is
  significant for the Hull-White delta in all four splits, and for the Whalley-Wilmott band and the smile-implied
  delta from 2024 on and on calls. The smile-implied delta is also significantly worse on puts: -0.17
  [-0.35, -0.04].
- **The rough-vol residual hedger is the only deep or classical hedger with a lower CVaR95 than BS delta on puts:**
  3.31 against 3.42, Δ +0.11 [-0.40, +0.66]. That is not significant, and it is worse on calls. (BS-DVOL is also
  lower on puts, 3.35, not significantly.)
- **Original deep hedgers.** The bootstrap and diffusion policies are significantly worse than BS delta in all four
  splits, as in runs 2 and 3.

### Compatibility check (arXiv 2608.20842), no new test run

**The claim.** arXiv 2608.20842 says a generator's statistical realism does not predict hedging performance, and that
a "compatibility gap" does. The gap is the difference in hedging risk of the same strategies between the synthetic
and the real environment.

**How it was measured here.** The definitions were declared in `PLAN.md` (change 17) before computing anything, and
the check uses frozen policies only.
- **Real environment:** the 1,388 options of 2021.
- **Synthetic environment:** each world's 20,000 training paths, hedging the same contracts.
- **Loss:** payoff minus hedge gains plus 0.05% perp costs (the paper's loss), in % of S0.
- **Gap:** CVaR95 on synthetic episodes minus CVaR95 on real ones, for three fixed strategies per world:
  - BS delta at the trailing 20-day realised vol (the worlds have no IV);
  - the world's original policy;
  - the world's residual policy.

| World | Gap, BS delta at trailing vol | Gap, original policy | Gap, residual policy |
|---|---|---|---|
| bootstrap | +0.46 | -0.31 | -0.31 |
| GARCH-t | +4.26 | +3.42 | +3.63 |
| rough vol | +2.26 | +1.87 | +1.86 |
| diffusion | +1.40 | -0.82 | +0.33 |

The CVaR95 on the real 2021 options is between 16.54 and 17.02 for every strategy (`results/compatibility_gaps.csv`).

**Does it rank the hedgers?** The table gives the Spearman correlation between each predictor and the run-3 test
CVaR95. A positive value means the predictor ranks the hedgers correctly. All predictors are in
`results/compatibility_ranks.csv`.

| Predictor (smaller = predicted better) | Original hedgers (n = 4) | Residual hedgers (n = 4) | All 8 |
|---|---|---|---|
| Compatibility: \|gap CVaR95\| of the hedger's own policy (primary) | -0.60 (p = 0.83) | -0.80 (p = 0.96) | -0.24 (p = 0.73) |
| Compatibility: \|gap CVaR95\| of BS delta at trailing vol | -0.60 (p = 0.83) | -0.80 (p = 0.96) | -0.34 (p = 0.82) |
| Realism: W1 on standardised returns (primary) | +0.80 (p = 0.17) | +0.40 (p = 0.38) | +0.29 (p = 0.26) |
| Realism: mean rank of 7 realism distances | -0.60 (p = 0.83) | -0.80 (p = 0.96) | -0.34 (p = 0.82) |
| Reference: the hedger's own 2021 validation CVaR95 | +0.60 (p = 0.21) | +0.60 (p = 0.21) | +0.62 (p = 0.057) |

- **The compatibility gap ranks the hedgers in the wrong direction in every group.** The GARCH-t world has by far
  the largest gap, yet its residual hedger is the second best of the eight.
- **Standardised W1 points the right way, but not significantly.** It picks the rough-vol world as best, though only
  by a hair over bootstrap (0.0457 against 0.0460 in `results/realism.csv`). The composite realism rank points the
  wrong way.
- **The best of the three is the plain 2021 validation score,** with Spearman +0.62 over 8 hedgers (one-sided
  p = 0.057). That is the number the project already selects on.
- **Small n, said plainly.** With n = 4, only a perfect ranking reaches p = 1/24 = 0.042, and n = 8 is not much
  better. This check can neither confirm nor refute the paper.
- **Two differences from the paper.** The paper's gap is a supremum over a whole strategy class, while here it is
  three fixed strategies. And this project's worlds have no implied vol, so the gaps mostly reflect how heavy each
  world's tails are.
- **Secondary target.** The own-policy gap does rank the residual hedgers correctly by their test standard deviation
  of loss (+0.80, `results/compatibility_ranks.csv`).

### Delta haircut check (arXiv 2605.21696), descriptive

**The claim and how it was tested.** arXiv 2605.21696 (SPX options) reports that deep hedgers learn a delta "haircut"
below BS delta, explained by spot-IV co-movement. The Hull-White adjustment is the textbook form of that co-movement
effect. So on the run-4 test entries, each deep hedger's correction at entry (its hedge minus BS delta at the entry
IV) is compared with the Hull-White adjustment (minimum-variance delta minus BS delta). Results are in
`results/haircut_check.csv`.

| Hedger | Mean correction, calls | Mean correction, puts | Haircut share, calls | Haircut share, puts | Pearson with HW adjustment | Same sign as HW | OLS slope (R²) |
|---|---|---|---|---|---|---|---|
| Hull-White adjustment itself | -0.018 | -0.005 | 88% | 47% | +1.00 | 100% | +1.00 (1.00) |
| Residual, bootstrap world | +0.052 | +0.122 | 16% | 100% | +0.06 | 36% | +0.26 (0.00) |
| Residual, GARCH-t world | +0.070 | -0.021 | 14% | 40% | -0.91 | 8% | -5.13 (0.82) |
| Residual, rough-vol world | +0.051 | +0.007 | 21% | 52% | -0.94 | 7% | -4.37 (0.87) |
| Residual, diffusion world | -0.050 | +0.083 | 85% | 97% | +0.17 | 59% | +0.96 (0.03) |

A haircut is |hedge| < |BS delta|: a negative correction on calls, a positive one on puts.

- **The 2021 Hull-White fit does imply a small haircut on calls,** -0.018 BTC on average and on 88% of calls. On puts
  it is close to zero (-0.005 on average, a slightly larger hedge).
- **The residual hedgers mostly do the opposite.**
  - The GARCH-t and rough-vol hedgers add delta to calls (+0.070 and +0.051). Their corrections are strongly
    anti-correlated with the Hull-White adjustment: Pearson -0.91 and -0.94, same sign on only 8% and 7% of options.
  - The bootstrap hedger haircuts puts but over-hedges calls, unrelated to the adjustment.
  - Only the diffusion hedger haircuts broadly (85% of calls, 97% of puts). Its correction is almost unrelated to the
    Hull-White adjustment (R² 0.03).
- **The original deep hedgers, a supplement,** are all negatively correlated with the adjustment (-0.42 to -0.90;
  `tables.md`).
- **What this means.**
  - P1's deep hedgers do not show the pattern of arXiv 2605.21696.
  - That is not a test of the paper: P1's worlds have no implied vol, so no spot-IV channel could be learned.
  - The adjustment itself did not help out of sample here: the minimum-variance delta is 0.18 worse in CVaR95.

### Simulator realism: worlds fitted on 2018-2021 vs real daily returns from 2022

| Series | Annual vol | Excess kurtosis | Mean ACF of \|r\|, lags 1-10 | 1% quantile | 99% quantile | W1, standardised |
|---|---|---|---|---|---|---|
| Real BTC, 2022 on | 0.49 | 3.2 | 0.107 | -0.082 | 0.072 | n/a |
| Bootstrap | 0.80 | 7.1 | 0.103 | -0.130 | 0.113 | 0.046 |
| GARCH-t | 0.81 | 22.1 | 0.160 | -0.122 | 0.120 | 0.105 |
| Rough vol | 0.82 | 6.7 | 0.162 | -0.126 | 0.116 | 0.046 |
| Diffusion | 0.80 | 36.6 | 0.224 | -0.124 | 0.113 | 0.172 |

- **The regime shift is large.** All four worlds learn from 2018-2021, so they are 60-70% more volatile than BTC
  was in 2022-2026 (annual vol 0.80-0.82 against 0.49).
- **The diffusion model is mixed.** Its raw return distribution is the closest to the real one (W1 0.0057, see
  `results/realism.csv`), but it has the heaviest tails and the most volatility clustering.
- **These statistics do not reliably predict hedging quality** (see the compatibility check above). Bootstrap and
  rough vol are equally close in standardised shape (0.046), yet the original rough-vol policy hedges far better than
  the bootstrap one (3.98 vs 5.07).

### Run history

| Run | Logged (UTC) | Config hash | What changed | Archive |
|---|---|---|---|---|
| 1, official | 2026-11-19 09:41 | `d2f906a2dba41748` | the seven original hedgers | `results/run1_official/` |
| 2, forced rerun | 2026-11-21 20:24 | `152ac45c09422193` | GARCH shock cap (a bug fix); GARCH CVaR95 20.15 to 4.51 | `results/run2_official/` |
| 3, forced rerun | 2026-11-22 14:15 | `d35f2054d7c27041` | four residual deep hedgers | `results/run3_official/` |
| 4, forced rerun | 2026-11-23 11:27 | `22c21a25ad4237ee` | three classical baselines | `results/` (latest) |

- **Run 1 to run 2.** In run 1's GARCH world, one Student-t draw of about 150 sd produced a +12 daily log return.
  - The drift correction is log E[exp(r)], so that single value set it to 0.115 per day, and every path fell about
    97% in 35 days.
  - The defect shows in the simulator's own output (`results/run1_official/simulators/final_garch.json`).
  - The fix truncates standardised shocks at the largest in-sample residual. The GARCH validation was redone
    (alpha = 0.9 again) and its policy retrained.
  - The other six hedgers matched run 1 to within 4e-9 (% of S0). Read the GARCH row as less clean than the others.
- **Runs 3 and 4 add hedgers and do not touch the old ones.** In run 3 the seven original hedgers reproduced run 2
  to within 1.7e-10 in any summary statistic (float32 rounding under a different thread count;
  `results/run3_official/reproduction_vs_run2.csv`). In run 4 all 11 reproduce run 3 exactly.

### More figures

- `results/fig_paired_improvement.png`: paired change in mean loss and CVaR95 loss vs BS delta, every hedger except no
  hedge.
- `results/fig_pnl_distributions.png`: loss distributions and right-tail quantiles, one row per family (original,
  residual, classical).
- `results/fig_hedge_ratio_vs_moneyness.png`: hedge ratio at entry vs moneyness, deep policies vs BS delta, one row
  per deep-hedger family.

## Method

**Question.** A deep-hedging policy (Buehler, Gonon, Teichmann & Wood, *Deep hedging*, Quantitative Finance
19(8), 2019) is only as good as the market it was trained in. Four training markets are compared, and every
policy is scored on the same thing: **real expired Deribit BTC options, hedged with the Binance USDT-M perpetual,
after fees and funding**.

**Training markets** (all fitted on Binance spot BTCUSDT daily 08:00-UTC log returns from 2018-01-02; every
generator is shifted so that E[exp(r)] = 1, a driftless price, so the policies learn to hedge rather than to bet on
the 2018-2021 trend):

| World | Model | Fitted parameters (final stage, data <= 2021-12-31) |
|---|---|---|
| (a) bootstrap | stationary block bootstrap (Politis & Romano), mean block 20 days | `results/simulators/final_bootstrap.json` |
| (b) GARCH-t | GARCH(1,1), zero mean, standardised Student-t, MLE with `arch`; each path starts from an in-sample filtered variance; conditional variance capped at the in-sample maximum (the fit is integrated, alpha + beta = 1, and uncapped paths exploded); standardised shocks truncated at the largest in-sample standardised residual (run 2 onwards) | `results/simulators/final_garch.json` |
| (c) rough vol | RFSV (Gatheral, Jaisson & Rosenbaum 2018, the P-model behind rough Bergomi): log-vol is a fractional OU process; H and vol-of-vol from the regression of log E[(log sigma_{t+D} - log sigma_t)^2] on log D, D = 1..30 days, with daily realised variance from 5-minute returns; leverage rho from corr(r_t/sqrt(RV_t), d log RV_{t+1}) | `results/simulators/final_rough.json` |
| (d) diffusion | DDPM, 1D U-Net with 1.09M parameters, cosine schedule (1,000 levels), DDIM sampling (50 steps, predicted clean window clipped at +-15 sd), trained on 64-day windows of standardised daily log returns taken at all 24 hourly anchors | `results/simulators/final_diffusion.json` |

**Original policy (runs 1-4).** One MLP (3 x 64, SiLU) shared by every contract.
- **Inputs at day t**, all computed from data <= t: log(S_t/K), time to expiry, the current hedge, 10-day realised
  vol (annualised, from daily 08:00 returns), a call/put flag, and z = log(S/K)/(rv sqrt(tau)) clipped to +-4 (a
  function of the other inputs).
- **Output.** A hedge ratio in [0, 1] for calls, [-1, 0] for puts (sigmoid).
- **Objective.** CVaR_alpha of the loss of a short option hedged with proportional costs (0.05% per trade),
  Rockafellar-Uryasev form.
- **Training contracts.** Maturity uniform on 7..35 days, log(K/S0) uniform on [-0.2, 0.2], half calls.
- **Budget.** 20,000 simulated 64-day paths per world, Adam, batch 4,096, 1,500 steps (about 4 minutes on a laptop
  CPU).
- **No IV input.** These policies do not see implied volatility (the simulators do not produce one), whereas the
  Black-Scholes benchmarks do.

**Residual policy (runs 3-4).** One more hedger per world, `dh_resid_<world>`:
q_t = BS delta(S_t, K, tau_t, sigma_entry) + c * tanh(MLP(x_t)), clipped to [0, 1] for calls and [-1, 0] for puts.
- **Network.** The MLP has the same shape (3 x 64, SiLU). It has 8 inputs: the six above, sigma_entry, and the base
  BS delta. Its last layer starts at zero, so the untrained policy *is* "BS delta, entry IV", and training can move
  the hedge by at most c BTC per option.
- **sigma_entry on real options.** The IV of the entry trade, known before the first hedge at 08:00.
- **sigma_entry on simulated paths, which have no IV.** Two stand-ins were declared before any validation run
  (`PLAN.md`, change 15):
  - `fwd_noisy`: the path's realised vol over the option's life times lognormal noise (eta = 0.3, not tuned), which
    imitates an IV as a noisy forecast;
  - `trailing`: the realised vol of the 20 days before entry.
- **Chosen on 2021 validation options:** the stand-in, c in {0.1, 0.3} and alpha in {0.5, 0.9, 0.99}.
- **Budget.** Paths, budget, objective and fees are the same as for the original policies. Every cell of a world uses
  the same seed.

**Classical baselines (run 4).** All three use sigma = entry IV, like BS delta. Each has at most three parameters,
fitted on 2021 data only and declared in `PLAN.md` (change 17) before the validation run. The code is in
`src/dh/hedging.py` (hedgers) and `src/dh/classical.py` (estimation).
- **Whalley-Wilmott band** (Whalley & Wilmott 1997).
  - Half-width H = (3 kappa S Gamma^2 / (2 gamma))^(1/3) around BS delta, with kappa = 0.05% (the perp fee).
  - The hedger holds its position while inside the band, otherwise it trades to the nearest edge.
  - One parameter, the risk aversion gamma. It was picked from {1, 3, 10, 30, 100, 300, 1000, 3000} by the lowest
    2021 validation CVaR95; gamma = 1000 was chosen.
- **Minimum-variance delta** (Hull & White 2017).
  - delta_BS + vega/(S sqrt(T)) (a + b delta_C + c delta_C^2), with delta_C the call delta, so one fit serves calls
    and puts (put-call parity).
  - a, b, c come from Hull & White's own regression, (Δf - delta_BS ΔS)/S on phi(d1) ΔS/S [1, delta_C, delta_C^2]
    (OLS, no intercept). It uses daily changes in Deribit mark prices of instruments traded in 07:00-08:00 UTC on two
    consecutive days of 2021, with filters of 2-35 days to expiry, |log(K/S)| < 0.2 and 0.05 <= delta_C <= 0.95.
  - **Deviation from Hull & White:** they used end-of-day S&P 500 mid quotes and rolling 36-month windows. Here there
    is one 2021 fit on mark prices observed once a day.
- **Smile-implied delta** (Alexander & Imeraj 2023, the model-free delta of scale-invariant models).
  - delta_BS - phi(d1) sqrt(tau) theta, where theta = d sigma / d log(K/S) at the option's entry moneyness.
  - theta comes from an OLS quadratic of median trade IV on log-moneyness, fitted to the same-expiry strikes traded
    in 07:00-08:00 UTC on the entry day. It needs at least 4 strikes and interpolation only, otherwise theta = 0.
  - The entry IV and theta stay fixed over the option's life.
  - A pre-declared reliability rule kept it (coverage >= 50% and split-half correlation >= 0.5 on 2021).
  - The paper's full text was not reachable here (publisher and repository pages refused automated access), so the
    variant was identified from its abstract and secondary sources.

**Benchmarks.** No hedge; Black-Scholes delta (r = 0) with the option's traded IV at entry; Black-Scholes delta with
that IV moved in parallel by the change in Deribit's DVOL index since entry (sigma_t = IV_0 + DVOL_t - DVOL_0).

**Rebalancing: daily at 08:00 UTC, for every hedger.** Deribit options settle at 08:00 UTC, 08:00 is a Binance
funding time, and the simulators are fitted on daily 08:00-to-08:00 returns. Hedging every 8 hours would triple the
number of trades at a 0.05% taker fee.

**Leak-free protocol.**
1. **Validation.** Fit the four simulators on data <= 2020-12-31 and train policies with alpha in {0.5, 0.9, 0.99}.
   Score them on the 2021 **validation** options (entry in 2021 and expiry <= 2021-12-31). For each world, keep the
   alpha with the lowest validation CVaR95 (`results/selected_hyperparameters.json`). No other choice used the data.
2. **Final stage.** Refit every simulator on data <= 2021-12-31 and retrain the four policies with the chosen alpha.
3. **Test.** Score the test period (entries from 2022-01-01) once per official run, all hedgers in one run, logged
   with its timestamp and config hash in `results/test_log.jsonl`. The script refuses a second run unless `--force`,
   which is logged too. There have been four runs (see "Run history").
4. **Causality is tested.**
   - `tests/test_leak.py` scrambles every price and DVOL value after day t and checks that the actions up to day t do
     not change (deep, BS-IV and BS-DVOL hedgers).
   - `tests/test_residual.py` does the same for the residual policy. It also checks that the policy's IV input is the
     IV of the entry trade, constant over the option's life, and that the residual policy equals BS delta bit for bit
     at initialisation.
   - `tests/test_classical.py` does the same for the three classical hedgers. It also checks that each equals BS
     delta bit for bit with zero band, a = b = c = 0 or a flat smile, and that the smile slope reads only the entry
     day's window.
5. **Residual hedgers (run 3)** went through the same steps:
   1. Validation on the 2021 options with simulators fitted on data <= 2020 (the cached paths of steps 1-2), in two
      pre-declared stages: every stand-in x c at alpha = 0.9, then the winner at alpha = 0.5 and 0.99.
   2. Freeze a new config hash.
   3. Retrain on the final-stage paths (data <= 2021).
   4. One forced test run that scores all 11 hedgers.
6. **Classical baselines (run 4)** also went through validation on 2021 only: the Hull-White regression, the smile
   reliability check and the band's risk-aversion grid. Then a new config hash was frozen, run 3 was archived, and
   one forced test run scored all 14 hedgers (`scripts/run4_classical_chain.sh`).

**Frozen configuration** (`results/frozen_config.json`, config hash `22c21a25ad4237ee`, written at 2026-11-23 11:27 UTC
before the test run).
- **Earlier frozen configs:** run 3's (`d35f2054d7c27041`) is in `results/run3_official/`, run 2's
  (`152ac45c09422193`) in `results/run2_official/` and run 1's (`d2f906a2dba41748`) in `results/run1_official/`.
- **How choices were made.** Every tuned choice below was picked as the lowest CVaR95 loss on the 1,388 validation
  options of 2021, with simulators fitted on data <= 2020. The one exception is the Hull-White coefficients, which
  are a regression fit and involve no selection.

*Original deep hedgers*: the CVaR level alpha of each world.

| World | CVaR95 loss on validation, % of S0: alpha = 0.5 / 0.9 / 0.99 | Chosen alpha |
|---|---|---|
| bootstrap | 8.41 / 7.64 / **7.31** | 0.99 |
| GARCH-t (run 2 on; run 1 was 8.41 / **6.08** / 16.59) | 6.10 / **5.85** / 7.90 | 0.9 |
| rough vol | 6.50 / **6.30** / 6.44 | 0.9 |
| diffusion | 12.32 / 7.95 / **7.20** | 0.99 |
| *benchmark: BS delta at entry IV* | *6.63* | |

*Residual deep hedgers*: stage 1 compares stand-in x c at alpha = 0.9; stage 2 runs the stage-1 winner at alpha = 0.5
and 0.99 (`results/validation/dh_resid_<world>.csv`).

| World | fwd_noisy, c = 0.1 | fwd_noisy, c = 0.3 | trailing, c = 0.1 | trailing, c = 0.3 | Stage-1 winner at alpha = 0.5 / 0.99 | Chosen (validation CVaR95) |
|---|---|---|---|---|---|---|
| bootstrap | 6.40 | **6.16** | 7.41 | 7.07 | 7.07 / 6.61 | fwd_noisy, c = 0.3, alpha = 0.9 (6.16) |
| GARCH-t | 6.16 | 6.28 | **5.55** | 6.00 | 5.93 / 5.83 | trailing, c = 0.1, alpha = 0.9 (5.55) |
| rough vol | 6.26 | 6.26 | **5.98** | 6.08 | 6.41 / 6.29 | trailing, c = 0.1, alpha = 0.9 (5.98) |
| diffusion | **7.40** | 7.43 | 7.58 | 8.10 | 7.03 / **6.72** | fwd_noisy, c = 0.1, alpha = 0.99 (6.72) |
| *benchmark: BS delta at entry IV* | | | | | | *6.63* |

*Classical baselines* (`results/validation/ww_band.csv`, `classical.csv`, `mv_delta_fit.json`,
`smile_slope_reliability.json`).
- **Whalley-Wilmott risk aversion,** validation CVaR95 for gamma = 1 / 3 / 10 / 30 / 100 / 300 / 1000 / 3000:
  8.21 / 7.30 / 6.92 / 6.73 / 6.62 / 6.58 / **6.56** / 6.58, so gamma = 1000 was chosen. Turnover rises from 0.96 to
  1.56 as the band narrows.
- **Hull-White:** a = -0.185, b = 0.407, c = -0.185 (n = 5,293, in-sample gain 1.1%).
- **Smile slope:** coverage 93.5%, split-half Pearson 0.94, so it was kept.
- **Validation CVaR95 at these settings:** 6.56 (band), 6.91 (minimum variance) and 7.00 (smile), against BS delta's
  6.63.

**Validation vs test.**
- On validation, two original policies (GARCH-t and rough vol), three residual policies (bootstrap, GARCH-t, rough
  vol) and the Whalley-Wilmott band had a lower CVaR95 than BS delta.
- On the 2022-2026 test options none of them is better than BS delta beyond noise. The closest, the rough-vol
  residual hedger, ties (+0.00 [-0.45, +0.53]), and the band is significantly worse.

Everything else (network, features, training budget, fees) was fixed in `PLAN.md` before any option was scored.

## Data

All free, no API keys. Raw responses are cached in `~/.cache/quant-research/p1/` (a rerun downloads nothing).

| Data | Source | Use |
|---|---|---|
| Option trades 07:00-08:00 UTC, every day from 2021-01-01 | `history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time` (BTC, kind=option), paged forward in time, <= 4 requests/s, retries with backoff | entry price (BTC), IV, index price; the classical baselines' Hull-White regression (2021 mark prices) and smile slope at entry |
| Delivery prices | `www.deribit.com/api/v2/public/get_delivery_prices?index_name=btc_usd` | payoff |
| DVOL | `www.deribit.com/api/v2/public/get_volatility_index_data` (hourly; history starts 2021-03-24) | BS-DVOL benchmark |
| BTCUSDT perp, 1h klines + funding | `data.binance.vision` futures/um monthly files, `.CHECKSUM` verified | hedge prices at 08:00, funding |
| BTCUSDT spot, 1h (2018-01 on) and 5-minute (2018-2021) klines | `data.binance.vision` spot monthly files, checksum verified | simulator training, realised variance |

**Option selection.**
- **Entry.** For each instrument, the last trade between 07:00 and 08:00 UTC gives the entry: premium = trade price
  in BTC x the index price on that trade; entry IV = the trade's IV.
- **Eligible.** 7 to 35 days to expiry, |log(K/S0)| < 0.2, price >= 0.0005 BTC, IV in [5%, 300%].
- **One entry per instrument**, on its first eligible day, so no option counts twice.
- **Payoff.** Every payoff uses Deribit's delivery price (no fallback was needed).
- **Lists.** The selected options are in `results/test_options.csv` and `results/validation_options.csv`.

**Inverse vs linear.** Deribit options are inverse: premium and payoff are paid in BTC. A call pays
(S_T - K)+/S_T BTC, worth exactly (S_T - K)+ USD at the delivery price, so the USD payoff is linear. We convert the
premium to USD at the entry index price and hedge with a linear USDT perp.
- **Not modelled:** BTC collateral (which adds delta), margin, and the basis between the Deribit index (payoff,
  entry) and the Binance perp (hedge).
- **Timing gap.** The trade happens up to an hour before 08:00, while the hedge starts at 08:00; that gap is the same
  for every hedger.

**Fees and funding** (charged in every test result unless stated):
- Deribit option taker fee: 0.03% of the underlying, capped at 12.5% of the option price, paid once at entry. The
  delivery fee is not charged (identical for all hedgers).
- Binance USDT-M taker fee: 0.05% of traded notional, on every rebalance including the first trade and the unwind
  at expiry.
- Perp funding: a position held over (t, t+1] pays position x rate x perp price at each of the three funding times
  (16:00, 00:00, 08:00 UTC).
- Fee sensitivity: both fees x0, x1, x2 at evaluation (positions unchanged, funding unchanged).

**P&L.** Short one option on 1 BTC: P&L = premium - payoff + sum_t q_t (P_{t+1} - P_t) - perp fees - option fee -
funding, divided by S0 (index at entry) and shown in **% of S0**. Loss = -P&L. The same function computes the
training loss and the real backtest (`src/dh/accounting.py`).

## What is new

Deep hedging has mostly been evaluated inside simulated markets. Related work that uses generative models as the
training market:
- Tanaka et al., *CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation*, arXiv:2503.04164
  (2025). The abstract reports that training on CoFinDiff-generated data improves performance in a deep-hedging task;
  it does not describe a test on real traded options with fees.
- Aghapour, Bayraktar & Yuan, *Dynamic data generation and dynamic portfolio selection: an application of a
  score-based diffusion model*, arXiv:2507.09916 (2025). A score-based diffusion model generates training scenarios
  for dynamic mean-variance portfolio selection (not option hedging), tested on synthetic and real market data.
- Tiwari, *Generative Diffusion Model for Risk-Neutral Derivative Pricing*, arXiv:2603.20582 (2026). A drift shift in
  the reverse diffusion gives martingale (risk-neutral) paths for pricing; the abstract reports tests against a GBM
  benchmark and does not mention hedging.

What this project adds:
- The same deep-hedging policy is trained in four markets (bootstrap, GARCH-t, rough vol, diffusion).
- **Each one is scored on about 9,500 real expired BTC options with real fees, funding and settlement prices.**
- The benchmarks are Black-Scholes deltas that use the market's implied vol, with block-bootstrap confidence intervals
  grouped by expiry date.
- Run 3 adds a residual variant that starts from the IV-based delta. Learning a correction on top of a classical hedge
  is a standard design and is not claimed as new here.
- Run 4 adds three classical BS-delta variants from the literature, fitted on 2021 data, to check that "BS delta wins"
  is not just a weak-benchmark artefact.

It is a small, honest comparison, not a new hedging method.

## Related work (2026)

- **Kumar, *Deep Hedging Under Realistic Market Frictions: A Regime-Conditional Empirical Study of Dynamic Option
  Hedging on Bitcoin Options*, [arXiv:2608.29025](https://arxiv.org/abs/2608.29025).** This is the closest
  competitor, and it reaches the same conclusion: no deep hedger beats a classical benchmark on real Deribit BTC
  options.
  - **Their setup.** Data from 2020-2024, hourly rebalancing over episodes of up to 24 hours, 5 bp round-trip costs,
    and 11,546 test episodes from September 2023 to December 2024.
  - **Their result for the band.** Its risk aversion was set "by inspection". It had the best mean P&L and CVaR95 of
    all strategies; the P&L gain over BS delta is not significant, and no test is reported for CVaR. It made about 8x
    fewer trades (2.47 against 20.40 per episode) and cut costs significantly.
  - **What P1 adds.** Sim-to-real training in four simulated markets; a test sample to October 2026; options held 7-35
    days to expiry; daily perp hedging with funding; residual hedgers; and their winning band, with its risk aversion
    chosen on 2021 validation and tested on 2022-2026.
  - **The difference in P1's result.** With daily rebalancing the band still trades 14% less, but its CVaR95 is
    significantly worse than BS delta's. With about 20 trades per episode, costs dominate; with one trade a day,
    tracking error does.
- **Hashimoto, Hirano, Ozaki & Imajo, *Rethinking Synthetic Scenario Realism: Compatibility, Not Fidelity, Drives
  Hedging Performance*, [arXiv:2608.20842](https://arxiv.org/abs/2608.20842).**
  - **Their claim.** A generator's statistical realism does not predict hedging performance, while a "compatibility
    gap" (the hedging-risk difference between synthetic and real environments) does.
  - **P1's check.** P1's four worlds and eight frozen hedgers, scored on the real test, give a small natural
    experiment (see "Compatibility check").
  - **Result.** Realism did not predict reliably, which agrees with them. A fixed-strategy version of the gap did not
    predict either. The 2021 validation score did best.
- **Zernikov, *What Does Deep Hedging Actually Learn? Delta Corrections, Regime Fragility, and Symbolic
  Distillation*, [arXiv:2605.21696](https://arxiv.org/abs/2605.21696).**
  - **Their result.** On S&P 500 options, TD3 agents learn a delta haircut linked to spot-IV co-movement, and it is
    fragile across regimes.
  - **P1's check.** P1 compares its hedgers' corrections with the Hull-White adjustment. They do not show the
    haircut, but P1's worlds contain no implied vol (see "Delta haircut check").
- **Poddar, *Uncertainty-Aware Deep Hedging*, [arXiv:2603.10137](https://arxiv.org/abs/2603.10137).**
  - **Their result.** A deep ensemble of five LSTMs under Heston with costs. Ensemble disagreement serves as
    confidence, and an uncertainty-weighted blend of the ensemble hedge and BS delta beats BS delta and the
    Whalley-Wilmott band in CVaR, in simulation.
  - **The link to P1.** P1's residual hedgers are a fixed-bound blend around BS delta with one seed per policy. On
    real options they tie BS delta at best. An ensemble-weighted blend is a natural untested extension.
- **Han, Zhang, Torres, Acero & Xu, *Diffusion models for dynamic volatility surface generation and data-driven
  hedging*, [arXiv:2609.13402](https://arxiv.org/abs/2609.13402).**
  - **Their work.** A diffusion model generates returns jointly with implied-volatility surfaces (S&P 500 options,
    2000-2023), and the scenarios are used for hedging.
  - **The link to P1.** P1's diffusion world generates returns only, with no IV, smile or variance risk premium. That
    is the main information gap named in "Limits", and exactly what a joint generator would supply.
- **Alexander & Imeraj, *Delta hedging bitcoin options with a smile*, Quantitative Finance 23(5):799-817 (2023),
  [doi:10.1080/14697688.2023.2181205](https://doi.org/10.1080/14697688.2023.2181205).**
  - **Their result.** On hourly bitcoin option prices over two one-year periods (2020-2022), smile-implied deltas beat
    BS delta in some periods, with efficiency gains above 30% for OTM puts hedged with the perpetual.
  - **P1's result.** P1's smile-implied delta (2022-2026, daily, held to expiry) is significantly worse than BS delta,
    on puts too.
- **Hull & White, *Optimal delta hedging for options*, Journal of Banking & Finance 82:180-190 (2017),
  [doi:10.1016/j.jbankfin.2017.05.006](https://doi.org/10.1016/j.jbankfin.2017.05.006).**
  - **Their result.** For S&P 500 options, the minimum-variance delta, which accounts for the expected change in
    implied vol given a price move, hedges with less variance than the practitioner BS delta.
  - **P1's result.** Fitted on 2021 Deribit mark prices, its in-sample gain is 1.1%, and out of sample it is 0.18 worse
    in CVaR95.

## Limits (read before quoting a number)

- **Information gap and IV stand-ins.**
  - The original deep hedgers see realised vol only. The Black-Scholes benchmarks see the option's implied vol,
    which is the market's own forecast.
  - The residual hedgers see the real entry IV at test time, but they learned on **stand-ins, not real Deribit IV**:
    `fwd_noisy` is the path's future realised vol with lognormal noise (eta = 0.3, a round number, not measured on
    option data); `trailing` is a backward-looking realised vol.
  - No simulated world has a variance risk premium or a volatility smile, so no policy learned how real IV relates
    to the vol that follows.
- **The test period has been scored four times.**
  - Runs 3 and 4 are post-hoc follow-ups: the residual idea came from reading run 2, and the classical baselines from
    2026 papers read after run 3. Both are disclosed as forced reruns (`forced_rerun: true` in
    `results/test_log.jsonl`).
  - Each new hedger's settings, grid and selection rule were declared in `PLAN.md` before its validation run, and no
    test number fed back into any choice.
  - Still, read the residual and classical rows as follow-ups, not as a clean first test.
- **Classical baselines, simplified.**
  - The band and the smile-implied delta use the entry IV for the whole life of the option, not the day's implied
    vol, because P1 has option prices only in the entry window.
  - The Hull-White fit uses one year of once-a-day mark prices.
  - The smile slope comes from one hour of trades on the entry day.
  - The Alexander-Imeraj variant was identified from the abstract, not the full text.
  - Each could do better with hourly option data, which P1 does not have.
- **Not a full grid.** The residual hyperparameters were chosen by a two-stage search (stand-in x c at alpha = 0.9,
  then alpha), 6 cells per world instead of the 12 of a full grid, to fit a shared machine.
- **The correction sits at its bound.** On test options the residual correction at entry is often at its bound, so
  c, not the network, limits how far these policies move from BS delta.
- **Regime shift.** The simulators are fitted on 2018-2021 (BTC vol ~80% a year); the test period 2022-2026 is
  calmer (see the realism table). The policies get realised vol as an input, but they extrapolate.
- **One seed per policy.** Each policy is trained once (fixed seed). Differences between two deep hedgers that sit
  inside each other's bootstrap interval should not be read as a ranking.
- **The descriptive checks are tiny.** The compatibility check ranks 4 hedgers per family (8 pooled). The haircut
  check is correlational, on entry hedges only.
- **Execution.** Option premium = the last traded price in 07:00-08:00 UTC (both buyer- and seller-initiated
  prints; no extra bid-ask cost beyond the fee). The perp trades at the 08:00 hourly open, no slippage. The hedge
  starts at 08:00, up to an hour after the option trade; this gap is identical for every hedger.
- **Not modelled.** BTC collateral of an inverse option account, margin, the Deribit-index vs Binance-perp basis,
  the option delivery fee, intraday rebalancing. Fee sensitivity (x0, x2) re-prices the same positions; the
  policies are not retrained for each fee level.
- **Fits.**
  - The 2018-2021 GARCH-t fit is integrated (alpha + beta = 1) with nu close to 3, so the GARCH world has very heavy
    tails.
  - The diffusion model learns from about 1,450 days (heavily overlapping 64-day windows); memorisation was not
    measured.
  - Rough-vol H is estimated from daily 5-minute realised variance, which includes the weekday/weekend pattern of
    BTC activity.
- **Sample end.** Binance publishes funding as monthly files only, so the test stops at expiries on or before
  2026-10-31.

## Simplifications and fixes made during the build

Most of these were made before the first test run, on training or validation data only. Three came after a test run:
- the GARCH shock cap, made after run 1 (see "Run history");
- the residual hedgers, added after run 2;
- the classical baselines, added after run 3 (see "Limits").

`PLAN.md` ("Changes during the build") has the details.

- **GARCH shock cap (after run 1).** Standardised Student-t shocks are truncated at the largest in-sample standardised
  residual (7.8 sd for data <= 2020, 8.2 sd for <= 2021). The GARCH validation, final policy and test were redone.
- **Residual hedgers (after run 2).** Four new hedgers, validated, frozen, trained and tested as described above. The
  seven original hedgers were not retrained.
- **Classical baselines (after run 3).** Three hedgers with at most three parameters each, fitted on 2021 data. Nothing
  else was changed.
- **Compute budget.**
  - Policies: batch 4,096, 1,500 Adam steps, 20,000 paths per world, three CVaR levels on validation.
  - Diffusion: 1.09M parameters (smallest size in the 1-5M brief), 8,000 steps of batch 128, DDIM with 50 steps.
  - One seed per policy.
- **GARCH variance cap.** The integrated GARCH-t fit produced exploding paths (daily log returns above 10); the
  conditional variance is now capped at its in-sample maximum.
- **Rough vol in RFSV form.** The non-stationary rBergomi form (flat forward variance, no mean reversion) gave
  64-day paths with excess kurtosis ~224. The stationary RFSV model of GJR 2018 is used instead, with a small mean
  reversion fitted to the dispersion of log-vol; H and vol-of-vol still come from the GJR regression.
- **Diffusion sampler clipping.** Unclipped DDIM sampling blew up (alpha_bar = 2.4e-9 at the last noise level); the
  predicted clean window is clipped at +-15 training standard deviations.
- **Thin slice on validation, not test,** so that the test period is scored only in official runs.
- **No implied-vol input to the original deep hedgers** (optional in the brief): the simulated worlds have no implied
  vol. The residual hedgers of run 3 take the entry IV, with simulated stand-ins in training.
- **Daily rebalancing only**; no 8-hour variant.

## How to run

Windows 11, Git Bash, `uv` 0.12, CPU only.

```bash
git clone https://github.com/mohymabrouk/deep-hedging-sim-to-real.git
cd deep-hedging-sim-to-real
uv sync                                                     # Python 3.12 venv, CPU torch (lockfile: uv.lock)
export PYTHONIOENCODING=utf-8
uv run python -m dh.cli download                            # ~10 min the first time, then cached
uv run python -m dh.cli build                               # option table: results/test_options.csv, validation_options.csv
uv run python -m dh.cli validate                            # fit <= 2020, alpha in {0.5, 0.9, 0.99}, score 2021 options
uv run python -m dh.cli validate_resid                      # residual hedgers: stand-in x c, then alpha, on 2021 options
uv run python -m dh.cli validate_classical                  # classical baselines: Hull-White fit, smile check, WW grid (2021)
uv run python -m dh.cli freeze                              # frozen config + hash (results/frozen_config.json)
uv run python -m dh.cli final                               # refit <= 2021, retrain with chosen alpha, realism table
uv run python -m dh.cli final_resid                         # retrain the residual hedgers with the frozen choice
uv run python -m dh.cli test                                # the single test run (logged; refuses to rerun without --force;
                                                            #   already run four times here, so a rerun needs --force)
uv run python -m dh.cli haircut                             # delta-haircut check on the test entries
uv run python -m dh.cli compat                              # compatibility check (needs results/run3_official/)
uv run python -m dh.cli report                              # tables.md, realism.csv, figures
uv run pytest                                               # tests
```

`validate --only bootstrap,garch,rough` and `validate --only diffusion` split the original validation work across two
processes (that is how it was run here, with `OMP_NUM_THREADS=4` each).
- **Residual stages (run 3)** ran as one process at a time with `OMP_NUM_THREADS=2` (`scripts/run3_resid_chain.sh`).
  Validation (24 policies) took about 3 hours and the four final policies 21 minutes, on a shared laptop CPU.
- **Run 4** (`scripts/run4_classical_chain.sh`: classical validation, freeze, test, haircut check, report) took about
  2 minutes.
- **Logs** go to `logs/`.

## Files

- `PLAN.md`: the plan written before coding, with the changes made during the build. Change 15 pre-declares the
  residual hedgers and change 16 records run 3. Change 17 pre-declares the classical baselines and both checks, and
  change 18 records run 4.
- `src/dh/`:
  - data: `download.py`, `deribit.py`, `binance.py`;
  - options and markets: `dataset.py` (options), `market.py` (08:00 series, funding);
  - P&L and hedging: `accounting.py` (P&L), `hedging.py` (all hedgers including the residual and classical ones,
    causal loop, deep-hedging training);
  - `classical.py`: the Hull-White regression and the smile slope at entry;
  - simulators: `sims.py` (bootstrap, GARCH, rough vol, realism statistics), `diffusion.py`;
  - `evaluate.py` (metrics, block bootstrap), `pipeline.py` (protocol stages), `checks.py` (compatibility and haircut
    checks), `report.py`.
- `scripts/`: `run3_resid_chain.sh` (residual validation, freeze and final training of run 3) and
  `run4_classical_chain.sh` (run 4).
- `results/`:
  - tables: `performance_test.csv`, `performance_splits.csv`, `fee_sensitivity.csv`, `realism.csv`,
    `test_pnl_per_option.csv`, `test_entry_hedge_ratios.csv` (with the smile slope at entry),
    `classical_vs_best_deep.csv`, `reproduction_vs_run3.csv`, `tables.md`;
  - checks: `compatibility_gaps.csv`, `compatibility_predictors.csv`, `compatibility_ranks.csv`, `haircut_check.csv`;
  - configuration and logs: `validation/` (including `run4_predeclared.json`), `simulators/`,
    `selected_hyperparameters.json`, `frozen_config.json`, `test_log.jsonl`;
  - figures `fig_*.png` (`fig_headline.png` is the one above);
  - archives: `run1_official/`, `run2_official/` and `run3_official/` hold every output of test runs 1, 2 and 3.
- `models/`: the `final/` and `validation/` policies.
  - `run1_official/` holds the run-1 GARCH policy that run 2 replaced.
  - `run2_official/` holds byte-identical copies of the four original policies used in runs 2 to 4.
- `tests/` (48 tests):
  - accounting (BS delta on GBM);
  - the deep hedger recovers BS delta on GBM;
  - delivery/payoff mapping and inverse-to-USD;
  - the leak test;
  - put-call parity on a saved Deribit snapshot;
  - `test_sims.py`: no simulated world can be dominated by one draw (martingale shift and path-collapse checks over six
    seeds, on synthetic and real BTC fits; the real-data checks skip without the data cache);
  - `test_residual.py`: the residual policy equals BS delta at initialisation and stays within its bound; its IV input
    is the entry trade's IV, static and causal; the stand-ins use the intended windows;
  - `test_classical.py`: each classical hedger equals BS delta bit for bit with no adjustment and is causal. It also
    checks the band rule and put-call parity of the minimum-variance delta, that the Hull-White and smile estimators
    recover known coefficients and apply the declared filters, and the smile reliability keep rule;
  - `test_reproduction.py`: run 4 reproduces run 3 for the 11 earlier hedgers, option by option (it reads the committed
    result files and skips without them).
