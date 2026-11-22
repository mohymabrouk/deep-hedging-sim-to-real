# PLAN: which simulated market trains deep hedgers that hold up on real BTC options?

Written before any code. It records the design, the leak-free protocol, the risks and the fallbacks.
Things that change during the build are noted at the end under "Changes during the build".

## 1. Question

Train one deep-hedging policy (Buehler, Gonon, Teichmann & Wood 2019) in each of four simulated
markets: (a) block bootstrap of BTC returns, (b) GARCH(1,1)-t, (c) rough volatility (rough
Bergomi-style dynamics under P, with H estimated as in Gatheral, Jaisson & Rosenbaum 2018),
(d) a small diffusion model (DDPM, 1-5M parameters). Then use each policy to hedge **real expired
Deribit BTC options** (2022 to the latest month with complete hedge data), with a **Binance
USDT-M perpetual** as the hedge, and compare the hedging error after fees and funding with three
benchmarks: no hedge, Black-Scholes delta at the entry IV, and Black-Scholes delta with the IV
moved by Deribit's DVOL index.

## 2. Data (free, no keys; raw files cached under `~/.cache/quant-research/p1/`)

| Item | Source | Use |
|---|---|---|
| Option trades (expired and live) | `history.deribit.com .../get_last_trades_by_currency_and_time` (kind=option), one request window per day: 07:00-08:00 UTC | entries: price (BTC), iv, index_price, instrument |
| Delivery prices | `www.deribit.com .../get_delivery_prices?index_name=btc_usd` (checked: 2,671 daily records, back to 2019) | option payoff at expiry |
| DVOL | `www.deribit.com .../get_volatility_index_data` (checked: starts 2021-03-24) | BS-DVOL benchmark |
| Perp hourly klines + funding | `data.binance.vision` futures/um monthly 1h klines and fundingRate, `.CHECKSUM` verified | hedge prices at 08:00 UTC, funding cash flows |
| Spot hourly + 5-minute klines | `data.binance.vision` spot monthly, 2018-01 onwards, checksum verified | simulator training data (daily 08:00 returns; 5-min realised variance for rough vol) |

The existing `~/.cache/qal/` panel is not needed (it holds daily data from 2020); it is left untouched.
Deribit requests: at most 4 per second, retries with exponential backoff, every raw response cached
as JSON so a rerun never downloads again. Pages are walked forward in time (`sorting=asc`), deduplicated
by `trade_id`.

## 3. Test options

* Snapshot: trades between 07:00 and 08:00 UTC each day. For each instrument the **last trade** in
  that hour gives the entry: premium = trade price (BTC) x `index_price` at that trade (USD),
  entry IV = trade `iv`, S0 = that index price.
* Eligible: 7 to 35 days to expiry (expiry is 08:00 UTC, so days are whole), |log(K/S0)| < 0.2,
  price >= 0.0005 BTC (one tick above the minimum), IV between 5% and 300%.
* One entry per instrument: its first eligible day. So no option is counted twice.
* Validation set: entry in 2021 **and** expiry <= 2021-12-31. Test set: entry >= 2022-01-01 and
  expiry no later than the last day with Binance perp + funding data.
* Payoff in USD = intrinsic value at the Deribit delivery price (30-min TWAP of the index before
  08:00 UTC). Fallback if a delivery price is missing: index at 08:00 from the trades feed (logged).

### Inverse to USD, basis

Deribit options are inverse: premium and payoff are paid in BTC. A call pays max(S_T - K, 0)/S_T BTC,
which is worth max(S_T - K, 0) USD at the delivery price, so the USD payoff is linear. We assume the BTC
premium is converted to USD at the entry index price immediately, and the hedge is a linear USDT perp.
Not modelled: holding BTC collateral (extra delta), margin, the index-vs-perp basis at entry and expiry
(the hedge unwinds at the perp price at 08:00, the payoff uses the index TWAP). Basis is common to all
hedgers that hold the same position, and it is reported, not hidden.

## 4. Hedging and P&L accounting (one function for simulation and real data)

* Rebalance **daily at 08:00 UTC**, for every hedger. Reasons: Deribit expiries settle at 08:00, an
  8-hour Binance funding time is 08:00, the simulators model daily 08:00-to-08:00 returns, and with a
  0.05% taker fee the cost of 3x more trades is not obviously paid back. Same step for all hedgers.
* We are **short one option on 1 BTC** and hedge with q_t perp contracts (BTC units) held from day t to
  day t+1.
* P&L (USD) = premium - payoff + sum_t q_t (P_{t+1} - P_t) - perp fees - option fee - funding.
  * Perp fee: 0.05% x |q_t - q_{t-1}| x P_t at each rebalance, including the first trade and the
    unwind at expiry.
  * Option fee (Deribit taker): min(0.03% x S0, 12.5% x premium), paid once at entry. The delivery fee
    is not charged (same for every hedger).
  * Funding: Binance pays at 00:00, 08:00, 16:00 UTC; a position q held on (t, t+1] pays
    q x rate x price at each of the three funding times in that interval (price = perp hourly open).
* Normalisation: P&L / S0 (index at entry), reported in % of S0. Loss = -P&L.
* Fee sensitivity: both fee types multiplied by 0, 1, 2 at evaluation (funding unchanged; policies
  are not retrained).

## 5. Hedgers

* **No hedge**: q = 0.
* **BS-IV**: Black-Scholes delta, r = 0, sigma = entry IV, S = perp price at 08:00, tau in calendar
  days / 365.
* **BS-DVOL**: same, sigma_t = entry IV + (DVOL_t - DVOL_entry) (parallel shift of the smile by the change
  of the 30-day vol index; floor 5%). DVOL at 08:00 UTC hourly close.
* **Deep hedgers (4)**: one MLP shared across all contracts (3 x 64, SiLU). Inputs at day t, all
  from data <= t: log(S_t/K), tau_t, q_{t-1}, 10-day realised vol (annualised, from daily 08:00 log
  returns), call flag, and the derived z = log(S/K)/(rv sqrt(tau)) clipped to +-4. Output: sigmoid ->
  hedge in [0, 1] for calls, minus 1 for puts.
* Objective: CVaR_alpha of the training loss with proportional perp costs (Rockafellar-Uryasev form,
  the threshold is a trained parameter). Funding is not in the training objective.
* Training contracts: maturity uniform on 7..35 days, log(K/S0) uniform on [-0.2, 0.2], 50% calls.
  Adam, batch 8192, fixed seeds. Budget: < 10 min per policy on CPU.

## 6. Simulators (all produce paths of 64 daily 08:00 log returns; 24 days of history for the vol
feature, then up to 35 days of option life)

All are made drift-free: after generation each generator's returns are shifted so that
E[exp(r)] = 1 (a martingale price). Otherwise 2018-2020 drift teaches the policy a directional bet.

* (a) **Stationary block bootstrap** (Politis-Romano), mean block 20 days, on daily returns.
* (b) **GARCH(1,1) with Student-t** innovations, fitted with `arch` (zero mean); 300-day burn-in.
* (c) **Rough vol**: daily realised variance from 5-min spot returns; H and nu from the slope and level
  of log E|log sigma_{t+D} - log sigma_t|^2 vs log D (D = 1..30 days, GJR 2018). Simulate
  v_t = xi0 exp(eta B^H_t - eta^2/2 t^{2H}), eta = 2 nu, fBm exact by Cholesky on the daily grid,
  xi0 drawn from the empirical 5-day-average RV; r_t = sqrt(v_t) z_t - v_t/2, with z_t correlated
  (rho) with the fGn increment driving log v (leverage). rho from corr(r_t / sqrt(RV_t), d log RV_{t+1}).
* (d) **Diffusion**: DDPM, epsilon-prediction, cosine schedule (1000 steps, DDIM 100 steps to sample),
  1D U-Net with ~2M parameters on standardised windows of 64 daily returns. Training windows use all 24
  hourly anchors of a "day" (08:00-08:00, 09:00-09:00, ...), which multiplies the overlapping windows
  by 24 without new information. Budget: < 30 min training on CPU.

## 7. Protocol (leak-free)

1. **Validation stage.** Fit all simulators on spot data <= 2020-12-31. Train each deep hedger for
   alpha in {0.5, 0.9, 0.99}. Score on the 2021 validation options. For each generator keep the alpha
   with the lowest validation CVaR95 of the loss (1x fees). No other choice is made from the data.
2. **Final stage.** Refit all simulators on data <= 2021-12-31, retrain the four policies with the
   chosen alpha.
3. **Test.** Run all seven hedgers on the test options **once**, in one script that appends
   {timestamp, config hash, code hash, counts, date range} to `results/test_log.jsonl` and refuses to
   run again unless `--force` (which is logged too).
4. Deviation from the suggested build order, on purpose: the thin slice (BS delta on real options) is
   run on the 2021 **validation** options, not on test, so the test period is scored exactly once, for
   all hedgers together.

## 8. Metrics (same test options for every hedger)

* Loss in % of S0: mean, std, CVaR95; mean P&L; turnover (sum |dq|, BTC per BTC notional); perp fees,
  option fee, funding (% of S0).
* Paired improvement vs BS-IV: mean(loss_BS - loss_h) and CVaR95(loss_BS) - CVaR95(loss_h), with 95% CIs
  from a **moving-block bootstrap over expiry dates** (sorted expiries, blocks of 4 consecutive
  expiries, about a month; all options of a resampled expiry come together), 2,000 resamples.
* Splits: 2022-23 vs 2024 onwards (by entry date); calls vs puts; fees x0/x1/x2.
* Realism table: for each final-stage generator vs held-out daily returns 2022-01-01 onwards:
  annualised vol, excess kurtosis, mean ACF of |r| over lags 1-10, leverage mean corr(r_t, r_{t+k}^2)
  over k = 1..10, 1% and 99% quantiles, Wasserstein-1 distance (raw and on standardised returns).

## 9. Tests (pytest)

1. BS delta on GBM, zero costs, fine rebalancing -> hedging error near zero, and shrinking with more steps
   (checks the shared P&L function).
2. Deep hedger trained on GBM (no costs) recovers BS delta within a tolerance.
3. Instrument parsing + payoff from the delivery price.
4. Inverse -> USD conversion.
5. Leak test: actions at steps <= t are unchanged when prices after t are replaced.
6. Put-call parity on a saved Deribit snapshot (loose tolerance).
7. Accounting function gives the same numbers on numpy and torch inputs.

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Deribit history API pages fail / rate limit | backoff + resume from cache; if a day keeps failing, skip it and log it |
| A delivery price missing | use the 08:00 index price from trades, flag the option |
| Binance data for the current month not published | end the test at the last full month, state it |
| DVOL missing on a day | forward-fill the last DVOL value (no look-ahead); before 2021-03-24 BS-DVOL = BS-IV |
| Few test options (< 1,000) | widen the window to 06:00-08:00 and say so |
| Diffusion model memorises 3 years of data / poor tails | report it in the realism table; no tuning on test |
| Training > 45 min | cut iterations, keep the same budget for every generator |
| Vol regime shift (2018-21 vol higher than 2022+) | policies see realised vol and z as inputs; realism table shows the gap |
| Policies learn a drift bet | martingale correction of every generator |

## 11. Deliverables

`README.md` (headline, performance table, realism table, method, data, fees, what is new, limits,
commands), `results/*.csv`, figures, `results/test_log.jsonl`, passing tests.

## Changes during the build

1. **Perp history** on data.binance.vision starts in 2020-01 (monthly 1h klines and funding), not 2019-09. Enough:
   validation entries start 2021-01-01 and need 24 days of history.
2. **Test end**: Binance publishes funding only as monthly files, the last full month is 2026-10, so the test
   keeps options that expire on or before 2026-10-31 (last expiry 2026-10-30). Option counts: 1,388 validation,
   9,527 test (251 Friday expiries).
3. **Thin slice on validation** (planned in section 7.4): BS benchmarks were first run on the 2021 options.
4. **Training speed**: policy batch 4,096 (not 8,192), 1,500 Adam steps (~3-4 min on CPU). Paths per generator:
   20,000 for all four (the diffusion sampler is the slow one).
5. **GARCH**: the 2018-2020 fit is integrated (alpha + beta = 1.000, nu = 2.94), so there is no finite
   unconditional variance to start from. Paths start from a conditional variance drawn from the in-sample
   filtered variances (no burn-in). The first validation run showed exploding paths (daily log returns up to 11.7,
   and a martingale shift of 0.098 per day that dragged every path down 97% in 35 days); that GARCH policy did not
   hedge at all (validation CVaR95 26.8%, same as no hedge). Fix, made on validation before any test scoring: the
   conditional variance is capped at the in-sample maximum. The GARCH validation run was redone from scratch.
6. **Rough vol**: the first version (rBergomi form with a flat forward variance per path and no mean reversion)
   gave excess kurtosis ~224 on 64-day paths because log-vol dispersion grows like t^H without bound. Replaced
   by the RFSV model of GJR 2018 (fractional OU log-vol with a small mean reversion, set so the stationary std of
   log-vol matches the data), which is the P-model behind rough Bergomi. H, nu still come from the GJR variogram
   regression. Checked on 2018-2020 data before any option was scored: vol 77% vs 79% real, excess kurtosis 7.9 vs 8.7.
7. **Diffusion size**: a 1D U-Net with 32 base channels (1.09M parameters, inside the 1-5M range), batch 128,
   8,000 steps, DDIM with 50 steps. 64 base channels (4.3M) measured at 0.7 s per step on this CPU: too slow.
8. **Diffusion sampler bug**: the first validation samples were garbage (median |log return| 8.5): with the cosine
   schedule alpha_bar at the last level is 2.4e-9, and the first DDIM step divides the noise-prediction error by
   its square root. Fix: clip the predicted clean window to +-15 training standard deviations at each step (the
   usual clip_denoised; the largest 2018-2021 move is 8.2 sd) and recompute the noise from the clipped value.
   The trained model was kept; only sampling was redone. Checked against 2018-2020 returns before use.
9. **Final-stage diffusion** was started before validation ended (its fit does not depend on alpha); the policy
   training step waits for the validation choice.
10. **Compute incidents** (no effect on any result): running 3-4 training processes at once on the 8-core laptop
   slowed everything 3-5x; the final-stage diffusion model took 51 min to train (8,000 steps) instead of ~20 min
   alone. Two background jobs (diffusion validation policies, final bootstrap policy) died around 22:28 and were
   relaunched from their cached paths and models. From then on at most two training processes ran at once.
11. **Citations**: the abstracts of arXiv 2507.09916 and 2603.20582 do not match the one-line descriptions in the
   brief (see README "What is new"); the README describes them from their abstracts.
12. **Frozen config and the test (run 1).** Session resumed on 2026-11-19: the frozen config (`results/frozen_config.json`,
   hash d2f906a2dba41748, alpha = 0.99 / 0.9 / 0.9 / 0.99 for bootstrap / GARCH / rough / diffusion) was written from
   validation only, all four worlds were refitted on data <= 2021, and the test ran once (logged 09:41 UTC).
13. **Bug found after run 1, and run 2.** The run-1 realism table showed the final-stage GARCH world was broken: one
   Student-t draw of about 150 standard deviations gave a +12 daily log return, and since the drift correction is
   log E[exp(r)], that single value set it to 0.115 per day, so every GARCH path fell ~97% in 35 days. The GARCH
   policy then barely hedged calls and the run-1 GARCH row equals "no hedge". The bug is visible from the simulator
   output alone (`results/run1_official/simulators/final_garch.json`), and my pre-test checks missed it because
   they used a different seed. Fix: truncate standardised GARCH shocks at the largest in-sample standardised
   residual (7.8 sd for data <= 2020, 8.2 sd for <= 2021). Then the GARCH validation was redone (alpha chosen again
   on 2021 only), the final GARCH policy retrained, and the test rerun with `--force`; the rerun is logged as
   `forced_rerun: true`. The other six hedgers are unchanged between the two runs. Run 1 outputs are kept in
   `results/run1_official/` and both runs are reported in the README.
14. **Run 2 done (2026-11-21).** Re-validation of the capped GARCH world gave CVaR95 6.10 / 5.85 / 7.90 for
   alpha 0.5 / 0.9 / 0.99, so alpha = 0.9 was chosen again. The frozen config was rewritten (hash `152ac45c09422193`),
   the final GARCH policy retrained (339 s), and the test rerun with `--force` (logged 20:24 UTC). Results:
   - The GARCH CVaR95 loss is 4.51, down from 20.15 in run 1.
   - The other six hedgers are unchanged, up to floating-point noise (at most 4e-9).
   Also fixed:
   - `report.py` read the *first* test-log entry for the header of `tables.md`; it now reads the latest.
   - Added the regression test `tests/test_sims.py`. It fails on the uncapped sampler.
15. **Residual deep hedgers, pre-declared on 2026-11-22 before any validation run of this change.**
   - **Why.** Run 2 showed that no deep hedger beats BS delta at the entry IV. BS delta sees the market's
     forward-looking implied vol; the policies see only a 10-day realised vol, learned in worlds that are 60-70% more
     volatile than 2022-2026.
   - **New hedgers.** One residual hedger per world, `dh_resid_<world>`. The seven existing hedgers stay unchanged and
     are not retrained.
   - **Policy.** q_t = BS delta(S_t, K, tau_t, sigma_entry) + c * tanh(MLP(x_t)), clipped to [0, 1] for calls and
     [-1, 0] for puts. The MLP has the same shape as before (3 x 64, SiLU). It has 8 inputs: the six existing features,
     sigma_entry, and the base BS delta. Its last layer starts at zero, so at initialisation the policy is exactly
     "BS delta, entry IV".
   - **sigma_entry on real options.** The IV of the entry trade, already in the option table. It is known at entry,
     before the first hedge at 08:00.
   - **sigma_entry on simulated paths.** The worlds have no IV, so two stand-ins are declared here and chosen per world
     on validation:
     - `fwd_noisy`: the path's realised vol over the option's life (entry to expiry) times exp(eta * eps - eta^2/2),
       with eps ~ N(0, 1) per contract and eta = 0.3. It mimics an IV as a noisy forecast of the vol the option will
       see. eta = 0.3 is a round number; it is not tuned and not measured on option data.
     - `trailing`: the realised vol of the 20 daily returns before entry.
   - **Grid.** Correction bound c in {0.1, 0.3}. CVaR level alpha in {0.5, 0.9, 0.99}, the same grid as before.
   - **Training.** The same budget as the existing policies: the cached 20,000 paths per world, batch 4,096,
     1,500 Adam steps, lr 2e-3 with cosine decay, 0.05% perp cost. Every cell of a world uses the same seed, so the
     cells see the same paths and contracts. The stand-in noise has its own random stream.
   - **Selection, on validation only.** It uses the 2021 options and the cached validation paths (simulators fitted on
     data <= 2020), and it runs in two stages:
     1. Train every stand-in x c at alpha = 0.9 (4 cells per world).
     2. Train the stage-1 winner at alpha = 0.5 and 0.99.
     The chosen cell is the one with the lowest validation CVaR95 loss at 1x fees, the same rule as for alpha before.
     That is 24 policies. A full 12-cell grid per world (48 policies, about 3 hours on 2 CPU threads) was judged too
     slow for a shared machine.
   - **Then.**
     1. Freeze a new config hash.
     2. Retrain the four chosen policies on the final-stage paths. These are the cached paths of run 2 (simulators
        fitted on data <= 2021), not refitted.
     3. Archive run 2 in `results/run2_official/` and `models/run2_official/`.
     4. Run the test with `--force` (test run 3, logged as a forced rerun) for all 11 hedgers.
     5. Check that the seven existing hedgers reproduce run 2 up to floating-point noise.
   - **Reporting.** The four new rows are reported next to the old ones whatever the result. The idea comes from
     reading run 2's diagnosis, so read run 3's new rows as a post-hoc follow-up, not as a clean first test.
16. **Run 3 done, 2026-11-22.**
   - **Validation**, which took 2 h 52 min on 2 CPU threads, chose per world:
     - bootstrap: `fwd_noisy`, c = 0.3, alpha = 0.9 (validation CVaR95 6.16);
     - GARCH-t: `trailing`, c = 0.1, alpha = 0.9 (5.55);
     - rough vol: `trailing`, c = 0.1, alpha = 0.9 (5.98);
     - diffusion: `fwd_noisy`, c = 0.1, alpha = 0.99 (6.72).
     BS delta at the entry IV scores 6.63 on the same options. Every cell is in `results/validation/dh_resid_<world>.csv`.
   - **Freeze and final training.** The config was frozen at 13:51 UTC with hash `d35f2054d7c27041`. The four final
     residual policies were then trained on the cached final-stage paths, in about 5 min each.
   - **Archive.** Run 2's test outputs and frozen config were moved to `results/run2_official/`. Its simulator and
     realism files and its selected hyperparameters were copied there. Byte-identical copies of the four original
     policies went to `models/run2_official/`; the originals stay in `models/final/` because run 3 uses them.
   - **Test run 3** ran with `--force` and was logged at 14:15 UTC as `forced_rerun: true`, with the frozen hash. The
     test stage now refuses to run if the hash differs from `frozen_config.json`.
   - **Result: no residual hedger beats BS delta.** CVaR95 loss vs BS delta's 3.64, with the change and its 95%
     interval:
     - rough vol: 3.63, +0.00 [-0.45, +0.53];
     - GARCH-t: 3.69, -0.05 [-0.51, +0.48];
     - diffusion: 3.93, -0.29 [-0.72, +0.20];
     - bootstrap: 3.97, -0.33 [-0.82, +0.18].
   - **Beyond the headline.**
     - All four residual hedgers have a significantly higher standard deviation of loss than BS delta (+0.23 to +0.32).
     - Their correction at entry on test options averages 0.06-0.09 BTC and often sits at the bound.
     - They narrow the gap of the original deep hedgers (for example diffusion, from -1.90 to -0.29), but no edge
       appears.
   - **Reproduction.** The seven original hedgers were not retrained.
     - The three benchmarks reproduce run 2 bit for bit.
     - The four original deep hedgers differ by at most 1.7e-10 (% of S0) in any summary statistic, 6.8e-10 in the
       split and fee tables, and 7.2e-7 in any single option's P&L. This is float32 rounding under a different
       thread count (`results/reproduction_vs_run2.csv`).
   - **Edits after the test.**
     - `report.py` got shorter titles for the hedge-ratio figure (one was clipped). It is text only, so the code hash
       is now `288cd6ea965c52a0`, against `6ebba392e2e4b00e` logged by the test.
     - The README's run-2 value 3.69 for BS-DVOL in 2022-23 was a rounding slip (3.685); it now reads 3.68.
   - **Tests.** 30 tests: 21 old and 9 in `tests/test_residual.py`.
