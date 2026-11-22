"""All constants of the study in one place. Every number here is part of the config hash in the test log."""

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("P1_CACHE", Path.home() / ".cache" / "quant-research" / "p1"))
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

# ---- protocol dates -------------------------------------------------------------------------
VAL_FIT_END = date(2020, 12, 31)  # simulators + policies of the validation stage use data <= this day
FINAL_FIT_END = date(2021, 12, 31)  # simulators + policies of the final stage use data <= this day
VAL_START = date(2021, 1, 1)  # validation options: entry in 2021 and expiry <= 2021-12-31
VAL_END = date(2021, 12, 31)
TEST_START = date(2022, 1, 1)  # test options: entry >= this day, expiry <= last day of hedge data
SUBPERIOD_SPLIT = date(2024, 1, 1)  # "2022-23" vs "2024->" by entry date

# ---- data ranges ------------------------------------------------------------------------------
SPOT_FIRST_MONTH = (2018, 1)  # Binance spot BTCUSDT, simulator training data
SPOT_5M_LAST_MONTH = (2021, 12)  # 5-minute bars are only needed for rough-vol fitting (<= final fit end)
PERP_FIRST_MONTH = (2020, 1)  # first monthly 1h kline file of the USDT-M perp on data.binance.vision
DERIBIT_FIRST_DAY = date(2021, 1, 1)
SNAPSHOT_HOUR_START = 7  # trades in [07:00, 08:00) UTC
SNAPSHOT_HOUR_END = 8
REBALANCE_HOUR = 8  # daily rebalancing at 08:00 UTC (Deribit expiry, Binance funding time)

# ---- option filters -----------------------------------------------------------------------------
MIN_DAYS = 7
MAX_DAYS = 35
MAX_ABS_LOG_MONEYNESS = 0.2
MIN_PRICE_BTC = 0.0005
MIN_IV, MAX_IV = 5.0, 300.0  # in vol points, as Deribit reports them

# ---- fees ------------------------------------------------------------------------------------
PERP_TAKER_FEE = 0.0005  # Binance USDT-M taker, 0.05% of traded notional
OPTION_TAKER_FEE = 0.0003  # Deribit option taker, 0.03% of the underlying ...
OPTION_FEE_CAP = 0.125  # ... capped at 12.5% of the option price
FEE_MULTIPLIERS = (0.0, 1.0, 2.0)

# ---- simulated paths and policy -------------------------------------------------------------------
HIST = 24  # days of history before entry in every simulated path (for the vol feature)
PATH_LEN = 64  # daily returns per simulated path (24 history + up to 35 option days + spare)
RV_WINDOW = 10  # realised-vol feature: last 10 daily 08:00 log returns, annualised
DAYS_PER_YEAR = 365.0
N_SIM_PATHS = 20_000  # paths drawn from each generator for policy training (same for all four)
GARCH_SHOCK_CAP = "max abs standardised in-sample residual"  # truncated Student-t (added after test run 1)
BOOTSTRAP_MEAN_BLOCK = 20  # stationary bootstrap mean block length (days)
ROUGH_MAX_LAG = 30  # lags (days) for the H regression
ALPHAS = (0.5, 0.9, 0.99)  # CVaR levels tried on validation
POLICY_HIDDEN = 64
POLICY_LAYERS = 3
POLICY_BATCH = 4096
POLICY_ITERS = 1500
POLICY_LR = 2e-3
SEED = 20260101

# ---- diffusion --------------------------------------------------------------------------------------
DIFF_STEPS = 1000  # DDPM noise levels (cosine schedule)
DIFF_SAMPLE_STEPS = 50  # DDIM sampling steps
DIFF_TRAIN_ITERS = 8_000
DIFF_BATCH = 128
DIFF_LR = 3e-4
DIFF_CHANNELS = 32  # 1.09M parameters
DIFF_X0_CLIP = 15.0  # clip of the predicted clean window during sampling, in training standard deviations

# ---- evaluation --------------------------------------------------------------------------------------
CVAR_LEVEL = 0.95
BOOT_REPS = 2000
BOOT_BLOCK = 4  # consecutive expiry dates per bootstrap block (about one month of weekly expiries)

GENERATORS = ("bootstrap", "garch", "rough", "diffusion")

# ---- residual deep hedgers (added for test run 3; pre-declared in PLAN.md, change 15, before any validation run) ----
# q = BS delta(sigma_entry) + bound * tanh(MLP(x)); sigma_entry = entry IV on real options, a stand-in on simulated paths
RESID_STAND_INS = ("fwd_noisy", "trailing")  # see hedging.entry_vol_stand_in
RESID_IV_NOISE = 0.3  # eta of the lognormal noise of the "fwd_noisy" stand-in (not tuned)
RESID_TRAIL_WINDOW = 20  # days of returns before entry for the "trailing" stand-in
RESID_BOUNDS = (0.1, 0.3)  # correction bound c (BTC per option)
RESID_STAGE1_ALPHA = 0.9  # validation stage 1: stand-in x bound at this CVaR level; stage 2: the other ALPHAS

HEDGERS = (("no_hedge", "bs_iv", "bs_dvol") + tuple(f"dh_{g}" for g in GENERATORS)
           + tuple(f"dh_resid_{g}" for g in GENERATORS))


def as_dict() -> dict:
    """Every upper-case constant (used for the config hash)."""
    return {k: (str(v) if isinstance(v, (Path, date)) else v) for k, v in globals().items() if k.isupper() and k not in {"ROOT", "CACHE", "RESULTS", "MODELS", "LOGS"}}
