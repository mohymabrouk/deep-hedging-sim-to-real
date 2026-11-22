"""Real option dataset: one entry per instrument (its first eligible day), payoff from the delivery price, and the
per-option market arrays (perp prices, funding cash, DVOL) aligned on the daily 08:00 UTC grid."""

import logging
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import torch

from dh import config, deribit, market
from dh.accounting import inverse_to_usd, option_fee_usd, payoff_usd
from dh.hedging import Contracts

log = logging.getLogger(__name__)
DERIVED = config.CACHE / "derived"


def parse_instrument(name: str) -> tuple[date, float, bool]:
    """'BTC-25FEB22-35000-P' -> (2022-02-25, 35000.0, False). Expiry is at 08:00 UTC on that date."""
    _, exp, strike, kind = name.split("-")
    expiry = datetime.strptime(exp.upper(), "%d%b%y").date()
    if kind not in ("C", "P"):
        raise ValueError(name)
    return expiry, float(strike), kind == "C"


def last_trades(trades: list[dict]) -> dict[str, dict]:
    """Last trade of each instrument in the window (by timestamp, then trade_seq)."""
    out = {}
    for t in sorted(trades, key=lambda t: (t["timestamp"], t.get("trade_seq", 0))):
        out[t["instrument_name"]] = t
    return out


def build_entries(first: date, last: date) -> pd.DataFrame:
    """Scan the daily snapshots in order; keep each instrument at its first eligible day."""
    seen, rows = set(), []
    n_days = (last - first).days + 1
    for k in range(n_days):
        day = first + timedelta(days=k)
        if not deribit.trades_cache_path(day).exists():
            log.warning("no cached trades for %s", day)
            continue
        for name, t in last_trades(deribit.day_trades(day)).items():
            if name in seen:
                continue
            try:
                expiry, strike, is_call = parse_instrument(name)
            except ValueError:
                continue
            days = (expiry - day).days
            s0 = float(t["index_price"])
            logm = math.log(strike / s0)
            iv = t.get("iv")
            ok = (config.MIN_DAYS <= days <= config.MAX_DAYS and abs(logm) < config.MAX_ABS_LOG_MONEYNESS
                  and t["price"] >= config.MIN_PRICE_BTC and iv is not None and config.MIN_IV <= iv <= config.MAX_IV)
            if not ok:
                continue
            seen.add(name)
            rows.append({"instrument": name, "entry_day": day, "entry_ts": int(t["timestamp"]), "expiry": expiry, "T": days,
                         "strike": strike, "is_call": is_call, "price_btc": float(t["price"]), "mark_price_btc": float(t["mark_price"]),
                         "iv": float(iv), "index_price": s0, "log_moneyness": logm, "direction": t.get("direction"),
                         "amount": float(t.get("amount", np.nan)), "trade_id": t["trade_id"]})
    return pd.DataFrame(rows)


def attach_payoffs(df: pd.DataFrame, delivery: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    dp = df["expiry"].map(lambda d: delivery.get(d.isoformat(), np.nan))
    df["delivery_price"] = dp
    df["payoff_source"] = np.where(dp.notna(), "deribit_delivery", "missing")
    df["premium_usd"] = inverse_to_usd(df["price_btc"], df["index_price"])
    df["payoff_usd"] = payoff_usd(df["delivery_price"], df["strike"], df["is_call"].to_numpy())
    df["option_fee_usd"] = option_fee_usd(df["premium_usd"], df["index_price"], config.OPTION_TAKER_FEE, config.OPTION_FEE_CAP)
    return df


def build_all() -> dict:
    """Build the option table for validation + test and save it (derived cache + results CSV)."""
    end = market.hedge_data_end()
    entries = build_entries(config.DERIBIT_FIRST_DAY, end)
    log.info("eligible first entries: %d", len(entries))
    delivery = deribit.fetch_delivery_prices()
    df = attach_payoffs(entries, delivery)
    df["split"] = np.select(
        [(df["entry_day"] >= config.VAL_START) & (df["expiry"] <= config.VAL_END),
         (df["entry_day"] >= config.TEST_START) & (df["expiry"] <= end)],
        ["validation", "test"], default="excluded")
    missing = df[(df["split"] != "excluded") & df["delivery_price"].isna()]
    if len(missing):
        log.warning("%d options without a delivery price -> excluded: %s", len(missing), missing["expiry"].unique()[:10])
        df.loc[missing.index, "split"] = "excluded"
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DERIVED / "options.parquet")
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    df[df["split"] == "test"].to_csv(config.RESULTS / "test_options.csv", index=False)
    df[df["split"] == "validation"].to_csv(config.RESULTS / "validation_options.csv", index=False)
    stats = df.groupby("split").agg(n=("instrument", "size"), first_entry=("entry_day", "min"), last_expiry=("expiry", "max"),
                                     calls=("is_call", "sum")).reset_index()
    log.info("\n%s", stats.to_string())
    return {"stats": stats}


def load_options(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DERIVED / "options.parquet")
    for col in ("entry_day", "expiry"):
        df[col] = pd.to_datetime(df[col]).dt.date
    return df[df["split"] == split].sort_values(["entry_day", "instrument"]).reset_index(drop=True)


def market_arrays(df: pd.DataFrame, H: int = config.HIST, tmax: int = config.MAX_DAYS):
    """Per-option arrays on the daily 08:00 grid, in units of S0 (index at entry):
    P [N, H+tmax+1] perp prices (padded with the expiry price after expiry), dvol [N, H+tmax+1] (fraction),
    funding [N, tmax] (cash per BTC long over (t, t+1], zero after expiry). Options with missing perp data are dropped."""
    perp, fund, dvol = market.perp_08(), market.funding_cash_daily(), market.dvol_08()
    P, F, V, keep = [], [], [], []
    for i, r in df.iterrows():
        days = [r["entry_day"] + timedelta(days=k) for k in range(-H, tmax + 1)]
        p = np.array([perp.get(d, np.nan) for d in days], float)
        T = int(r["T"])
        p[H + T + 1:] = p[H + T]
        f = np.array([fund.get(r["entry_day"] + timedelta(days=k), np.nan) if k < T else 0.0 for k in range(tmax)], float)
        v = np.array([dvol.get(d, np.nan) for d in days], float) / 100.0
        if np.isnan(p).any() or np.isnan(f).any():
            log.warning("dropping %s: missing perp/funding data", r["instrument"])
            continue
        s0 = r["index_price"]
        P.append(p / s0), F.append(f / s0), V.append(v), keep.append(i)
    df = df.loc[keep].reset_index(drop=True)
    mkt = {"P": torch.as_tensor(np.array(P)), "dvol": torch.as_tensor(np.array(V))}
    dvol0 = mkt["dvol"][:, H].numpy()
    c = Contracts(df["strike"].to_numpy() / df["index_price"].to_numpy(), df["T"].to_numpy(), df["is_call"].to_numpy(),
                  iv0=df["iv"].to_numpy() / 100.0, dvol0=dvol0)
    return df, mkt, torch.as_tensor(np.array(F)), c
