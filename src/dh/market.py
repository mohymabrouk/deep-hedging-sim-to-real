"""Market series built from the cached raw files: perp prices at 08:00 UTC, daily funding cash, DVOL at 08:00,
spot daily returns (any anchor hour) and daily realised variance from 5-minute bars."""

import json
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from dh import binance, config, deribit


def _end_month() -> tuple[int, int]:
    s = json.loads((config.CACHE / "download_summary.json").read_text())
    d = date.fromisoformat(s["hedge_data_end"])
    return d.year, d.month


def hedge_data_end() -> date:
    return date.fromisoformat(json.loads((config.CACHE / "download_summary.json").read_text())["hedge_data_end"])


@lru_cache(maxsize=None)
def perp_hourly() -> pd.DataFrame:
    return binance.load_klines("perp", "1h", config.PERP_FIRST_MONTH, _end_month())


@lru_cache(maxsize=None)
def spot_hourly() -> pd.DataFrame:
    return binance.load_klines("spot", "1h", config.SPOT_FIRST_MONTH, _end_month())


@lru_cache(maxsize=None)
def spot_5m() -> pd.DataFrame:
    return binance.load_klines("spot", "5m", config.SPOT_FIRST_MONTH, config.SPOT_5M_LAST_MONTH)


def _hourly_price(df: pd.DataFrame) -> pd.Series:
    """Price at each full hour = open of the hourly bar; short gaps (<= 3 h) filled with the last close."""
    grid = pd.date_range(df.index[0], df.index[-1], freq="h")
    o = df["open"].reindex(grid)
    last_close = df["close"].reindex(grid).shift(1).ffill(limit=3)
    return o.fillna(last_close)


def daily_at_hour(df: pd.DataFrame, hour: int) -> pd.Series:
    """Price at `hour`:00 UTC each day, indexed by date."""
    p = _hourly_price(df)
    p = p[p.index.hour == hour]
    return pd.Series(p.to_numpy(), index=[t.date() for t in p.index], name=f"px_{hour:02d}")


@lru_cache(maxsize=None)
def perp_08() -> pd.Series:
    return daily_at_hour(perp_hourly(), config.REBALANCE_HOUR)


@lru_cache(maxsize=None)
def funding_cash_daily() -> pd.Series:
    """USD paid by a 1-BTC long perp position held over (d 08:00, d+1 08:00], indexed by d:
    sum over funding events in that interval of rate x perp price at the event time."""
    y, m = _end_month()
    rates = binance.load_funding(config.PERP_FIRST_MONTH, (y, m))
    px = _hourly_price(perp_hourly())
    cash = rates * px.reindex(rates.index)
    if cash.isna().any():
        raise RuntimeError(f"{int(cash.isna().sum())} funding events without a perp price")
    # event at time e belongs to the interval (d 08:00, d+1 08:00] with d = date(e - 8h - 1 min)
    owner = [(e - pd.Timedelta(hours=config.REBALANCE_HOUR, minutes=1)).date() for e in cash.index]
    out = cash.groupby(owner).sum()
    out.index = list(out.index)
    return out


@lru_cache(maxsize=None)
def funding_rate_events() -> pd.Series:
    return binance.load_funding(config.PERP_FIRST_MONTH, _end_month())


@lru_cache(maxsize=None)
def dvol_08() -> pd.Series:
    """DVOL (vol points) at 08:00 UTC each day: open of the hourly candle starting at 08:00 (= value at 08:00)."""
    rows = []
    end = hedge_data_end()
    y, m = 2021, 3
    while (y, m) <= (end.year, end.month):
        rows += deribit.fetch_dvol_month(y, m)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    s = pd.Series({pd.Timestamp(int(r[0]), unit="ms", tz="UTC"): float(r[1]) for r in rows}).sort_index()
    grid = pd.date_range(s.index[0].normalize() + pd.Timedelta(hours=8), s.index[-1], freq="D")
    v = s.reindex(s.index.union(grid)).ffill().reindex(grid)  # last value at or before 08:00 (no look-ahead)
    return pd.Series(v.to_numpy(), index=[t.date() for t in grid], name="dvol")


def spot_daily_returns(hour: int = config.REBALANCE_HOUR, end: date | None = None, start: date | None = None) -> pd.Series:
    """Daily log returns between consecutive `hour`:00 UTC prices; index = day of the later price.
    Only days whose closing price is at or before `end` (inclusive) are kept."""
    p = daily_at_hour(spot_hourly(), hour)
    r = np.log(p).diff()
    idx = pd.Index(p.index)
    gap = pd.Series([(b - a).days for a, b in zip(idx[:-1], idx[1:])], index=idx[1:])
    r = r.iloc[1:][gap.to_numpy() == 1].dropna()
    if end is not None:
        r = r[[d <= end for d in r.index]]
    if start is not None:
        r = r[[d >= start for d in r.index]]
    return r


def spot_rv_daily(end: date) -> pd.Series:
    """Realised variance of day d = sum of squared 5-minute log returns over (d-1 08:00, d 08:00]."""
    c = spot_5m()["close"]
    lr = np.log(c).diff().dropna()
    lr = lr[lr.index.to_series().diff().dt.total_seconds().fillna(300).to_numpy() <= 300 + 1]
    # bar with open time o closes at o+5min; it belongs to day d if o+5min in (d-1 08:00, d 08:00]
    close_t = lr.index + pd.Timedelta(minutes=5)
    owner = (close_t - pd.Timedelta(hours=config.REBALANCE_HOUR, seconds=1)).date + timedelta(days=1)
    df = pd.DataFrame({"r2": lr.to_numpy() ** 2, "d": owner})
    g = df.groupby("d")["r2"].agg(["sum", "count"])
    g = g[g["count"] >= 0.9 * 288]  # days with at least 90% of the 5-minute bars
    rv = g["sum"]
    rv = rv[[d <= end for d in rv.index]]
    return rv
