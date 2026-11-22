"""Binance public data (data.binance.vision): download monthly zips, verify the .CHECKSUM, load as DataFrames.

Files are cached under CACHE/binance/<same path as on the server>. A file is downloaded once; its sha256 is
checked against the published .CHECKSUM before it is used.
"""

import hashlib
import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from dh import config

BASE = "https://data.binance.vision/"
log = logging.getLogger(__name__)

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def months(first: tuple[int, int], last: tuple[int, int]) -> list[tuple[int, int]]:
    out, (y, m) = [], first
    while (y, m) <= last:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def last_full_month(today: date | None = None) -> tuple[int, int]:
    today = today or date.today()
    return (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)


def kline_path(market: str, interval: str, y: int, m: int, symbol: str = "BTCUSDT") -> str:
    root = "data/spot/monthly/klines" if market == "spot" else "data/futures/um/monthly/klines"
    return f"{root}/{symbol}/{interval}/{symbol}-{interval}-{y:04d}-{m:02d}.zip"


def funding_path(y: int, m: int, symbol: str = "BTCUSDT") -> str:
    return f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y:04d}-{m:02d}.zip"


def _get(url: str, tries: int = 5) -> requests.Response:
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # network hiccup: back off and retry
            log.warning("GET %s failed (%s), retry %d", url, e, k + 1)
            time.sleep(2**k)
    raise RuntimeError(f"giving up on {url}")


def fetch(rel: str) -> Path | None:
    """Download BASE+rel and its checksum into the cache (once). Returns the local path, or None if 404."""
    local = config.CACHE / "binance" / rel
    ok_marker = local.with_suffix(local.suffix + ".verified")
    if ok_marker.exists() and local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    r = _get(BASE + rel)
    if r.status_code == 404:
        return None
    c = _get(BASE + rel + ".CHECKSUM")
    if c.status_code == 404:
        raise RuntimeError(f"no checksum published for {rel}")
    expected = c.text.split()[0].strip().lower()
    got = hashlib.sha256(r.content).hexdigest()
    if got != expected:
        raise RuntimeError(f"checksum mismatch for {rel}: {got} != {expected}")
    local.write_bytes(r.content)
    (local.parent / (local.name + ".CHECKSUM")).write_text(c.text)
    ok_marker.write_text(got)
    return local


def fetch_many(rels: list[str], workers: int = 4) -> list[Path | None]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fetch, rels))


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    has_header = not first[:1].isdigit()
    return pd.read_csv(io.StringIO(raw), header=0 if has_header else None)


def _to_utc(ms_or_us: pd.Series) -> pd.Series:
    """Binance switched spot timestamps to microseconds in 2025; detect the unit per value."""
    v = ms_or_us.astype("int64").to_numpy()
    v = np.where(v > 10**14, v // 1000, v)
    return pd.to_datetime(v, unit="ms", utc=True)


def load_klines(market: str, interval: str, first: tuple[int, int], last: tuple[int, int]) -> pd.DataFrame:
    """Klines for every month in [first, last] that exists on the server; index = bar open time (UTC)."""
    frames = []
    for (y, m), p in zip(months(first, last), fetch_many([kline_path(market, interval, y, m) for y, m in months(first, last)])):
        if p is None:
            log.warning("missing %s %s %04d-%02d", market, interval, y, m)
            continue
        df = _read_zip_csv(p)
        df = df.iloc[:, :12]
        df.columns = KLINE_COLS
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = _to_utc(df["open_time"])
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_funding(first: tuple[int, int], last: tuple[int, int]) -> pd.Series:
    """Funding rate per funding event (time rounded to the hour, UTC)."""
    frames = []
    for (y, m), p in zip(months(first, last), fetch_many([funding_path(y, m) for y, m in months(first, last)])):
        if p is None:
            log.warning("missing funding %04d-%02d", y, m)
            continue
        df = _read_zip_csv(p)
        df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][: df.shape[1]]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    t = _to_utc(df["calc_time"]).round("h")
    s = pd.Series(df["last_funding_rate"].astype(float).to_numpy(), index=t, name="funding_rate")
    return s[~s.index.duplicated()].sort_index()
