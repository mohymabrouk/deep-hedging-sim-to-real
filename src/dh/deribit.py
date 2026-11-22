"""Deribit public API: option trades in the daily snapshot window, delivery prices, DVOL.

Polite client: at most 4 requests per second across threads, retries with exponential backoff, every raw
response cached as JSON so a rerun never downloads again.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import requests

from dh import config

HISTORY = "https://history.deribit.com/api/v2/public/"
MAIN = "https://www.deribit.com/api/v2/public/"
MAX_RPS = 4.0
log = logging.getLogger(__name__)

_lock = threading.Lock()
_last = [0.0]
_session = requests.Session()


def _throttle() -> None:
    with _lock:
        wait = _last[0] + 1.0 / MAX_RPS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()


def get(base: str, method: str, params: dict, tries: int = 6) -> dict:
    for k in range(tries):
        _throttle()
        try:
            r = _session.get(base + method, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"status {r.status_code}")
            r.raise_for_status()
            body = r.json()
            if "result" not in body:
                raise requests.HTTPError(f"no result: {str(body)[:200]}")
            return body["result"]
        except (requests.RequestException, ValueError) as e:
            log.warning("%s %s failed (%s); retry %d", method, params, e, k + 1)
            time.sleep(min(60, 2 ** (k + 1)))
    raise RuntimeError(f"giving up on {method} {params}")


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def trades_cache_path(day: date):
    return config.CACHE / "deribit" / "trades" / f"{day.isoformat()}.json"


def fetch_day_trades(day: date) -> dict:
    """All BTC option trades in [day 07:00, day 08:00) UTC. Cached as the list of raw result pages."""
    path = trades_cache_path(day)
    if path.exists():
        return json.loads(path.read_text())
    start = _ms(datetime(day.year, day.month, day.day, config.SNAPSHOT_HOUR_START, tzinfo=timezone.utc))
    end = _ms(datetime(day.year, day.month, day.day, config.SNAPSHOT_HOUR_END, tzinfo=timezone.utc)) - 1
    pages, cursor = [], start
    while True:
        res = get(HISTORY, "get_last_trades_by_currency_and_time",
                  {"currency": "BTC", "kind": "option", "start_timestamp": cursor, "end_timestamp": end,
                   "count": 1000, "sorting": "asc"})
        pages.append(res)
        trades = res.get("trades", [])
        if not res.get("has_more") or not trades:
            break
        last_ts = trades[-1]["timestamp"]
        cursor = last_ts if last_ts > cursor else cursor + 1  # inclusive restart; dedupe by trade_id later
        if len(pages) > 200:
            raise RuntimeError(f"too many pages for {day}")
    out = {"day": day.isoformat(), "window_ms": [start, end], "pages": pages}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    return out


def day_trades(day: date) -> list[dict]:
    """Deduplicated trades of the cached day."""
    seen, out = set(), []
    for page in fetch_day_trades(day)["pages"]:
        for t in page.get("trades", []):
            if t["trade_id"] not in seen:
                seen.add(t["trade_id"])
                out.append(t)
    return out


def fetch_trades_range(first: date, last: date, workers: int = 3) -> list[date]:
    """Download every day in [first, last]; returns the days that failed (logged, not fatal)."""
    days = [first + timedelta(d) for d in range((last - first).days + 1)]
    todo = [d for d in days if not trades_cache_path(d).exists()]
    log.info("deribit trades: %d days, %d to download", len(days), len(todo))
    failed = []

    def one(d: date):
        try:
            fetch_day_trades(d)
        except Exception as e:  # noqa: BLE001 - log and continue, the day is reported as missing
            log.error("day %s failed: %s", d, e)
            failed.append(d)

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 100 == 0:
                log.info("  %d / %d", i, len(todo))
    return sorted(failed)


def fetch_delivery_prices() -> dict[str, float]:
    """Daily BTC delivery prices {YYYY-MM-DD: price}. Cached (one file; delete it to refresh)."""
    path = config.CACHE / "deribit" / "delivery_prices_btc_usd.json"
    if path.exists():
        return json.loads(path.read_text())
    out, offset = {}, 0
    while True:
        res = get(MAIN, "get_delivery_prices", {"index_name": "btc_usd", "offset": offset, "count": 1000})
        for row in res["data"]:
            out[row["date"]] = float(row["delivery_price"])
        offset += len(res["data"])
        if not res["data"] or offset >= res["records_total"]:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, sort_keys=True))
    return out


def fetch_dvol_month(y: int, m: int) -> list[list[float]]:
    """Hourly DVOL candles [ts_ms, open, high, low, close] for one month (cached)."""
    path = config.CACHE / "deribit" / "dvol" / f"{y:04d}-{m:02d}.json"
    if path.exists():
        return json.loads(path.read_text())
    start = datetime(y, m, 1, tzinfo=timezone.utc)
    nxt = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=timezone.utc)
    rows, end = [], _ms(nxt) - 1
    while True:
        res = get(MAIN, "get_volatility_index_data",
                  {"currency": "BTC", "start_timestamp": _ms(start), "end_timestamp": end, "resolution": 3600})
        rows.extend(res["data"])
        cont = res.get("continuation")
        if not cont or not res["data"]:
            break
        end = int(cont)  # the API pages backwards in time
    rows = sorted({int(r[0]): r for r in rows}.values(), key=lambda r: r[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows
