"""Download (once) every raw input of the study into the cache."""

import json
import logging
from datetime import date, timedelta

from dh import binance, config, deribit

log = logging.getLogger(__name__)


def last_data_day() -> date:
    """Last day of the last full month published on data.binance.vision (perp klines and funding)."""
    y, m = binance.last_full_month()
    while binance.fetch(binance.funding_path(y, m)) is None or binance.fetch(binance.kline_path("perp", "1h", y, m)) is None:
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    nxt = date(y + (m == 12), 1 if m == 12 else m + 1, 1)
    return nxt - timedelta(days=1)


def run() -> dict:
    end_day = last_data_day()
    last_month = (end_day.year, end_day.month)
    log.info("hedge data available up to %s", end_day)

    log.info("binance spot 1h, spot 5m, perp 1h, funding")
    spot_1h = binance.fetch_many([binance.kline_path("spot", "1h", y, m) for y, m in binance.months(config.SPOT_FIRST_MONTH, last_month)])
    spot_5m = binance.fetch_many([binance.kline_path("spot", "5m", y, m) for y, m in binance.months(config.SPOT_FIRST_MONTH, config.SPOT_5M_LAST_MONTH)])
    perp_1h = binance.fetch_many([binance.kline_path("perp", "1h", y, m) for y, m in binance.months(config.PERP_FIRST_MONTH, last_month)])
    fund = binance.fetch_many([binance.funding_path(y, m) for y, m in binance.months(config.PERP_FIRST_MONTH, last_month)])

    log.info("deribit delivery prices + DVOL")
    delivery = deribit.fetch_delivery_prices()
    for y, m in binance.months((2021, 3), last_month):
        deribit.fetch_dvol_month(y, m)

    log.info("deribit option trades, daily 07:00-08:00 UTC windows")
    failed = deribit.fetch_trades_range(config.DERIBIT_FIRST_DAY, end_day)

    summary = {
        "hedge_data_end": end_day.isoformat(),
        "spot_1h_files": sum(p is not None for p in spot_1h),
        "spot_5m_files": sum(p is not None for p in spot_5m),
        "perp_1h_files": sum(p is not None for p in perp_1h),
        "funding_files": sum(p is not None for p in fund),
        "delivery_prices": len(delivery),
        "delivery_first": min(delivery),
        "delivery_last": max(delivery),
        "deribit_days_failed": [d.isoformat() for d in failed],
    }
    (config.CACHE / "download_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("download summary: %s", summary)
    return summary
