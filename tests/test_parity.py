"""Put-call parity on a real Deribit snapshot (saved fixture: BTC option trades, 2022-03-01 07:00-08:00 UTC).

For the same expiry and strike, (C - P) in USD = F - K (rates ~ 0 over weeks), so the implied forward
F = K + C_usd - P_usd must sit close to the index. Mark prices are used (no bid/ask bounce); each leg is converted
with the index price of its own trade. Loose tolerance: the basis of a <= 1-quarter BTC forward is a few %.
"""

import json
from pathlib import Path

import numpy as np

from dh.dataset import last_trades, parse_instrument

FIXTURE = Path(__file__).parent / "fixtures" / "deribit_btc_options_2022-03-01_0700_0800.json"


def test_put_call_parity_on_a_deribit_snapshot():
    trades = json.loads(FIXTURE.read_text())
    last = last_trades(trades)
    legs = {}
    for name, t in last.items():
        expiry, strike, is_call = parse_instrument(name)
        legs.setdefault((expiry, strike), {})["C" if is_call else "P"] = t
    devs = []
    for (expiry, strike), pair in legs.items():
        if len(pair) < 2:
            continue
        c, p = pair["C"], pair["P"]
        c_usd = c["mark_price"] * c["index_price"]
        p_usd = p["mark_price"] * p["index_price"]
        s = 0.5 * (c["index_price"] + p["index_price"])
        devs.append((strike + c_usd - p_usd) / s - 1.0)
    devs = np.array(devs)
    assert len(devs) >= 5, "need at least 5 call/put pairs in the snapshot"
    assert np.all(np.abs(devs) < 0.05), devs  # implied forward within 5% of the index
    assert abs(np.median(devs)) < 0.01  # and typically within 1%
