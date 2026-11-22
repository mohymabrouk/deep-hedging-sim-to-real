"""Instrument parsing, delivery-price payoff mapping and the inverse (BTC) -> USD conversion."""

from datetime import date

import numpy as np
import pandas as pd

from dh.accounting import inverse_payoff_btc, inverse_to_usd, option_fee_usd, payoff_usd
from dh.dataset import attach_payoffs, last_trades, parse_instrument


def test_parse_instrument():
    assert parse_instrument("BTC-25FEB22-35000-P") == (date(2022, 2, 25), 35000.0, False)
    assert parse_instrument("BTC-2JAN22-46000-C") == (date(2022, 1, 2), 46000.0, True)
    assert parse_instrument("BTC-31DEC21-50000-C") == (date(2021, 12, 31), 50000.0, True)
    assert parse_instrument("BTC-27MAR26-100000-P") == (date(2026, 3, 27), 100000.0, False)


def test_payoff_uses_the_delivery_price_of_the_expiry_date():
    df = pd.DataFrame({"expiry": [date(2022, 2, 25), date(2022, 2, 25), date(2022, 3, 4)],
                       "strike": [35000.0, 40000.0, 40000.0], "is_call": [False, True, True],
                       "price_btc": [0.02, 0.03, 0.01], "index_price": [38000.0, 38000.0, 39000.0]})
    delivery = {"2022-02-25": 39100.0, "2022-02-24": 1.0, "2022-02-26": 1e9}
    out = attach_payoffs(df, delivery)
    assert out["delivery_price"].iloc[0] == 39100.0
    assert out["payoff_usd"].iloc[0] == 0.0  # put 35000 at 39100: out of the money
    assert out["payoff_usd"].iloc[1] == 0.0  # call 40000 at 39100
    assert np.isnan(out["delivery_price"].iloc[2]) and out["payoff_source"].iloc[2] == "missing"
    out2 = attach_payoffs(df.iloc[:2], {"2022-02-25": 41000.0})
    assert out2["payoff_usd"].tolist() == [0.0, 1000.0]


def test_inverse_to_usd_conversion():
    assert inverse_to_usd(0.0185, 46963.86) == 0.0185 * 46963.86
    # an inverse call pays (S-K)+/S BTC; converted at S it is the linear USD payoff
    for S, K, call in [(50000, 45000, True), (50000, 55000, False), (50000, 55000, True), (40000, 41000, False)]:
        btc = inverse_payoff_btc(S, K, call)
        assert np.isclose(btc * S, payoff_usd(S, K, call))
    assert np.isclose(inverse_payoff_btc(50000, 45000, True), 0.1)


def test_option_fee_is_capped_at_12_5_percent_of_premium():
    # 0.03% of 50,000 = 15 USD; a 100 USD option pays 12.5 USD (cap), a 1,000 USD option pays 15 USD
    assert np.isclose(option_fee_usd(100.0, 50000.0), 12.5)
    assert np.isclose(option_fee_usd(1000.0, 50000.0), 15.0)


def test_last_trade_per_instrument():
    trades = [{"instrument_name": "A", "timestamp": 2, "trade_seq": 1, "price": 1},
              {"instrument_name": "A", "timestamp": 5, "trade_seq": 2, "price": 2},
              {"instrument_name": "B", "timestamp": 3, "trade_seq": 1, "price": 3},
              {"instrument_name": "A", "timestamp": 5, "trade_seq": 3, "price": 4}]
    lt = last_trades(trades)
    assert lt["A"]["price"] == 4 and lt["B"]["price"] == 3
