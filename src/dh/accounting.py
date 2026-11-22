"""P&L of a short option hedged with a linear perpetual. One function for simulated and real paths.

Works on numpy arrays and on torch tensors (only arithmetic, abs and .sum(1) are used), so the training
objective and the real-data backtest share the exact same accounting.

Sign convention: we are SHORT one option on 1 BTC and LONG q_t perp contracts (BTC units) over (t, t+1].
    P&L = premium - payoff + sum_t q_t (P_{t+1} - P_t) - perp fees - option fee - funding
    perp fee at time t = cost_rate * |q_t - q_{t-1}| * P_t, with q_{-1} = 0 and q_T = 0 (unwind at expiry)
    funding over (t, t+1] = q_t * funding_cash_t  (funding_cash_t = sum of rate x price at the funding times)
"""

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _pad(q, front: bool):
    """Add a zero column at the front (positions before the first trade) or the back (unwind at expiry)."""
    if _is_torch(q):
        z = torch.zeros_like(q[:, :1])
        return torch.cat([z, q] if front else [q, z], 1)
    z = np.zeros_like(q[:, :1])
    return np.concatenate([z, q] if front else [q, z], 1)


def hedge_pnl(P, q, payoff, premium=0.0, funding_cash=None, option_fee=0.0, cost_rate=0.0005) -> dict:
    """P: [N, T+1] hedge prices at the rebalancing times 0..T. After an option's expiry, pad with its expiry
    price (any value works, since q must be 0 there). q: [N, T] positions; 0 after the option's expiry.
    payoff, premium, option_fee: [N] or scalars. funding_cash: [N, T] or None. All in the same currency unit."""
    after = _pad(q, front=False)  # position after the trade at t = 0..T
    before = _pad(q, front=True)  # position before the trade at t = 0..T
    trades = abs(after - before)
    perp_fees = cost_rate * (trades * P).sum(1)
    gains = (q * (P[:, 1:] - P[:, :-1])).sum(1)
    funding = (q * funding_cash).sum(1) if funding_cash is not None else 0.0 * gains
    pnl = premium - payoff + gains - perp_fees - option_fee - funding
    return {"pnl": pnl, "gains": gains, "perp_fees": perp_fees, "funding": funding, "turnover": trades.sum(1),
            "option_fee": option_fee + 0.0 * gains, "premium": premium + 0.0 * gains, "payoff": payoff + 0.0 * gains}


def option_fee_usd(premium_usd, index_price, rate=0.0003, cap=0.125):
    """Deribit option taker fee in USD: 0.03% of the underlying, capped at 12.5% of the option price."""
    return np.minimum(rate * index_price, cap * premium_usd)


def inverse_to_usd(price_btc, index_price):
    """Deribit option prices are quoted in BTC; their USD value at the index price."""
    return price_btc * index_price


def payoff_usd(delivery_price, strike, is_call):
    """USD payoff of a Deribit option at the delivery price. The inverse payoff (S-K)+/S BTC converted at S
    is the linear payoff (S-K)+ USD, so this equals inverse_payoff_btc * delivery_price."""
    d, k = np.asarray(delivery_price, float), np.asarray(strike, float)
    return np.where(is_call, np.maximum(d - k, 0.0), np.maximum(k - d, 0.0))


def inverse_payoff_btc(delivery_price, strike, is_call):
    d, k = np.asarray(delivery_price, float), np.asarray(strike, float)
    return np.where(is_call, np.maximum(d - k, 0.0), np.maximum(k - d, 0.0)) / d
