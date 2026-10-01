from math import exp, log, pi, sqrt, erf
from dataclasses import dataclass
from typing import Literal, get_args

Kind = Literal["call", "put"]

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x**2)

def _validate(forward_price: float, strike_price: float, time_to_expiry: float, vol: float, kind: Kind) -> None:
    if forward_price <= 0.0 or strike_price <= 0.0:
        raise ValueError(
            "Black-76 requires forward_price > 0 and strike_price > 0 (lognormal model; "
            "use a Bachelier/normal model for negative prices."
        )
    if time_to_expiry < 0.0:
        raise ValueError("time_to_expiry must be >= 0.")
    if vol < 0.0:
        raise ValueError("vol must be >= 0")
    if kind not in get_args(Kind):
        raise ValueError("kind must be 'call' or 'put'.")

def _d1_d2(forward_price: float, strike_price: float, time_to_expiry: float, vol: float) -> tuple[float, float]:
    vol_sqrt_t = vol * sqrt(time_to_expiry)
    d1 = (log(forward_price / strike_price) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t

def black76_price(
    forward_price: float,   # (F) futures/forward price for option's expiry
    strike_price: float,    # (K) strike
    time_to_expiry: float,  # (T) in years
    risk_free_rate: float,  # (r) continuously compounded
    vol: float,             # (sigma) lognormal volatility as decimal
    kind: Kind = "call"
) -> float:
    _validate(forward_price, strike_price, time_to_expiry, vol, kind)
    df = exp(-risk_free_rate * time_to_expiry)    #risk-free discount factor
    if time_to_expiry == 0.0 or vol == 0.0:
        intrinsic = max(forward_price - strike_price, 0.0) if kind == "call" else max(strike_price - forward_price, 0.0)
        return df * intrinsic
    d1, d2 = _d1_d2(forward_price, strike_price, time_to_expiry, vol)
    if kind == "call":
        return df * (forward_price * _norm_cdf(d1) - strike_price * _norm_cdf(d2))
    return df * (strike_price * _norm_cdf(-d2) - forward_price * _norm_cdf(-d1))

#-------------------------- GREEKS --------------------------
@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

def black76_greeks(
    forward_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    vol: float,
    kind: Kind = "call"
) -> Greeks:
    _validate(forward_price, strike_price, time_to_expiry, vol, kind)
    if time_to_expiry == 0.0 or vol == 0.0:
        raise ValueError("Greeks require time_to_expiry > 0 and vol > 0")

    df = exp(-risk_free_rate * time_to_expiry)
    d1, d2 = _d1_d2(forward_price, strike_price, time_to_expiry, vol)

    if kind == "call":
        price = df * (forward_price * _norm_cdf(d1) - strike_price * _norm_cdf(d2))
        delta = df * _norm_cdf(d1)
    else:
        price = df * (strike_price * _norm_cdf(-d2) - forward_price * _norm_cdf(-d1))
        delta = -df * _norm_cdf(-d1)

    gamma = df * _norm_pdf(d1) / (forward_price * vol * sqrt(time_to_expiry))
    vega = df * forward_price * _norm_pdf(d1) * sqrt(time_to_expiry)
    theta = risk_free_rate * price - df * forward_price * _norm_pdf(d1) * vol / (2.0 * sqrt(time_to_expiry))
    rho = -time_to_expiry * price
    return Greeks(price, delta, gamma, vega, theta, rho)

    


