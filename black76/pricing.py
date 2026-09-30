from math import exp, log, pi, sqrt, erf
from dataclasses import dataclass
from typing import Literal

Kind = Literal["call", "put"]

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x**2)

def _validate(F: float, K: float, T: float, sigma: float, kind:str) -> None:
    if F<=0.0 or K <= 0.0:
        raise ValueError(
            "Black-76 requires F > 0 and K > 0 (lognormal model; "
            "use a Bachelier/normal model for negative prices."
        )
    if T < 0.0:
        raise ValueError("T must be >= 0.")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'.")

def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t

def black76_price(
    F: float,   #current future price
    K: float,   #strike price
    T: float,   #time to option expiry
    r: float,   #continuously compounded risk-free interest rate
    sigma: float,   #volatility
    kind: Kind = "call"
) -> float:
    _validate(F, K, T, sigma, kind)
    df = exp(-r * T)    #risk-free discount factor
    if T == 0.0 or sigma == 0.0:
        intrinsic = max(F - K, 0.0) if kind == "call" else max(K - F, 0.0)
        return df * intrinsic
    d1, d2 = _d1_d2(F, K, T, sigma)
    if kind == "call":
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))

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
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: Kind = "call"
) -> Greeks:
    _validate(F, K, T, sigma, kind)
    if T == 0.0 or sigma == 0.0:
        raise ValueError("Greeks require T > 0 and sigma > 0")

    df = exp(-r * T)
    d1, d2 = _d1_d2(F, K, T, sigma)

    if kind == "call":
        price = df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
        delta = df * _norm_cdf(d1)
    else:
        price = df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))
        delta = -df * _norm_cdf(-d1)

    gamma = df * _norm_pdf(d1) / (F * sigma * sqrt(T))
    vega = df * F * _norm_pdf(d1) * sqrt(T)
    theta = r * price - df * F * _norm_pdf(d1) * sigma / (2.0 * sqrt(T))
    rho = -T * price
    return Greeks(price, delta, gamma, vega, theta, rho)

    


