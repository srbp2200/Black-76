from math import pi, sqrt, exp
from .pricing import Kind, black76_greeks, black76_price

_VEGA_FLOOR = 1e-12

def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    kind: Kind = "call",
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Solve for sigma such that black76_price(F, K, T, r, sigma, kind) == price.
 
    Newton steps are used while they stay inside a shrinking bracket; otherwise
    the method falls back to bisection, so it always converges for a price that
    lies strictly inside the no-arbitrage bounds.
 
    Raises
    ------
    ValueError  if T <= 0 or the price is not strictly between the discounted
                intrinsic value and the upper bound (df*F for calls, df*K for puts).
    RuntimeError if the solver fails to converge within max_iter.
    """
    if T <= 0.0:
        raise ValueError("T must be > 0 to imply a volatility")
    if F <= 0.0 or K <= 0.0:
        raise ValueError("Black-76 requires F > 0 and K > 0")

    df = exp(-r * T)
    if kind == "call":
        lower, upper = df * max(F - K, 0.0), df * F
    else:
        lower, upper = df * max(K - F, 0.0), df * K
    if not (lower < price < upper):
        raise ValueError(
            f"Price {price} outside no-arbitrage bounds ({lower}, {upper}); "
            "implied vol is undefined"
        )

    # Bracket the root
    lo, hi = 1e-8, 1.0
    while black76_price(F, K, T, r, hi, kind) < price:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError("Could not bracket implied volatility (sigma > 1000)")

    # Brenner-Subrahmanyam starting guess
    sigma = sqrt(2.0 * pi / T) * price / (df * F)
    sigma = min(max(sigma, lo), hi)

    for _ in range(max_iter):
        diff = black76_price(F, K, T, r, sigma, kind) - price
        if abs(diff) < tol:
            return sigma
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        if hi - lo < 1e-15:
            return sigma

        vega = black76_greeks(F, K, T, r, sigma, kind).vega
        step = sigma - diff / vega if vega > _VEGA_FLOOR else lo - 1.0
        sigma = step if lo < step < hi else 0.5 * (lo + hi)

    raise RuntimeError("Implied volatility solver did not converge")