import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from math import exp, log1p, sqrt
from typing import NamedTuple, Sequence, get_args

import numpy as np
import numpy.typing as npt

from .pricing import Kind, black76_price

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float

class _Averaging(NamedTuple):
    fixing_times: FloatArray # years to each remaining fixing
    fixed_sum: float # sum of already fixed values
    n_total: int # total number of fixings (including already fixed values)
    payment_time: float # years until the option pays out (may be different from the last fixing time)

def _prepare(
    forward_price: float,           # (F)
    strike_price: float,            # (K)
    fixing_times: Sequence[float],  # (t_i) observation times
    fixed_prices: Sequence[float],  # (S_i) fixed values (e.g. already observed prices)
    vol: float,                     # (sigma) volatility
    kind: Kind,
    payment_time: float | None,     # payment time (T_pay) for the option, if different from the last observation time
) -> _Averaging:
    if forward_price <= 0:
        raise ValueError("Forward price must be positive.")
    if vol < 0:
        raise ValueError("Volatility must be non-negative.")
    if strike_price <= 0:
        raise ValueError("Strike price must be positive.")
    if kind not in get_args(Kind):
        raise ValueError("Kind must be 'call' or 'put'.")
    
    fixing_times_array: FloatArray = np.asarray(fixing_times, dtype=np.float64)     
    if fixing_times_array.size > 0 and (fixing_times_array[0] < 0 or bool(np.any(np.diff(fixing_times_array) < 0.0))):
        raise ValueError("Times must be non-negative and sorted in ascending order.")
    
    n_total: int = int(fixing_times_array.size) + len(fixed_prices)
    if n_total == 0:
        raise ValueError("At least one observation time or fixed value must be provided.")
    
    last_fixing_time: float = float(fixing_times_array[-1]) if fixing_times_array.size > 0 else 0.0
    payment_time = last_fixing_time if payment_time is None else payment_time
    if payment_time < last_fixing_time:
        raise ValueError("Payment time must be greater than or equal to the last observation time.")
    
    return _Averaging(fixing_times_array, float(sum(fixed_prices)), n_total, payment_time)

def asian_price(
    forward_price: float,           # (F)
    strike_price: float,            # (K)
    fixing_times: Sequence[float],  # (t_i) observation times
    fixed_prices: Sequence[float],  # (S_i) fixed values (e.g. already observed prices)
    risk_free_rate: float,          # (r) continuously compounded
    vol: float,                     # (sigma) volatility
    kind: Kind = "call",            # option type
    payment_time: float | None = None,  # payment time (T_pay) for the option, if different from the last observation time
) -> float:
    # Two-moment lognormal approximation for an option on a partially fixed average.

    avg = _prepare(forward_price, strike_price, fixing_times, fixed_prices, vol, kind, payment_time)
    df: float = exp(-risk_free_rate * avg.payment_time)
    fixed_contribution: float = avg.fixed_sum / avg.n_total
    n_open: int = int(avg.fixing_times.size)

    if n_open == 0: # everything is already fixed, so the option is either in or out of the money
        intrinsic: float = max(fixed_contribution - strike_price, 0.0) if kind == "call" else max(strike_price - fixed_contribution, 0.0)
        return df * intrinsic

    # First two moments of the open part of the average, Y
    open_mean: float = n_open * forward_price / avg.n_total
    min_times: FloatArray = np.minimum(avg.fixing_times[:, None], avg.fixing_times[None, :])
    open_var: float = (forward_price / avg.n_total) ** 2 * float(np.expm1(vol**2 * min_times).sum())
    total_variance: float = log1p(open_var / open_mean**2)

    # Fixed days are moved out of the average, so the mean is shifted by the fixed contribution
    residual_strike: float = strike_price - fixed_contribution
    if residual_strike <= 0.0:
        return df * (fixed_contribution - strike_price + open_mean) if kind == "call" else 0.0

    # Use Black-76 on the lognormal approximation of the open part of the average
    return df * black76_price(open_mean, residual_strike, 1, 0.0, sqrt(total_variance), kind)

def asian_price_mc(
    forward_price: float,           # (F)
    strike_price: float,            # (K)
    fixing_times: Sequence[float],  # (t_i) observation times
    fixed_prices: Sequence[float],  # (S_i) fixed values (e.g. already observed prices)
    risk_free_rate: float,          # (r) continuously compounded
    vol: float,                     # (sigma) volatility
    kind: Kind = "call",            # option type
    payment_time: float | None = None,  # payment time (T_pay) for the option, if different from the last observation time
    n_pairs: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Monte Carlo price of the same payoff, with antithetic variates.
 
    Fixings are simulated exactly: S_i = F * exp(-vol^2 t_i / 2 + vol W(t_i)),
    with W a Brownian motion sampled at the fixing times. `n_pairs` is the number
    of antithetic pairs, so 2 * n_pairs paths are used.
    """
    avg = _prepare(forward_price, strike_price, fixing_times, fixed_prices, vol, kind, payment_time)
    df: float = exp(-risk_free_rate * avg.payment_time)
    fixed_contribution: float = avg.fixed_sum / avg.n_total
    n_open: int = int(avg.fixing_times.size)

    def payoff(average: FloatArray) -> FloatArray:
        diff = average - strike_price if kind == "call" else strike_price - average
        return np.maximum(diff, 0.0)
    
    if n_open == 0: # everything is already fixed, so the option is either in or out of the money
        return MCResult(df * float(payoff(np.array([fixed_contribution]))[0]), 0.0)

    rng: np.random.Generator = np.random.default_rng(seed)
    time_steps: FloatArray = np.diff(avg.fixing_times, prepend=0.0)
    z: FloatArray = rng.standard_normal((n_pairs, n_open))
    brownian: FloatArray = np.cumsum(np.sqrt(time_steps) * z, axis=1)
    drift: FloatArray = -0.5 * vol**2 * avg.fixing_times

    def average_of(paths: FloatArray) -> FloatArray:
        fixings: FloatArray = forward_price * np.exp(drift + vol * paths)
        return (avg.fixed_sum + fixings.sum(axis=1)) / avg.n_total

    payoffs: FloatArray = 0.5 * (payoff(average_of(brownian)) + payoff(average_of(-brownian)))
    price: float = df * float(payoffs.mean())
    std_error: float = df * float(payoffs.std(ddof=1)) / sqrt(n_pairs)

    return MCResult(price, std_error)

def delivery_month_fixings(
    valuation_date: date,
    year: int,
    month: int,
    day_count: float = 365.0,
) -> tuple[int, list[float]]:
    """Split a delivery month's daily day-ahead fixings into fixed and open.
 
    The day-ahead price for delivery day D is set the day before (D - 1), so that
    is taken as its fixing date. Fixings dated on or before `valuation_date` count
    as already fixed; the rest become times in years from `valuation_date`.
 
    Returns (number of fixed days, times of the remaining fixings).
    """
    n_days: int = calendar.monthrange(year, month)[1]
    n_fixed: int = 0
    fixing_times: list[float] = []
    for day in range(1, n_days + 1):
        fixing_date = date(year, month, day) - timedelta(days=1)
        if fixing_date <= valuation_date:
            n_fixed += 1
        else:
            fixing_times.append((fixing_date - valuation_date).days / day_count)
    return n_fixed, fixing_times
