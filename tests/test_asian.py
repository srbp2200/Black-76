from datetime import date
from math import exp

import pytest

from black76 import (
    asian_price,
    asian_price_mc,
    black76_price,
    delivery_month_fixings,
)
from black76.pricing import Kind

DAY = 1.0 / 365.0
FIXING_TIMES = [i * DAY for i in range(1, 31)]  # 30 daily fixings, 1..30 days out


def test_single_fixing_equals_black76() -> None:
    time_to_fixing = 0.25
    asian = asian_price(
        forward_price=85.0,
        strike_price=90.0,
        fixing_times=[time_to_fixing],
        fixed_prices=[],
        risk_free_rate=0.03,
        vol=0.45,
        kind="call",
    )
    vanilla = black76_price(85.0, 90.0, time_to_fixing, 0.03, 0.45, "call")
    assert asian == pytest.approx(vanilla, rel=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_identical_fixing_times_equal_black76(kind: Kind) -> None:
    # Averaging N copies of the same lognormal is that lognormal.
    time_to_fixing = 0.2
    asian = asian_price(100.0, 95.0, [time_to_fixing] * 5, [], 0.02, 0.5, kind)
    vanilla = black76_price(100.0, 95.0, time_to_fixing, 0.02, 0.5, kind)
    assert asian == pytest.approx(vanilla, rel=1e-10)


def test_all_fixed_is_discounted_intrinsic() -> None:
    fixed_prices = [80.0, 100.0, 90.0]  # average 90
    call = asian_price(100.0, 85.0, [], fixed_prices, 0.0, 0.4, "call")
    put = asian_price(100.0, 85.0, [], fixed_prices, 0.0, 0.4, "put")
    assert call == pytest.approx(5.0)
    assert put == 0.0


@pytest.mark.parametrize("strike_price", [40.0, 70.0, 100.0, 130.0])
@pytest.mark.parametrize("fixed_prices", [[], [95.0] * 10, [-5.0] * 10, [300.0] * 10])
def test_put_call_parity(strike_price: float, fixed_prices: list[float]) -> None:
    forward_price, risk_free_rate, vol = 100.0, 0.03, 0.6
    fixing_times = FIXING_TIMES[len(fixed_prices):]
    n_total = len(fixing_times) + len(fixed_prices)
    expected_average = (sum(fixed_prices) + len(fixing_times) * forward_price) / n_total
    df = exp(-risk_free_rate * fixing_times[-1])
    call = asian_price(forward_price, strike_price, fixing_times, fixed_prices, risk_free_rate, vol, "call")
    put = asian_price(forward_price, strike_price, fixing_times, fixed_prices, risk_free_rate, vol, "put")
    assert call - put == pytest.approx(df * (expected_average - strike_price), abs=1e-9)


def test_average_is_cheaper_than_final_fixing_option() -> None:
    # Averaging reduces effective volatility, so an ATM average option is cheaper.
    asian = asian_price(100.0, 100.0, FIXING_TIMES, [], 0.0, 0.6, "call")
    vanilla = black76_price(100.0, 100.0, FIXING_TIMES[-1], 0.0, 0.6, "call")
    assert asian < vanilla


def test_fixed_days_reduce_price_for_atm_option() -> None:
    # Locking in part of the average removes variance.
    all_open = asian_price(100.0, 100.0, FIXING_TIMES, [], 0.0, 0.6, "call")
    half_fixed = asian_price(100.0, 100.0, FIXING_TIMES[15:], [100.0] * 15, 0.0, 0.6, "call")
    assert half_fixed < all_open


CASES = [
    # forward_price, strike, n_fixed, vol, kind, model-error allowance (relative to price)
    (100.0, 100.0, 0, 0.30, "call", 0.01),
    (100.0, 100.0, 0, 0.80, "call", 0.01),
    (100.0, 100.0, 15, 0.60, "call", 0.01),
    (100.0, 90.0, 20, 0.80, "put", 0.01),
    # Far out of the money, the two-moment lognormal approximation is visibly
    # less accurate (about 10% here), so those cases get a looser allowance.
    (100.0, 120.0, 0, 0.60, "call", 0.15),
    (100.0, 80.0, 0, 0.60, "put", 0.25),
]


@pytest.mark.parametrize("forward_price,strike_price,n_fixed,vol,kind,model_tol", CASES)
def test_approximation_matches_monte_carlo(
    forward_price: float,
    strike_price: float,
    n_fixed: int,
    vol: float,
    kind: Kind,
    model_tol: float,
) -> None:
    risk_free_rate = 0.02
    fixed_prices = [105.0] * n_fixed
    fixing_times = FIXING_TIMES[n_fixed:]
    analytic = asian_price(
        forward_price, strike_price, fixing_times, fixed_prices, risk_free_rate, vol, kind
    )
    mc = asian_price_mc(
        forward_price,
        strike_price,
        fixing_times,
        fixed_prices,
        risk_free_rate,
        vol,
        kind,
        n_pairs=200_000,
        seed=7,
    )
    # The analytic price is an approximation, so allow a model error on top of
    # the Monte Carlo sampling error.
    assert abs(analytic - mc.price) <= 4.0 * mc.std_error + model_tol * mc.price


def test_mc_is_reproducible_with_seed() -> None:
    first = asian_price_mc(100.0, 100.0, FIXING_TIMES, [], 0.0, 0.5, "call", n_pairs=1000, seed=1)
    second = asian_price_mc(100.0, 100.0, FIXING_TIMES, [], 0.0, 0.5, "call", n_pairs=1000, seed=1)
    assert first == second


def test_delivery_month_fixings_mid_month() -> None:
    n_fixed, fixing_times = delivery_month_fixings(date(2026, 10, 15), 2026, 10)
    assert n_fixed == 16  # fixing dates 30 Sep .. 15 Oct
    assert len(fixing_times) == 15
    assert fixing_times[0] == pytest.approx(1 / 365)
    assert fixing_times[-1] == pytest.approx(15 / 365)


def test_delivery_month_fixings_before_month() -> None:
    n_fixed, fixing_times = delivery_month_fixings(date(2026, 9, 1), 2026, 10)
    assert n_fixed == 0
    assert len(fixing_times) == 31


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        asian_price(100.0, 100.0, [], [], 0.0, 0.3)  # no fixings at all
    with pytest.raises(ValueError):
        asian_price(100.0, 100.0, [0.2, 0.1], [], 0.0, 0.3)  # unsorted times
    with pytest.raises(ValueError):
        asian_price(100.0, 100.0, [0.1], [], 0.0, 0.3, payment_time=0.05)  # pays before last fixing
    with pytest.raises(ValueError):
        asian_price(-1.0, 100.0, [0.1], [], 0.0, 0.3)  # non-positive forward
    with pytest.raises(ValueError):
        asian_price(100.0, 100.0, [0.1], [], 0.0, 0.3, kind="straddle")  # type: ignore[arg-type]
