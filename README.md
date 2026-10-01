# black76

Black-76 option pricing, Greeks and implied volatility in typed, tested Python,
extended to average-price (Asian) options with a Monte Carlo benchmark. Built as
a foundation for pricing options on Danish and wider European power and gas
futures.

## Status

- [x] Black-76 price for European calls and puts on a forward/future
- [x] Analytic Greeks: delta, gamma, vega, theta, rho
- [x] Implied volatility solver (Newton with bisection fallback)
- [x] Monthly average-price (Asian) options, including partially fixed averages,
      with a Monte Carlo benchmark
- [ ] Bachelier (normal-vol) model for prices that can go negative
- [ ] Zonal spread options (e.g. DK1 vs. DK2): Kirk vs. Bachelier vs. Monte Carlo
- [ ] Forward curve and volatility term structure
- [ ] Book-level risk (delta ladder, scenarios) and dashboard

## Install

```
pip install -e ".[dev]"
pytest
mypy black76 tests
```

Requires Python 3.10+ and NumPy.

## Usage

Arguments are shown positionally, in the order of each function's signature.

```python
from black76 import black76_greeks, black76_price, implied_vol

# Call on a forward at 85 EUR/MWh: strike 90, six months, 3% rate, 45% vol
price = black76_price(85.0, 90.0, 0.5, 0.03, 0.45, "call")
greeks = black76_greeks(85.0, 90.0, 0.5, 0.03, 0.45, "call")
vol = implied_vol(price, 85.0, 90.0, 0.5, 0.03, "call")
```

### Average-price options on a delivery month

Cash-settled month futures such as the Danish DK1/DK2 contracts settle on the
average of the daily day-ahead price. Part-way through the month, some days are
already fixed and the rest are still open.

```python
from datetime import date
from black76 import asian_price, asian_price_mc, delivery_month_fixings

# Valued on 15 October 2026 on the October contract
n_fixed, fixing_times = delivery_month_fixings(date(2026, 10, 15), 2026, 10)
fixed_prices = [92.0] * n_fixed  # replace with the observed day-ahead prices

# forward price of the open days, strike, fixing times, fixed prices,
# discount rate, vol, call/put
price = asian_price(88.0, 90.0, fixing_times, fixed_prices, 0.03, 0.60, "call")

# Monte Carlo benchmark: returns a price and its standard error
mc = asian_price_mc(88.0, 90.0, fixing_times, fixed_prices, 0.03, 0.60, "call", seed=1)
print(price, mc.price, mc.std_error)
```

## Conventions

| Quantity | Meaning |
|----------|---------|
| forward price | forward/futures price for the option's expiry (for Asian options: the price of the still-open days) |
| strike | contract strike. For Asian options it is compared with the average of all fixings |
| time to expiry | years, e.g. 30 days = 30/365 |
| risk-free rate | continuously compounded rate, used to discount the premium |
| vol | lognormal (Black) volatility as a decimal, e.g. 0.30 |

Greeks are with respect to the forward price, not spot:

- **delta**: dV/dF, including the discount factor
- **gamma**: d²V/dF²
- **vega**: dV/dvol per 1.00 of vol (divide by 100 for per vol point)
- **theta**: dV/dt per year, with the forward price and vol held fixed
- **rho**: dV/dr per 1.00 of rate with the forward price held fixed, equal to `-T * V`

## How the Asian option is priced

The average is `A = (fixed prices + open fixings) / N`. The open fixings are
lognormal, but their sum is not, so it is approximated by a lognormal with the
same first two moments (the Levy / Turnbull-Wakeman approach). The already-fixed
days only shift the strike, and the shifted option is priced with Black-76. The
day-ahead price for delivery day D is set on D-1, and that is used as its fixing
date. `asian_price_mc` simulates the same payoff exactly, with antithetic
variates, and is used to test the approximation.

## Validation

- Put-call parity: `C - P = exp(-rT) * (F - K)` (and the average-based
  equivalent for Asian options, including negative fixed prices)
- Every Greek checked against central finite differences, for calls and puts
- Implied vol round-trips (price to vol and back) across strikes, vols and maturities
- Expiry and zero-vol limits return discounted intrinsic value
- Asian option with one fixing, or several fixings at the same time, equals plain Black-76
- Asian approximation compared with Monte Carlo, within sampling error plus a
  stated model-error allowance

## Limitations

- The Asian option uses a flat volatility and a flat forward across the averaging
  window. The two-moment lognormal approximation agrees with Monte Carlo to about
  1% near the money, but is noticeably less accurate far out of the money
  (roughly 10-20% off in the tails of the test cases).
- Black-76 is lognormal, so it requires a positive forward and strike. It does
  not fit spot or hourly power prices, which can go negative. A Bachelier model
  is planned for those products.
- A price at or outside its no-arbitrage bounds has no implied vol, and the
  solver raises `ValueError`. Prices that are almost entirely intrinsic value
  identify volatility only very weakly.

## Project layout

```
black76/
├── pyproject.toml
├── README.md
├── black76/
│   ├── __init__.py
│   ├── pricing.py       # Black-76 price and Greeks
│   ├── implied_vol.py   # implied volatility solver
│   └── asian.py         # Asian options, Monte Carlo benchmark, month fixings
└── tests/
    ├── test_pricing.py
    ├── test_implied_vol.py
    └── test_asian.py
```