# black76

Black-76 option pricing, Greeks and implied volatility in typed, tested Python,
built as a foundation for pricing options on Danish and wider European power
and gas futures.

## Status

Core is done. Energy-specific instruments are in progress.

- [x] Black-76 price for European calls and puts on a forward/future
- [x] Analytic Greeks: delta, gamma, vega, theta, rho
- [x] Implied volatility solver (Newton with bisection fallback)
- [ ] Monthly average-price (Asian) options, including partially fixed averages
- [ ] Bachelier (normal-vol) model for prices that can go negative
- [ ] Zonal spread options (e.g. DK1 vs. DK2), Kirk vs. Bachelier vs. Monte Carlo
- [ ] Forward curve and volatility term structure
- [ ] Book-level risk (delta ladder, scenarios) and dashboard

## Install

```
pip install -e ".[dev]"
pytest
mypy black76 tests
```

## Usage

```python
from black76 import black76_greeks, black76_price, implied_vol

# Call on a forward at 85 EUR/MWh, strike 90, six months, 3% rate, 45% vol
price = black76_price(F=85.0, K=90.0, T=0.5, r=0.03, sigma=0.45, kind="call")
greeks = black76_greeks(F=85.0, K=90.0, T=0.5, r=0.03, sigma=0.45, kind="call")
vol = implied_vol(price, F=85.0, K=90.0, T=0.5, r=0.03, kind="call")
```

## Conventions

| Symbol | Meaning |
|--------|---------|
| `F` | forward/futures price for the option's expiry |
| `K` | strike |
| `T` | time to expiry in years |
| `r` | continuously compounded discount rate |
| `sigma` | lognormal (Black) volatility as a decimal, e.g. 0.30 |

Greeks are with respect to the forward price, not spot:

- **delta**: dV/dF, including the discount factor
- **gamma**: d²V/dF²
- **vega**: dV/dsigma per 1.00 of vol (divide by 100 for per vol point)
- **theta**: dV/dt per year, with F and sigma held fixed
- **rho**: dV/dr per 1.00 of rate with F held fixed, equal to `-T * V`

## Validation

- Put-call parity: `C - P = exp(-rT) * (F - K)`
- Every Greek checked against central finite differences, for calls and puts
- Implied vol round-trips (price to vol and back) across strikes, vols and maturities
- Expiry and zero-vol limits return discounted intrinsic value

## Limitations

- Black-76 is lognormal, so it requires `F > 0` and `K > 0`. It does not fit
  spot or hourly power prices, which can go negative. A Bachelier model is
  planned for those products.
- A price at or outside its no-arbitrage bounds has no implied vol, and the
  solver raises `ValueError`. Prices that are almost entirely intrinsic value
  identify volatility only very weakly.