from math import exp
 
import pytest
 
from black76 import black76_price, implied_vol
from black76.pricing import Kind
 
 
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 110.0, 160.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8, 2.5])
@pytest.mark.parametrize("T", [0.05, 1.0, 3.0])
def test_round_trip(kind: Kind, K: float, sigma: float, T: float) -> None:
    F, r = 100.0, 0.03
    px = black76_price(F, K, T, r, sigma, kind)
    df = exp(-r * T)
    intrinsic = df * (max(F - K, 0.0) if kind == "call" else max(K - F, 0.0))
    if px - intrinsic < 1e-4:  # vol is not identifiable from such a price
        pytest.skip("time value too small to identify volatility")
    assert implied_vol(px, F, K, T, r, kind) == pytest.approx(sigma, rel=1e-5)
 
 
def test_price_outside_bounds_raises() -> None:
    with pytest.raises(ValueError):
        implied_vol(0.0, 100, 100, 1.0, 0.0, "call")  # at lower bound
    with pytest.raises(ValueError):
        implied_vol(100.0, 100, 100, 1.0, 0.0, "call")  # at upper bound (F)
    with pytest.raises(ValueError):
        implied_vol(5.0, 100, 100, 0.0, 0.0, "call")  # T == 0