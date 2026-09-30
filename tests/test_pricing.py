from math import exp
 
import pytest
 
from black76 import black76_greeks, black76_price
from black76.pricing import Kind
 
CASES = [
    (100.0, 100.0, 1.0, 0.03, 0.20),
    (80.0, 100.0, 0.25, 0.01, 0.45),
    (120.0, 100.0, 2.0, 0.05, 0.30),
    (3.5, 3.0, 0.5, 0.0, 0.60),  # gas-like levels, zero rate
]
 
 
def test_reference_value_atm_zero_rate() -> None:
    # r=0, F=K: price = F * (2 N(sigma*sqrt(T)/2) - 1)
    assert black76_price(100, 100, 1.0, 0.0, 0.2, "call") == pytest.approx(7.965567, abs=1e-5)
 
 
@pytest.mark.parametrize("F,K,T,r,sigma", CASES)
def test_put_call_parity(F: float, K: float, T: float, r: float, sigma: float) -> None:
    c = black76_price(F, K, T, r, sigma, "call")
    p = black76_price(F, K, T, r, sigma, "put")
    assert c - p == pytest.approx(exp(-r * T) * (F - K), abs=1e-10)
 
 
def test_expiry_and_zero_vol_give_discounted_intrinsic() -> None:
    assert black76_price(110, 100, 0.0, 0.05, 0.3, "call") == pytest.approx(10.0)
    assert black76_price(90, 100, 0.0, 0.05, 0.3, "call") == 0.0
    assert black76_price(100, 110, 1.0, 0.0, 0.0, "put") == pytest.approx(10.0)
 
 
@pytest.mark.parametrize("F,K,T,r,sigma", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(
    F: float, K: float, T: float, r: float, sigma: float, kind: Kind
) -> None:
    g = black76_greeks(F, K, T, r, sigma, kind)
 
    def px(F_: float = F, T_: float = T, r_: float = r, s_: float = sigma) -> float:
        return black76_price(F_, K, T_, r_, s_, kind)
 
    hF, hs, hT, hr = F * 1e-4, 1e-5, 1e-5, 1e-6
    assert g.price == pytest.approx(px())
    assert g.delta == pytest.approx((px(F_=F + hF) - px(F_=F - hF)) / (2 * hF), rel=1e-5)
    assert g.gamma == pytest.approx(
        (px(F_=F + hF) - 2 * px() + px(F_=F - hF)) / hF**2, rel=1e-4
    )
    assert g.vega == pytest.approx((px(s_=sigma + hs) - px(s_=sigma - hs)) / (2 * hs), rel=1e-5)
    assert g.theta == pytest.approx(-(px(T_=T + hT) - px(T_=T - hT)) / (2 * hT), rel=1e-4)
    assert g.rho == pytest.approx((px(r_=r + hr) - px(r_=r - hr)) / (2 * hr), rel=1e-4)
 
 
def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        black76_price(-1, 100, 1, 0.0, 0.2)
    with pytest.raises(ValueError):
        black76_price(100, 100, -1, 0.0, 0.2)
    with pytest.raises(ValueError):
        black76_greeks(100, 100, 0.0, 0.0, 0.2)
 