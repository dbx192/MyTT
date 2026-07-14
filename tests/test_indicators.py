from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from MyTT import (
    ASI,
    ATR,
    BBI,
    BIAS,
    BOLL,
    BRAR,
    CCI,
    CR,
    DFMA,
    DMI,
    DPO,
    EMV,
    EXPMA,
    KDJ,
    KTN,
    MACD,
    MASS,
    MFI,
    MTM,
    OBV,
    PSY,
    ROC,
    RSI,
    TAQ,
    TRIX,
    VR,
    WR,
    XSII,
)


def _arrays(value) -> tuple[np.ndarray, ...]:
    return value if isinstance(value, tuple) else (value,)


@pytest.mark.parametrize(
    ("indicator", "inputs"),
    [
        (MACD, ("close",)),
        (KDJ, ("close", "high", "low")),
        (RSI, ("close",)),
        (WR, ("close", "high", "low")),
        (BIAS, ("close",)),
        (BOLL, ("close",)),
        (PSY, ("close",)),
        (CCI, ("close", "high", "low")),
        (ATR, ("close", "high", "low")),
        (BBI, ("close",)),
        (DMI, ("close", "high", "low")),
        (TAQ, ("high", "low")),
        (KTN, ("close", "high", "low")),
        (TRIX, ("close",)),
        (VR, ("close", "vol")),
        (CR, ("close", "high", "low")),
        (EMV, ("high", "low", "vol")),
        (DPO, ("close",)),
        (BRAR, ("open", "close", "high", "low")),
        (DFMA, ("close",)),
        (MTM, ("close",)),
        (MASS, ("high", "low")),
        (ROC, ("close",)),
        (EXPMA, ("close",)),
        (OBV, ("close", "vol")),
        (MFI, ("close", "high", "low", "vol")),
        (ASI, ("open", "close", "high", "low")),
        (XSII, ("close", "high", "low")),
    ],
)
def test_all_indicators_return_aligned_arrays(
    indicator: Callable, inputs: tuple[str, ...], market: dict[str, np.ndarray]
) -> None:
    kwargs = {"N": 20} if indicator is TAQ else {}
    results = _arrays(indicator(*(market[name] for name in inputs), **kwargs))
    assert results
    for result in results:
        assert isinstance(result, np.ndarray)
        assert result.shape == market["close"].shape
        assert np.isfinite(result[-1])


def test_macd_reference_values() -> None:
    dif, dea, histogram = MACD([1, 2, 3, 4], SHORT=2, LONG=3, M=2)
    np.testing.assert_allclose(dif, [0, 0.167, 0.306, 0.394], atol=0.001)
    np.testing.assert_allclose(dea, [0, 0.111, 0.241, 0.343], atol=0.001)
    np.testing.assert_allclose(histogram, [0, 0.111, 0.13, 0.102], atol=0.001)


def test_boll_reference_values() -> None:
    upper, middle, lower = BOLL([1, 2, 3, 4], N=2, P=2)
    np.testing.assert_allclose(upper, [np.nan, 2.5, 3.5, 4.5], equal_nan=True)
    np.testing.assert_allclose(middle, [np.nan, 1.5, 2.5, 3.5], equal_nan=True)
    np.testing.assert_allclose(lower, [np.nan, 0.5, 1.5, 2.5], equal_nan=True)


def test_zero_denominators_are_reported_as_nan() -> None:
    flat = np.ones(40)
    assert np.isnan(RSI(flat, 6)[-1])
    assert np.isnan(CCI(flat, flat, flat, 6)[-1])
    assert np.isnan(WR(flat, flat, flat, 6)[0][-1])
