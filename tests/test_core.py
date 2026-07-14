from __future__ import annotations

import numpy as np
import pytest

from MyTT import (
    ABS,
    AVEDEV,
    CONST,
    COS,
    DIFF,
    DMA,
    EMA,
    FORCAST,
    HHV,
    HHVBARS,
    IF,
    LLV,
    LLVBARS,
    LN,
    MA,
    MAX,
    MIN,
    POW,
    RD,
    REF,
    RET,
    SIN,
    SLOPE,
    SMA,
    SQRT,
    STD,
    SUM,
    TAN,
    WMA,
)


def assert_array(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_elementwise_functions_accept_lists_and_scalars() -> None:
    values = [1.0, 2.0, 4.0]
    assert_array(ABS([-1, -2]), [1, 2])
    assert_array(LN(values), np.log(values))
    assert_array(POW(values, 2), [1, 4, 16])
    assert_array(SQRT(values), [1, np.sqrt(2), 2])
    assert_array(SIN(values), np.sin(values))
    assert_array(COS(values), np.cos(values))
    assert_array(TAN(values), np.tan(values))
    assert_array(MAX(values, 2), [2, 2, 4])
    assert_array(MIN(values, 2), [1, 2, 2])
    assert_array(IF([True, False, True], values, 0), [1, 0, 4])
    assert_array(RD([1.23456], 2), [1.23])


def test_reference_diff_return_and_const() -> None:
    assert_array(REF([10, 20, 35, 50], 2), [np.nan, np.nan, 10, 20])
    assert_array(DIFF([10, 20, 35, 50], 2), [np.nan, np.nan, 25, 30])
    assert RET([1, 2, 3]) == 3
    assert RET([1, 2, 3], 2) == 2
    assert_array(CONST([1, 2, 9]), [9, 9, 9])


def test_rolling_aggregations_have_explicit_warmup() -> None:
    values = [1, 2, 3, 4]
    assert_array(MA(values, 3), [np.nan, np.nan, 2, 3])
    assert_array(SUM(values, 3), [np.nan, np.nan, 6, 9])
    assert_array(SUM(values, 0), [1, 3, 6, 10])
    assert_array(STD(values, 2), [np.nan, np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)])
    assert_array(HHV(values, 2), [np.nan, 2, 3, 4])
    assert_array(LLV(values, 2), [np.nan, 1, 2, 3])
    assert_array(AVEDEV(values, 2), [np.nan, 0.5, 0.5, 0.5])


def test_dynamic_extremes_and_bar_distances() -> None:
    values = [3, 1, 4, 4, 2]
    periods = [np.nan, 2, 2, 3, 2]
    assert_array(HHV(values, periods), [np.nan, 3, 4, 4, 4])
    assert_array(LLV(values, periods), [np.nan, 1, 1, 1, 2])
    assert_array(HHVBARS(values, 3), [np.nan, np.nan, 0, 0, 1])
    assert_array(LLVBARS(values, 3), [np.nan, np.nan, 1, 2, 0])


def test_moving_averages_match_hand_calculation() -> None:
    values = [1, 2, 3, 4]
    assert_array(EMA(values, 3), [1, 1.5, 2.25, 3.125])
    assert_array(SMA(values, 4, 2), [1, 1.5, 2.25, 3.125])
    assert_array(WMA(values, 3), [np.nan, np.nan, 14 / 6, 20 / 6])
    assert_array(DMA(values, 0.5), [1, 1.5, 2.25, 3.125])
    assert_array(DMA(values, [0, 0.5, 1, 0.25]), [1, 1.5, 3, 3.25])


def test_regression_functions() -> None:
    values = [1, 3, 5, 7, 9]
    assert_array(SLOPE(values, 3), [np.nan, np.nan, 2, 2, 2])
    assert_array(FORCAST(values, 3), [np.nan, np.nan, 5, 7, 9])
    assert_array(SLOPE(values, 1), [0, 0, 0, 0, 0])


@pytest.mark.parametrize("function", [MA, STD, EMA, WMA])
@pytest.mark.parametrize("period", [0, -1, 1.5, True])
def test_invalid_periods_are_rejected(function, period) -> None:
    expected = TypeError if isinstance(period, (float, bool)) else ValueError
    with pytest.raises(expected):
        function([1, 2, 3], period)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        MA([[1, 2], [3, 4]], 2)
    with pytest.raises(ValueError, match="same length"):
        MAX([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="must not be empty"):
        CONST([])
    with pytest.raises(IndexError):
        RET([1], 2)
    with pytest.raises(ValueError, match="less than or equal"):
        SMA([1, 2], 2, 3)
    with pytest.raises(ValueError, match="match S length"):
        DMA([1, 2], [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
        DMA([1, 2], 1.1)
    with pytest.raises(ValueError, match="dynamic N"):
        HHV([1, 2, 3], [1, 2])


def test_tongdaxin_zero_period_and_dynamic_reference_semantics() -> None:
    assert_array(HHV([3, 1, 4, 2], 0), [3, 3, 4, 4])
    assert_array(LLV([3, 1, 4, 2], 0), [3, 1, 1, 1])
    assert_array(REF([10, 20, 30, 40], [0, 1, 2, 1]), [10, 10, 10, 30])
