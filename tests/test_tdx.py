from __future__ import annotations

import numpy as np
import pytest

from MyTT import (
    ACOS,
    ALIGNRIGHT,
    AMA,
    ASIN,
    ATAN,
    BACKSET,
    BARSCOUNT,
    BARSLASTS,
    BARSNEXT,
    BARSSINCE,
    BARSTATUS,
    BETAEX,
    CEILING,
    CON2STR,
    COVAR,
    CURRBARSCOUNT,
    DATETODAY,
    DAYTODATE,
    DEVSQ,
    DOWNNDAY,
    EXISTR,
    EXP,
    EXPMEMA,
    FILTERX,
    FINDHIGH,
    FINDHIGHBARS,
    FINDLOW,
    FINDLOWBARS,
    FINDSTR,
    FLOOR,
    FRACPART,
    HHVLLV,
    HOD,
    IFF,
    IFN,
    INTPART,
    ISLASTBAR,
    ISVALID,
    LOD,
    LOG,
    MAX6,
    MEMA,
    MIN6,
    MOD,
    MULAR,
    NDAY,
    NOT,
    RANGE,
    REFX,
    RELATE,
    REVERSE,
    ROUND,
    ROUND2,
    SECTOTIME,
    SIGN,
    STDDEV,
    STDP,
    STR2CON,
    STRCAT,
    STRCAT6,
    STRCMP,
    STRLEN,
    STRSPACE,
    SUBSTR,
    SUMBARS,
    SUMBARSX,
    TIMETOSEC,
    TMA,
    TOTALBARSCOUNT,
    TR,
    UPDOWN,
    UPNDAY,
    VAR,
    VAR2STR,
    VARCAT,
    VARCAT6,
    VARP,
)


def assert_array(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_math_functions() -> None:
    assert_array(REVERSE([1, -2]), [-1, 2])
    assert_array(ACOS([1, 0]), [0, np.pi / 2])
    assert_array(ASIN([0, 1]), [0, np.pi / 2])
    assert_array(ATAN([0, 1]), [0, np.pi / 4])
    assert_array(EXP([0, 1]), [1, np.e])
    assert_array(LOG([1, 100]), [0, 2])
    assert_array(CEILING([1.2, -1.2]), [2, -1])
    assert_array(FLOOR([1.2, -1.2]), [1, -2])
    assert_array(INTPART([1.8, -1.8]), [1, -1])
    assert_array(FRACPART([1.25, -1.25]), [0.25, -0.25])
    assert_array(ROUND([1.2, 1.8]), [1, 2])
    assert_array(ROUND2([1.234, 2.345], 2), [1.23, 2.35])
    assert_array(SIGN([-2, 0, 3]), [-1, 0, 1])
    assert_array(MOD([5, 7], 3), [2, 1])
    assert np.isnan(MOD([1], 0)[0])
    assert_array(MAX6([1, 9], 2, 3, 4, 5, 6), [6, 9])
    assert_array(MIN6([1, 9], 2, 3, 4, 5, 6), [1, 2])


def test_conditions_and_validity() -> None:
    assert_array(RANGE([2, 1, 3], 1, 3), [True, False, False])
    assert_array(IFF([True, False], [1, 2], 0), [1, 0])
    assert_array(IFN([True, False], [1, 2], 0), [0, 2])
    assert_array(NOT([0, 1, 2]), [True, False, False])
    assert_array(ISVALID([1, np.nan, np.inf]), [True, False, False])
    assert_array(UPDOWN([2, 3, 3, 1]), [0, 1, 0, -1])
    assert_array(UPNDAY([1, 2, 3, 2, 4], 2), [False, False, True, False, False])
    assert_array(DOWNNDAY([3, 2, 1, 2, 0], 2), [False, False, True, False, False])
    assert_array(NDAY([2, 3, 1, 4], [1, 2, 2, 3], 2), [False, True, False, False])
    assert_array(EXISTR([False, True, False, False], 2, 1), [False, False, True, True])
    assert_array(EXISTR([False, True, False], 0, 0), [False, True, True])


def test_bar_position_functions() -> None:
    values = [np.nan, 10, 20, 30]
    assert_array(BARSCOUNT(values), [0, 1, 2, 3])
    assert_array(CURRBARSCOUNT(values), [4, 3, 2, 1])
    assert_array(TOTALBARSCOUNT(values), [4, 4, 4, 4])
    assert_array(ISLASTBAR(values), [False, False, False, True])
    assert_array(BARSTATUS(values), [1, 0, 0, 2])
    assert_array(BARSTATUS([1]), [2])
    assert_array(BARSSINCE([False, True, False, True]), [np.nan, 0, 1, 2])
    assert_array(BARSLASTS([False, True, False, True, False], 2), [np.nan] * 3 + [2, 3])
    assert_array(BARSLASTS([True, False, True, False], [1, 1, 2, 1]), [0, 1, 2, 1])


def test_date_time_and_string_functions() -> None:
    assert_array(DATETODAY([901219, 901220, 1000101]), [0, 1, 3300])
    assert_array(DAYTODATE([0, 1, 3300]), [901219, 901220, 1000101])
    assert_array(TIMETOSEC([93000, 235959]), [34200, 86399])
    assert_array(SECTOTIME([34200, 86399]), [93000, 235959])
    assert CON2STR([1.2, 3.456], 2) == "3.46"
    np.testing.assert_array_equal(VAR2STR([1.2, 3.456], 2), ["1.20", "3.46"])
    assert STR2CON("12.5") == 12.5
    assert STRLEN("通达信") == 3
    assert STRCAT("多头", "开仓") == "多头开仓"
    assert STRCAT6("A", "B", "C", "D", "E", "F") == "ABCDEF"
    assert STRSPACE("A") == "A "
    assert SUBSTR("ABCDE", 2, 3) == "BCD"
    np.testing.assert_array_equal(VARCAT(["A", "B"], "1"), ["A1", "B1"])
    np.testing.assert_array_equal(
        VARCAT6(["A", "B"], "1", "2", "3", "4", "5"), ["A12345", "B12345"]
    )
    assert STRCMP("行业", "行业")
    assert FINDSTR("多头开仓", "开仓")


def test_explicit_future_functions() -> None:
    assert_array(BARSNEXT([False, True, False, False, True]), [1, 0, 2, 1, 0])
    assert_array(BACKSET([False, False, True, False], 2), [False, True, True, False])
    assert_array(ALIGNRIGHT([1, np.nan, 2, 3]), [np.nan, 1, 2, 3])
    assert_array(REFX([10, 20, 30, 40], 2), [30, 40, np.nan, np.nan])
    assert_array(REFX([10, 20, 30], [0, 1, 0]), [10, 30, 30])


def test_cumulative_filter_and_smoothing_functions() -> None:
    assert_array(MULAR([1, 2, 3, 4], 0), [1, 2, 6, 24])
    assert_array(MULAR([1, 2, 3, 4], 2), [np.nan, 2, 6, 12])
    assert_array(FILTERX([True, False, True, True, False], 2), [True, False, False, True, False])
    assert_array(SUMBARS([1, 2, 3, 4], 5), [1, 2, 2, 2])
    assert_array(SUMBARS([1, 2, 3], [0, 3, 9]), [0, 2, 3])
    assert_array(SUMBARSX([1, 2, 3], 4), [np.nan, np.nan, 2])
    assert_array(TMA([1, 2, 3], 0.5, 0.5), [1, 1.5, 2.25])
    assert_array(MEMA([1, 2, 3], 2), [1, 1.5, 2.25])
    assert_array(AMA([1, 2, 3], 0.5), [1, 1.5, 2.25])
    assert_array(EXPMEMA([1, 2, 3, 4], 3), [np.nan, np.nan, 2.25, 3.125])


def test_true_range_and_rolling_statistics() -> None:
    assert_array(TR([9, 10, 12], [10, 12, 13], [8, 9, 11]), [2, 3, 3])
    values = [1, 2, 3, 4]
    assert_array(DEVSQ(values, 3), [np.nan, np.nan, 2, 2])
    assert_array(STDP(values, 3), [np.nan, np.nan, np.sqrt(2 / 3), np.sqrt(2 / 3)])
    assert_array(STDDEV(values, 3), [np.nan, np.nan, 1, 1])
    assert_array(VAR(values, 3), [np.nan, np.nan, 1, 1])
    assert_array(VARP(values, 3), [np.nan, np.nan, 2 / 3, 2 / 3])
    x, y = [1, 2, 3, 4], [2, 4, 6, 8]
    assert_array(COVAR(x, y, 3), [np.nan, np.nan, 2, 2])
    assert_array(RELATE(x, y, 3), [np.nan, np.nan, 1, 1])
    assert_array(BETAEX(x, y, 3), [np.nan, np.nan, 0.5, 0.5])


def test_rank_and_extreme_search_functions() -> None:
    values = [3, 1, 4, 2, 5]
    assert_array(HOD(values, 3), [np.nan, np.nan, 1, 2, 1])
    assert_array(LOD(values, 3), [np.nan, np.nan, 3, 2, 3])
    assert_array(HOD(values, 0), [1, 2, 1, 3, 1])
    assert_array(HHVLLV(values, 0, 3, 1), [np.nan] * 3 + [4, 4])
    assert_array(HHVLLV(values, 1, 0, 1), [np.nan, 3, 1, 1, 1])
    assert_array(FINDHIGH(values, 1, 3, 2), [np.nan] * 3 + [3, 2])
    assert_array(FINDHIGHBARS(values, 1, 3, 2), [np.nan] * 3 + [3, 1])
    assert_array(FINDLOW(values, 1, 3, 1), [np.nan] * 3 + [1, 1])
    assert_array(FINDLOWBARS(values, 1, 3, 1), [np.nan] * 3 + [2, 3])


def test_new_function_validation() -> None:
    with pytest.raises(TypeError, match="integer"):
        ROUND2([1], 1.5)
    with pytest.raises(ValueError, match="one-dimensional"):
        NOT([[1]])
    with pytest.raises(ValueError, match="dynamic N"):
        BARSLASTS([True, False], [1])
    with pytest.raises(ValueError, match="dynamic A"):
        REFX([1, 2], [1])
    with pytest.raises(ValueError, match="non-negative"):
        SUMBARS([1, -1], 2)
    with pytest.raises(ValueError, match="match X"):
        SUMBARS([1, 2], [1, 2, 3])
    with pytest.raises(TypeError, match="real scalars"):
        TMA([1], "bad", 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        TMA([1], 1, 0.5)
    with pytest.raises(ValueError, match="greater than"):
        EXISTR([True], 1, 2)
    with pytest.raises(ValueError, match="T must be 0"):
        HHVLLV([1], 2, 1, 0)
    with pytest.raises(ValueError, match="less than or equal"):
        FINDHIGH([1, 2], 0, 1, 2)
