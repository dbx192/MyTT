from __future__ import annotations

import numpy as np
import pytest

from MyTT import (
    BARSLAST,
    BARSLASTCOUNT,
    BARSSINCEN,
    BETWEEN,
    COUNT,
    CROSS,
    EVERY,
    EXIST,
    FILTER,
    LAST,
    LONGCROSS,
    LOWRANGE,
    TOPRANGE,
    VALUEWHEN,
)


def test_boolean_window_functions() -> None:
    flags = np.array([True, True, False, True, True])
    np.testing.assert_allclose(COUNT(flags, 3), [np.nan, np.nan, 2, 2, 2], equal_nan=True)
    np.testing.assert_array_equal(EVERY(flags, 2), [False, True, False, False, True])
    np.testing.assert_array_equal(EXIST(flags, 2), [False, True, True, True, True])
    np.testing.assert_array_equal(LAST(flags, 2, 1), [False, False, True, False, False])


def test_filter_is_deterministic_and_does_not_mutate_input() -> None:
    source = np.array([True, True, False, True, False, True])
    original = source.copy()
    np.testing.assert_array_equal(FILTER(source, 2), [True, False, False, True, False, False])
    np.testing.assert_array_equal(source, original)


def test_event_distance_functions() -> None:
    flags = [False, False, True, True, False, False, True]
    np.testing.assert_array_equal(BARSLAST(flags), [1, 2, 0, 0, 1, 2, 0])
    np.testing.assert_array_equal(BARSLASTCOUNT(flags), [0, 0, 1, 2, 0, 0, 1])
    np.testing.assert_allclose(
        BARSSINCEN(flags, 4), [np.nan, np.nan, np.nan, 1, 2, 3, 3], equal_nan=True
    )


def test_crosses_and_ranges() -> None:
    left = [1, 1, 3, 2, 5]
    right = [2, 2, 2, 3, 4]
    np.testing.assert_array_equal(CROSS(left, right), [False, False, True, False, True])
    np.testing.assert_array_equal(LONGCROSS(left, right, 2), [False, False, True, False, False])
    np.testing.assert_array_equal(BETWEEN([2, 2, 5], [1, 3, 5], [3, 1, 6]), [True, True, False])
    np.testing.assert_array_equal(TOPRANGE([3, 1, 2, 4, 2]), [0, 0, 1, 3, 0])
    np.testing.assert_array_equal(LOWRANGE([3, 4, 2, 1, 2]), [0, 0, 2, 3, 0])


def test_valuewhen_forward_fills_after_first_event() -> None:
    actual = VALUEWHEN([False, True, False, True], [10, 20, 30, 40])
    np.testing.assert_allclose(actual, [np.nan, 20, 20, 40], equal_nan=True)


def test_condition_validation() -> None:
    with pytest.raises(ValueError, match="0 <= B <= A"):
        LAST([True], 1, 2)
    with pytest.raises(ValueError, match="same length"):
        CROSS([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="same length"):
        VALUEWHEN([True], [1, 2])
