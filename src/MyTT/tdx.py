"""Additional TongDaXin-compatible functions with no client data dependency."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np

from .core import DMA, EMA, SMA, STD, _array, _period, _period_or_zero, _same

Array = np.ndarray


def REVERSE(X: Any) -> Array:
    return -_array(X)


def ACOS(X: Any) -> Array:
    return np.arccos(_array(X))


def ASIN(X: Any) -> Array:
    return np.arcsin(_array(X))


def ATAN(X: Any) -> Array:
    return np.arctan(_array(X))


def EXP(X: Any) -> Array:
    return np.exp(_array(X))


def LOG(X: Any) -> Array:
    return np.log10(_array(X))


def CEILING(X: Any) -> Array:
    return np.ceil(_array(X))


def FLOOR(X: Any) -> Array:
    return np.floor(_array(X))


def INTPART(X: Any) -> Array:
    return np.trunc(_array(X))


def FRACPART(X: Any) -> Array:
    values = _array(X)
    return values - np.trunc(values)


def ROUND(X: Any) -> Array:
    return np.round(_array(X))


def ROUND2(X: Any, N: int) -> Array:
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
        raise TypeError("N must be an integer")
    return np.round(_array(X), int(N))


def SIGN(X: Any) -> Array:
    return np.sign(_array(X))


def MOD(M: Any, N: Any) -> Array:
    left, right = _same(M, N)
    result = np.full(left.size, np.nan)
    np.remainder(left, right, out=result, where=np.isfinite(right) & (right != 0))
    return result


def MAX6(A: Any, B: Any, C: Any, D: Any, E: Any, F: Any) -> Array:
    return np.maximum.reduce(_same(A, B, C, D, E, F))


def MIN6(A: Any, B: Any, C: Any, D: Any, E: Any, F: Any) -> Array:
    return np.minimum.reduce(_same(A, B, C, D, E, F))


def RANGE(A: Any, B: Any, C: Any) -> Array:
    value, lower, upper = _same(A, B, C)
    return (value > lower) & (value < upper)


def IFF(X: Any, A: Any, B: Any) -> Array:
    return np.where(np.asarray(X, dtype=bool), A, B)


def IFN(X: Any, A: Any, B: Any) -> Array:
    return np.where(np.asarray(X, dtype=bool), B, A)


def NOT(X: Any) -> Array:
    values = np.asarray(X)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("X must be one-dimensional")
    return np.logical_not(values)


def ISVALID(X: Any) -> Array:
    return np.isfinite(_array(X))


def DATETODAY(DATE: Any) -> Array:
    """Convert TongDaXin dates to days since 1990-12-19."""
    values = _array(DATE)
    result = np.full(values.size, np.nan)
    epoch = date(1990, 12, 19)
    for i, value in enumerate(values):
        if np.isfinite(value) and value == int(value):
            raw = int(value) + 19_000_000
            try:
                current = date(raw // 10_000, raw // 100 % 100, raw % 100)
            except ValueError:
                continue
            result[i] = (current - epoch).days
    return result


def DAYTODATE(N: Any) -> Array:
    """Convert days since 1990-12-19 to TongDaXin date values."""
    values = _array(N)
    result = np.full(values.size, np.nan)
    epoch = date(1990, 12, 19)
    for i, value in enumerate(values):
        if np.isfinite(value) and value == int(value):
            try:
                current = epoch + timedelta(days=int(value))
            except OverflowError:
                continue
            result[i] = current.year * 10_000 + current.month * 100 + current.day - 19_000_000
    return result


def TIMETOSEC(TIME: Any) -> Array:
    values = _array(TIME)
    result = np.full(values.size, np.nan)
    for i, value in enumerate(values):
        if not np.isfinite(value) or value != int(value):
            continue
        raw = int(value)
        hour, minute, second = raw // 10_000, raw // 100 % 100, raw % 100
        if 0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60:
            result[i] = hour * 3600 + minute * 60 + second
    return result


def SECTOTIME(N: Any) -> Array:
    values = _array(N)
    result = np.full(values.size, np.nan)
    valid = np.isfinite(values) & (values == np.trunc(values)) & (values >= 0) & (values < 86400)
    seconds = values[valid].astype(int)
    result[valid] = (seconds // 3600) * 10_000 + (seconds // 60 % 60) * 100 + seconds % 60
    return result


def CON2STR(A: Any, N: int) -> str:
    digits = _period_or_zero(N)
    values = _array(A)
    if not values.size:
        raise ValueError("A must not be empty")
    return f"{values[-1]:.{digits}f}"


def VAR2STR(A: Any, N: int) -> Array:
    digits = _period_or_zero(N)
    return np.asarray([f"{value:.{digits}f}" for value in _array(A)])


def STR2CON(S: Any) -> float:
    return float(S)


def STRLEN(S: Any) -> int:
    return len(str(S))


def STRCAT(A: Any, B: Any) -> str:
    return str(A) + str(B)


def STRCAT6(A: Any, B: Any, C: Any, D: Any, E: Any, F: Any) -> str:
    return "".join(map(str, (A, B, C, D, E, F)))


def STRSPACE(A: Any) -> str:
    return f"{A} "


def SUBSTR(S: Any, A: int, N: int) -> str:
    start = _period(A, "A")
    length = _period_or_zero(N)
    return str(S)[start - 1 : start - 1 + length]


def _string_array(value: Any) -> Array:
    result = np.asarray(value, dtype=str)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1:
        raise ValueError("string series inputs must be one-dimensional")
    return result


def VARCAT(A: Any, B: Any) -> Array:
    left, right = _string_array(A), _string_array(B)
    length = max(left.size, right.size)
    if left.size not in (1, length) or right.size not in (1, length):
        raise ValueError("string series inputs must have the same length")
    return np.char.add(np.resize(left, length), np.resize(right, length))


def VARCAT6(A: Any, B: Any, C: Any, D: Any, E: Any, F: Any) -> Array:
    result = VARCAT(A, B)
    for value in (C, D, E, F):
        result = VARCAT(result, value)
    return result


def STRCMP(A: Any, B: Any) -> bool:
    return str(A) == str(B)


def FINDSTR(A: Any, B: Any) -> bool:
    return str(B) in str(A)


def BARSCOUNT(X: Any) -> Array:
    values = _array(X)
    valid = np.isfinite(values)
    result = np.zeros(values.size, dtype=int)
    if np.any(valid):
        first = int(np.argmax(valid))
        result[first:] = np.arange(1, values.size - first + 1)
    return result


def CURRBARSCOUNT(X: Any) -> Array:
    values = _array(X)
    return np.arange(values.size, 0, -1)


def TOTALBARSCOUNT(X: Any) -> Array:
    values = _array(X)
    return np.full(values.size, values.size, dtype=int)


def ISLASTBAR(X: Any) -> Array:
    values = _array(X)
    result = np.zeros(values.size, dtype=bool)
    if values.size:
        result[-1] = True
    return result


def BARSTATUS(X: Any) -> Array:
    values = _array(X)
    result = np.zeros(values.size, dtype=int)
    if values.size:
        result[0] = 1
        result[-1] = 2
    return result


def BARSSINCE(X: Any) -> Array:
    condition = np.asarray(X, dtype=bool)
    if condition.ndim != 1:
        raise ValueError("X must be one-dimensional")
    result = np.full(condition.size, np.nan)
    matches = np.flatnonzero(condition)
    if matches.size:
        first = int(matches[0])
        result[first:] = np.arange(condition.size - first)
    return result


def BARSNEXT(X: Any) -> Array:
    """Return distance to the next match; this is a future function."""
    condition = np.asarray(X, dtype=bool)
    if condition.ndim != 1:
        raise ValueError("X must be one-dimensional")
    result = np.full(condition.size, np.nan)
    distance = np.nan
    for i in range(condition.size - 1, -1, -1):
        distance = 0 if condition[i] else distance + 1
        result[i] = distance
    return result


def BACKSET(X: Any, N: int) -> Array:
    """Set the current and previous N-1 bars; this is a future function."""
    condition = np.asarray(X, dtype=bool)
    if condition.ndim != 1:
        raise ValueError("X must be one-dimensional")
    n = _period(N)
    result = np.zeros(condition.size, dtype=bool)
    for i in np.flatnonzero(condition):
        result[max(0, i - n + 1) : i + 1] = True
    return result


def ALIGNRIGHT(X: Any) -> Array:
    values = _array(X)
    result = np.full(values.size, np.nan)
    valid = values[np.isfinite(values)]
    if valid.size:
        result[-valid.size :] = valid
    return result


def REFX(X: Any, A: Any) -> Array:
    """Reference future values; this must not be used in trading signals."""
    values = _array(X)
    periods = np.asarray(A)
    if periods.ndim == 0:
        periods = np.full(values.size, periods)
    else:
        periods = _array(periods)
        if periods.size != values.size:
            raise ValueError("dynamic A must match X length")
    result = np.full(values.size, np.nan)
    for i, raw_period in enumerate(periods):
        if np.isfinite(raw_period) and raw_period == int(raw_period) and raw_period >= 0:
            target = i + int(raw_period)
            if target < values.size:
                result[i] = values[target]
    return result


def BARSLASTS(X: Any, N: Any) -> Array:
    condition = np.asarray(X, dtype=bool)
    if condition.ndim != 1:
        raise ValueError("X must be one-dimensional")
    periods = np.asarray(N)
    if periods.ndim == 0:
        periods = np.full(condition.size, periods)
    else:
        periods = _array(periods)
        if periods.size != condition.size:
            raise ValueError("dynamic N must match X length")
    result = np.full(condition.size, np.nan)
    matches: list[int] = []
    for i, (flag, raw_period) in enumerate(zip(condition, periods, strict=True)):
        if flag:
            matches.append(i)
        if np.isfinite(raw_period) and raw_period == int(raw_period) and raw_period >= 1:
            period = int(raw_period)
            if len(matches) >= period:
                result[i] = i - matches[-period]
    return result


def MULAR(X: Any, N: int) -> Array:
    values = _array(X)
    n = _period_or_zero(N)
    if n == 0:
        return np.cumprod(values, dtype=float)
    result = np.full(values.size, np.nan)
    for i in range(n - 1, values.size):
        window = values[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            result[i] = np.prod(window)
    return result


def FILTERX(X: Any, N: int) -> Array:
    n = _period(N)
    result = np.asarray(X, dtype=bool).copy()
    if result.ndim != 1:
        raise ValueError("X must be one-dimensional")
    for i in range(result.size - 1, -1, -1):
        if result[i]:
            result[max(0, i - n) : i] = False
    return result


def _thresholds(A: Any, length: int) -> Array:
    thresholds = np.asarray(A, dtype=float)
    if thresholds.ndim == 0:
        return np.full(length, float(thresholds))
    thresholds = _array(thresholds)
    if thresholds.size != length:
        raise ValueError("A must be scalar or match X length")
    return thresholds


def _sumbars(X: Any, A: Any, *, invalid_when_short: bool) -> Array:
    values = _array(X)
    thresholds = _thresholds(A, values.size)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("X must contain finite non-negative values")
    if np.any(~np.isfinite(thresholds)) or np.any(thresholds < 0):
        raise ValueError("A must contain finite non-negative values")
    result = np.full(values.size, np.nan)
    for i, threshold in enumerate(thresholds):
        if threshold == 0:
            result[i] = 0
            continue
        total = 0.0
        for start in range(i, -1, -1):
            total += values[start]
            if total >= threshold:
                result[i] = i - start + 1
                break
        if not invalid_when_short and np.isnan(result[i]):
            result[i] = i + 1
    return result


def SUMBARS(X: Any, A: Any) -> Array:
    return _sumbars(X, A, invalid_when_short=False).astype(int)


def SUMBARSX(X: Any, A: Any) -> Array:
    return _sumbars(X, A, invalid_when_short=True)


def TMA(X: Any, A: float, B: float) -> Array:
    scalar_types = (int, float, np.integer, np.floating)
    if (
        isinstance(A, bool)
        or isinstance(B, bool)
        or not isinstance(A, scalar_types)
        or not isinstance(B, scalar_types)
    ):
        raise TypeError("A and B must be real scalars")
    a, b = float(A), float(B)
    if not 0 <= a < 1 or not 0 <= b < 1:
        raise ValueError("A and B must be scalar values in [0, 1)")
    values = _array(X)
    result = np.full(values.size, np.nan)
    if values.size:
        result[0] = values[0]
    for i in range(1, values.size):
        result[i] = a * result[i - 1] + b * values[i]
    return result


def MEMA(X: Any, N: int) -> Array:
    return SMA(X, N, 1)


def AMA(X: Any, A: Any) -> Array:
    return DMA(X, A)


def EXPMEMA(X: Any, N: int) -> Array:
    n = _period(N)
    result = EMA(X, n)
    result[: n - 1] = np.nan
    return result


def UPDOWN(X: Any) -> Array:
    values = _array(X)
    result = np.zeros(values.size, dtype=int)
    if values.size > 1:
        result[1:] = np.sign(values[1:] - values[:-1]).astype(int)
    return result


def UPNDAY(X: Any, M: int) -> Array:
    values = _array(X)
    m = _period(M, "M")
    result = np.zeros(values.size, dtype=bool)
    for i in range(m, values.size):
        result[i] = np.all(np.diff(values[i - m : i + 1]) > 0)
    return result


def DOWNNDAY(X: Any, M: int) -> Array:
    values = _array(X)
    m = _period(M, "M")
    result = np.zeros(values.size, dtype=bool)
    for i in range(m, values.size):
        result[i] = np.all(np.diff(values[i - m : i + 1]) < 0)
    return result


def NDAY(X: Any, Y: Any, N: int) -> Array:
    left, right = _same(X, Y)
    n = _period(N)
    condition = left > right
    result = np.zeros(left.size, dtype=bool)
    for i in range(n - 1, left.size):
        result[i] = np.all(condition[i - n + 1 : i + 1])
    return result


def EXISTR(X: Any, A: int, B: int) -> Array:
    condition = np.asarray(X, dtype=bool)
    if condition.ndim != 1:
        raise ValueError("X must be one-dimensional")
    a = _period_or_zero(A, "A")
    b = _period_or_zero(B, "B")
    if a != 0 and a < b:
        raise ValueError("A must be zero or greater than or equal to B")
    result = np.zeros(condition.size, dtype=bool)
    for i in range(condition.size):
        start = 0 if a == 0 else i - a
        end = i - b
        if start >= 0 and end >= start:
            result[i] = np.any(condition[start : end + 1])
    return result


def TR(CLOSE: Any, HIGH: Any, LOW: Any) -> Array:
    close, high, low = _same(CLOSE, HIGH, LOW)
    previous = np.empty(close.size)
    if close.size:
        previous[0] = close[0]
        previous[1:] = close[:-1]
    return np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous)))


def _rolling_stat(X: Any, N: int, operation: str) -> Array:
    values = _array(X)
    n = _period(N)
    result = np.full(values.size, np.nan)
    for i in range(n - 1, values.size):
        window = values[i - n + 1 : i + 1]
        if not np.all(np.isfinite(window)):
            continue
        if operation == "devsq":
            result[i] = np.sum(np.square(window - np.mean(window)))
        elif operation == "var":
            result[i] = np.var(window, ddof=1) if n > 1 else np.nan
        elif operation == "varp":
            result[i] = np.var(window, ddof=0)
    return result


def DEVSQ(X: Any, N: int) -> Array:
    return _rolling_stat(X, N, "devsq")


def STDP(X: Any, N: int) -> Array:
    values = _array(X)
    n = _period(N)
    result = np.full(values.size, np.nan)
    for i in range(n - 1, values.size):
        window = values[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            result[i] = np.std(window, ddof=0)
    return result


def STDDEV(X: Any, N: int) -> Array:
    return STD(X, N)


def VAR(X: Any, N: int) -> Array:
    return _rolling_stat(X, N, "var")


def VARP(X: Any, N: int) -> Array:
    return _rolling_stat(X, N, "varp")


def _rolling_pair(X: Any, Y: Any, N: int, operation: str) -> Array:
    left, right = _same(X, Y)
    n = _period(N)
    result = np.full(left.size, np.nan)
    for i in range(n - 1, left.size):
        x = left[i - n + 1 : i + 1]
        y = right[i - n + 1 : i + 1]
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or n < 2:
            continue
        if operation == "covar":
            result[i] = np.cov(x, y, ddof=1)[0, 1]
        elif operation == "relate":
            denominator = np.std(x, ddof=0) * np.std(y, ddof=0)
            if denominator != 0:
                result[i] = np.mean((x - np.mean(x)) * (y - np.mean(y))) / denominator
        elif operation == "beta":
            denominator = np.var(y, ddof=1)
            if denominator != 0:
                result[i] = np.cov(x, y, ddof=1)[0, 1] / denominator
    return result


def COVAR(X: Any, Y: Any, N: int) -> Array:
    return _rolling_pair(X, Y, N, "covar")


def RELATE(X: Any, Y: Any, N: int) -> Array:
    return _rolling_pair(X, Y, N, "relate")


def BETAEX(X: Any, Y: Any, N: int) -> Array:
    return _rolling_pair(X, Y, N, "beta")


def HOD(X: Any, N: int) -> Array:
    return _rank_current(X, N, descending=True)


def LOD(X: Any, N: int) -> Array:
    return _rank_current(X, N, descending=False)


def _rank_current(X: Any, N: int, *, descending: bool) -> Array:
    values = _array(X)
    n = _period_or_zero(N)
    result = np.full(values.size, np.nan)
    start = 0 if n == 0 else n - 1
    for i in range(start, values.size):
        window = values[: i + 1] if n == 0 else values[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            comparison = window > values[i] if descending else window < values[i]
            result[i] = 1 + np.count_nonzero(comparison)
    return result


def HHVLLV(X: Any, T: int, N1: int, N2: int) -> Array:
    values = _array(X)
    if T not in (0, 1):
        raise ValueError("T must be 0 (highest) or 1 (lowest)")
    n1 = _period_or_zero(N1, "N1")
    n2 = _period_or_zero(N2, "N2")
    if n1 != 0 and n1 < n2:
        raise ValueError("N1 must be zero or greater than or equal to N2")
    result = np.full(values.size, np.nan)
    for i in range(values.size):
        start = 0 if n1 == 0 else i - n1
        end = i - n2
        if start >= 0 and end >= start:
            window = values[start : end + 1]
            if np.all(np.isfinite(window)):
                result[i] = np.max(window) if T == 0 else np.min(window)
    return result


def _find_extreme(X: Any, N: int, M: int, T: int, *, high: bool, bars: bool) -> Array:
    values = _array(X)
    n = _period_or_zero(N)
    m = _period(M, "M")
    t = _period(T, "T")
    if t > m:
        raise ValueError("T must be less than or equal to M")
    result = np.full(values.size, np.nan)
    for i in range(n + m - 1, values.size):
        end = i - n + 1
        start = end - m
        window = values[start:end]
        if not np.all(np.isfinite(window)):
            continue
        order = np.argsort(window, kind="stable")
        selected = order[-t] if high else order[t - 1]
        result[i] = i - (start + selected) if bars else window[selected]
    return result


def FINDHIGH(X: Any, N: int, M: int, T: int) -> Array:
    return _find_extreme(X, N, M, T, high=True, bars=False)


def FINDHIGHBARS(X: Any, N: int, M: int, T: int) -> Array:
    return _find_extreme(X, N, M, T, high=True, bars=True)


def FINDLOW(X: Any, N: int, M: int, T: int) -> Array:
    return _find_extreme(X, N, M, T, high=False, bars=False)


def FINDLOWBARS(X: Any, N: int, M: int, T: int) -> Array:
    return _find_extreme(X, N, M, T, high=False, bars=True)


__all__ = [
    "ACOS",
    "ALIGNRIGHT",
    "AMA",
    "ASIN",
    "ATAN",
    "BACKSET",
    "BARSCOUNT",
    "BARSLASTS",
    "BARSNEXT",
    "BARSSINCE",
    "BARSTATUS",
    "BETAEX",
    "CEILING",
    "CON2STR",
    "COVAR",
    "CURRBARSCOUNT",
    "DATETODAY",
    "DAYTODATE",
    "DEVSQ",
    "DOWNNDAY",
    "EXISTR",
    "EXP",
    "EXPMEMA",
    "FILTERX",
    "FINDHIGH",
    "FINDHIGHBARS",
    "FINDLOW",
    "FINDLOWBARS",
    "FINDSTR",
    "FLOOR",
    "FRACPART",
    "HHVLLV",
    "HOD",
    "IFF",
    "IFN",
    "INTPART",
    "ISLASTBAR",
    "ISVALID",
    "LOD",
    "LOG",
    "MAX6",
    "MEMA",
    "MIN6",
    "MOD",
    "MULAR",
    "NDAY",
    "NOT",
    "RANGE",
    "REFX",
    "RELATE",
    "REVERSE",
    "ROUND",
    "ROUND2",
    "SECTOTIME",
    "SIGN",
    "STDDEV",
    "STDP",
    "STR2CON",
    "STRCAT",
    "STRCAT6",
    "STRCMP",
    "STRLEN",
    "STRSPACE",
    "SUBSTR",
    "SUMBARS",
    "SUMBARSX",
    "TIMETOSEC",
    "TMA",
    "TOTALBARSCOUNT",
    "TR",
    "UPDOWN",
    "UPNDAY",
    "VAR",
    "VAR2STR",
    "VARCAT",
    "VARCAT6",
    "VARP",
]
