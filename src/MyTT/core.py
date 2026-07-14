"""Numerical primitives and technical indicators used by :mod:`mytt`.

The public function names intentionally follow the terminology used by
TongDaXin, Wenhua and similar formula engines.  Every function accepts
array-like values and returns a NumPy array (or a tuple of arrays).
"""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray


def _array(value: Any, *, dtype: Any = float) -> Array:
    result = np.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1:
        raise ValueError("series inputs must be one-dimensional")
    return result


def _period(value: Any, name: str = "N") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _period_or_zero(value: Any, name: str = "N") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _same(*values: Any) -> tuple[Array, ...]:
    arrays = tuple(_array(v) for v in values)
    length = max((len(v) for v in arrays), default=0)
    if any(len(v) not in (1, length) for v in arrays):
        raise ValueError("series inputs must have the same length")
    return tuple(
        np.full(length, v[0], dtype=float) if len(v) == 1 and length != 1 else v for v in arrays
    )


def _rolling(series: Array, window: int, operation: str) -> Array:
    window = _period(window)
    out = np.full(series.size, np.nan, dtype=float)
    if window > series.size:
        return out
    for end in range(window - 1, series.size):
        values = series[end - window + 1 : end + 1]
        if operation == "sum":
            out[end] = np.sum(values) if np.all(np.isfinite(values)) else np.nan
        elif operation == "mean":
            out[end] = np.mean(values) if np.all(np.isfinite(values)) else np.nan
        elif operation == "std":
            out[end] = (
                np.std(values, ddof=1)
                if values.size > 1 and np.all(np.isfinite(values))
                else np.nan
            )
        elif operation == "stdp":
            out[end] = np.std(values, ddof=0) if np.all(np.isfinite(values)) else np.nan
        elif operation == "max":
            out[end] = np.max(values) if np.all(np.isfinite(values)) else np.nan
        elif operation == "min":
            out[end] = np.min(values) if np.all(np.isfinite(values)) else np.nan
        elif operation == "mad":
            out[end] = (
                np.mean(np.abs(values - np.mean(values))) if np.all(np.isfinite(values)) else np.nan
            )
        else:
            raise ValueError(f"unsupported rolling operation: {operation}")
    return out


def _safe_divide(numerator: Any, denominator: Any) -> Array:
    num, den = _same(numerator, denominator)
    result = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, den, out=result, where=np.isfinite(den) & (den != 0))
    return result


def RD(N: Any, D: int = 3) -> Array:
    """Round values to ``D`` decimal places."""
    if isinstance(D, bool) or not isinstance(D, (int, np.integer)):
        raise TypeError("D must be an integer")
    return np.round(N, int(D))


def RET(S: Any, N: int = 1) -> Any:
    """Return the Nth value counted from the end (N starts at one)."""
    n = _period(N, "N")
    values = _array(S)
    if n > values.size:
        raise IndexError("N is larger than the series length")
    return values[-n]


def ABS(S: Any) -> Array:
    return np.abs(_array(S))


def LN(S: Any) -> Array:
    return np.log(_array(S))


def POW(S: Any, N: Any) -> Array:
    return np.power(_array(S), N)


def SQRT(S: Any) -> Array:
    return np.sqrt(_array(S))


def SIN(S: Any) -> Array:
    return np.sin(_array(S))


def COS(S: Any) -> Array:
    return np.cos(_array(S))


def TAN(S: Any) -> Array:
    return np.tan(_array(S))


def MAX(S1: Any, S2: Any) -> Array:
    a, b = _same(S1, S2)
    return np.maximum(a, b)


def MIN(S1: Any, S2: Any) -> Array:
    a, b = _same(S1, S2)
    return np.minimum(a, b)


def IF(S: Any, A: Any, B: Any) -> Array:
    condition = np.asarray(S, dtype=bool)
    return np.where(condition, A, B)


def REF(S: Any, N: Any = 1) -> Array:
    values = _array(S)
    result = np.full(values.size, np.nan, dtype=float)
    if np.asarray(N).ndim == 0:
        n = _period_or_zero(N)
        if n == 0:
            return values.copy()
        if n < values.size:
            result[n:] = values[:-n]
        return result
    periods = _array(N)
    if periods.size != values.size:
        raise ValueError("dynamic N must match S length")
    for i, raw_period in enumerate(periods):
        if np.isfinite(raw_period) and raw_period == int(raw_period) and 0 <= raw_period <= i:
            result[i] = values[i - int(raw_period)]
    return result


def DIFF(S: Any, N: int = 1) -> Array:
    n = _period(N)
    values = _array(S)
    result = np.full(values.size, np.nan, dtype=float)
    if n < values.size:
        result[n:] = values[n:] - values[:-n]
    return result


def STD(S: Any, N: int) -> Array:
    return _rolling(_array(S), N, "std")


def SUM(S: Any, N: int) -> Array:
    values = _array(S)
    n = _period_or_zero(N)
    if n == 0:
        return np.cumsum(values, dtype=float)
    return _rolling(values, n, "sum")


def CONST(S: Any) -> Array:
    values = _array(S)
    if not values.size:
        raise ValueError("S must not be empty")
    return np.full(values.size, values[-1], dtype=float)


def _dynamic_extreme(S: Any, N: Any, operation: str) -> Array:
    values = _array(S)
    if np.asarray(N).ndim == 0:
        if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
            raise TypeError("N must be a non-negative integer or a period series")
        n = _period_or_zero(N)
        if n == 0:
            return (
                np.maximum.accumulate(values)
                if operation == "max"
                else np.minimum.accumulate(values)
            )
        return _rolling(values, n, operation)
    periods = _array(N)
    if periods.size != values.size:
        raise ValueError("dynamic N must match S length")
    out = np.full(values.size, np.nan)
    for i, raw_period in enumerate(periods):
        if not np.isfinite(raw_period) or raw_period != int(raw_period) or raw_period < 1:
            continue
        period = int(raw_period)
        if period <= i + 1:
            window = values[i - period + 1 : i + 1]
            if np.all(np.isfinite(window)):
                out[i] = np.max(window) if operation == "max" else np.min(window)
    return out


def HHV(S: Any, N: Any) -> Array:
    return _dynamic_extreme(S, N, "max")


def LLV(S: Any, N: Any) -> Array:
    return _dynamic_extreme(S, N, "min")


def HHVBARS(S: Any, N: int) -> Array:
    values = _array(S)
    n = _period_or_zero(N)
    out = np.full(values.size, np.nan)
    start = 0 if n == 0 else n - 1
    for i in range(start, values.size):
        window = values[: i + 1] if n == 0 else values[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = int(np.argmax(window[::-1]))
    return out


def LLVBARS(S: Any, N: int) -> Array:
    values = _array(S)
    n = _period_or_zero(N)
    out = np.full(values.size, np.nan)
    start = 0 if n == 0 else n - 1
    for i in range(start, values.size):
        window = values[: i + 1] if n == 0 else values[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = int(np.argmin(window[::-1]))
    return out


def MA(S: Any, N: int) -> Array:
    return _rolling(_array(S), N, "mean")


def EMA(S: Any, N: int) -> Array:
    values = _array(S)
    n = _period(N)
    alpha = 2.0 / (n + 1.0)
    out = np.full(values.size, np.nan)
    if not values.size:
        return out
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = (
            values[i]
            if not np.isfinite(out[i - 1])
            else alpha * values[i] + (1 - alpha) * out[i - 1]
        )
    return out


def SMA(S: Any, N: int, M: int = 1) -> Array:
    n = _period(N)
    m = _period(M, "M")
    if m > n:
        raise ValueError("M must be less than or equal to N")
    values = _array(S)
    alpha = m / n
    out = np.full(values.size, np.nan)
    if values.size:
        out[0] = values[0]
    for i in range(1, values.size):
        out[i] = (
            values[i]
            if not np.isfinite(out[i - 1])
            else alpha * values[i] + (1 - alpha) * out[i - 1]
        )
    return out


def WMA(S: Any, N: int) -> Array:
    values = _array(S)
    n = _period(N)
    out = np.full(values.size, np.nan)
    weights = np.arange(1, n + 1, dtype=float)
    denominator = weights.sum()
    for i in range(n - 1, values.size):
        window = values[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = np.dot(window, weights) / denominator
    return out


def DMA(S: Any, A: Any) -> Array:
    values = _array(S)
    alpha = np.asarray(A, dtype=float)
    if alpha.ndim == 0:
        alpha = np.full(values.size, float(alpha))
    else:
        alpha = _array(alpha)
        if alpha.size != values.size:
            raise ValueError("A must be scalar or match S length")
    alpha = np.where(np.isnan(alpha), 1.0, alpha)
    if np.any(~np.isfinite(alpha)) or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("A must contain values between 0 and 1")
    out = np.full(values.size, np.nan)
    if values.size:
        out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha[i] * values[i] + (1 - alpha[i]) * out[i - 1]
    return out


def AVEDEV(S: Any, N: int) -> Array:
    return _rolling(_array(S), N, "mad")


def SLOPE(S: Any, N: int) -> Array:
    values = _array(S)
    n = _period(N)
    out = np.full(values.size, np.nan)
    if n == 1:
        out[np.isfinite(values)] = 0.0
        return out
    x = np.arange(n, dtype=float)
    x -= x.mean()
    denominator = np.dot(x, x)
    for i in range(n - 1, values.size):
        window = values[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = np.dot(x, window - window.mean()) / denominator
    return out


def FORCAST(S: Any, N: int) -> Array:
    values = _array(S)
    n = _period(N)
    slope = SLOPE(values, n)
    out = np.full(values.size, np.nan)
    for i in range(n - 1, values.size):
        window = values[i - n + 1 : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = window.mean() + slope[i] * ((n - 1) - (n - 1) / 2)
    return out


def LAST(S: Any, A: int, B: int) -> Array:
    a = _period(A, "A")
    b = int(B)
    if b < 0 or b > a:
        raise ValueError("B must satisfy 0 <= B <= A")
    values = np.asarray(S, dtype=bool)
    if values.ndim != 1:
        raise ValueError("S must be one-dimensional")
    out = np.zeros(values.size, dtype=bool)
    for i in range(a, values.size):
        out[i] = np.all(values[i - a : i - b + 1])
    return out


def COUNT(S: Any, N: int) -> Array:
    return SUM(np.asarray(S, dtype=float), N)


def EVERY(S: Any, N: int) -> Array:
    return SUM(np.asarray(S, dtype=float), N) == _period(N)


def EXIST(S: Any, N: int) -> Array:
    return SUM(np.asarray(S, dtype=float), N) > 0


def FILTER(S: Any, N: int) -> Array:
    n = _period(N)
    out = np.asarray(S, dtype=bool).copy()
    if out.ndim != 1:
        raise ValueError("S must be one-dimensional")
    for i in range(out.size):
        if out[i]:
            out[i + 1 : i + 1 + n] = False
    return out


def BARSLAST(S: Any) -> Array:
    values = np.asarray(S, dtype=bool)
    if values.ndim != 1:
        raise ValueError("S must be one-dimensional")
    out = np.zeros(values.size, dtype=int)
    distance = 0
    for i, flag in enumerate(values):
        distance = 0 if flag else distance + 1
        out[i] = distance
    return out


def BARSLASTCOUNT(S: Any) -> Array:
    values = np.asarray(S, dtype=bool)
    out = np.zeros(values.size, dtype=int)
    count = 0
    for i, flag in enumerate(values):
        count = count + 1 if flag else 0
        out[i] = count
    return out


def BARSSINCEN(S: Any, N: int) -> Array:
    values = np.asarray(S, dtype=bool)
    n = _period(N)
    out = np.full(values.size, np.nan)
    for i in range(n - 1, values.size):
        window = values[i - n + 1 : i + 1]
        if np.any(window):
            out[i] = n - 1 - int(np.argmax(window))
    return out


def CROSS(S1: Any, S2: Any) -> Array:
    left, right = _same(S1, S2)
    current = left > right
    out = np.zeros(left.size, dtype=bool)
    if left.size > 1:
        out[1:] = (~current[:-1]) & current[1:]
    return out


def LONGCROSS(S1: Any, S2: Any, N: int) -> Array:
    n = _period(N)
    left, right = _same(S1, S2)
    out = np.zeros(left.size, dtype=bool)
    for i in range(n, left.size):
        out[i] = np.all(left[i - n : i] < right[i - n : i]) and left[i] > right[i]
    return out


def VALUEWHEN(S: Any, X: Any) -> Array:
    condition = np.asarray(S, dtype=bool)
    values = _array(X)
    if condition.ndim != 1 or condition.size != values.size:
        raise ValueError("S and X must have the same length")
    out = np.full(values.size, np.nan)
    last = np.nan
    for i, flag in enumerate(condition):
        if flag:
            last = values[i]
        out[i] = last
    return out


def BETWEEN(S: Any, A: Any, B: Any) -> Array:
    values, a, b = _same(S, A, B)
    return ((a <= values) & (values <= b)) | ((a >= values) & (values >= b))


def TOPRANGE(S: Any) -> Array:
    values = _array(S)
    out = np.zeros(values.size, dtype=int)
    for i in range(1, values.size):
        j = i - 1
        while j >= 0 and values[j] <= values[i]:
            j -= 1
        out[i] = i - j - 1
    return out


def LOWRANGE(S: Any) -> Array:
    values = _array(S)
    out = np.zeros(values.size, dtype=int)
    for i in range(1, values.size):
        j = i - 1
        while j >= 0 and values[j] >= values[i]:
            j -= 1
        out[i] = i - j - 1
    return out


def MACD(CLOSE: Any, SHORT: int = 12, LONG: int = 26, M: int = 9):
    dif = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
    dea = EMA(dif, M)
    return RD(dif), RD(dea), RD((dif - dea) * 2)


def KDJ(CLOSE: Any, HIGH: Any, LOW: Any, N: int = 9, M1: int = 3, M2: int = 3):
    close, high, low = _same(CLOSE, HIGH, LOW)
    denominator = HHV(high, N) - LLV(low, N)
    rsv = _safe_divide((close - LLV(low, N)) * 100, denominator)
    k = EMA(rsv, 2 * _period(M1, "M1") - 1)
    d = EMA(k, 2 * _period(M2, "M2") - 1)
    return k, d, 3 * k - 2 * d


def RSI(CLOSE: Any, N: int = 24) -> Array:
    close = _array(CLOSE)
    dif = close - REF(close, 1)
    return RD(_safe_divide(SMA(np.maximum(dif, 0), N), SMA(np.abs(dif), N)) * 100)


def WR(CLOSE: Any, HIGH: Any, LOW: Any, N: int = 10, N1: int = 6):
    close, high, low = _same(CLOSE, HIGH, LOW)
    return RD(_safe_divide(HHV(high, N) - close, HHV(high, N) - LLV(low, N)) * 100), RD(
        _safe_divide(HHV(high, N1) - close, HHV(high, N1) - LLV(low, N1)) * 100
    )


def BIAS(CLOSE: Any, L1: int = 6, L2: int = 12, L3: int = 24):
    close = _array(CLOSE)
    return tuple(RD(_safe_divide(close - MA(close, n), MA(close, n)) * 100) for n in (L1, L2, L3))


def BOLL(CLOSE: Any, N: int = 20, P: float = 2):
    close = _array(CLOSE)
    mid = MA(close, N)
    deviation = STD(close, N) * P
    return RD(mid + deviation), RD(mid), RD(mid - deviation)


def PSY(CLOSE: Any, N: int = 12, M: int = 6):
    close = _array(CLOSE)
    psy = COUNT(close > REF(close, 1), N) / N * 100
    return RD(psy), RD(MA(psy, M))


def CCI(CLOSE: Any, HIGH: Any, LOW: Any, N: int = 14) -> Array:
    close, high, low = _same(CLOSE, HIGH, LOW)
    tp = (high + low + close) / 3
    return _safe_divide(tp - MA(tp, N), 0.015 * AVEDEV(tp, N))


def ATR(CLOSE: Any, HIGH: Any, LOW: Any, N: int = 20) -> Array:
    close, high, low = _same(CLOSE, HIGH, LOW)
    prev = REF(close, 1)
    tr = np.maximum.reduce((high - low, np.abs(prev - high), np.abs(prev - low)))
    return MA(tr, N)


def BBI(CLOSE: Any, M1: int = 3, M2: int = 6, M3: int = 12, M4: int = 20) -> Array:
    close = _array(CLOSE)
    return (MA(close, M1) + MA(close, M2) + MA(close, M3) + MA(close, M4)) / 4


def DMI(CLOSE: Any, HIGH: Any, LOW: Any, M1: int = 14, M2: int = 6):
    close, high, low = _same(CLOSE, HIGH, LOW)
    prev_close = REF(close, 1)
    tr = SUM(
        np.maximum.reduce((high - low, np.abs(high - prev_close), np.abs(low - prev_close))), M1
    )
    hd = high - REF(high, 1)
    ld = REF(low, 1) - low
    dmp = SUM(IF((hd > 0) & (hd > ld), hd, 0), M1)
    dmm = SUM(IF((ld > 0) & (ld > hd), ld, 0), M1)
    pdi = _safe_divide(dmp * 100, tr)
    mdi = _safe_divide(dmm * 100, tr)
    adx = MA(_safe_divide(np.abs(mdi - pdi) * 100, pdi + mdi), M2)
    return pdi, mdi, adx, (adx + REF(adx, M2)) / 2


def TAQ(HIGH: Any, LOW: Any, N: int):
    high, low = _same(HIGH, LOW)
    up, down = HHV(high, N), LLV(low, N)
    return up, (up + down) / 2, down


def KTN(CLOSE: Any, HIGH: Any, LOW: Any, N: int = 20, M: int = 10):
    close, high, low = _same(CLOSE, HIGH, LOW)
    mid = EMA((high + low + close) / 3, N)
    atr = ATR(close, high, low, M)
    return mid + 2 * atr, mid, mid - 2 * atr


def TRIX(CLOSE: Any, M1: int = 12, M2: int = 20):
    close = _array(CLOSE)
    tr = EMA(EMA(EMA(close, M1), M1), M1)
    trix = _safe_divide(tr - REF(tr, 1), REF(tr, 1)) * 100
    return trix, MA(trix, M2)


def VR(CLOSE: Any, VOL: Any, M1: int = 26) -> Array:
    close, vol = _same(CLOSE, VOL)
    previous = REF(close, 1)
    return (
        _safe_divide(SUM(IF(close > previous, vol, 0), M1), SUM(IF(close <= previous, vol, 0), M1))
        * 100
    )


def CR(CLOSE: Any, HIGH: Any, LOW: Any, N: int = 20) -> Array:
    close, high, low = _same(CLOSE, HIGH, LOW)
    mid = REF(high + low + close, 1) / 3
    return _safe_divide(SUM(np.maximum(0, high - mid), N), SUM(np.maximum(0, mid - low), N)) * 100


def EMV(HIGH: Any, LOW: Any, VOL: Any, N: int = 14, M: int = 9):
    high, low, vol = _same(HIGH, LOW, VOL)
    volume = _safe_divide(MA(vol, N), vol)
    mid = _safe_divide(100 * (high + low - REF(high + low, 1)), high + low)
    emv = MA(_safe_divide(mid * volume * (high - low), MA(high - low, N)), N)
    return emv, MA(emv, M)


def DPO(CLOSE: Any, M1: int = 20, M2: int = 10, M3: int = 6):
    close = _array(CLOSE)
    dpo = close - REF(MA(close, M1), M2)
    return dpo, MA(dpo, M3)


def BRAR(OPEN: Any, CLOSE: Any, HIGH: Any, LOW: Any, M1: int = 26):
    open_, close, high, low = _same(OPEN, CLOSE, HIGH, LOW)
    prev = REF(close, 1)
    return _safe_divide(SUM(high - open_, M1), SUM(open_ - low, M1)) * 100, _safe_divide(
        SUM(np.maximum(0, high - prev), M1), SUM(np.maximum(0, prev - low), M1)
    ) * 100


def DFMA(CLOSE: Any, N1: int = 10, N2: int = 50, M: int = 10):
    close = _array(CLOSE)
    dif = MA(close, N1) - MA(close, N2)
    return dif, MA(dif, M)


def MTM(CLOSE: Any, N: int = 12, M: int = 6):
    close = _array(CLOSE)
    mtm = close - REF(close, N)
    return mtm, MA(mtm, M)


def MASS(HIGH: Any, LOW: Any, N1: int = 9, N2: int = 25, M: int = 6):
    high, low = _same(HIGH, LOW)
    spread = high - low
    mass = SUM(_safe_divide(MA(spread, N1), MA(MA(spread, N1), N1)), N2)
    return mass, MA(mass, M)


def ROC(CLOSE: Any, N: int = 12, M: int = 6):
    close = _array(CLOSE)
    roc = _safe_divide(100 * (close - REF(close, N)), REF(close, N))
    return roc, MA(roc, M)


def EXPMA(CLOSE: Any, N1: int = 12, N2: int = 50):
    close = _array(CLOSE)
    return EMA(close, N1), EMA(close, N2)


def OBV(CLOSE: Any, VOL: Any) -> Array:
    close, vol = _same(CLOSE, VOL)
    previous = REF(close, 1)
    return SUM(IF(close > previous, vol, IF(close < previous, -vol, 0)), 0) / 10000


def MFI(CLOSE: Any, HIGH: Any, LOW: Any, VOL: Any, N: int = 14) -> Array:
    close, high, low, vol = _same(CLOSE, HIGH, LOW, VOL)
    typ = (high + low + close) / 3
    previous = REF(typ, 1)
    ratio = _safe_divide(
        SUM(IF(typ > previous, typ * vol, 0), N), SUM(IF(typ < previous, typ * vol, 0), N)
    )
    return 100 - _safe_divide(100, 1 + ratio)


def ASI(OPEN: Any, CLOSE: Any, HIGH: Any, LOW: Any, M1: int = 26, M2: int = 10):
    open_, close, high, low = _same(OPEN, CLOSE, HIGH, LOW)
    lc = REF(close, 1)
    aa = np.abs(high - lc)
    bb = np.abs(low - lc)
    cc = np.abs(high - REF(low, 1))
    dd = np.abs(lc - REF(open_, 1))
    r = IF(
        (aa > bb) & (aa > cc),
        aa + bb / 2 + dd / 4,
        IF((bb > cc) & (bb > aa), bb + aa / 2 + dd / 4, cc + dd / 4),
    )
    x = close - lc + (close - open_) / 2 + lc - REF(open_, 1)
    si = _safe_divide(16 * x * np.maximum(aa, bb), r)
    asi = SUM(si, M1)
    return asi, MA(asi, M2)


def XSII(CLOSE: Any, HIGH: Any, LOW: Any, N: int = 102, M: int = 7):
    close, high, low = _same(CLOSE, HIGH, LOW)
    aa = MA((2 * close + high + low) / 4, 5)
    td1 = aa * N / 100
    td2 = aa * (200 - N) / 100
    cc = _safe_divide(np.abs((2 * close + high + low) / 4 - MA(close, 20)), MA(close, 20))
    dd = DMA(close, cc)
    return td1, td2, (1 + M / 100) * dd, (1 - M / 100) * dd


__all__ = [
    "ABS",
    "ASI",
    "ATR",
    "AVEDEV",
    "BARSLAST",
    "BARSLASTCOUNT",
    "BARSSINCEN",
    "BBI",
    "BETWEEN",
    "BIAS",
    "BOLL",
    "BRAR",
    "CCI",
    "CONST",
    "COS",
    "COUNT",
    "CR",
    "CROSS",
    "DFMA",
    "DIFF",
    "DMA",
    "DMI",
    "DPO",
    "EMA",
    "EMV",
    "EVERY",
    "EXIST",
    "EXPMA",
    "FILTER",
    "FORCAST",
    "HHV",
    "HHVBARS",
    "IF",
    "KDJ",
    "KTN",
    "LAST",
    "LLV",
    "LLVBARS",
    "LN",
    "LONGCROSS",
    "LOWRANGE",
    "MA",
    "MACD",
    "MASS",
    "MAX",
    "MFI",
    "MIN",
    "MTM",
    "OBV",
    "POW",
    "PSY",
    "RD",
    "REF",
    "RET",
    "ROC",
    "RSI",
    "SIN",
    "SLOPE",
    "SMA",
    "SQRT",
    "STD",
    "SUM",
    "TAN",
    "TAQ",
    "TOPRANGE",
    "TRIX",
    "VALUEWHEN",
    "VR",
    "WMA",
    "WR",
    "XSII",
]
