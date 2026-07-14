"""Advanced indicators that require stateful algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import DMA, SUM, _array, _period, _same


def DSMA(X: Any, N: int) -> np.ndarray:
    """Deviation-scaled moving average."""
    values = _array(X)
    n = _period(N)
    a1 = np.exp(-1.414 * np.pi * 2 / n)
    b1 = 2 * a1 * np.cos(1.414 * np.pi * 2 / n)
    c2, c3 = b1, -(a1**2)
    c1 = 1 - c2 - c3
    zeros = np.pad(values[2:] - values[:-2], (2, 0))
    filtered = np.zeros(values.size)
    for i in range(values.size):
        previous = filtered[i - 1] if i >= 1 else 0.0
        previous2 = filtered[i - 2] if i >= 2 else 0.0
        zero_previous = zeros[i - 1] if i >= 1 else 0.0
        filtered[i] = c1 * (zeros[i] + zero_previous) / 2 + c2 * previous + c3 * previous2
    rms = np.sqrt(SUM(np.square(filtered), n) / n)
    alpha = np.full(values.size, np.nan)
    np.divide(np.abs(filtered) * 5, n * rms, out=alpha, where=np.isfinite(rms) & (rms != 0))
    return DMA(values, np.clip(alpha, 0, 1))


def SUMBARSFAST(X: Any, A: Any) -> np.ndarray:
    """Return periods needed for the backward sum of ``X`` to reach ``A``."""
    values = _array(X)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("X must contain finite positive values")
    threshold = np.asarray(A, dtype=float)
    if threshold.ndim == 0:
        threshold = np.full(values.size, float(threshold))
    else:
        threshold = _array(threshold)
        if threshold.size != values.size:
            raise ValueError("A must be scalar or match X length")
    if np.any(~np.isfinite(threshold)) or np.any(threshold < 0):
        raise ValueError("A must contain finite non-negative values")
    result = np.zeros(values.size, dtype=int)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    for i in range(values.size):
        target = cumulative[i + 1] - threshold[i]
        start = int(np.searchsorted(cumulative[: i + 1], target, side="right") - 1)
        if start >= 0 and cumulative[i + 1] - cumulative[start] >= threshold[i]:
            result[i] = i - start + 1
    return result


def SAR(HIGH: Any, LOW: Any, N: int = 10, S: float = 2, M: float = 20) -> np.ndarray:
    """Parabolic stop-and-reverse indicator with an N-period extreme point."""
    high, low = _same(HIGH, LOW)
    n = _period(N)
    if not 0 < S <= M <= 100:
        raise ValueError("S and M must satisfy 0 < S <= M <= 100")
    result = np.full(high.size, np.nan)
    if high.size <= n:
        return result
    step, maximum = S / 100, M / 100
    is_long = high[n - 1] > high[n - 2]
    first = True
    acceleration = step
    for i in range(n, high.size):
        recent_high = np.max(high[i - n : i])
        recent_low = np.min(low[i - n : i])
        if first:
            acceleration = step
            result[i] = recent_low if is_long else recent_high
            first = False
        else:
            extreme = recent_high if is_long else recent_low
            prior_extreme = np.max(high[i - n : i - 1]) if is_long else np.min(low[i - n : i - 1])
            if (is_long and extreme > prior_extreme) or (not is_long and extreme < prior_extreme):
                acceleration = min(acceleration + step, maximum)
            result[i] = result[i - 1] + acceleration * (extreme - result[i - 1])
        if (is_long and low[i] < result[i]) or (not is_long and high[i] > result[i]):
            is_long = not is_long
            first = True
    return result


def TDX_SAR(High: Any, Low: Any, iAFStep: float = 2, iAFLimit: float = 20) -> np.ndarray:
    """TongDaXin-compatible parabolic SAR."""
    high, low = _same(High, Low)
    if not 0 < iAFStep <= iAFLimit <= 100:
        raise ValueError("iAFStep and iAFLimit must satisfy 0 < step <= limit <= 100")
    if not high.size:
        return np.array([], dtype=float)
    step, limit = iAFStep / 100, iAFLimit / 100
    result = np.zeros(high.size)
    bull, acceleration, extreme = True, step, high[0]
    result[0] = low[0]
    for i in range(1, high.size):
        if bull and high[i] > extreme:
            extreme = high[i]
            acceleration = min(acceleration + step, limit)
        elif not bull and low[i] < extreme:
            extreme = low[i]
            acceleration = min(acceleration + step, limit)
        result[i] = result[i - 1] + acceleration * (extreme - result[i - 1])
        if bull:
            result[i] = max(result[i - 1], min(result[i], low[i], low[i - 1]))
        else:
            result[i] = min(result[i - 1], max(result[i], high[i], high[i - 1]))
        if bull and low[i] < result[i]:
            bull = False
            prior_extreme = extreme
            extreme, acceleration = low[i], step
            result[i] = (
                prior_extreme
                if high[i - 1] == prior_extreme
                else prior_extreme + acceleration * (extreme - prior_extreme)
            )
        elif not bull and high[i] > result[i]:
            bull = True
            extreme, acceleration = high[i], step
            result[i] = min(low[i], low[i - 1])
    return result


__all__ = ["DSMA", "SAR", "SUMBARSFAST", "TDX_SAR"]
