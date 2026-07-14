from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def market() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260715)
    close = 100 + np.cumsum(rng.normal(0.15, 1.2, 240))
    spread = rng.uniform(0.4, 2.0, close.size)
    return {
        "close": close,
        "open": close + rng.normal(0, 0.6, close.size),
        "high": close + spread,
        "low": close - spread,
        "vol": rng.integers(10_000, 1_000_000, close.size).astype(float),
    }
