from __future__ import annotations

import numpy as np
import pytest

from MyTT import DSMA, SAR, SUMBARSFAST, TDX_SAR


def test_sumbarsfast_scalar_and_dynamic_threshold() -> None:
    np.testing.assert_array_equal(SUMBARSFAST([1, 2, 3, 4], 3), [0, 2, 1, 1])
    np.testing.assert_array_equal(SUMBARSFAST([1, 2, 3, 4], [1, 3, 5, 10]), [1, 2, 2, 4])


def test_stateful_indicators_return_stable_shapes(market) -> None:
    dsma = DSMA(market["close"], 20)
    sar = SAR(market["high"], market["low"], 10)
    tdx_sar = TDX_SAR(market["high"], market["low"])
    assert dsma.shape == sar.shape == tdx_sar.shape == market["close"].shape
    assert np.isfinite(dsma[-1])
    assert np.isfinite(sar[-1])
    assert np.all(np.isfinite(tdx_sar))


def test_tdx_sar_first_value_and_empty_input() -> None:
    assert TDX_SAR([], []).size == 0
    assert TDX_SAR([2, 3], [1, 2])[0] == 1


@pytest.mark.parametrize("function", [SAR, TDX_SAR])
def test_sar_parameter_validation(function) -> None:
    with pytest.raises(ValueError, match=r"step|S and M"):
        function([2, 3], [1, 2], 10, 30, 20) if function is SAR else function(
            [2, 3], [1, 2], 30, 20
        )


def test_sumbarsfast_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        SUMBARSFAST([1, 0], 2)
    with pytest.raises(ValueError, match="match X length"):
        SUMBARSFAST([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="non-negative"):
        SUMBARSFAST([1, 2], -1)
