# MyTT

MyTT is a typed, NumPy-first implementation of the technical-analysis
formulas commonly used by TongDaXin, Wenhua and 同花顺. It targets Python
3.12+ and has no TA-Lib or pandas dependency.

## Install

```bash
python -m pip install MyTT
```

The package exposes the canonical formula names directly:

```python
import MyTT

close = [10, 10.5, 11, 10.8, 11.4]
fast, slow, histogram = MyTT.MACD(close)
signal = MyTT.CROSS(MyTT.MA(close, 5), MyTT.MA(close, 10))
```

## API rules

- Inputs are one-dimensional array-like values. Outputs are `numpy.ndarray`
  instances, with one output value per input row.
- Rolling functions use an explicit warm-up policy: values are `nan` until a
  complete window is available. This avoids silently treating incomplete
  history as valid data.
- Division by zero produces `nan`; it never produces an unbounded `inf`.
- Inputs are never modified in place. In particular, `FILTER` returns a new
  boolean array.
- Invalid periods, mismatched lengths and invalid smoothing factors raise
  `TypeError` or `ValueError` with an actionable message.

## Available formulas

The core layer includes `MA`, `EMA`, `SMA`, `WMA`, `DMA`, `REF`, `SUM`, `HHV`,
`LLV`, `CROSS`, `BARSLAST`, `VALUEWHEN`, `MACD`, `KDJ`, `RSI`, `BOLL`, `CCI`,
`ATR`, `DMI`, `TRIX`, `MFI`, `OBV`, `ASI`, `XSII` and the other common
TongDaXin formula names. Stateful extensions `DSMA`, `SAR`, `TDX_SAR` and
`SUMBARSFAST` are exported from the same namespace.

See the function docstrings and the tests in `tests/` for exact warm-up,
missing-data and boundary semantics.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check src tests
ruff format --check src tests
mypy src
python -m build
```

The test suite requires at least 90% branch-aware coverage and currently
exceeds that threshold.
