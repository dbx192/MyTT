# MyTT_plus

MyTT_plus 是一个基于 NumPy 的通达信、同花顺和文华财经技术分析公式函数库。
它提供常用指标、序列计算、条件判断、统计、日期时间和字符串函数，适合公式迁移、
行情研究、量化回测和策略原型开发。

项目不依赖 TA-Lib 和 pandas，支持 Python 3.12 及以上版本。

> PyPI 发行包名是 `MyTT_plus`，安装后的 Python 导入名仍为 `MyTT`。
> 保留原导入名是为了兼容已有代码。

## 安装

```bash
python -m pip install MyTT_plus
```

升级到最新版本：

```bash
python -m pip install --upgrade MyTT_plus
```

确认安装结果：

```python
import MyTT

print(MyTT.__version__)
```

## 快速开始

```python
import numpy as np
import MyTT

close = np.array([10.0, 10.5, 11.0, 10.8, 11.4, 11.9])
high = close + 0.3
low = close - 0.3

# 常用指标
dif, dea, macd = MyTT.MACD(close)
k, d, j = MyTT.KDJ(close, high, low)

# 通达信式序列公式
ma5 = MyTT.MA(close, 5)
signal = MyTT.CROSS(ma5, MyTT.MA(close, 10))

# 动态引用周期
previous = MyTT.REF(close, [0, 1, 1, 2, 2, 3])

# N=0 表示从第一个有效值累计
history_high = MyTT.HHV(close, 0)
```

## API 约定

- 输入一般是一维 list、tuple 或 `numpy.ndarray`。
- 序列函数返回与输入等长的 `numpy.ndarray`。
- 窗口历史不足时返回 `nan`，不会把不完整窗口误当成有效结果。
- 除数为零时返回 `nan`，不会生成无限值。
- 函数不会原地修改输入数组。
- 非法周期、长度不一致或无效平滑参数会抛出明确的 `TypeError` 或 `ValueError`。
- `OPEN`、`HIGH`、`LOW`、`CLOSE`、`VOL` 等行情数据需要调用者显式传入。

## 主要能力

### 基础序列和条件函数

包括 `MA`、`EMA`、`SMA`、`WMA`、`DMA`、`REF`、`SUM`、`HHV`、`LLV`、
`COUNT`、`EVERY`、`EXIST`、`FILTER`、`CROSS`、`LONGCROSS`、`BARSLAST`、
`VALUEWHEN`、`BETWEEN` 等。

### 常用技术指标

包括 `MACD`、`KDJ`、`RSI`、`BOLL`、`CCI`、`ATR`、`DMI`、`TRIX`、`MFI`、
`OBV`、`ASI`、`BIAS`、`WR`、`PSY`、`BBI`、`VR`、`CR`、`EMV`、`ROC`、
`SAR`、`TDX_SAR` 等。

### 通达信兼容扩展

4.1.0 新增 77 个可独立计算的通达信函数，主要包括：

- 数学与逻辑：`ACOS`、`ASIN`、`ATAN`、`EXP`、`LOG`、`CEILING`、`FLOOR`、
  `INTPART`、`FRACPART`、`ROUND`、`ROUND2`、`SIGN`、`MOD`、`MAX6`、`MIN6`、
  `REVERSE`、`RANGE`、`IFF`、`IFN`、`NOT`、`ISVALID`。
- 窗口与定位：`BARSCOUNT`、`CURRBARSCOUNT`、`TOTALBARSCOUNT`、`BARSTATUS`、
  `BARSSINCE`、`BARSLASTS`、`MULAR`、`FILTERX`、`SUMBARS`、`SUMBARSX`、
  `HOD`、`LOD`、`HHVLLV`、`FINDHIGH`、`FINDLOW` 及其位置函数。
- 条件与趋势：`UPDOWN`、`UPNDAY`、`DOWNNDAY`、`NDAY`、`EXISTR`。
- 均线与统计：`TMA`、`MEMA`、`AMA`、`EXPMEMA`、`TR`、`DEVSQ`、`STDP`、
  `STDDEV`、`VAR`、`VARP`、`COVAR`、`RELATE`、`BETAEX`。
- 日期和字符串：`DATETODAY`、`DAYTODATE`、`TIMETOSEC`、`SECTOTIME`、
  `CON2STR`、`VAR2STR`、`STR2CON`、`STRCAT`、`VARCAT`、`SUBSTR`、`FINDSTR` 等。

当前公开 152 个公式和指标名称，其中 119 个名称与通达信官网函数表直接对应。

## 4.1.0 改动说明

### 新增函数

- 新增独立的通达信兼容层 `MyTT.tdx`，并从 `MyTT` 顶层统一导出。
- 增加数学、统计、周期定位、条件、日期时间和字符串共 77 个函数。
- 增加 `BACKSET`、`BARSNEXT`、`REFX` 三个显式未来函数，便于迁移和核对旧公式。
- `EXPMA(X, N)` 现在可作为通达信 `EMA(X, N)` 的别名使用。
- `EXPMA(X)` 仍返回 MyTT 传统的 12/50 双线指标，保持旧代码兼容。

### 兼容性修复

- `REF(X, N)` 支持动态周期数组，并支持 `N=0` 返回当前值。
- `HHV`、`LLV`、`HHVBARS`、`LLVBARS` 支持 `N=0` 的历史累计语义。
- `STD` 按通达信定义修正为样本标准差。
- `BETWEEN` 修正为包含上下边界的闭区间判断。
- `SUMBARS` 和 `SUMBARSX` 分别实现累计不足时返回已有周期数和无效值的语义。
- 增加输入长度、周期、平滑参数、除零和缺失数据检查。

### 工程化改进

- 使用 `src/` 包布局和 `pyproject.toml` 构建。
- 提供类型标记，支持 mypy 检查。
- 使用 Ruff 统一代码检查和格式。
- 测试覆盖基础函数、条件、指标、状态函数和兼容扩展，分支覆盖率超过 90%。
- 可同时构建 wheel 和 sdist 安装包。

## 未来函数警告

`BACKSET`、`BARSNEXT` 和 `REFX` 会使用当前周期之后才能确定的数据。
它们只应用于公式迁移、图形复现或离线研究，不得用于历史回测中的可交易信号，
更不能直接用于实盘决策。

## 尚不能由本库独立实现的功能

以下能力依赖通达信客户端、外部数据库或专有算法，不能只靠 OHLCV 数组等价实现：

- `DYNAINFO(*)`、`FINANCE(*)`、Level-2 和实时盘口数据。
- 行业、概念、地域、自定义板块及横向统计。
- 跨品种、跨周期、其他公式输出和云公式引用。
- `COST`、`WINNER` 等筹码分布模型及除权除息数据库。
- `DRAWKLINE`、`DRAWTEXT`、`STICKLINE` 等客户端绘图指令。
- 账户、下单、撤单、DLL 和客户端交易状态机。
- 部分算法细节未公开的未来函数，如 `ZIG`、`PEAK`、`TROUGH`。

详细审计结果见
[通达信函数覆盖审计](https://github.com/dbx192/MyTT/blob/main/docs/tdx-function-coverage.md)。

## 开发和测试

```bash
git clone https://github.com/dbx192/MyTT.git
cd MyTT
python -m pip install -e '.[dev]'
pytest
ruff check src tests
ruff format --check src tests
mypy src
python -m build
```

## 许可证

本项目采用 GPL-3.0-or-later 许可证。
