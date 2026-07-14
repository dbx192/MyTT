# 通达信函数覆盖审计

审计日期：2026-07-15

审计来源：通达信官网
[`functionlist.html`](https://help.tdx.com.cn/gspt/docs/markdown/redword/functionlist.html)
当前函数表，以及仓库重构前的
`MyTT.py`、`MyTT_plus.py`。官网原表有 604 个不同条目，但其中同时包含行情变量、
运算符、绘图指令、财务数据编号和客户端状态，并不等于 604 个可由数值库实现的函数。

## 结论

- 重构前 MyTT 的公开函数没有遗失，旧版函数均已迁入 4.0.0。
- 本次补充 77 个直接可计算的通达信函数。当前包公开 152 个公式/指标名称，其中
  119 个名称与官网函数表直接对应，另有 33 个 MyTT 指标或辅助名称。
- `REF` 现已支持动态周期；`HHV`、`LLV`、`HHVBARS`、`LLVBARS` 支持通达信的
  `N=0` 累计语义。
- 修正 `STD` 为官网定义的样本标准差，修正 `BETWEEN` 为包含边界。
- `EXPMA(X)` 保留 MyTT 旧双线指标，`EXPMA(X,N)` 按官网定义作为 `EMA(X,N)` 别名。

## 本次实现

数学与逻辑：

`ACOS`、`ASIN`、`ATAN`、`EXP`、`LOG`、`CEILING`、`FLOOR`、`INTPART`、
`FRACPART`、`ROUND`、`ROUND2`、`SIGN`、`MOD`、`MAX6`、`MIN6`、`REVERSE`、
`RANGE`、`IFF`、`IFN`、`NOT`、`ISVALID`。

窗口、定位与条件：

`BARSCOUNT`、`CURRBARSCOUNT`、`TOTALBARSCOUNT`、`ISLASTBAR`、`BARSTATUS`、
`BARSSINCE`、`BARSLASTS`、`MULAR`、`FILTERX`、`SUMBARS`、`SUMBARSX`、
`UPDOWN`、`UPNDAY`、`DOWNNDAY`、`NDAY`、`EXISTR`、`HOD`、`LOD`、
`HHVLLV`、`FINDHIGH`、`FINDHIGHBARS`、`FINDLOW`、`FINDLOWBARS`。

均线与统计：

`TMA`、`MEMA`、`AMA`、`EXPMEMA`、`TR`、`DEVSQ`、`STDP`、`STDDEV`、
`VAR`、`VARP`、`COVAR`、`RELATE`、`BETAEX`。

日期、时间与字符串：

`DATETODAY`、`DAYTODATE`、`TIMETOSEC`、`SECTOTIME`、`CON2STR`、`VAR2STR`、
`STR2CON`、`STRLEN`、`STRCAT`、`STRCAT6`、`STRSPACE`、`SUBSTR`、`VARCAT`、
`VARCAT6`、`STRCMP`、`FINDSTR`。

显式未来函数：

`BACKSET`、`BARSNEXT`、`REFX`。这些函数为兼容公式迁移而提供，使用了当前周期之后
才可能确认的数据，不能用于历史可交易信号。`ALIGNRIGHT` 也已实现，但它只是数据对齐，
不是交易信号函数。

## 仍未实现及原因

### 行情和客户端上下文

`OPEN/HIGH/LOW/CLOSE/VOL/AMOUNT` 在 MyTT 中是调用者传入的数组，不是全局变量。
`ADVANCE`、`DECLINE`、`DYNAINFO(*)`、`TOTALHQINFO(*)`、`MAINZSHQ(*)`、
`BUYVOL`、`SELLVOL`、`PERIOD`、`FROMOPEN`、`TOTALFZNUM`、`MACHINEDATE` 等需要
实时行情、品种交易日历或客户端状态。MyTT 单独无法生成这些数据；应由 mootdx、券商 API
或其他行情源提供后，再传给公式函数。

### 财务、品种元数据和板块

`FINANCE(*)`、专业财务函数、`CAPITAL`、`TOTALCAPITAL`、`CODE`、`STKNAME`、
`NAMELIKE`、`CODELIKE`、`INBLOCK`、`HYBLOCK`、`DYBLOCK`、`GNBLOCK`、
`BLOCKSETNUM`、`HORCALC`、`INSORT`、`INSUM` 等依赖通达信数据库、板块定义和版本权限。
仅凭 OHLCV 数组不能等价实现。

### 跨品种、跨周期和其他公式

`CALCSTOCKINDEX`、`$` 品种引用、`#` 跨周期引用、`XXX.XX` 指标输出引用，以及
`INDEX*`、`HY_INDEX*` 系列都需要数据加载、交易日对齐、复权和公式注册环境。它们属于
数据/执行引擎能力，不属于单序列 NumPy 函数。

### 筹码、除权和涨跌停

`COST`、`WINNER`、`LWINNER`、`PWINNER`、`COSTEX`、`PPART` 需要通达信筹码分布
模型；`SPLIT`、`SPLITBARS`、`DIVFACTOR` 需要公司行为数据；`ZTPRICE`、`DTPRICE`
还需要品种最小价位和市场舍入规则。这些不能从现有函数参数可靠推导。

### 绘图、公式控制和交易状态机

`DRAWKLINE`、`STICKLINE`、`DRAWICON`、`DRAWTEXT*`、`DRAWNUMBER*`、`DRAWBAND`、
`DRAWGBK*`、`RGB` 等依赖通达信绘图窗口。`TESTSKIP`、`IFC` 是公式解释器控制流；
`TFILTER`、`TTFILTER` 与专家系统的开平仓状态和同周期指令优先级绑定。数值库可以做近似
状态机，但不能声称与客户端执行引擎等价，因此没有伪实现。

### 其余未来函数和未公开算法细节

`DHIGH/DOPEN/DLOW/DCLOSE/DVOL`、`XMA`、`ZIG/ZIGA`、`PEAK/PEAKBARS`、
`TROUGH/TROUGHBARS`、`DRAWLINE`、`REFXV` 等依赖未来数据；其中转向点、平滑边界和
同值处理的细节不足以从函数表唯一确定。`REFV` 的“平滑处理”边界语义也未在当前官网表中
完整规定。这些函数只有在取得客户端对照数据并建立逐项回归测试后才适合加入。

### DLL、账户和专有能力

`TDXDLL*`、`USERFUNC*`、云函数、账户函数、下单/撤单函数以及 Level-2 专有函数依赖
外部二进制、登录账户或付费权限，MyTT 无法独立实现。

## 兼容边界

MyTT 是函数库，不是通达信公式解析器。它不解析 `:=`、输出线、颜色后缀、`#`、`$` 或
公式文件，也不隐式提供 OHLCV。新增函数遵循项目现有约定：一维数组输入、对齐数组输出、
无完整窗口时返回 `nan`、除零返回 `nan`。未来函数虽可调用，但历史回测和实盘信号必须禁用。
