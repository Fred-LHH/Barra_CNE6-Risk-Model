# Barra_CNE6-Risk-Model

## Factor

- ### **Size**


  - LNCAP: 流通市值的自然对数

  * MIDCAP：中市值.首先取Size exposure的立方, 再WLS对Size正交, 去极值, 标准化
- ### Volatility


  - Beta:股票收益率对沪深300收益率进行时间序列回归,取回归系数,回归时间窗口为252个交易日,半衰期63个交易日
  - Hist_sigma:在计算Beta所进行的时序回归中,取回归残差收益率的波动率
  - Daily_std:日收益率在过去252个交易日的波动率, 半衰期42个交易日
  - Cumulative_range:累计收益范围

  如果最近252个交易日内,股票样本点少于63个,则不进行计算。股票收益率数据存在缺失值,这将导致回归参数估计失败(矩阵运算结果为NaN)。所以,在任意交易日,我们筛选出具有完整数据的股票,进行批量的加权最小二乘估计;而不具有完整数据的股票,则分别取出,将数据对齐,再进行参数估计。
- 

## utils

- ### Function

  - **_convert_ttm()**
  - **_transfer_rpt_dates()**
  - **_winsorize()**
  - **_regress()**
  - **_exp_weight()**
  - **_get_trade_dates()**
  - **_align_trade_pub_date()**
  - **_cal_industry_RS()**
  - **_panel_rolling_apply()**
  - **_cumstd()**
  - **_t_reg()**
  - **_get_dates_map()**
  - **_cummean()**
