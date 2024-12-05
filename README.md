# Barra_CNE6-Risk-Model

## Factor

所用的数据来自于Tushare，国泰安

- ### **Size**


  - LNCAP: 流通市值的自然对数

  * MIDCAP：中市值.首先取Size exposure的立方, 再WLS对Size正交, 去极值, 标准化
- ### Volatility


  - Beta:股票收益率对沪深300收益率进行时间序列回归,取回归系数,回归时间窗口为252个交易日,半衰期63个交易日
  - Hist_sigma:在计算Beta所进行的时序回归中,取回归残差收益率的波动率
  - Daily_std:日收益率在过去252个交易日的波动率, 半衰期42个交易日
  - Cumulative_range:累计收益范围

  如果最近252个交易日内,股票样本点少于63个,则不进行计算。股票收益率数据存在缺失值,这将导致回归参数估计失败(矩阵运算结果为NaN)。所以,在任意交易日,我们筛选出具有完整数据的股票,进行批量的加权最小二乘估计;而不具有完整数据的股票,则分别取出,将数据对齐,再进行参数估计。
- ### Dividend


  - 股息率： 最近12个月的每股股息除以上个月月末的股价
  - 分析师预测分红价格比： 预测12个月的每股股息(DPS)除以当前价格 (缺少数据，故不做计算)
- ### Growth


  - 分析师预测长期盈利增长率:分析师预测的长期(3-5)年利润增长率
  - 每股收益增长率:过去5个财政年度的每股收益对时间回归的斜率除以平均每股年收益
  - 每股营业收入增长率:过去5个财政年度的每股年营业收入对时间回归斜率除以平均每股年营业收入
- ### Liquidity


  - Monthly share turnover月换手率.对最近21个交易日的股票换手率求和,然后取对数
  - Quarterly share turnover 季换手率 T=3
  - Annual share turnover 年换手率 T=12
  - Annualized traded value ratio 年化交易量比率. 对日换手率进行加权求和,时间窗口为252个交易日,半衰期为63个交易日
- ### Momentum


  - Short Term reversal短期反转.最近一个月的加权累计对数日收益率
  - Seasonality 季节因子.过去5年的已实现次月收益率的平均值
  - Industry Momentum 行业动量.该指标描述个股相对中信一级行业的强度
  - Relative strength 相对于市场强度
  - Historical alpha 在BETA计算所进行的时间序列回归中取回归截距项
- ### Quality


  - Leverage

    - Market Leverage(3) 市场杠杆
    - Book Leverage(3) 账面杠杆
    - Debt to asset ratio(3) 资产负债比
  - Earning Variability

    - Variation in Sales(3) 营业收入波动率
    - Variation in Earning(3) 盈利波动率
    - Variation in cashflows(3) 现金流波动率
    - Standard deviation of analyst Earnings-to-Price(3) 分析师预测盈市率标准差
  - Earnings Quality

    - Accruals Balancesheet version(3) 资产负债表应计项目
    - Accruals Cashflow version(3) 现金流量表应计项目
  - 盈利能力Profitability

    - 资产周转率Asset turnover
    - 资产毛利率Gross profitability
    - 销售毛利率Gross Profit Margin
    - 总资产收益率 Return on assets
  - Investment Quality

    - 总资产增长率Total Assets Growth Rate：最近5个财政年度的总资产对时间的回归的斜率值，除以平均总资产，最后取相反数
    - 股票发行量增长率Issuance growth：最近5个财政年度的流通股本对时间的回归的斜率值，除以平均流通股本，最后取相反数
    - 资本支出增长率Capital expenditure growth：将过去5个财政年度的资本支出对时间的回归的斜率值，除以平均资本支出，最后取相反数

    PS:资本支出是指用于购买各种长期资产(长期投资固定资产无形资产和其他长期资产)的支出然后再减去无息长期负债(各种不需支付利息的长期应付款专项应付款等)的增加额。
- ### Value


  - 账面市值比（Book to price）：将最近报告期的普通股账面价值除以当前市值
  - Earnings-to-price Ratio：过去12个月的盈利除以当前市值
  - 分析师预测EP比：预测12个月的盈利除以当前市值
  - Cash earnings to price：过去12个月的现金盈利除以当前市值
  - Enterprise multiple：上一财政年度的息税前利润（EBIT）除以当前企业价值（EV）
  - 长期相对强度
  - 长期历史Alpha

  PS:长期指标均采用750为回望窗口大小。


## Risk Model




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
