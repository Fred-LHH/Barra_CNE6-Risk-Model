import numpy as np
import pandas as pd
import time
from functools import wraps
from joblib import Parallel, delayed
import tushare as ts
ts.set_token('dfb6e9f4f9a3db86c59a3a0f680a9bdc46ed1b5adbf1e354c7faa761')
pro = ts.pro_api()
import talib as ta

def try_except(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            return np.nan
    return wrapper

def _forecast_custom_fill(x: pd.DataFrame, 
                          col: str) -> pd.DataFrame:
    """ 
    用未来的值进行填充 (backfill) 
    当计算分析师预期因子时使用
    """
    x['trade_date'].fillna(x[col], inplace=True)
    return x.sort_values(by='trade_date').bfill().dropna(subset=['trade_date'])

def _custom_fill(x: pd.DataFrame, col: str) -> pd.DataFrame:
    """ 
    用过去的值进行填充 (forward fill) 
    """
    x['trade_date'].fillna(x[col], inplace=True)
    return x.sort_values(by='trade_date').ffill().dropna(subset=['trade_date'])


class Function:

    @staticmethod
    def _convert_ttm(data, cols=None):
        '''
        算法:
            (1)最新报告期是年报,则TTM=年报；
            (2)最新报告期不是年报,则TTM=本期+(上年年报-上年同期)，如果本期、上年年报、上年同期存在空值，则不计算，返回空值；
            (3)最新报告期通过财报发布时间进行判断,防止前视偏差。

        df: 默认为 code, ann_date, end_date, factors
        '''
        df = data.copy()
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

        f_cols = [df.columns[3:] if cols is None else cols]
        df = df.assign(**{f'{col}_TTM': np.nan for col in f_cols})

        df = df.assign(month=df['end_date'].dt.month, year=df['end_date'].dt.year)
        df.sort_values(by=['code', 'end_date'], inplace=True)

        def _get_ttm(group):
            annual_reports = group[group['month'] == 12] 
            group['last_annual_report'] = group['year'].map(
            lambda y: annual_reports[annual_reports['year'] == y - 1]
            )
            group['last_period'] = group['year'].map(
            lambda y: group[(group['year'] == y - 1) & (group['month'] == group['month'])]
            )

            for col in f_cols:
                group[f'{col}_TTM'] = group.apply(
                    lambda row: row[col] 
                    if row['month'] == 12 else 
                    (row[col] + (row['last_annual_report'][col].values[0] - row['last_same_period'][col].values[0]) 
                    if pd.notna(row[col]) and pd.notna(row['last_annual_report'][col].values[0]) 
                    and pd.notna(row['last_same_period'][col].values[0]) 
                    else np.nan), 
                    axis=1
                )

            group.drop(columns=['last_annual_report', 'last_period'], inplace=True)
            return group

        TTM = df.groupby('code').apply(_get_ttm).reset_index(drop=True)

        return TTM[['code', 'ann_date', 'end_date'] + [f'{col}_TTM' for col in f_cols]]
    

    @staticmethod
    def _transfer_rpt_dates(date):
        month = date.month
        bias = {3: [0, 5, 1], 6: [0, 9, 1], 9: [0, 11, 1], 12: [1, 5, 1]}
        return pd.Timestamp(date.year+bias[month][0], bias[month][1], bias[month][2])

    @staticmethod
    def _winsorize(x, multiplier=5):
        """MAD去除极端值"""
        x = x.replace([np.inf, -np.inf], np.nan)
        x_M = np.nanmedian(x)
        x_MAD = np.nanmedian(np.abs(x-x_M))
        upper = x_M + multiplier * x_MAD
        lower = x_M - multiplier * x_MAD
        x[x>upper] = upper
        x[x<lower] = lower
        return x
    
    @staticmethod
    def _regress(df: pd.DataFrame, 
                 x: str, 
                 y: str, 
                 w: str = None, 
                 winsorize=True, 
                 standardize=True):
        y = df[y].values
        x = np.c_[np.ones((len(y), 1)), df[x].values]
        W = np.diag(np.sqrt(df[w]))
        beta = np.linalg.pinv(x.T @ W @ x) @ x.T @ W @ y
        resid = y - x @ beta
        if winsorize:
            resid = Function._winsorize(resid)
        if standardize:
            resid -= np.nanmean(resid)
            resid /= np.nanstd(resid)

        return pd.Series(resid, index=df['code'])
    
    @staticmethod
    def _exp_weight(window, 
                    half_life):
        Weights = np.asarray([1/2 ** (1 / half_life)] * window) ** np.arange(window)
        return Weights[::-1] / np.sum(Weights)
    
    @staticmethod
    def _get_trade_dates(start_date, end_date=None, count=None):
        start_date = [start_date.strftime('%Y%m%d') 
                      if isinstance(start_date, pd.Timestamp)
                      else start_date]

        end_date = [pd.Timestamp.today().strftime('%Y%m%d')
                    if end_date is None 
                    else end_date.strftime('%Y%m%d')]
        trade_dates = pro.trade_cal(start_date=start_date, end_date=end_date)
        trade_dates = trade_dates.loc[trade_dates['is_open'] == 1].sort_values('cal_date')['cal_date'].tolist()
        if count is not None:
            trade_dates = trade_dates[count:]
        return trade_dates
    

    @staticmethod
    def _align_trade_pub_date(df: pd.DataFrame, 
                              pubDate_col: str | None, 
                              end_date: str, 
                              analysis_pred=False) -> pd.DataFrame:
        """
        将公告时间与交易时间对齐
        pubDate_col: 公告时间列名
        end_date: 全局的截止日期
        analysis_pred: 是否为分析师预期因子
        """
        if pubDate_col is None:
            start_date = f"{end_date[:4]}0101"
            trade_dates = Function._get_trade_dates(start_date=start_date, end_date=end_date)
        else:
            df[pubDate_col] = pd.to_datetime(df[pubDate_col])
            trade_dates = Function._get_trade_dates(start_date=df[pubDate_col].min(), end_date=end_date)

        time_code = pd.DataFrame([(date, code) for date in trade_dates for code in df['code'].unique()],
                             columns=['trade_date', 'code'])
        time_code['trade_date'] = pd.to_datetime(time_code['trade_date'])

        if pubDate_col is None:
            res = df.merge(time_code, left_on=['code', 'end_date'], right_on=['code', 'trade_date'], how='outer')
            res = res.groupby('code', as_index=False).apply(
            lambda g: _forecast_custom_fill(g, pubDate_col) if analysis_pred else _custom_fill(g, 'end_date')
        ).reset_index(drop=True)
            field = ['code', 'trade_date', 'end_date']
        else:
            res = df.merge(time_code, left_on=['code', pubDate_col], right_on=['code', 'trade_date'], how='outer')
            res = res.groupby('code', as_index=False).apply(
            lambda g: _forecast_custom_fill(g, pubDate_col) if analysis_pred else _custom_fill(g, pubDate_col)
        ).reset_index(drop=True)
            field = ['code', 'trade_date', pubDate_col]

        return res[field + [c for c in res.columns if c not in field]]

    @staticmethod
    def _cal_industry_RS(x):
        ind_RS = x.groupby('industry_code').apply(
        lambda y: y['RS'].dot(y['circ_mv']) / y['circ_mv'].sum()
        ).reset_index(name='ind_RS')
        x = pd.merge(x, ind_RS)
        x['Industry_Momentum'] = x['ind_RS'] - x['RS']
        return x[['code', 'Industry_Momentum']].set_index('code')

    @staticmethod
    def _panel_rolling_apply(
        df: pd.DataFrame,
        value_col: str,
        window: int, 
        apply_func: callable,
        rolling_kwargs=None,
        dropna: bool = True,
        fillna_value: float = np.nan,
        fillna_method: str = 'ffill',
        parallel: bool = False,
        min_periods: int = None
    ):
        '''
        面板数据滚动应用函数, 支持并行
        '''
        rolling_kwargs = rolling_kwargs or {}
        min_periods = min_periods or window
        
        @try_except
        def _apply_func(group):
            group_name = group.index[-1]
            if len(group) < min_periods:
                return pd.Series(np.nan, index=group.columns, name=group_name)
            group = apply_func(group, axis=0)
            group.name = group_name
            return group
        
        tmp = pd.pivot_table(df, values=value_col, index='trade_date', columns='code')
        tmp_rolling = tmp.rolling(window, **rolling_kwargs)

        if parallel:
            tmp = Parallel(n_jobs=-1)(
                delayed(_apply_func)(group) 
                if len(group) >= min_periods
                else pd.Series(np.nan, index=group.columns)
                for group in tmp_rolling)
            tmp = pd.concat(tmp, axis=1).T
            tmp.index.name = 'trade_date'
        else:
            tmp = tmp_rolling.apply(lambda group: apply_func(group) if len(group) >= min_periods else pd.Series(np.nan, index=group.columns))
        
        if (fillna_value is not None) ^ (fillna_method is not None): # 逻辑异或
            tmp = tmp.fillna(fillna_value, method=fillna_method)
        
        if dropna:
            tmp = tmp.dropna(how='all')

        return pd.melt(tmp.reset_index(), id_vars='trade_date', value_name=value_col).dropna().reset_index(drop=True)

    @staticmethod
    def _cumstd(x):
        def _sub_cumstd(y):
            f_ = y['Fnetpro'].expanding().apply(lambda s: np.nan if len(s) < 5 else np.nanstd(s), raw=True)
            return f_
        np_std = x.groupby('Fenddt').apply(_sub_cumstd)
        np_std.name = 'np_std'
        return np_std.dropna()

    @staticmethod
    def _t_reg(x, field, min_period):
        # 时间序列回归斜率除平均值
        x = x[field].dropna()
        if len(x) < min_period:
            return np.nan
        return ta.LINEARREG_SLOPE(x, timeperiod=len(x)).iloc[-1] / x.mean()

    @staticmethod
    def _get_dates_map(start_date, end_date, input_dates, method='prev'):
        '''
        method: 'prev' or 'next'
        '''
        cal_dates = pro.trade_cal(start_date=start_date, end_date=end_date)
        cal_dates = cal_dates[['cal_date', 'is_open']]
        cal_dates['cal_date'] = pd.to_datetime(cal_dates['cal_date'], format='%Y%m%d')
        cal_dates.set_index('cal_date', inplace=True)
        cal_dates.sort_index(inplace=True)
        input_dates = pd.to_datetime(input_dates, format='%Y%m%d')

        def _get_prev_date(date):
            if cal_dates.loc[date, 'is_open'] == 1:
                return date
            else:
                prev_dates = cal_dates.loc[:date].index
                last_trade_date = prev_dates[cal_dates.loc[prev_dates, 'is_open'] == 1].max()
                return last_trade_date
    
        def _get_next_date(date):
            if cal_dates.loc[date, 'is_open'] == 1:
                return date
            else:
                next_dates = cal_dates.loc[date:].index
                next_trade_date = next_dates[cal_dates.loc[next_dates, 'is_open'] == 1].min()
                return next_trade_date
        if method == 'prev':
            return input_dates.map(_get_prev_date)
        elif method == 'next':
            return input_dates.map(_get_next_date)
        else:
            raise ValueError('method must be "prev" or "next"')

    @staticmethod
    def _cummean(x, field, multi_periods=False):
        def _sub_cummean(y):
            cummean = y[field].expanding().apply(lambda s: np.nan if len(s) < 5 else np.nanmean(s), raw=True)
            return cummean
        group_field = 'base_date' if multi_periods else 'Fenddt'
        np_mean = x.groupby(group_field).apply(_sub_cummean)
        np_mean.name = 'np_mean'
        return np_mean.dropna()