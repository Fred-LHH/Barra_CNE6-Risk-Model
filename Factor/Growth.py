import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import Function

class Growth(Function):

    def __init__(self, 
                 forcast_analysis: pd.DataFrame,
                 financial_df: pd.DataFrame):
        
        self.financial_df = financial_df
        self.forcast_analysis = forcast_analysis

        self.forecast_roe_mean = self.forecast_roe_mean()
        self.revenue_growth_rate, self.income_growth = self.growth_factor()


    def forecast_roe_mean(self):
        forecast_roe_mean = []

        for year in tqdm(range(2014, 2025)):
            years = ['{}1231'.format(year+i) for i in range(3)]
            mask = (self.forcast_analysis['Fenddt'].isin(years))
            tmp = self.forcast_analysis[mask].copy()
            tmp['base_date'] = pd.to_datetime(str(year)+'1231')
            growth_mean = tmp.groupby('Stkcd').apply(self._cummean, 'FROE', multi_periods=True).reset_index()
            growth_mean.rename(columns={'Stkcd': 'code', 'base_date':'end_date'}, inplace=True)
            growth_mean = growth_mean[['code', 'end_date', 'np_mean']]
            growth_mean = self._pubDate_align_tradedate(growth_mean, pubDate_col=None, end_date=str(year)+'1231', analysis_pred=True)
            forecast_roe_mean.append(growth_mean)

        forecast_roe_mean = pd.concat(forecast_roe_mean)
        forecast_roe_mean.sort_values(['code', 'trade_date'], inplace=True)
        forecast_roe_mean.reset_index(drop=True, inplace=True)
        return forecast_roe_mean
    
    def growth_factor(self):
        periods = ['{}{}'.format(year, date) for year in range(2009, 2025) for date in ['0331', '0630', '0930', '1231']]
        periods = periods[:-2]
        periods = pd.to_datetime(periods)
        data['end_date'] = pd.to_datetime(data['end_date'])
        data = self._convert_ttm(data)

        def _sub_calc_factor(time):
            # 这里的date为半年度频率
            # 获取date的前19个季度的数据
            idx = periods.get_loc(time)
            dates = periods[idx-19:idx+1]
            df = data.loc[data['end_date'].isin(dates)].copy()
            df['discDate'] = df['end_date'].apply(self._transfer_rpt_dates)
            df = df.query('ann_date<discDate').drop(columns='ann_date')\
            .rename(columns={'discDate':'trade_date'})\
            .sort_values(by=['code', 'end_date'])
            tmp = df.copy()
            tmp.reset_index(drop=True, inplace=True)
            revenue_Growth_Rate = - tmp.groupby('code').apply(
                self._t_reg, field='revenue_TTM', min_period=6).fillna(0.)
            income_growth = - tmp.groupby('code').apply(
                self._t_reg, field='n_income_attr_p_TTM', min_period=6).fillna(0.)
        
            sub_factor = pd.concat([revenue_Growth_Rate, income_growth], axis=1)
            sub_factor.columns = ['revenue_Growth_Rate', 'income_growth']
            return sub_factor
    
        factor = []
        for date in tqdm(periods, desc='计算Growth...:'):
            if date.year < 2014:
                continue
            elif date.month != 6 and date.month != 12:
                continue
            else:
                sub_factor = _sub_calc_factor(date)
                if date.month == 6:
                    trade_date = str(date.year) + '0901'
                    sub_factor['trade_date'] = trade_date
                elif date.month == 12:
                    trade_date = str(date.year+1) + '0501'
                    sub_factor['trade_date'] = trade_date
                factor.append(sub_factor.reset_index())
        factor = self._pubDate_align_tradedate(pd.concat(factor), 'trade_date', '20240831')
        factor.reset_index(drop=True, inplace=True) 
        return   factor[['code', 'trade_date', 'revenue_Growth_Rate']], factor[['code', 'trade_date', 'income_growth']]















def calc_growth_factor(data):
    periods = ['{}{}'.format(year, date) for year in range(2009, 2025) for date in ['0331', '0630', '0930', '1231']]
    periods = periods[:-2]
    periods = pd.to_datetime(periods)
    data['end_date'] = pd.to_datetime(data['end_date'])
    data = ut._calculate_ttm(data)

    def _sub_calc_factor(time):
        # 这里的date为半年度频率
        # 获取date的前19个季度的数据
        idx = periods.get_loc(time)
        dates = periods[idx-19:idx+1]
        df = data.loc[data['end_date'].isin(dates)].copy()
        df['discDate'] = df['end_date'].apply(ut._discDate)
        df = df.query('ann_date<discDate').drop(columns='ann_date')\
            .rename(columns={'discDate':'trade_date'})\
            .sort_values(by=['code', 'end_date'])
        tmp = df.copy()
        tmp.reset_index(drop=True, inplace=True)
        revenue_Growth_Rate = - tmp.groupby('code').apply(
            ut._t_reg, field='revenue_TTM', min_period=6).fillna(0.)
        income_growth = - tmp.groupby('code').apply(
            ut._t_reg, field='n_income_attr_p_TTM', min_period=6).fillna(0.)
        
        sub_factor = pd.concat([revenue_Growth_Rate, income_growth], axis=1)
        sub_factor.columns = ['revenue_Growth_Rate', 'income_growth']
        return sub_factor
    
    factor = []
    for date in tqdm(periods, desc='计算Growth...:'):
        if date.year < 2014:
            continue
        elif date.month != 6 and date.month != 12:
            continue
        else:
            sub_factor = _sub_calc_factor(date)
            if date.month == 6:
                trade_date = str(date.year) + '0901'
                sub_factor['trade_date'] = trade_date
            elif date.month == 12:
                trade_date = str(date.year+1) + '0501'
                sub_factor['trade_date'] = trade_date
            factor.append(sub_factor.reset_index())
    factor = ut._pubDate_align_tradedate(pd.concat(factor), 'trade_date', '20240831')
    return factor.reset_index(drop=True)   










