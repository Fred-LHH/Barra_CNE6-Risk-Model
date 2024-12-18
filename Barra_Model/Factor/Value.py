import pandas as pd
import numpy as np
import RiskModel.Factor.utils as Function
from tqdm import tqdm
from joblib import Parallel, delayed

class Value(Function):

    def __init__(self, 
                 data: pd.DataFrame,
                 forecast: pd.DataFrame,
                 indicator: pd.DataFrame,
                 price: pd.DataFrame):
        # input data columns : code	ann_date end_date n_cashflow_act_TTM trade_date	pe_ttm pb total_mv
        # price includes stocks and index close price
        data['total_mv'] = data['total_mv'] * 1e4
        price['ret'] = price['close'] / price['pre_close']
        ret = pd.pivot_table(price, values='ret', index='trade_date', columns='code')

        self.book_to_price, self.earning_to_price, self.cash_earning_to_price = self.Value_Factor(data)
        total_mv = indicator[['code', 'trade_date', 'total_mv']].copy()
        self.forecast_ep_mean = self.Forecast_EP_Mean(forecast, total_mv)
        self.enterprise_multiple = self.Enterprise_Multiple(data, indicator)
        self.lp_relative_strength = self.LP_Relative_Strength(ret)
        self.long_alpha = self.Long_Alpha(ret) 

    def Value_Factor(self, data):
        data['Book_to_price'] = 1 / data['pb']
        data['Earning_to_price'] = 1 / data['pe_ttm']
        data['Cash_earning_to_price'] = data['n_cashflow_act_TTM'] / data['total_mv']
        return data[['code', 'trade_date', 'Book_to_price']], data[['code', 'trade_date', 'Earning_to_price']], data[['code', 'trade_date', 'Cash_earning_to_price']]
    
    def Forecast_EP_Mean(self, forecast, indicator):
        # indicator columns: code trade_date total_mv
        forecast_EP_mean = []
        for year in tqdm(range(2014, 2025), desc='分析师预测EP比...'):
            mask = (forecast['Fenddt'] == pd.to_datetime('{}1231'.format(year)))
            tmp = forecast[mask].copy()
            tmp['Fnetpro'] /= 1e8
            tmp.rename(columns={'Stkcd': 'code', 'Fenddt':'end_date'}, inplace=True)
            np_mean = tmp.groupby('code').apply(self._cummean).reset_index()
            np_mean = self._pubDate_align_tradedate(np_mean, pubDate_col=None, end_date=str(year)+'1231', analysis_pred=True)
            total_mv = indicator[indicator['trade_date'].dt.year == year]
            np_mean = np_mean.merge(total_mv, on=['code', 'trade_date'])
            np_mean['forecast_EP_mean'] = np_mean.eval('np_mean/total_mv')
            forecast_EP_mean.append(np_mean)
        forecast_EP_mean = pd.concat(forecast_EP_mean, axis=0)
        forecast_EP_mean = forecast_EP_mean[['code', 'trade_date', 'forecast_EP_mean']]
        return forecast_EP_mean
    
    def Enterprise_Multiple(self, data, indicators):
        data.rename(columns={'ts_code':'code', 'f_ann_date':'ann_date'}, inplace=True)
        data['ann_date'] = pd.to_datetime(data['ann_date'])
        data['end_date'] = pd.to_datetime(data['end_date'])
        data['discDate'] = data['end_date'].apply(self._transfer_rpt_dates)
        data.rename(columns={'ts_code':'code'}, inplace=True)
        data = data.query('ann_date<discDate').drop(columns='ann_date')\
            .rename(columns={'discDate':'ann_date'})\
            .sort_values(by=['code', 'end_date'])
        data = self._pubDate_align_tradedate(data, 'ann_date', '20240831')
        data['code'] = data['code'].apply(lambda x: x.split('.')[0])
        data = pd.merge(data, indicators, on=['code', 'trade_date'])
        data['ebit'] = data['ebit'] / 1e8
        data['end_bal_cash_equ'] = data['end_bal_cash_equ'] / 1e8
        data['total_liab'] = data['total_liab'] / 1e8
        data['EV'] = data.eval('total_liab + total_mv- end_bal_cash_equ')
        data['Enterprise_multiple'] = data.eval('ebit / EV')
        Enterprise_multiple = data[['code', 'trade_date', 'Enterprise_multiple']]
        return Enterprise_multiple

    def LP_Relative_Strength(self, ret, window=750, half_life=260):
        W = self._exp_weight(window=window, half_life=half_life)
        relative_strength = {}
        for i in tqdm(range(len(ret) - window - 1), desc='长期非滞后相对强度……'):
            tmp = ret.iloc[i:i+window, :]
            tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / window < 0.1].fillna(0.)
            relative_strength[tmp.index[-1]] = pd.Series(np.sum(W.reshape(-1, 1) * tmp.values, axis=0), index=tmp.columns)
        relative_strength = pd.DataFrame(relative_strength).T
        relative_strength.index.name = 'date'
        relative_strength = relative_strength.shift(273)
        relative_strength = relative_strength.rolling(11).mean().dropna(how='all').mul(-1)
        relative_strength = pd.melt(relative_strength.reset_index(), id_vars='date', value_name='LP_Relative_strength').dropna().reset_index(drop=True)
        relative_strength.columns = ['date', 'code', 'LP_Relative_strength']
        return relative_strength
    
    def Long_Alpha(self, ret, window=750, half_life=260):
        W = self._exp_weight(window=window, half_life=half_life)

        def _calc_Alpha(tmp):
            W_f = np.diag(W)
            Y_f = tmp.dropna(axis=1).drop(columns='399300.SZ')
            idx_f, Y_f = Y_f.columns, Y_f.values
            X_f = np.c_[np.ones((window, 1)), tmp.loc[:, '399300.SZ'].values]
            beta_f = np.linalg.pinv(X_f.T @ W_f @ X_f) @ X_f.T @ W_f @ Y_f
            alpha_f = pd.Series(beta_f[0], index=idx_f, name=tmp.index[-1])
        
            alpha_l = {}
            for c in set(tmp.columns) - set(idx_f) - set('399300.SZ'):
                tmp_ = tmp.loc[:, [c, '399300.SZ']].copy()
                tmp_.loc[:, 'W'] = W
                tmp_ = tmp_.dropna()
                W_l = np.diag(tmp_['W'])
                if len(tmp_) < half_life:
                    continue
                X_l = np.c_[np.ones(len(tmp_)), tmp_['399300.SZ'].values]
                Y_l = tmp_[c].values
                beta_l = np.linalg.pinv(X_l.T @ W_l @ X_l)@ X_l.T @ W_l @ Y_l
                alpha_l[c] = beta_l[0]
            alpha_l = pd.Series(alpha_l, name=tmp.index[-1])  
            alpha = pd.concat([alpha_f, alpha_l]).sort_index()
            return alpha
    
        Alpha = Parallel(6, verbose=10)(delayed(_calc_Alpha)(
            ret.iloc[i:i+window, :].copy()) for i in 
            tqdm(range(len(ret)-window+1), desc='正在计算alpha...'))

        Alpha = (pd.concat(Alpha, axis=1).T
         .apply(pd.to_numeric, errors='coerce')
         .shift(273)
         .rolling(11).mean()
         .dropna(how='all', axis=1)
         .mul(-1))
        Alpha = pd.melt(Alpha.reset_index(), id_vars='index', value_name='Longterm_Alpha').dropna().reset_index(drop=True)
        Alpha.columns = ['date', 'code', 'Longterm_Alpha']
        return Alpha
    
    



