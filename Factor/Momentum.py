import pandas as pd
import numpy as np
from utils import Function
from tqdm import tqdm
from joblib import Parallel, delayed


class Momentum(Function):

    def __init__(self, pv_data: pd.DataFrame):
        
        self.data = pv_data[pv_data['code'] != '399300.SZ']
        self.trade_dates = self._get_trade_dates(start_date='20080101', end_date='20240831')
        self.trade_dates = pd.to_datetime(self.trade_dates, format='%Y%m%d')

        self.data['ret'] = self.data['close'] / self.data['pre_close'] - 1
        self.ret = pd.pivot_table(self.data, values='ret', index='trade_date', columns='code')
        self.ret = self.ret[self.ret.index > pd.to_datetime('2013-12-31')]

        self.short_term_reversal = self.Short_Term_Reversal()
        self.seasonality = self.Seasonality()
        self.industry_momentum = self.Industry_Momentum()
        self.relative_strength = self.Relative_strength()
        self.historical_alpha = self.Historical_alpha()

    def Short_Term_Reversal(self):
        r_n = self.ret.rolling(21).mean().dropna(how='all')
        W = self._exp_weight(window=21, half_life=5)
        STREV = []
        for i in tqdm(range(len(r_n) - 20), desc='计算短期反转...'):
            tmp = np.log(1 + r_n.iloc[i:i+21, :].copy())
            tmp = pd.Series(np.sum(W.reshape(-1, 1) * tmp.values, axis=0), name=tmp.index[-1], index=tmp.columns)
            STREV.append(tmp)
        STREV = pd.concat(STREV, axis=1).T
        STREV.index.name = 'trade_date'
        STREV = pd.melt(STREV.reset_index(), id_vars='trade_date', value_name='Short_term_reversal').dropna()
        return STREV
    
    def Seasonality(self):
        trade_dates = self._get_trade_dates(start_date=self.ret.index[0], end_date=self.ret.index[-1])
        trade_dates = pd.to_datetime(trade_dates, format='%Y%m%d')

        def get_specific_dates(date):
            if date in self.trade_dates:
                index = self.trade_dates.get_loc(date)
            else:
                index = self.trade_dates.searchsorted(date)
    
            return self.trade_dates[index:index+21]
    
        seasonality = {}
        for td in tqdm(trade_dates, desc='正在计算季节性因子...'):
            r_y = []
            for i in range(1, 6):
                td_shift = get_specific_dates(td-pd.Timedelta(days=365*i))
                price = self.data.loc[self.data['trade_date'].isin(td_shift), ['trade_date', 'code', 'close']]
                price = pd.pivot_table(price, values='close', index='trade_date', columns='code').ffill()
                r_y.append(price.iloc[-1, :] / price.iloc[0, :] - 1)
            seasonality[pd.to_datetime(td)] = pd.concat(r_y, axis=1).mean(axis=1)
        seasonality = pd.DataFrame(seasonality).T
        seasonality.index.name = 'trade_date'
        seasonality = pd.melt(seasonality.reset_index(), id_vars='trade_date', value_name='Seasonality').dropna()
        return seasonality

    def Industry_Momentum(self):
        ## 计算个股强度
        Weights = self._exp_weight(window=126, half_life=21)
        RS = {}
        for i in tqdm(range(len(self.ret) - 125), desc='正在计算个股强度...'):
            tmp = self.ret.iloc[i:i+126, :].copy()
            # 缺失值在10%以内
            tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / 252 <= 0.1].fillna(0.)
            tmp = np.log(1 + tmp)
            RS[tmp.index[-1]] = pd.Series(np.sum(Weights.reshape(-1, 1) * tmp.values, axis=0), index=tmp.columns)
        RS = pd.DataFrame(RS).T
        RS.index.name = 'trade_date'
        RS = pd.melt(RS.reset_index(), id_vars='trade_date', value_name='RS').dropna().reset_index(drop=True)

        mv = self.data[self.data['trade_date'] > pd.to_datetime('20131231')][['code', 'trade_date', 'circ_mv']]
        RS = pd.merge(RS, mv, on=['code', 'trade_date'])
        # 计算行业相对强度
        RS['month'] = RS['trade_date'].apply(lambda x: x.strftime('%Y-%m-01'))
        RS['circ_mv'] = np.sqrt(RS['circ_mv'])
        ind = self.data[['code', 'industry_code']].drop_duplicates().reset_index(drop=True)
        RS = RS.merge(ind, on='code')
        INDMOM = []
        for m, tmp_RS in tqdm(RS.groupby('month'), desc='正在计算行业动量...'):
            INDMOM.append(tmp_RS.groupby('trade_date').apply(self._industry_RS).reset_index())
        INDMOM = pd.concat(INDMOM).reset_index(drop=True)
        return INDMOM

    def Relative_strength(self):
        # 相对强度 Relative strength
        w = self._exp_weight(window=252, half_life=126)
        ret_ = np.log(self.ret + 1)
        Relative_strength = {}
        for i in tqdm(range(len(ret_) - 251), desc='正在计算非滞后相对强度...'):
            tmp = ret_.iloc[i:i+252, :]
            tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / 252 <= 0.1].fillna(0.)
            Relative_strength[tmp.index[-1]] = pd.Series(np.sum(w.reshape(-1, 1) * tmp.values, axis=0), index=tmp.columns)
        Relative_strength = pd.DataFrame(Relative_strength).T
        Relative_strength.index.name = 'trade_date'
        Relative_strength = Relative_strength.rolling(11).mean().dropna(how='all')
        Relative_strength = pd.melt(Relative_strength.reset_index(), id_vars='trade_date', value_name='Relative_strength').dropna().reset_index(drop=True)
        return Relative_strength
    
    def Historical_alpha(self):
        # Historical alpha
        Historical_alpha = []
        for i in tqdm(range(len(self.ret) - 251), desc='正在计算Historical alpha...'):
            tmp = self.ret.iloc[i:i+252, :].copy()
            W = self._exp_weight(252, 63)
            Y_f = tmp.dropna(axis=1).drop(columns='399300.SZ')
            idx_f, Y_f = Y_f.columns, Y_f.values
            X_f = np.c_[np.ones((252, 1)), tmp.loc[:, '399300.SZ'].values]
            W_f = np.diag(W)
            beta_f = np.linalg.pinv(X_f.T @ W_f @ X_f) @ X_f.T @ W_f @ Y_f
            historical_alpha_f = pd.Series(beta_f[0], index=idx_f, name=tmp.index[-1])

            historical_alpha_l = {}
            for c in set(tmp.columns) - set(idx_f) - set('399300.SZ'):
                tmp_ = tmp.loc[:, [c, '399300.SZ']].copy()
                tmp_.loc[:, 'Weight'] = W
                tmp_.dropna(inplace=True)
                W_l = np.diag(tmp_['Weight'])
                if len(tmp_) < 63:
                    continue
                X_l = np.c_[np.ones(len(tmp_)), tmp_['399300.SZ'].values]
                Y_l = tmp_[c].values
                beta_tmp = np.linalg.pinv(X_l.T @ W_l @ X_l) @ X_l.T @ W_l @ Y_l
                historical_alpha_l[c] = beta_tmp[0]
            historical_alpha_l = pd.Series(historical_alpha_l, name=tmp.index[-1])
            Historical_alpha.append(pd.concat([historical_alpha_f, historical_alpha_l]).sort_index())
        Historical_alpha = pd.concat(Historical_alpha, axis=1).T
        Historical_alpha = pd.melt(Historical_alpha.reset_index(), id_vars='index').dropna()
        Historical_alpha.columns = ['trade_date', 'code', 'Historical_alpha']
        return Historical_alpha