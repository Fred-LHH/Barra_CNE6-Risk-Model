import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import Function

class Volatility(Function):

    def __init__(self,
                 close_code: pd.DataFrame,
                 close_index: pd.DataFrame,
                 window: int = 252,
                 half_life: int = 63):
        
        self.price = pd.concat([close_code, close_index], axis=0).reset_index(drop=True)
        self.price['ret'] = self.price['close'] / self.price['pre_close']
        self.ret = pd.pivot_table(self.price, values='ret', index='trade_date', columns='code')
        self.BETA, self.Hist_sigma = self.Beta_HS(window, half_life)
        self.Daily_std = self.Daily_std(window)
        self.CMRA = self.Cumulative_range(window)
    
    def Beta_HS(self, window, half_life):
        Weight = np.diag(self._exp_weight(window, half_life))

        def _calc_factor(tmp):
            # 不存在缺失值的股票
            Y_f = tmp.dropna(axis=1).drop(columns='399300.SZ')
            idx_f, Y_f = Y_f.columns, Y_f.values
            X_f = np.c_[np.ones((window, 1)), tmp.loc[:, '399300.SZ'].values]
            beta_f = np.linalg.pinv(X_f.T @ Weight @ X_f) @ X_f.T @ Weight @ Y_f
            hist_sigma_f = pd.Series(np.std(Y_f - X_f @ beta_f, axis=0), index=idx_f, name=tmp.index[-1])
            beta_f = pd.Series(beta_f[1], index=idx_f, name=tmp.index[-1])
            # 对于不具有完整数据的股票, 则取出进行数据对齐, 如果最近252个交易日内股票样本点少于63, 则不进行估计
            beta_l, hist_sigma_l = {}, {}
            for c in set(tmp.columns) - set(idx_f) - set('399300.SZ'):
                tmp_ = tmp.loc[:, [c, '399300.SZ']].copy()
                tmp_.loc[:, 'Weight'] = Weight
                tmp_.dropna(inplace=True)
                W_l = np.diag(tmp_['Weight'])
                if len(tmp_) < half_life:
                    continue
                X_l = np.c_[np.ones(len(tmp_)), tmp_['399300.SZ'].values]
                Y_l = tmp_[c].values
                beta_tmp = np.linalg.pinv(X_l.T @ W_l @ X_l) @ X_l.T @ W_l @ Y_l
                hist_sigma_l[c] = np.std(Y_l - X_l @ beta_tmp)
                beta_l[c] = beta_tmp[1]
            beta_l = pd.Series(beta_l, name=tmp.index[-1])
            hist_sigma_l = pd.Series(hist_sigma_l, name=tmp.index[-1])
            beta = pd.concat([beta_f, beta_l]).sort_index()
            hist_sigma = pd.concat([hist_sigma_f, hist_sigma_l]).sort_index()
            return beta, hist_sigma
        
        args = [(self.ret.iloc[i:i+window, :].copy()) for i in range(len(self.ret) - window + 1)]
        res = Parallel(n_jobs=-1)(delayed(_calc_factor)(*arg) for arg in args)

        betas = [result[0] for result in res]
        hist_sigmas = [result[1] for result in res]
        beta = pd.concat(betas, axis=1).T
        hist_sigma = pd.concat(hist_sigmas, axis=1).T
        beta = pd.melt(beta.reset_index(), id_vars='index').dropna()
        beta.columns = ['trade_date', 'code', 'BETA']
        hist_sigma = pd.melt(hist_sigma.reset_index(), id_vars='index').dropna()
        hist_sigma.columns = ['trade_date', 'code', 'Hist_sigma']
        return betas, hist_sigmas

    def Daily_std(self, window):
        L = 0.5 ** (1 / 42)
        daily_std = {}
        #### 计算Daily std
        ### 采用EWMA估计股票ret的波动率,半衰期42个交易日
        for i in tqdm(range(len(self.ret) - window + 1), desc='计算Daily std...'):
            ret_ = self.ret.iloc[i:i + window]
            init_var = ret_.var(axis=0)
            tmp = init_var.copy()
        
            for t, k in ret_.iterrows():
                tmp = tmp * L + k ** 2 * (1 - L)
            daily_std[ret_.index[-1]] = np.sqrt(tmp)

        daily_std = pd.DataFrame(daily_std).T
        daily_std.index.name = 'trade_date'
        daily_std = pd.melt(daily_std.reset_index(), id_vars='trade_date', value_name='Daily_std').dropna()
        daily_std.columns = ['trade_date', 'code', 'Daily_std']
    
        return daily_std

    def Cumulative_range(self, window):
        close = pd.pivot_table(self.price, values='close', index='trade_date', columns='code').fillna(method='ffill', limit=10)
        pre_close = pd.pivot_table(self.price, values='pre_close', index='trade_date', columns='code').fillna(method='ffill', limit=10)
        idx = close.index
        CMRA = {}
        for i in tqdm(range(252, len(close)), desc='计算CMRA...'):
            close_ = close.iloc[i-window:i, :]
            pre_close_ = pre_close.iloc[i-window, :]
            pre_close_.name = pre_close_.name - pd.Timedelta(days=1)
            close_ = pd.concat([close_, pre_close_.to_frame().T], axis=0).sort_index().iloc[list(range(0, 253, 21)), :]
            r_tau = close_.pct_change().dropna(how='all')
            Z_T = np.log(r_tau + 1).iloc[::-1].cumsum(axis=0)
            CMRA[idx[i-1]] = Z_T.max(axis=0) - Z_T.min(axis=0)
        CMRA = pd.DataFrame(CMRA).T
        CMRA.index.name = 'trade_date'
        CMRA = pd.melt(CMRA.reset_index(), id_vars='trade_date', value_name='Cumulative_range').dropna()
        return CMRA




