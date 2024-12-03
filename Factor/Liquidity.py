import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import Function

class Liquidity(Function):

    def __init__(self,
                 turnover: pd.DataFrame):
        '''
        PS: 换手率为百分比形式
        '''
        turnover = pd.pivot_table(turnover, index='trade_date', columns='code', values='turnover_rate')
        self.monthly_share_turnover, self.quarterly_share_turnover, self.annual_share_turnover = self.Share_turnover(turnover)
        self.annualized_traded_value_ratio = self.Annualized_traded_value_ratio(turnover)
        
    
    def Share_turnover(self, data: pd.DataFrame):
        monthly_share_turnover = np.log(data.rolling(21).sum())

        idx = list(range(20, 252, 21))
        quarterly_share_turnover, annual_share_turnover = {}, {}
        for i in tqdm(range(len(data) - 251), desc='计算季度、年度换手率...'):
            t = data.index[i+251]
            mst = np.exp(monthly_share_turnover.iloc[i:i+252, :].iloc[idx, :])
            quarterly_share_turnover[t] = np.log(mst.iloc[-3:, :].mean(axis=0))
            annual_share_turnover[t] = np.log(mst.mean(axis=0))
        quarterly_share_turnover = pd.DataFrame(quarterly_share_turnover).T
        annual_share_turnover = pd.DataFrame(annual_share_turnover).T
        quarterly_share_turnover.index.name = 'trade_date'
        annual_share_turnover.index.name = 'trade_date'

        monthly_share_turnover = pd.melt(monthly_share_turnover.reset_index(), id_vars='trade_date', value_name='Monthly_share_turnover').dropna()
        quarterly_share_turnover = pd.melt(quarterly_share_turnover.reset_index(), id_vars='trade_date', value_name='Quarterly_share_turnover').dropna()
        annual_share_turnover = pd.melt(annual_share_turnover.reset_index(), id_vars='trade_date', value_name='Annual_share_turnover').dropna()

        return monthly_share_turnover, quarterly_share_turnover, annual_share_turnover
    

    def Annualized_traded_value_ratio(self, data: pd.DataFrame):
        weight = self._get_exp_weight(252, 63)
        annualized_traded_value_ratio = []
        for i in tqdm(range(len(data)-251), desc='计算年化交易量比率...'):
            tmp = data.iloc[i:i+252, :].copy()
            annualized_traded_value_ratio.append(
                pd.Series(np.nansum(tmp.values * weight.reshape(-1, 1), axis=0), index=data.columns, name=tmp.index[-1])
            )

        annualized_traded_value_ratio = pd.concat(annualized_traded_value_ratio, axis=1).T
        annualized_traded_value_ratio.index.name = 'trade_date'
        annualized_traded_value_ratio = pd.melt(annualized_traded_value_ratio.reset_index(), id_vars='trade_date', value_name='Annualized_traded_value_ratio').dropna()

        return annualized_traded_value_ratio
