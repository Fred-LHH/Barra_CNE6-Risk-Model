import pandas as pd
import numpy as np
from utils import *

def CrossSection_OneDay(
        date: str,
        data: pd.DataFrame,
        style_factors: pd.DataFrame,
        industry_factors: pd.DataFrame | None,
):
    '''
    逐日进行截面回归

    Params:
    -------------
    date: '%Y-%m-%d' str
    data: pd.DataFrame
        columns include (['code', 'date', 'market_cap', 'ret'])
    style_factors: pd.DataFrame
        columns include (['Size', 'Volatility', ...])
    industry_factors: pd.DataFrame
        columns include (['industry_1', 'industry_2', ...])
    '''
    cap = data.market_cap.values
    next_ret = data.ret.values  # t+1期的收益率

    if industry_factors is not None:
        industry_factors = industry_factors.values
        I = industry_factors.shape[1]

    S = style_factors.shape[1]
    style_factors = cap_norm(style_factors.values, cap)
    country_factors = np.array([[1]] * data.shape[0])

    Weights = np.diag(np.sqrt(cap) / sum(np.sqrt(cap))) # WLS权重

    # 截面回归求解多因子模型
    if I > 0:
        # 每个行业的总市值
        industry_cap = np.array([sum(industry_factors[:, i] * cap) for i in range(I)])

        # 因引入国家因子, 故model存在行业共线性, 引入行业中性限制对应的变换矩阵
        R = np.eye(1 + S + I)
        R[S, 1:(1+S)] = -industry_cap / industry_cap[-1]
        R = np.delete(R, S, axis=1)

        factors = np.matrix(np.hstack([country_factors, industry_factors, style_factors]))
        tran_factors = factors @ R
        # 纯因子组合权重
        pure_factor_portfolio_weight = R @ np.linalg.inv(tran_factors.T @ Weights @ tran_factors) @ tran_factors.T @ Weights
        '''
        第1行: 国家因子纯因子组合
        第2-2+p行: 行业因子纯因子组合
        其他行: 风格因子纯因子组合
        '''

    else:
        factors = np.matrix(np.hstack([country_factors, style_factors]))
        pure_factor_portfolio_weight = np.linalg.inv(factors.T @ Weights @ factors) @ factors.T @ Weights

    # 纯因子收益
    factor_ret = pure_factor_portfolio_weight @ next_ret
    factor_ret = np.array(factor_ret)[0]
    # 纯因子组合在各个因子上的暴露
    pure_factor_portfolio_exposure = pure_factor_portfolio_weight @ factors   
    # 个股特异收益
    specific_ret = next_ret - np.array(factors @ factor_ret.T)[0]
    # R square
    R2 = 1 - np.var(specific_ret) / np.var(next_ret)            

    return factor_ret, specific_ret, pure_factor_portfolio_exposure, R2

def CrossSection(data: pd.DataFrame,
                 S: int,
                 I: int):
    '''
    逐期截面回归求解多因子模型

    Params:
    -------------
    data: pd.DataFrame
        columns include (['code', 'date', 'market_cap', 'ret', 'Size', 'Volatility', ..., 'industry_1', 'industry_2', ...])
    S: int
        number of style factors
    I: int 
        number of industry factors
    '''
    factor_ret = []
    specific_ret = []
    R2 = []

    dates = pd.to_datetime(data.date.values)
    






    

    



