from collections import OrderedDict
import pandas as pd


def compute_exposures(
        positions,
        factor_loadings):
    '''
    compute daily risk factor exposures

    Params:
    -----------
    positions: pd.Series
        daily holdings as percentages
        index: multi-index(1 - date, 2 - code), value: weight

    factor_loadings: pd.DataFrame
        factor loadings for all days in the date range
        index: multi-index(1 - date, 2 - code), columns: factor, value: factor_loading

    Returns:
    -----------
    risk_exposures_portfolio: pd.DataFrame
        index: date, columns: factor, values: portfolio risk exposures
    '''
    risk_exposures = factor_loadings.multiply(positions, axis='rows')
    return risk_exposures.groupby(level='date').sum()



def risk_attribute(ret,
                   positions,
                   factor_ret,
                   factor_loadings):
    '''
    Params:
    -----------
    ret: pd.Series
        returns for each day in the date range
        index: date, value: return

    positions: pd.Series
        daily holdings in percentage
        index: multi-index(1 - date, 2 - code), value: weight

    factor_ret: pd.DataFrame
        factor's return
        index: date, columns: factor, value: factor_return
    
    factor_loadings: pd.DataFrame
        factor loadings for all days in the date range
        index: multi-index(1 - date, 2 - code), columns: factor, value: factor_loading

    Returns:
    -----------
    (risk_exposures_portfolio, risk_attribution): Tuple

    risk_exposures_portfolio: pd.DataFrame
        index: date, columns: factor, values:portfolio risk exposures

    risk_attribution: pd.DataFrame
        index: date, columns: factor + common_returns + specific_returns, values: risk attribution
    '''
    
    start, end = ret.index[0], ret.index[-1]   
    factor_ret = factor_ret.loc[start:end]
    factor_loadings = factor_loadings.loc[start:end]

    factor_loadings = factor_loadings.index.set_names(['date', 'code'])
    positions = positions.copy()
    positions = positions.index.set_names(['date', 'code'])

    risk_exposures_portfolio = compute_exposures(positions, factor_loadings)

    risk_attribute_by_factor = risk_exposures_portfolio.multiply(factor_ret)
    common_returns = risk_attribute_by_factor.sum(axis='columns')

    tilt_exposure = risk_exposures_portfolio.mean()
    tilt_returns = factor_ret.multiply(tilt_exposure).sum(axis='columns')
    timing_returns = common_returns - tilt_returns
    specific_returns = ret - common_returns

    ret_df = pd.DataFrame(OrderedDict([
        ('total_returns', ret),
        ('common_returns', common_returns),
        ('specific_returns', specific_returns),
        ('tilt_returns', tilt_returns),
        ('timing_returns', timing_returns)
    ]))

    return (risk_exposures_portfolio,
            pd.concat([risk_attribute_by_factor, ret_df], axis='columns'))












    
