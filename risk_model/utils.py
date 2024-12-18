import pandas as pd
import numpy as np

def cap_norm(factors,
             market_cap):
    '''
    市值标准化

    Params:
    -------------
    factors: np.ndarray
    market_cap: np.ndarray
    '''
    f_mean = np.sum(factors * market_cap, axis=0) / np.sum(market_cap, axis=0)
    f_std = np.sqrt(np.sum(market_cap * (factors - f_mean) ** 2, axis=0) / np.sum(market_cap, axis=0))
    
    return (factors - f_mean) / f_std













