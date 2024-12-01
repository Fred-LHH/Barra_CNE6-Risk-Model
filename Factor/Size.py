import pandas as pd
import numpy as np
from utils import Function

class Size(Function):

    def __init__(self,
                 df: pd.DataFrame):
        '''
        传入带有流通市值的DataFrame, columns: ['code', 'trade_date', 'circ_mv']
        '''
        df['circ_mv'] = df['circ_mv'] / 1e4 # 以亿元为单位
        self.df = df
        self.df['LNCAP'] = np.log(df['circ_mv']+1)
        self.df = self.MIDCAP()
        self.df['SIZE'] = (self.df['LNCAP'] + self.df['MIDCAP']) / 2
    
    def MIDCAP(self):
        self.df['sub_MIDCAP'] = self.df['LNCAP'] ** 3
        MIDCAP = self.df.groupby('trade_date').apply(self._regress, 'sub_MIDCAP', 'LNCAP', 'circ_mv')
        MIDCAP.name = 'MIDCAP'
        self.df = self.df.merge(MIDCAP.reset_index())
        return self.df
    
    def get_single_factor(self, col):
        return self._get_factor(self.df, col)