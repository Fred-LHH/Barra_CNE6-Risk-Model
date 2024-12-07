import pandas as pd
import numpy as np
from utils import Function

class Dividend(Function):

    def __init__(self,
                 dividend: pd.DataFrame,
                 price: pd.DataFrame):
        
        self.dividend = dividend[dividend['div_proc'] == '实施'][['code', 'end_date', 'ann_date', 'cash_div_tax']]
        self.dividend.reset_index(drop=True, inplace=True)
        self.price = price

        self.DTOP = self.DTOP()

    def DTOP(self):
        ttm_dividend = self._convert_ttm(self.dividend, cols=['cash_div_tax'])

        close = self.get_price_last_month_end(self.price)
        ttm_dividend = self._pubDate_align_tradedate(ttm_dividend, 'ann_date', '20240831')
        dividend = pd.merge(ttm_dividend, close, on=['code', 'trade_date'], how='left')
        dividend['DTOP'] = dividend.eval('cash_div_tax_TTM / last_month_end_close')
        DTOP = dividend[['code', 'trade_date', 'DTOP']]

        return DTOP
