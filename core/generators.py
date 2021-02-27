import pandas as pd
from sklearn.base import BaseEstimator

from ira.analysis.tools import srows, scols
from qlearn.core.base import signal_generator
from qlearn.core.data_utils import pre_close_time_shift
from qlearn.core.utils import _check_frame_columns


def crossup(x, t):
    return x[(x > t) & (x.shift(1) <= t)].index


def crossdown(x, t):
    return x[(x < t) & (x.shift(1) >= t)].index


@signal_generator
class RangeBreakoutDetector(BaseEstimator):
    """
    Detects breaks of rolling range. +1 for breaking upper range and -1 for bottom one.
    """

    def __init__(self, threshold=0, filter_indicator=None):
        self.threshold = threshold
        self.filter_indicator = filter_indicator

    def fit(self, X, y, **fit_params):
        return self

    def _ohlc_breaks(self, X):
        U, B = X.RangeTop + self.threshold, X.RangeBot - self.threshold
        open, close, high, low = X.open, X.close, X.high, X.low

        b1_bU = high.shift(1) <= U.shift(1)
        b1_aL = low.shift(1) >= B.shift(1)
        l_c = (b1_bU | (open <= U)) & (close > U)
        s_c = (b1_aL | (open >= B)) & (close < B)
        l_o = (b1_bU & (open > U))
        s_o = (b1_aL & (open < B))

        pre_close = pre_close_time_shift(X)

        return srows(
            pd.Series(+1, X[l_o].index), pd.Series(+1, X[(l_c & ~l_o)].index + pre_close),
            pd.Series(-1, X[s_o].index), pd.Series(-1, X[(s_c & ~s_o)].index + pre_close),
        )

    def _ticks_breaks(self, X):
        U, B = X.RangeTop + self.threshold, X.RangeBot - self.threshold
        a, b = X.ask, X.bid

        break_up = (a.shift(1) <= U.shift(1)) & (a > U)
        break_dw = (b.shift(1) >= B.shift(1)) & (b < B)

        return srows(pd.Series(+1, X[break_up].index), pd.Series(-1, X[break_dw].index))

    def predict(self, X):
        # take control on how we produce timestamps for signals
        self.exact_time = True

        try:
            _check_frame_columns(X, 'RangeTop', 'RangeBot', 'open', 'high', 'low', 'close')
            y0 = self._ohlc_breaks(X)

        except ValueError:
            _check_frame_columns(X, 'RangeTop', 'RangeBot', 'bid', 'ask')
            y0 = self._ticks_breaks(X)

        # if we want to filter out some signals
        if self.filter_indicator is not None and self.filter_indicator in X.columns:
            ms = scols(X[self.filter_indicator], y0, names=['F', 'S']).dropna()
            y0 = ms[(ms.F > 0) & (ms.S != 0)].S

        return y0


@signal_generator
class PivotsBreakoutDetector(BaseEstimator):
    @staticmethod
    def _to_list(x):
        return [x] if not isinstance(x, (list, tuple)) else x

    def __init__(self, resistances, supports):
        self.res_levels = self._tolist(resistances)
        self.sup_levels = self._tolist(supports)

    def fit(self, X, y, **fit_params):
        return self

    def predict(self, x):
        _check_frame_columns(x, 'open', 'close')

        t = scols(x, x.shift(1)[['open', 'close']].rename(columns={'open': 'open_1', 'close': 'close_1'}))
        cols = x.columns
        breaks = srows(
            # breaks up levels specified as resistance
            *[pd.Series(+1, t[(t.open_1 < t[ul]) & (t.close_1 < t[ul]) & (t.close > t[ul])].index) for ul in
              self.res_levels if ul in cols],

            # breaks down levels specified as supports
            *[pd.Series(-1, t[(t.open_1 > t[bl]) & (t.close_1 > t[bl]) & (t.close < t[bl])].index) for bl in
              self.sup_levels if bl in cols],
            keep='last')
        return breaks
