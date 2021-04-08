from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator

from ira.analysis.timeseries import smooth, rsi
from ira.analysis.tools import srows, scols
from qlearn.core.base import signal_generator
from qlearn.core.data_utils import pre_close_time_shift
from qlearn.core.utils import _check_frame_columns


def crossup(x, t: Union[pd.Series, float]):
    t1 = t.shift(1) if isinstance(t, pd.Series) else t
    return x[(x > t) & (x.shift(1) <= t1)].index


def crossdown(x, t: Union[pd.Series, float]):
    t1 = t.shift(1) if isinstance(t, pd.Series) else t
    return x[(x < t) & (x.shift(1) >= t1)].index


@signal_generator
class RangeBreakoutDetector(BaseEstimator):
    """
    Detects breaks of rolling range. +1 for breaking upper range and -1 for bottom one.
    """

    def __init__(self, threshold=0):
        self.threshold = threshold

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


@signal_generator
class CrossingMovings(BaseEstimator):
    def __init__(self, fast, slow, fast_type='sma', slow_type='sma'):
        self.fast = fast
        self.slow = slow
        self.fast_type = fast_type
        self.slow_type = slow_type

    def fit(self, x, y, **fit_args):
        return self

    def predict(self, x):
        price_col = self.market_info_.column
        fast_ma = smooth(x[price_col], self.fast_type, self.fast)
        slow_ma = smooth(x[price_col], self.slow_type, self.slow)

        return srows(
            pd.Series(+1, crossup(fast_ma, slow_ma)),
            pd.Series(-1, crossdown(fast_ma, slow_ma))
        )


@signal_generator
class Rsi(BaseEstimator):
    def __init__(self, period, lower=25, upper=75, smoother='sma'):
        self.period = period
        self.upper = upper
        self.lower = lower
        self.smoother = smoother

    def fit(self, x, y, **fit_args):
        return self

    def predict(self, x):
        price_col = self.market_info_.column
        r = rsi(x[price_col], self.period, smoother=self.smoother)
        return srows(pd.Series(+1, crossup(r, self.lower)), pd.Series(-1, crossdown(r, self.upper)))
