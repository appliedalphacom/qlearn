import pandas as pd
from sklearn.base import BaseEstimator

from ira.analysis.tools import srows
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
    Detects breaks of range
    """

    def __init__(self, threshold=0):
        self.threshold = threshold

    def fit(self, X, y, **fit_params):
        return self

    def predict(self, X):
        _check_frame_columns(X, 'RangeTop', 'RangeBot', 'open', 'high', 'low', 'close')

        open, close, high, low = X.open, X.close, X.high, X.low
        U, B = X.RangeTop + self.threshold, X.RangeBot - self.threshold

        b1_bU = high.shift(1) <= U.shift(1)
        b1_aL = low.shift(1) >= B.shift(1)
        l_c = (b1_bU | (open <= U)) & (close > U)
        s_c = (b1_aL | (open >= B)) & (close < B)
        l_o = (b1_bU & (open > U))
        s_o = (b1_aL & (open < B))

        pre_close = pre_close_time_shift(X)

        # take control on how we produce timestamps for signals
        self.exact_time = True

        return srows(
            pd.Series(+1, X[l_o].index), pd.Series(+1, X[(l_c & ~l_o)].index + pre_close),
            pd.Series(-1, X[s_o].index), pd.Series(-1, X[(s_c & ~s_o)].index + pre_close),
        )
