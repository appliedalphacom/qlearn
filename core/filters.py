import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ira.analysis.timeseries import adx, atr
from ira.analysis.tools import ohlc_resample
from qlearn.core.base import Filter
from qlearn.core.utils import _check_frame_columns


class AdxFilter(BaseEstimator, Filter):
    def __init__(self, timeframe, period, threshold, smoother='kama'):
        self.timeframe = timeframe
        self.period = period
        self.threshold = threshold
        self.smoother = smoother

    def get_filter(self, x):
        _check_frame_columns(x, 'open', 'high', 'low', 'close')

        a = adx(ohlc_resample(x[['open', 'high', 'low', 'close']], self.timeframe),
                self.period, smoother=self.smoother, as_frame=True).shift(1)
        return np.sign(a.ADX.where(a.ADX > self.threshold, 0))


class AcorrFilter(BaseEstimator, Filter):
    """
    Regime based returns series autocorrelation
      -1: mean reversion regime (negative correlation < t_mr)
       0: uncertain (t_mr < corr < r_mo)
      +1: momentum regime (positive correlation > t_mo)
    """

    def __init__(self, lag, period, mr, mo, timeframe=None):
        self.lag = lag
        self.period = period
        self.mr = mr
        self.mo = mo
        self.timeframe = timeframe

    def fit(self, x, y, **kwargs):
        return self

    def rolling_autocorrelation(self, x, lag, period):
        """
        Timeseries rolling autocorrelation indicator
        :param period: rolling window
        :param lag: lagged shift used for finding correlation coefficient
        """
        return x.rolling(period).corr(x.shift(lag))

    def get_filter(self, x):
        _check_frame_columns(x, 'open', 'high', 'low', 'close')

        xr = x
        if self.timeframe:
            xr = ohlc_resample(x[['open', 'high', 'low', 'close']], self.timeframe)

        returns = xr.close.pct_change()
        ind = self.rolling_autocorrelation(returns, self.lag, self.period).shift(1)
        r = pd.Series(0, index=ind.index)
        r[ind <= self.mr] = -1
        r[ind >= self.mo] = +1

        return r


class VolatilityFilter(BaseEstimator, Filter):
    """
    Regime based on volatility
       0: flat
      +1: volatile market
    """

    def __init__(self, timeframe, instant_period, typical_period, factor=1):
        self.instant_period = instant_period
        self.typical_period = typical_period
        self.factor = factor
        self.timeframe = timeframe

    def fit(self, x, y, **kwargs):
        return self

    def get_filter(self, x):
        _check_frame_columns(x, 'open', 'high', 'low', 'close')

        xr = x
        if self.timeframe:
            xr = ohlc_resample(x[['open', 'high', 'low', 'close']], self.timeframe)

        inst_vol = atr(xr, self.instant_period).shift(1)
        typical_vol = atr(xr, self.typical_period).shift(1)
        r = pd.Series(0, index=xr.index)
        r[inst_vol > typical_vol * self.factor] = +1

        return r


class AtrFilter(BaseEstimator, Filter):
    def __init__(self, timeframe, period, threshold, tz='UTC'):
        self.timeframe = timeframe
        self.period = period
        self.threshold = threshold
        self.tz = tz

    def fit(self, x, y, **kwargs):
        return self

    def get_filter(self, x):
        _check_frame_columns(x, 'open', 'high', 'low', 'close')

        a = atr(ohlc_resample(x[['open', 'high', 'low', 'close']], self.timeframe, resample_tz=self.tz),
                self.period).shift(1)

        r = pd.Series(0, index=x.index)
        r[a > self.threshold] = +1
        
        return r
