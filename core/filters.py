from sklearn.base import BaseEstimator

from ira.analysis.timeseries import adx, atr
from ira.analysis.tools import ohlc_resample


class AdxFilter(BaseEstimator):
    """
    ADX based trend filter. When adx > threshold
    """

    def __init__(self, timeframe, period, threshold, smoother='ema'):
        self.timeframe = timeframe
        self.period = period
        self.threshold = threshold
        self.smoother = smoother

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        a = adx(ohlc_resample(x, self.timeframe), self.period, smoother=self.smoother, as_frame=True).shift(1)
        return a.ADX > self.threshold


class AcorrFilter(BaseEstimator):
    """
    Autocorrelation filter on returns series
     If above is True (default) returns True for acorr > threshold
     If above is False returns True for acorr < threshold
    """

    def __init__(self, timeframe, lag, period, threshold, above=True):
        self.lag = lag
        self.period = period
        self.threshold = threshold
        self.timeframe = timeframe
        self.above = above

    def fit(self, x, y, **kwargs):
        return self

    def rolling_autocorrelation(self, x, lag, period):
        """
        Timeseries rolling autocorrelation indicator
        :param period: rolling window
        :param lag: lagged shift used for finding correlation coefficient
        """
        return x.rolling(period).corr(x.shift(lag))

    def predict(self, x):
        xr = ohlc_resample(x[['open', 'high', 'low', 'close']], self.timeframe)
        returns = xr.close.pct_change()
        ind = self.rolling_autocorrelation(returns, self.lag, self.period).shift(1)
        return (ind > self.threshold) if self.above else (ind < self.threshold)


class VolatilityFilter(BaseEstimator):
    """
    Regime based on volatility
       False: flat
       True:  volatile market
    """

    def __init__(self, timeframe, instant_period, typical_period, factor=1):
        self.instant_period = instant_period
        self.typical_period = typical_period
        self.factor = factor
        self.timeframe = timeframe

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        xr = ohlc_resample(x, self.timeframe)
        inst_vol = atr(xr, self.instant_period).shift(1)
        typical_vol = atr(xr, self.typical_period).shift(1)
        return inst_vol > typical_vol * self.factor


class AtrFilter(BaseEstimator):
    """
    Raw ATR filter
    """
    def __init__(self, timeframe, period, threshold, tz='UTC'):
        self.timeframe = timeframe
        self.period = period
        self.threshold = threshold
        self.tz = tz

    def fit(self, x, y, **kwargs):
        return self

    def get_filter(self, x):
        a = atr(ohlc_resample(x, self.timeframe, resample_tz=self.tz), self.period).shift(1)
        return a > self.threshold
