import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from ira.analysis.tools import ohlc_resample, scols
from qlearn.core.base import signal_generator
from qlearn.core.utils import _check_frame_columns


@signal_generator
class RollingRange(TransformerMixin, BaseEstimator):
    """
    Produces rolling high/low range (top/bottom) indicator
    """

    def __init__(self, timeframe, period, tz='UTC'):
        self.period = period
        self.timeframe = timeframe
        self.tz = tz

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x):
        """
        Attaches RangeTop, RangeBot columns to source dataframe
        """
        _check_frame_columns(x, 'open', 'high', 'low', 'close')
        ohlc = ohlc_resample(x, self.timeframe, resample_tz=self.tz)
        hilo = scols(
            ohlc.rolling(self.period, min_periods=self.period).high.max(),
            ohlc.rolling(self.period, min_periods=self.period).low.min(),
            names=['RangeTop', 'RangeBot'])
        hilo.index = hilo.index + pd.Timedelta(self.timeframe)
        hilo = x.combine_first(hilo).fillna(method='ffill')[hilo.columns]
        return x.assign(RangeTop=hilo.RangeTop, RangeBot=hilo.RangeBot)
