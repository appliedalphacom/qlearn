from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from ira.analysis.tools import scols
from qlearn.core.base import MarketInfo
from qlearn.core.data_utils import detect_data_type, ohlc_to_flat_price_series, forward_timeseries


class ForwardDirectionScoring:
    """
    Forward returns direction scoring class (binary classifiction)
    """

    def __init__(self, period: Union[str, pd.Timedelta], min_threshold=0):
        self.period = pd.Timedelta(period) if isinstance(period, str) else period
        self.min_threshold = min_threshold

    def _extract_market_info(self, estimator):
        q_obj = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
        return getattr(q_obj, 'market_info_', None)

    def __call__(self, estimator, data, _):
        pred = estimator.predict(data)

        # we skip empty signals set
        if len(pred) == 0:
            return 0

        dt = detect_data_type(data)
        if dt.type == 'ohlc':
            mi: MarketInfo = self._extract_market_info(estimator)
            freq = pd.Timedelta(dt.freq)
            prices = ohlc_to_flat_price_series(data, freq, mi.session_start, mi.session_end)
            f_prices = forward_timeseries(prices, self.period)

        elif dt.type == 'ticks':
            # here we will use midprices as some first approximation
            prices = 0.5 * (data.bid + data.ask)
            f_prices = forward_timeseries(prices, self.period)

        else:
            raise ValueError(f"Don't know how to derive forward returns from '{dt.type}' data")

        # forward changes
        dp = f_prices - prices
        if self.min_threshold > 0:
            dp = dp.where(abs(dp) >= self.min_threshold, 0)

        returns = np.sign(dp)
        # ni = [returns.index.get_loc(i, method='ffill') for i in pred.index]
        # rp = returns.iloc[ni]
        # rp = returns.loc[pred.index]

        _returns = returns[~returns.index.duplicated(keep='first')]
        rp = _returns.reindex(pred.index).dropna()

        yc = scols(rp, pred, keys=['rp', 'pred']).dropna()

        return accuracy_score(yc.rp, yc.pred)
