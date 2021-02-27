from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from ira.analysis.tools import scols
from qlearn.core.base import MarketInfo
from qlearn.core.data_utils import detect_data_type, ohlc_to_flat_price_series, forward_timeseries


def _extract_market_info(estimator):
    q_obj = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    return getattr(q_obj, 'market_info_', None)


class ForwardDataProvider:

    def __init__(self, period: Union[str, pd.Timedelta]):
        self.period = pd.Timedelta(period) if isinstance(period, str) else period

    def get_forward_data(self, estimator, data):
        dt = detect_data_type(data)
        if dt.type == 'ohlc':
            mi: MarketInfo = _extract_market_info(estimator)
            freq = pd.Timedelta(dt.freq)
            prices = ohlc_to_flat_price_series(data, freq, mi.session_start, mi.session_end)
            f_prices = forward_timeseries(prices, self.period)

        elif dt.type == 'ticks':
            # here we will use midprices as some first approximation
            prices = 0.5 * (data.bid + data.ask)
            f_prices = forward_timeseries(prices, self.period)

        else:
            raise ValueError(f"Don't know how to derive forward returns from '{dt.type}' data")
        return prices, f_prices


class ForwardDirectionScoring(ForwardDataProvider):
    """
    Forward returns direction scoring class (binary classifiction)
    """

    def __init__(self, period: Union[str, pd.Timedelta], min_threshold=0):
        super().__init__(period)
        self.min_threshold = min_threshold

    def __call__(self, estimator, data, _):
        pred = estimator.predict(data)

        # we skip empty signals set
        if len(pred) == 0:
            return 0

        # get prices / forward prices
        prices, f_prices = self.get_forward_data(estimator, data)

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


class ForwardReturnsSharpeScoring(ForwardDataProvider):
    COMMS = {'bitmex': (0.075, True), 'okex': (0.05, True),
             'binance': (0.04, True), 'dukas': (35 * 100 / 1e6, False)}

    def __init__(self, period: Union[str, pd.Timedelta], commissions=0, crypto_futures=False):
        super().__init__(period)

        # possible to pass name of exchange
        comm = commissions
        if isinstance(commissions, str):
            comm_info = ForwardReturnsSharpeScoring.COMMS.get(commissions, (0, False))
            comm = comm_info[0]
            crypto_futures = comm_info[1]

            # commissions are required in percentages
        self.commissions = comm / 100
        self.crypto_futures = crypto_futures

    def calculate_returns(self, estimator, data):
        pred = estimator.predict(data)

        # we skip empty signals set
        if len(pred) == 0:
            return None

        # get prices / forward prices
        prices, f_prices = self.get_forward_data(estimator, data)

        if self.crypto_futures:
            # pnl on crypto is calculated as following
            dp = 1 / prices - 1 / f_prices

            # commissions are dependent on prices
            dpc = scols(dp, self.commissions * 1 / f_prices, names=['D', 'C'])
        else:
            dp = f_prices - prices

            # commissions are fixed
            dpc = scols(dp, pd.Series(self.commissions, dp.index), names=['D', 'C'])

        # drop duplicated indexes if exist (may happened on tick data)
        dpc = dpc[~dpc.index.duplicated(keep='first')]
        rpc = dpc.reindex(pred.index).dropna()

        yc = scols(rpc, pred.rename('pred')).dropna()
        return yc.D * yc.pred - yc.C

    def __call__(self, estimator, data, _):
        rets = self.calculate_returns(estimator, data)

        if rets is None:
            return -1e6

        # measure is proratio to Sharpe
        std = np.nanstd(rets)
        return (np.nanmean(rets) / std) if std != 0 else -1e6
