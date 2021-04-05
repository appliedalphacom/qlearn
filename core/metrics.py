from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from ira.analysis.tools import scols
from qlearn.core.base import MarketInfo, _FIELD_MARKET_INFO
from qlearn.core.data_utils import detect_data_type, ohlc_to_flat_price_series, forward_timeseries
from qlearn.core.utils import debug_output


def _find_estimator(x):
    """
    Tries to find estimator in nested pipelines
    """
    if isinstance(x, Pipeline):
        return _find_estimator(x.steps[-1][1])
    return x


class ForwardDataProvider:

    def __init__(self, period: Union[str, pd.Timedelta]):
        self.period = pd.Timedelta(period) if isinstance(period, str) else period

    def preprocess_price_data(self, estimator, data):
        dt = detect_data_type(data)
        if dt.type == 'ohlc':
            mi: MarketInfo = getattr(_find_estimator(estimator), _FIELD_MARKET_INFO, None)
            if mi is None:
                raise Exception(f"Can't exctract market info data from {estimator}")
            freq = pd.Timedelta(dt.freq)
            prices = ohlc_to_flat_price_series(data, freq, mi.session_start, mi.session_end)

        elif dt.type == 'ticks':
            # here we will use midprices as some first approximation
            prices = 0.5 * (data.bid + data.ask)

        else:
            raise ValueError(f"Don't know how to derive forward returns from '{dt.type}' data")
        return prices

    def get_forward_data(self, estimator, data):
        prices = self.preprocess_price_data(estimator, data)
        f_prices = forward_timeseries(prices, self.period)
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

        # drop nan's
        dp[np.isnan(dp)] = 0
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

    def __init__(self, period: Union[str, pd.Timedelta], commissions=0, crypto_futures=False, debug=False):
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
        self.debug = debug

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
        sharpe_metric = -1e6
        if rets is not None:
            # measure is proratio to Sharpe
            std = np.nanstd(rets)
            sharpe_metric = (np.nanmean(rets) / std) if std != 0 else -1e6

            if self.debug:
                debug_output(data, 'Metric data', time_info=True)
                print(f'\t->> Estimator: {estimator}')
                print(f'\t->> Metric: {sharpe_metric:.4f}')

        return sharpe_metric


class ReverseSignalsSharpeScoring(ForwardReturnsSharpeScoring):
    """
    Scoring for reversive signals
    """

    def __init__(self, commissions=0, crypto_futures=False, debug=False):
        super().__init__(None, commissions, crypto_futures, debug)

    def calculate_returns(self, estimator, data):
        pred = estimator.predict(data)

        # we skip empty signals set
        if len(pred) == 0:
            return None

        # we need only points where position is reversed
        revere_pts = pred[pred.diff() != 0].dropna()

        # price series
        price = self.preprocess_price_data(estimator, data)
        prices = price.loc[revere_pts.index]
        f_prices = prices.shift(-1)

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
        rpc = dpc.reindex(revere_pts.index).dropna()

        yc = scols(rpc, revere_pts.rename('pred')).dropna()
        return yc.D * yc.pred - yc.C
