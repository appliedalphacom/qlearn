import unittest

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline

from ira.analysis.timeseries import atr
from ira.analysis.tools import srows, ohlc_resample, scols
from qlearn.core.base import MarketDataComposer, signal_generator, _FIELD_FILTER_INDICATOR, QLEARN_VERSION
from qlearn.core.filters import AdxFilter
from qlearn.core.metrics import ForwardDirectionScoring
from qlearn.core.pickers import SingleInstrumentPicker
from qlearn.core.utils import debug_output


@signal_generator
class WeekOpenRangeTest(TransformerMixin, BaseEstimator):

    @staticmethod
    def find_week_start_time(data, week_start_day=6):
        d1 = data.assign(time=data.index)
        return d1[d1.index.weekday == week_start_day].groupby(pd.Grouper(freq='1d')).first().dropna().time.values

    def __init__(self, open_interval, tick_size=0.25):
        self.open_interval = open_interval
        self.tick_size = tick_size

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, data):
        ul = {}
        op_int = pd.Timedelta(self.open_interval) if isinstance(self.open_interval, str) else self.open_interval
        for wt0 in self.find_week_start_time(data):
            wt1 = data[wt0 + op_int:].index[0]
            ws = data[wt0:wt1]
            ul[wt1] = {'RangeTop': ws.high.max() + self.tick_size,
                       'RangeBot': ws.low.min() - self.tick_size}
        ulf = pd.DataFrame.from_dict(ul, orient='index')
        return data.combine_first(ulf).fillna(method='ffill')


@signal_generator
class RangeBreakoutDetectorTest(BaseEstimator):
    def fit(self, X, y, **fit_params):
        return self

    def predict(self, X):
        if not all([c in X.columns for c in ['RangeTop', 'RangeBot']]):
            raise ValueError("Can't find 'RangeTop', 'RangeBot' in input data !")
        U, B = X.RangeTop, X.RangeBot
        l_i = ((X.close.shift(1) <= U.shift(1)) & (X.close > U)) | ((X.open <= U) & (X.close > U))
        s_i = ((X.close.shift(1) >= B.shift(1)) & (X.close < B)) | ((X.open >= B) & (X.close < B))
        return srows(pd.Series(+1, X[l_i].index), pd.Series(-1, X[s_i].index))


class BaseFunctionalityTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/ES.csv.gz', parse_dates=True, index_col=['time'])
        debug_output(self.data, 'ES')

    def test_prediction_alignment(self):
        wor = make_pipeline(WeekOpenRangeTest('4Min', 0.25), RangeBreakoutDetectorTest())
        m1 = MarketDataComposer(wor, SingleInstrumentPicker(), debug=True)
        debug_output(m1.fit(self.data, None).predict(self.data), 'Predicted')

        g1 = GridSearchCV(
            cv=TimeSeriesSplit(2),
            estimator=wor,
            scoring=ForwardDirectionScoring('30Min'),
            param_grid={
                'weekopenrangetest__open_interval': [pd.Timedelta(x) - pd.Timedelta('1Min') for x in [
                    '5Min', '10Min', '15Min', '20Min', '25Min', '30Min', '35Min', '40Min', '45Min'
                ]],
            }, verbose=True
        )

        mds = MarketDataComposer(g1, SingleInstrumentPicker(), 'close', debug=True)
        mds.fit(self.data, None)
        print(g1.best_score_)
        print(g1.best_params_)
        self.assertAlmostEqual(0.4893939, g1.best_score_, delta=1e-5)

    def test_filters(self):
        f_wor = make_pipeline(
            WeekOpenRangeTest('4Min', 0.25),
            AdxFilter('15Min', 20, 25, 'ema').apply_to(RangeBreakoutDetectorTest())
        )

        m1 = MarketDataComposer(f_wor, SingleInstrumentPicker(), debug=True)
        debug_output(m1.fit(self.data, None).predict(self.data), 'Predicted')

        g1 = GridSearchCV(
            cv=TimeSeriesSplit(2),
            estimator=f_wor,
            scoring=ForwardDirectionScoring('30Min'),
            param_grid={
                'weekopenrangetest__open_interval': [pd.Timedelta(x) - pd.Timedelta('1Min') for x in [
                    '5Min', '10Min', '15Min', '20Min', '25Min', '30Min', '35Min', '40Min', '45Min'
                ]],
                'pipeline__AdxFilter__period': [20, 30],
                'pipeline__AdxFilter__timeframe': ['5Min', '15Min', '30Min']
            }, verbose=True
        )

        mds = MarketDataComposer(g1, SingleInstrumentPicker(), 'close')
        mds.fit(self.data, None)
        print(g1.best_score_)
        print(g1.best_params_)
        self.assertAlmostEqual(0.7692, g1.best_score_, delta=1e-4)
