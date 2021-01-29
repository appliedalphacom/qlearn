import unittest

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline

from ira.analysis.tools import srows
from qlearn.core.base import BasicMarketEstimator, MarketDataComposer
from qlearn.core.forward_returns import ForwardReturnsDirection
from qlearn.core.pickers import SingleInstrumentPicker
from qlearn.core.utils import debug_output


class _WeekOpenRange(TransformerMixin, BaseEstimator):

    @staticmethod
    def find_week_start_time(data, week_start_day=6):
        d1 = data.assign(time=data.index)
        return d1[d1.index.weekday == week_start_day].groupby(pd.Grouper(freq='1d')).first().dropna().time.values

    def __init__(self, open_interval, tick_size=0.25):
        # self.open_interval = pd.Timedelta(open_interval)
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


class _RangeBreakoutDetector(BasicMarketEstimator):
    def fit(self, X, y, **fit_params):
        return self

    def predict(self, X):
        meta = self.metadata()
        if not all([c in X.columns for c in ['RangeTop', 'RangeBot']]):
            raise ValueError("Can't find 'RangeTop', 'RangeBot' in input data !")
        U, B = X.RangeTop, X.RangeBot
        l_i = ((X.close.shift(1) <= U.shift(1)) & (X.close > U)) | ((X.open <= U) & (X.close > U))
        s_i = ((X.close.shift(1) >= B.shift(1)) & (X.close < B)) | ((X.open >= B) & (X.close < B))
        return srows(pd.Series(+1, X[l_i].index), pd.Series(-1, X[s_i].index))


class BaseFunctionalityTests(unittest.TestCase):

    def test_prediction_alignment(self):
        data = pd.read_csv('data/ES.csv.gz', parse_dates=True, index_col=['time'])
        debug_output(data, 'ES')

        wor = make_pipeline(_WeekOpenRange('4Min', 0.25), _RangeBreakoutDetector().fillna(0).as_classifier())
        m1 = MarketDataComposer(wor, SingleInstrumentPicker(), ForwardReturnsDirection(horizon=60 * 60), debug=True)
        debug_output(m1.fit(data, None).predict(data), 'Predicted')

        g1 = GridSearchCV(
            cv=TimeSeriesSplit(2),
            estimator=wor,
            param_grid={
                'weekopenrange__open_interval': [pd.Timedelta(x) - pd.Timedelta('1Min') for x in [
                    '5Min', '10Min', '15Min', '20Min', '25Min', '30Min', '35Min', '40Min', '45Min'
                ]],
            }, verbose=True
        )

        mds = MarketDataComposer(g1,
                                 SingleInstrumentPicker(),
                                 ForwardReturnsDirection(horizon=60, debug=True), 'close', debug=True)
        mds.fit(data, None)
        print(g1.best_score_)
        print(g1.best_params_)
