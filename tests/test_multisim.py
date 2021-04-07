from unittest import TestCase
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline

from ira.analysis.tools import srows, ohlc_resample
from qlearn.core.base import MarketDataComposer, signal_generator
from qlearn.core.generators import RangeBreakoutDetector
from qlearn.core.metrics import ForwardDirectionScoring, ForwardReturnsSharpeScoring
from qlearn.core.pickers import SingleInstrumentPicker
from qlearn.core.transformers import RollingRange
from qlearn.core.utils import debug_output
from qlearn.simulation.multisim import simulation


class Test(TestCase):
    def test_simulation(self):
        data = pd.read_csv('data/ES.csv.gz', parse_dates=True, index_col=['time'])

        bs = make_pipeline(RollingRange('1H', 12), RangeBreakoutDetector())
        m = MarketDataComposer(bs, SingleInstrumentPicker(), debug=True)
        r = simulation(m, {'ES': data}, 'forex', 'Test1')
        debug_output(r[0].portfolio, 'Portfolio')
        self.assertAlmostEqual(24.5, r[0].portfolio['ES_PnL'].sum())
