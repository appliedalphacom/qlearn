from unittest import TestCase

import pandas as pd
from sklearn.pipeline import make_pipeline

from qlearn import TimeExpirationTracker, FixedTrader
from qlearn.core.base import MarketDataComposer
from qlearn.core.generators import RangeBreakoutDetector
from qlearn.core.pickers import SingleInstrumentPicker
from qlearn.core.transformers import RollingRange
from qlearn.core.utils import debug_output
from qlearn.simulation.multisim import simulation


class Test(TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/ES.csv.gz', parse_dates=True, index_col=['time'])
        self.ds = {'ES': self.data}

    def test_simulation(self):
        m1 = MarketDataComposer(make_pipeline(RollingRange('1H', 12), RangeBreakoutDetector()),
                                SingleInstrumentPicker(), debug=True).fit(self.ds)

        m2 = MarketDataComposer(make_pipeline(RollingRange('3H', 4), RangeBreakoutDetector()), SingleInstrumentPicker(),
                                debug=True).fit(self.ds)

        r = simulation({
            'exp1 [simple break]': m1,
            'exp2 [time tracker]': [m2, TimeExpirationTracker('5H')]
        }, self.ds, 'forex', 'Test1')
        # debug_output(r.results[0].portfolio, 'Portfolio')

        self.assertAlmostEqual(24.50, r.results[0].portfolio['ES_PnL'].sum())
        self.assertAlmostEqual(46.75, r.results[1].portfolio['ES_PnL'].sum())

        r.report(1000, only_report=True)

    def test_simulation_fixed(self):
        m1 = MarketDataComposer(make_pipeline(RollingRange('1H', 12), RangeBreakoutDetector()),
                                SingleInstrumentPicker(), debug=True).fit(self.ds)

        r = simulation({
            'exp1 [FIXED TRADER]': [m1, FixedTrader(10, 30, 10, 1)]
        }, self.ds, 'forex', 'Test1')

        r.report(1000, only_report=True)

        print(r.results[0].trackers_stat)
        self.assertEqual(3, r.results[0].trackers_stat['ES']['takes'])
        self.assertEqual(41, r.results[0].trackers_stat['ES']['stops'])
        self.assertAlmostEqual(1255.0, r.results[0].portfolio['ES_PnL'].sum())
