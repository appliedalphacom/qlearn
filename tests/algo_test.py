import unittest

import pandas as pd
from sklearn.pipeline import make_pipeline

from qlearn.core.base import MarketDataComposer
from qlearn.core.generators import RangeBreakoutDetector
from qlearn.core.pickers import SingleInstrumentPicker
from qlearn.core.transformers import RollingRange, FractalsRange
from qlearn.core.utils import debug_output


class AlgoTests(unittest.TestCase):

    def test_rolling_range_transformer(self):
        data = pd.read_csv('data/XBTUSD.csv.gz', parse_dates=True, index_col=['time'])
        # tf, period, f_shift = '10S', 30, 6
        # rr = RollingRange(tf, period, f_shift)

        tf, nf = '1Min', 2
        rr = FractalsRange(tf, nf)

        td = rr.transform(data)
        debug_output(td, 'RollingRange')

        bs = make_pipeline(rr, RangeBreakoutDetector(1))
        m2 = MarketDataComposer(bs, SingleInstrumentPicker(), None, debug=True)
        y0 = m2.fit(data, None).predict(data)

        debug_output(y0, 'Breaks')

        if False:
            import matplotlib.pyplot as plt
            _Z = slice('2021-02-07 00:50:45', '2021-02-07 00:50:46')
            rbt = td[['RangeBot', 'RangeTop']]
            plt.plot(data[['bid', 'ask']], '.', ms=0.5)
            plt.plot(rbt, lw=0.3)
            pT = data.loc[y0[y0 > 0].dropna().index].ask
            pB = data.loc[y0[y0 < 0].dropna().index].bid
            plt.plot(pT, '^', c='g', ms=5)
            plt.plot(pB, 'v', c='r', ms=5)
            plt.show()
