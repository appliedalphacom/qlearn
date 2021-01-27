import unittest

import pandas as pd

from ira.utils.nb_functions import z_backtest
from qlearn.tracking.trackers import TakeStopTracker, DispatchTracker, PipelineTracker, Tracker, TimeExpirationTracker


class Trackers_test(unittest.TestCase):

    def test_dispathcher(self):

        class MyFixedPositionTracker(Tracker):
            def __init__(self, size):
                self.size = size

            def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
                return signal_qty * self.size

        class TrendFixedPositionTracker(TakeStopTracker):

            def __init__(self, size, tick_size):
                super().__init__(True)
                self.size = size
                self.tick_size = tick_size

            def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
                if signal_qty > 0:
                    self.stop_at(signal_time, (bid + ask) / 2 - 50 * self.tick_size)
                elif signal_qty < 0:
                    self.stop_at(signal_time, (bid + ask) / 2 + 50 * self.tick_size)

                return signal_qty * self.size

        data = pd.read_csv('data/EURUSD.csv', parse_dates=True, header=0, index_col='time')
        # print(data.head())

        s = pd.DataFrame.from_dict({
            '2020-08-17 04:10:00': {'EURUSD': 'regime:trend'},
            '2020-08-17 04:19:59': {'EURUSD': +1},
            '2020-08-17 14:19:59': {'EURUSD': -1},
            '2020-08-17 15:00:00': {'EURUSD': 'regime:mr'},
            '2020-08-17 18:19:59': {'EURUSD': 1},
            '2020-08-17 22:19:59': {'EURUSD': 0},
        }, orient='index')
        s.index = pd.DatetimeIndex(s.index)

        p = z_backtest(s, {'EURUSD': data}, 'forex', spread=0, execution_logger=True,
                       trackers=DispatchTracker(
                           {
                               'regime:trend': TrendFixedPositionTracker(10000, 1e-5),
                               'regime:mr': PipelineTracker(
                                   TimeExpirationTracker('1h', True),
                                   MyFixedPositionTracker(10000)
                               )

                           }, None, flat_position_on_activate=True, debug=True)
                       )

        print(p.executions)

        self.assertListEqual(
            ['stop long at 1.18445', 'stop short at 1.1879499999999998',
             'TimeExpirationTracker:: position 10000 is expired'],
            list(filter(lambda x: x != '', p.executions.comment.values)))
