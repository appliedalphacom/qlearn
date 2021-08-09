import unittest

import pandas as pd

from ira.utils.utils import mstruct

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

from ira.utils.nb_functions import z_backtest
from qlearn.tracking.trackers import (TakeStopTracker, DispatchTracker, PipelineTracker, Tracker, TimeExpirationTracker,
                                      TriggeredOrdersTracker, TriggerOrder)


def _read_csv_ohlc(symbol):
    return {symbol: pd.read_csv(f'data/{symbol}.csv', parse_dates=True, header=0, index_col='time')}


def _signals(sdata):
    s = pd.DataFrame.from_dict(sdata, orient='index')
    s.index = pd.DatetimeIndex(s.index)
    return s


class _Test_NoRiskManagementTracker(Tracker):
    def __init__(self, size):
        self.size = size

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        return signal_qty * self.size


class _Test_StopTakeTracker(TakeStopTracker):

    def __init__(self, size, stop_points, take_points, tick_size):
        super().__init__(True)
        self.size = size
        self.tick_size = tick_size
        self.stop_points = stop_points
        self.take_points = take_points

    def on_take(self, timestamp, price, user_data=None):
        print(f" ----> TAKE: {user_data}")

    def on_stop(self, timestamp, price, user_data=None):
        print(f" ----> STOP: {user_data}")

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        mp = (bid + ask) / 2
        if signal_qty > 0:
            if self.stop_points is not None:
                self.debug(f' >> STOP LONG at {mp - self.stop_points * self.tick_size}')
                self.stop_at(signal_time, mp - self.stop_points * self.tick_size, "Stop user data long")

            if self.take_points is not None:
                self.take_at(signal_time, mp + self.take_points * self.tick_size, "Take user data long")

        elif signal_qty < 0:
            if self.stop_points is not None:
                self.debug(f' >> STOP SHORT at {mp - self.stop_points * self.tick_size}')
                self.stop_at(signal_time, mp + self.stop_points * self.tick_size, "Stop user data short")

            if self.take_points is not None:
                self.take_at(signal_time, mp - self.take_points * self.tick_size, "Take user data short")

        return signal_qty * self.size


class Trackers_test(unittest.TestCase):

    def test_dispatcher(self):
        data = _read_csv_ohlc('EURUSD')

        s = _signals({
            '2020-08-17 04:10:00': {'EURUSD': 'regime:trend'},
            '2020-08-17 04:19:59': {'EURUSD': +1},
            '2020-08-17 14:19:59': {'EURUSD': -1},
            '2020-08-17 14:55:59': {'EURUSD': +1},  # this should be flat !
            '2020-08-17 15:00:00': {'EURUSD': 'regime:mr'},
            '2020-08-17 18:19:59': {'EURUSD': 1},
            '2020-08-17 20:19:59': {'EURUSD': 'empty'},
            '2020-08-17 20:24:59': {'EURUSD': 1},  # this should be passed !
            '2020-08-17 23:19:59': {'EURUSD': 0},
        })

        p = z_backtest(s, data, 'forex', spread=0, execution_logger=True,
                       trackers=DispatchTracker(
                           {
                               'regime:trend': _Test_StopTakeTracker(10000, 50, None, 1e-5),
                               'regime:mr': PipelineTracker(
                                   TimeExpirationTracker('1h', True),
                                   _Test_NoRiskManagementTracker(777)
                               ),
                               'empty': None
                           }, None, flat_position_on_activate=True, debug=True)
                       )

        print(p.executions)
        print(p.trackers_stat)
        execs_log = list(filter(lambda x: x != '', p.executions.comment.values))
        print(execs_log)

        self.assertListEqual(
            ['stop long at 1.18445',
             'stop short at 1.1879499999999998',
             '<regime:mr> activated and flat position',
             'TimeExpirationTracker:: position 777 is expired'],
            execs_log)

    def test_triggered_orders(self):

        class StopOrdersTestTracker(TriggeredOrdersTracker):
            def __init__(self, tick_size):
                super().__init__(True)
                self.tick_size = tick_size
                self.to = None
                self._fired = 0

            def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
                if signal_qty > 0:
                    entry = ask + 50 * self.tick_size
                    self.to = self.stop_order(
                        entry, 1000, entry - 25 * self.tick_size, entry + 25 * self.tick_size,
                        comment='My test Order', user_data=mstruct(entry_number=1, test=1)
                    )
                return None

            def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
                super().on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

                if self.to is not None:
                    if self.to.fired:
                        print(quote_time, self.to)
                        self.to = None

            def on_trigger_fired(self, timestamp, order: TriggerOrder):
                print(f"\n\t---(FIRED)--> {timestamp} | {order} => {order.user_data} ")
                self._fired += 1

            def statistics(self):
                return {'fired': self._fired, **super().statistics()}

        data = _read_csv_ohlc('EURUSD')

        s = _signals({
            '2020-08-17 04:19:59': {'EURUSD': +1},
            '2020-08-17 23:19:59': {'EURUSD': 0},
        })

        track = StopOrdersTestTracker(1e-5)
        p = z_backtest(s, data, 'forex', spread=0, execution_logger=True, trackers=track)

        print(p.executions)
        print(p.trackers_stat)

        self.assertTrue(p.trackers_stat['EURUSD']['fired'] > 0)

    def test_take_stop_orders(self):
        data = _read_csv_ohlc('RM1')
        s = _signals({
            '2020-08-17 00:00:01': {'RM1': +1},
            '2020-08-17 00:22:00': {'RM1': 0},
        })

        p = z_backtest(s, data, 'forex', spread=0, execution_logger=True,
                       trackers=_Test_StopTakeTracker(10000, None, 16, 1))

        print(p.executions)
        print(p.trackers_stat)
