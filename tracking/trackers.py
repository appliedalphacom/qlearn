from typing import Dict

import numpy as np

from ira.simulator.SignalTester import Tracker


class TakeStopTracker(Tracker):
    """
    Simple stop/take tracker provider
    """

    def __init__(self, debug=False):
        self.take = None
        self.stop = None
        self.n_stops = 0
        self.n_takes = 0
        self.times_to_take = []
        self.times_to_stop = []
        # what is last triggered event: 'stop' or 'take' (None if nothing was triggered yet)
        self.last_triggered_event = None
        if debug:
            self.debug = print

    def debug(self, *args, **kwargs):
        pass

    def stop_at(self, trade_time, stop_price: float):
        self.stop = stop_price

    def take_at(self, trade_time, take_price: float):
        self.take = take_price

    def trade(self, trade_time, quantity):
        if quantity == 0:
            self.stop = None
            self.take = None

        # call super method
        super().trade(trade_time, quantity)

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        if self._position.quantity > 0:
            if bid >= self.take:
                self.debug(f' + take long [{self._instrument}] at {bid:.5f}')
                self.times_to_take.append(quote_time - self._last_trade_time)
                self.trade(quote_time, 0)
                self.n_takes += 1
                self.last_triggered_event = 'take'
                return

            if ask <= self.stop:
                self.debug(f' - stop long [{self._instrument}] at {ask:.5f}')
                self.times_to_stop.append(quote_time - self._last_trade_time)
                self.trade(quote_time, 0)
                self.n_stops += 1
                self.last_triggered_event = 'stop'
                return

        if self._position.quantity < 0:
            if ask <= self.take:
                self.debug(f' + take short [{self._instrument}] at {ask:.5f}')
                self.times_to_take.append(quote_time - self._last_trade_time)
                self.trade(quote_time, 0)
                self.n_takes += 1
                self.last_triggered_event = 'take'
                return

            if bid >= self.stop:
                self.debug(f' - stop short [{self._instrument}] at {bid:.5f}')
                self.times_to_stop.append(quote_time - self._last_trade_time)
                self.trade(quote_time, 0)
                self.n_stops += 1
                self.last_triggered_event = 'stop'
                return

    def statistics(self) -> Dict:
        return {
            'takes': self.n_takes,
            'stops': self.n_stops,
            'average_time_to_take': np.mean(self.times_to_take) if self.times_to_take else np.nan,
            'average_time_to_stop': np.mean(self.times_to_stop) if self.times_to_stop else np.nan,
        }


class ProgressionTracker(Tracker):
    pass


class TurtlesTracker(Tracker):
    pass
