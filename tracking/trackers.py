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
            if self.take and bid >= self.take:
                self.debug(f' -> [{quote_time}] take long [{self._instrument}] at {bid:.5f}')
                self.times_to_take.append(quote_time - self._last_trade_time)
                self.trade(quote_time, 0)
                self.n_takes += 1
                self.last_triggered_event = 'take'
                return

            if self.stop and ask <= self.stop:
                self.debug(f' -> [{quote_time}] stop long [{self._instrument}] at {ask:.5f}')
                self.times_to_stop.append(quote_time - self._last_trade_time)
                self.trade(quote_time, 0)
                self.n_stops += 1
                self.last_triggered_event = 'stop'
                return

        if self._position.quantity < 0:
            if self.take and ask <= self.take:
                self.debug(f' -> [{quote_time}] take short [{self._instrument}] at {ask:.5f}')
                self.times_to_take.append(quote_time - self._last_trade_time)
                self.trade(quote_time, 0)
                self.n_takes += 1
                self.last_triggered_event = 'take'
                return

            if self.stop and bid >= self.stop:
                self.debug(f' -> [{quote_time}] stop short [{self._instrument}] at {bid:.5f}')
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


class TurtleTracker(TakeStopTracker):
    """
    Our modifiaction of turtles money managenet algo
    """
    def __init__(self,
                 account_size, dollar_per_point, max_units=4, risk_capital_pct=0.01, reinvest_pnl_pct=0,
                 contract_size=100, max_allowed_contracts=200,
                 atr_timeframe='1d', pull_stops_on_incr=False, after_lose_only=False, debug=False):
        """
        :param accoun_size: starting amount in USD
        :param dollar_per_point: price of 1 point (for example 12.5 for ES mini) if none crypto sizing would be used
        :param max_untis: maximal number of position inreasings
        :param risk_capital_pct: percent of capital in risk (0.01 is 1%)
        :param reinvest_pnl_pct: percent of reinvestment pnl to trading (default is 0)
        :param contract_size: contract size in USD
        :param max_allowed_contracts: maximal allowed contracts to trade
        :param atr_timeframe: timeframe of ATR calculations
        :param pull_stops_on_incr: if true it pull up stop on position's increasing
        :param after_lose_only: if true
        :param debug: if true it prints debug messages
        """
        super().__init__(debug)
        self.account_size = account_size
        self.dollar_per_point = dollar_per_point
        self.atr_timeframe = atr_timeframe
        self.max_units = max_units
        self.trading_after_lose_only = after_lose_only
        self.pull_stops = pull_stops_on_incr
        self.risk_capital_pct = risk_capital_pct
        self.max_allowed_contracts = max_allowed_contracts
        self.reinvest_pnl_pct = reinvest_pnl_pct
        self.contract_size = contract_size

        if dollar_per_point is None:
            self.calculate_trade_size = self._calculate_trade_size_crypto

    def initialize(self):
        self.days = self.get_ohlc_series(self.atr_timeframe)
        self.N = None
        self.__TR_init_sum = 0
        self._n_entries = 0
        self._last_entry_price = np.nan

    def _get_size_at_risk(self):
        return (self.account_size + self.reinvest_pnl_pct * self._position.pnl) * self.risk_capital_pct

    def _calculate_trade_size_crypto(self, direction, vlt, price):
        price2 = price + direction * vlt
        return np.clip(round(self._get_size_at_risk() / ((price2 / price - 1) * self.contract_size)),
                       -self.max_allowed_contracts, self.max_allowed_contracts)

    def _calculate_trade_size_on_dollar_cost(self, direction, vlt, price):
        return min(self.max_units, round(self._get_size_at_risk() / (vlt * self.dollar_per_point))) * direction

    def calculate_trade_size(self, direction, vlt, price):
        return self._calculate_trade_size_on_dollar_cost(direction, vlt, price)

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        daily, n_days = self.days, len(self.days)
        today, yest = daily[1], daily[2]

        if yest is None or today is None:
            return

        if daily.is_new_bar:
            now_date = daily[0].time
            TR = max(today.high - today.low, today.high - yest.close, yest.close - today.low)
            if self.N is None:
                if n_days <= 21:
                    self.__TR_init_sum += TR
                else:
                    self.N = self.__TR_init_sum / 19
            else:
                self.N = (19 * self.N + TR) / 20

        # increasing position size if possible
        pos = self._position.quantity
        if pos != 0 and self._n_entries < self.max_units:
            n_2 = self.N / 2
            if (pos > 0 and ask > self._last_entry_price + n_2) or (pos < 0 and bid < self._last_entry_price - n_2):
                self._last_entry_price = ask if pos > 0 else bid
                t_size = self.calculate_trade_size(np.sign(pos), self.N, self._last_entry_price)
                self._n_entries += 1

                # increase inventory
                self.trade(quote_time, pos + t_size)

                # average position price
                avg_price = self._position.cost_usd / abs(self._position.quantity)

                # pull stops
                if self.pull_stops:
                    self.stop_at(quote_time, avg_price - self.N * 2 * np.sign(pos))

                self.debug(
                    f"\t[{quote_time}] -> [#{self._n_entries}] {self._instrument} <{avg_price:.2f}> increasing to "
                    f"{pos + t_size} @ {self._last_entry_price} x {self.stop:.2f}")

        # call stop/take tracker to process sl/tp if need
        super().on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

    def on_signal(self, signal_time, signal, quote_time, bid, ask, bid_size, ask_size):
        if self.N is None:
            return None

        s_type, s_direction, t_size = abs(signal), np.sign(signal), None
        position = self._position.quantity

        # when we want to enter position
        if position == 0 and s_type == 1:
            if not self.trading_after_lose_only or self.last_triggered_event != 'stop':
                self._last_entry_price = ask if s_direction > 0 else bid
                t_size = self.calculate_trade_size(s_direction, self.N, self._last_entry_price)
                self.stop_at(signal_time, self._last_entry_price - self.N * 2 * s_direction)
                self.last_triggered_event = None
                self._n_entries = 1
                self.debug(
                    f'\t[{signal_time}] -> [#{self._n_entries}] {self._instrument} {t_size} @ '
                    f'{self._last_entry_price:.2f} x {self.stop:.2f}')

        # when we got to exit signal
        if (position > 0 and signal == -2) or (position < 0 and signal == +2):
            self.last_triggered_event = 'take'
            self._n_entries = 0
            t_size = 0
            self.debug(f'[{signal_time}] -> Close in profit {self._instrument} @ {bid if position > 0 else ask}')
            self.times_to_take.append(signal_time - self._last_trade_time)
            self.n_takes += 1

        return t_size
