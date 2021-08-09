from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from dataclasses import dataclass

from ira.series.Indicators import ATR
from ira.simulator.SignalTester import Tracker
from ira.strategies.exec_core_api import Quote


class TakeStopTracker(Tracker):
    """
    Simple stop/take tracker provider
    """

    def __init__(self, debug=False):
        self.take = None
        self._take_user_data = None
        self.stop = None
        self._stop_user_data = None
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

    def stop_at(self, trade_time, stop_price: float, user_data=None):
        self.stop = stop_price
        self._stop_user_data = user_data

    def take_at(self, trade_time, take_price: float, user_data=None):
        self.take = take_price
        self._take_user_data = user_data

    def trade(self, trade_time, quantity, comment='', exact_price=None):
        if quantity == 0:
            self._cleanup()

        # call super method
        super().trade(trade_time, quantity, comment, exact_price=exact_price)

    def on_take(self, timestamp, price, user_data=None):
        """
        Default handler on take hit
        """
        pass

    def on_stop(self, timestamp, price, user_data=None):
        """
        Default handler on stop hit
        """
        pass

    def _cleanup(self):
        self.take = None
        self.stop = None
        self._take_user_data = None
        self._stop_user_data = None

    def __exec_risk_management(self, timestamp, exec_price, is_take, is_long):
        pos_dir = 'long' if is_long else 'short'
        if is_take:
            self.debug(f' -> [{timestamp}] take {pos_dir} [{self._instrument}] at {exec_price:.5f}')
            self.times_to_take.append(timestamp - self._service.last_trade_time)
            # self.trade(timestamp, 0, f'take long at {exec_price}', exact_price=exec_price)
            super().trade(timestamp, 0, f'take {pos_dir} at {exec_price}', exact_price=exec_price)
            self.n_takes += 1
            self.last_triggered_event = 'take'
            self.on_take(timestamp, exec_price, self._take_user_data)
        else:
            self.debug(f' -> [{timestamp}] stop {pos_dir} [{self._instrument}] at {exec_price:.5f}')
            self.times_to_stop.append(timestamp - self._service.last_trade_time)
            # self.trade(timestamp, 0, f'stop long at {exec_price}', exact_price=exec_price)
            super().trade(timestamp, 0, f'stop {pos_dir} at {exec_price}', exact_price=exec_price)
            self.n_stops += 1
            self.last_triggered_event = 'stop'
            self.on_stop(timestamp, exec_price, self._stop_user_data)

        # here we need to clean up stops/takes because position is closed
        self._cleanup()

    def update_market_data(self, instrument: str, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs):
        # check active stop/take
        # self.debug(f' ~~> {self._position.quantity} [{quote_time}] {bid}|{ask}({"S" if is_service_quote else "O"})')
        if self._position.quantity > 0:
            if self.take and bid >= self.take:
                self.__exec_risk_management(quote_time, self.take, is_take=True, is_long=True)

            if self.stop and ask <= self.stop:
                self.__exec_risk_management(quote_time, ask, is_take=False, is_long=True)

        if self._position.quantity < 0:
            if self.take and ask <= self.take:
                self.__exec_risk_management(quote_time, self.take, is_take=True, is_long=False)

            if self.stop and bid >= self.stop:
                self.__exec_risk_management(quote_time, bid, is_take=False, is_long=False)

        # call super method
        super().update_market_data(instrument, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs)

    def statistics(self) -> Dict:
        return {
            'takes': self.n_takes,
            'stops': self.n_stops,
            'average_time_to_take': np.mean(self.times_to_take) if self.times_to_take else np.nan,
            'average_time_to_stop': np.mean(self.times_to_stop) if self.times_to_stop else np.nan,
        }


@dataclass
class TriggerOrder:
    price: float  # trigger price
    quantity: int  # position to open
    stop: float  # stop level (if None not used)
    take: float  # take level (if None not used)
    comment: str = ''  # user comment
    user_data: Any = None  # any user data
    fired: bool = False  # true if order was triggered

    def __str__(self):
        return f"[{'FIRED' if self.fired else 'ACTIVE'}] TriggerOrder for {'buy' if self.quantity > 0 else 'sell'} " \
               f"of {self.quantity} @ {self.price} (T|S: {self.take} | {self.stop})"


class TriggeredOrdersTracker(TakeStopTracker):
    """
    Buy/Sell Stop trigger orders tracker implementation
    """

    def __init__(self, debug=False):
        super().__init__(debug)
        self.last_quote = Quote(np.nan, np.nan, np.nan, np.nan, np.nan)
        self.orders: List[TriggerOrder] = list()
        self.fired: List[TriggerOrder] = list()

    def trade(self, trade_time, quantity, comment='', exact_price=None):
        if quantity == 0:
            self._cleanup()

        pnl = 0
        if np.isfinite(quantity):
            pnl = self._position.update_position_bid_ask(
                trade_time, quantity, self.last_quote.bid, self.last_quote.ask, exec_price=exact_price,
                **self._service.get_aux_quote(), comment=comment)

            # set last trade time
            self._service.last_trade_time = trade_time
        return pnl

    def update_market_data(self, instrument: str, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs):
        # store data
        q = self.last_quote

        # check active orders
        if np.isfinite(q.ask) and self.orders:
            n_orders = []
            for o in self.orders:
                if (o.quantity > 0 and q.ask < o.price <= ask) or (o.quantity < 0 and q.bid > o.price >= bid):
                    o.fired = True
                    self.trade(quote_time, o.quantity, comment=o.comment, exact_price=o.price)
                    self.stop_at(quote_time, o.stop, o.user_data)
                    self.take_at(quote_time, o.take, o.user_data)
                    self.fired.append(o)
                    # call callback on trigger fire
                    self.on_trigger_fired(quote_time, o)
                else:
                    n_orders.append(o)
            self.orders = n_orders

        # update last quote and series
        q.time, q.bid, q.ask, q.bid_ask, q.ask_size = quote_time, bid, ask, bid_size, ask_size
        super().update_market_data(instrument, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs)

    def on_trigger_fired(self, timestamp, order: TriggerOrder):
        pass

    def cancel(self, order: TriggerOrder):
        if order in self.orders:
            self.orders.remove(order)

    def stop_order(self, price, quantity, stop=None, take=None, comment='', user_data=None):
        is_buy = quantity > 0
        if is_buy and price < self.last_quote.ask:
            raise ValueError(f"Can't send {'buy' if is_buy else 'sell'} stop order at price {price}"
                             f" market now is {self.last_quote} | {comment}")

        if (is_buy and (stop >= price or take <= price)) or (quantity < 0 and (stop <= price or take >= price)):
            raise ValueError(f"Wrong stop/take ({stop}/{take}) for {'buy' if is_buy else 'sell'} stop order at {price}")

        to = TriggerOrder(price, quantity, stop, take, comment, user_data)
        self.orders.append(to)
        return to

    def statistics(self) -> Dict:
        return {'triggers': len(self.fired) + len(self.orders), 'fired': len(self.fired), **super().statistics()}


class FixedTrader(TakeStopTracker):
    """
    Fixed trader tracker:
     fixed position size, fixed stop and take
    """

    def __init__(self, size, take, stop, tick_size=1, debug=False):
        super().__init__(debug)
        self.position_size = size
        self.fixed_take = take * tick_size
        self.fixed_stop = stop * tick_size

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        if signal_qty > 0:
            if self.fixed_stop > 0:
                self.stop_at(signal_time, ask - self.fixed_stop)

            if self.fixed_take > 0:
                self.take_at(signal_time, ask + self.fixed_take)

        elif signal_qty < 0:
            if self.fixed_stop > 0:
                self.stop_at(signal_time, bid + self.fixed_stop)

            if self.fixed_take > 0:
                self.take_at(signal_time, bid - self.fixed_take)

        # call super method
        return signal_qty * self.position_size


class FixedPctTrader(TakeStopTracker):
    """
    Fixed trader tracker (take and stop as percentage from entry price):
     fixed position size, fixed stop and take
    """

    def __init__(self, size, take, stop, debug=False):
        super().__init__(debug)
        self.position_size = size
        self.fixed_take = abs(take)
        self.fixed_stop = abs(stop)

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        if signal_qty > 0:
            if self.fixed_stop > 0:
                self.stop_at(signal_time, ask * (1 - self.fixed_stop))

            if self.fixed_take > 0:
                self.take_at(signal_time, ask * (1 + self.fixed_take))

        elif signal_qty < 0:
            if self.fixed_stop > 0:
                self.stop_at(signal_time, bid * (1 + self.fixed_stop))

            if self.fixed_take > 0:
                self.take_at(signal_time, bid * (1 - self.fixed_take))

        # call super method
        return signal_qty * self.position_size


class TimeExpirationTracker(Tracker):
    """
    Expiration exits
    """

    def __init__(self, timeout, debug=False):
        self.timeout = pd.Timedelta(timeout)
        self.debug = debug

    def initialize(self):
        self.n_expired_profit_ = 0
        self.n_expired_loss_ = 0

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        if self._position.quantity != 0 and quote_time - self._service.last_trade_time >= self.timeout:
            if self.debug:
                print(f' > {self._instrument} position {self._position.quantity} is expired at {quote_time}')
            pnl = self.trade(quote_time, 0, f'TimeExpirationTracker:: position {self._position.quantity} is expired')
            if pnl > 0:
                self.n_expired_profit_ += 1
            elif pnl < 0:
                self.n_expired_loss_ += 1

    def statistics(self) -> Dict:
        return {
            'expired': self.n_expired_loss_ + self.n_expired_profit_,
            'expired_profitable': self.n_expired_profit_,
            'expired_loss': self.n_expired_loss_
        }


class ProgressionTracker(Tracker):
    pass


class TurtleTracker(TakeStopTracker):
    """
    Our modifiction of turtles money management system
    """

    def __init__(self,
                 account_size, dollar_per_point, max_units=4, risk_capital_pct=0.01, reinvest_pnl_pct=0,
                 contract_size=100, max_allowed_contracts=200,
                 atr_timeframe='1d', pull_stops_on_incr=False, after_lose_only=False, debug=False):
        """
        Turtles strategy position tracking
        ----------------------------------

        >>> from sklearn.base import TransformerMixin
        >>> from sklearn.pipeline import make_pipeline
        >>> b_e1h = MarketDataComposer(make_pipeline(RollingRange('1h', 10), RangeBreakoutDetector()),
        >>>                                          SingleInstrumentPicker(), None).fit(data, None).predict(data)
        >>> b_x1h = MarketDataComposer(make_pipeline(RollingRange('1h', 6), RangeBreakoutDetector()),
        >>>                                          SingleInstrumentPicker(), None).fit(data, None).predict(data)
        >>> s1h = shift_signals(srows(1 * b_e1h, 2 * b_x1h), '4M59Sec')
        >>> p1h = z_backtest(s1h, data, 'crypto_futures', spread=0.5, execution_logger=True,
        >>>                  trackers=TurtleTracker(3000, None, max_units=4, risk_capital_pct=0.05,
        >>>                                         atr_timeframe='1h',
        >>>                                         max_allowed_contracts=1000, pull_stops_on_incr=True, debug=False))

        It processes signals as following:
          - signals in [-1, +1] designated for open positions
          - signals in [-2, +2] designated for positions closing

        :param accoun_size: starting amount in USD
        :param dollar_per_point: price of 1 point (for example 12.5 for ES mini) if none crypto sizing would be used
        :param max_untis: maximal number of position inreasings
        :param risk_capital_pct: percent of capital in risk (0.01 is 1%)
        :param reinvest_pnl_pct: percent of reinvestment pnl to trading (default is 0)
        :param contract_size: contract size in USD
        :param max_allowed_contracts: maximal allowed contracts to trade
        :param atr_timeframe: timeframe of ATR calculations
        :param pull_stops_on_incr: if true it pull up stop on position's increasing
        :param after_lose_only: if true it's System1 otherwise System2
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
        self.__init_days_counted = 1
        self.__TR_init_sum = 0
        self._n_entries = 0
        self._last_entry_price = np.nan

    def _get_size_at_risk(self):
        return (self.account_size + max(self.reinvest_pnl_pct * self._position.pnl, 0)) * self.risk_capital_pct

    def _calculate_trade_size_crypto(self, direction, vlt, price):
        price2 = price + direction * vlt
        return np.clip(round(self._get_size_at_risk() / ((price2 / price - 1) * self.contract_size)),
                       -self.max_allowed_contracts, self.max_allowed_contracts)

    def _calculate_trade_size_on_dollar_cost(self, direction, vlt, price):
        return min(self.max_units, round(self._get_size_at_risk() / (vlt * self.dollar_per_point))) * direction

    def calculate_trade_size(self, direction, vlt, price):
        return self._calculate_trade_size_on_dollar_cost(direction, vlt, price)

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        daily = self.days
        today, yest = daily[1], daily[2]

        if yest is None or today is None:
            return

        if daily.is_new_bar:
            TR = max(today.high - today.low, today.high - yest.close, yest.close - today.low)
            if self.N is None:
                if self.__init_days_counted <= 21:
                    self.__TR_init_sum += TR
                    self.__init_days_counted += 1
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

                # new position
                new_pos = pos + t_size

                # increase inventory
                self.trade(quote_time, new_pos, f'increased position to {new_pos} at {self._last_entry_price}')

                # average position price
                avg_price = self._position.cost_usd / self._position.quantity

                # pull stops
                if self.pull_stops:
                    self.stop_at(quote_time, avg_price - self.N * 2 * np.sign(pos))

                self.debug(
                    f"\t[{quote_time}] -> [#{self._n_entries}] {self._instrument} <{avg_price:.2f}> increasing to "
                    f"{new_pos} @ {self._last_entry_price} x {self.stop:.2f}")

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
                self._n_entries = 1
                self.debug(
                    f'\t[{signal_time}] -> [#{self._n_entries}] {self._instrument} {t_size} @ '
                    f'{self._last_entry_price:.2f} x {self.stop:.2f}')
            # clear previous state
            self.last_triggered_event = None

        # when we got to exit signal
        if (position > 0 and signal == -2) or (position < 0 and signal == +2):
            self.last_triggered_event = 'take'
            self._n_entries = 0
            t_size = 0
            self.debug(f'[{signal_time}] -> Close in profit {self._instrument} @ {bid if position > 0 else ask}')
            self.times_to_take.append(signal_time - self._service.last_trade_time)
            self.n_takes += 1

        return t_size

    def statistics(self) -> Dict:
        r = dict()
        return r


class DispatchTracker(Tracker):
    """
    Dispatching tracker
    """

    def __init__(self, trackers: Dict[str, Tracker], active_tracker: str, flat_position_on_activate=False, debug=False):
        self.trackers = trackers
        self.active_tracker = None
        self.flat_position_on_activate = flat_position_on_activate

        # statistics
        self.n_activations_ = defaultdict(lambda: 0)

        if active_tracker is not None:
            if active_tracker not in trackers:
                raise ValueError(f"Tracker '{active_tracker}' specified as active not found in trackers dict !")

            # active tracker
            self.active_tracker = trackers[active_tracker]
            self.debug(f' .-> {active_tracker} tracker is activated')
            self.n_activations_[active_tracker] += 1

        if debug:
            self.debug = print

    def debug(self, *args, **kwargs):
        pass

    def setup(self, service):
        super().setup(service)
        for t in self.trackers.values():
            if t and hasattr(t, 'setup'):
                t.setup(service)

    def on_info(self, info_time, info_data, **kwargs):
        if info_data in self.trackers:
            n_tracker = self.trackers[info_data]
            mesg = ''
            if self.flat_position_on_activate and n_tracker != self.active_tracker and self._position.quantity != 0:
                self.trade(info_time, 0, f'<{info_data}> activated and flat position')
                mesg = ' position is closed'

            self.active_tracker = n_tracker
            self.debug(f' [D]-> [{info_time}] {info_data} tracker is activated {mesg}')
            self.n_activations_[info_data] += 1

    def update_market_data(self, instrument: str, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs):
        # update series in all trackers
        [t._update_series(quote_time, bid, ask, bid_size, ask_size) for t in self.trackers.values() if t]

        # call handler if it's not service quote
        if not is_service_quote and self.active_tracker is not None:
            self.active_tracker.update_market_data(instrument, quote_time, bid, ask, bid_size, ask_size,
                                                   is_service_quote, **kwargs)
            # self.active_tracker.on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        if self.active_tracker:
            return self.active_tracker.on_signal(signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size)
        return signal_qty

    def statistics(self) -> Dict:
        r = dict(self.n_activations_)
        for t in self.trackers.values():
            if t is not None:
                s = t.statistics()
                if s is not None:
                    r.update(s)
        return r


class PipelineTracker(Tracker):
    """
    Provides ability to make pipeline of trackers

    >>> class MyTracker(Tracker):
    >>>     def __init__(self, size):
    >>>         self.size = size
    >>>
    >>>    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
    >>>        return signal_qty * self.size
    >>>
    >>> s = pd.DataFrame.from_dict({
    >>>     pd.Timestamp('2020-08-17 04:19:59'): {'EURUSD': +1},
    >>>     pd.Timestamp('2020-08-21 14:19:59'): {'EURUSD': -1},
    >>>     pd.Timestamp('2020-08-30 14:19:59'): {'EURUSD':  0}}, orient='index')
    >>> p = z_backtest(s, {'EURUSD': data}, 'forex', spread=0.0, execution_logger=True,
    >>>                trackers = PipelineTracker(
    >>>                   TimeExpirationTracker('1h', True),
    >>>                   MyTracker(10000))
    >>>          )
    """

    def __init__(self, *trackers):
        self.trackers = [t for t in trackers if isinstance(t, Tracker)]

    def setup(self, service):
        super().setup(service)
        for t in self.trackers:
            t.setup(service)

    def update_market_data(self, instrument: str, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs):
        for t in self.trackers:
            t.update_market_data(instrument, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs)

    def on_info(self, info_time, info_data, **kwargs):
        for t in self.trackers:
            t.on_info(info_time, info_data, **kwargs)

    def on_quote(self, quote_time, bid, ask, bid_size, ask_size, **kwargs):
        for t in self.trackers:
            t.on_quote(quote_time, bid, ask, bid_size, ask_size, **kwargs)

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        """
        Callback on new signal
        """
        processed_signal = signal_qty
        for t in self.trackers:
            processed_signal = t.on_signal(signal_time, processed_signal, quote_time, bid, ask, bid_size, ask_size)

            # if some tracker shutdowns signal we break the chain
            if processed_signal is None or np.isnan(processed_signal):
                break

        return processed_signal

    def statistics(self) -> Dict:
        r = dict()
        for t in self.trackers:
            s = t.statistics()
            if s is not None:
                r.update(s)
        return r


class ATRTracker(TakeStopTracker):
    """
    ATR based risk management
    Take at entry +/- ATR[1] * take_target
    Stop at entry -/+ ATR[1] * stop_rosk
    """

    def __init__(self, size, timeframe, period, take_target, stop_risk, atr_smoother='sma', debug=False):
        super().__init__(debug)
        self.timeframe = timeframe
        self.period = period
        self.position_size = size
        self.take_target = take_target
        self.stop_risk = stop_risk
        self.atr_smoother = atr_smoother

    def initialize(self):
        self.atr = ATR(self.period, self.atr_smoother)
        self.ohlc = self.get_ohlc_series(self.timeframe)
        self.ohlc.attach(self.atr)

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        av = self.atr[1]
        if av is None or not np.isfinite(av):
            # skip if ATR is not calculated yet
            return None

        if signal_qty > 0:
            if self.stop_risk > 0:
                self.stop_at(signal_time, ask - self.stop_risk * av)

            if self.take_target > 0:
                self.take_at(signal_time, ask + self.take_target * av)

        elif signal_qty < 0:
            if self.stop_risk > 0:
                self.stop_at(signal_time, bid + self.stop_risk * av)

            if self.take_target > 0:
                self.take_at(signal_time, bid - self.take_target * av)

        # call super method
        return signal_qty * self.position_size
