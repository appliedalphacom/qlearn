from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from ira.series.Indicators import ATR
from ira.simulator.SignalTester import Tracker
from ira.strategies.exec_core_api import Quote
from ira.utils.utils import mstruct


class TakeStopTracker(Tracker):
    """
    Simple stop/take tracker provider
    
    Now more sophisticated version is actual (MultiTakeStopTracker)
    """

    def __init__(self, debug=False, take_by_limit_orders=False):
        self.take = None
        self._take_user_data = None
        self.stop = None
        self._stop_user_data = None
        self.n_stops = 0
        self.n_takes = 0
        self.times_to_take = []
        self.times_to_stop = []
        # if set to true it will execute stops exactly at set level otherwise at current quote
        # second case may introduce additional costs if being tested on OHLC bars
        self.accurate_stops = False
        # what is last triggered event: 'stop' or 'take' (None if nothing was triggered yet)
        self.last_triggered_event = None
        # if we use limit order for take profit ?
        self.take_by_limit_orders = take_by_limit_orders

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

    def trade(self, trade_time, quantity, comment='', exact_price=None, market_order=True):
        if quantity == 0:
            self._cleanup()

        # call super method
        super().trade(trade_time, quantity, comment, exact_price=exact_price, market_order=market_order)

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
            super().trade(timestamp, 0,
                          f'take {pos_dir} at {exec_price} by {"LIMIT" if self.take_by_limit_orders else "MARKET"}',
                          exact_price=exec_price,
                          market_order=not self.take_by_limit_orders)
            self.n_takes += 1
            self.last_triggered_event = 'take'
            self.on_take(timestamp, exec_price, self._take_user_data)
        else:
            self.debug(f' -> [{timestamp}] stop {pos_dir} [{self._instrument}] at {exec_price:.5f}')
            self.times_to_stop.append(timestamp - self._service.last_trade_time)
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
                self.__exec_risk_management(quote_time, self.stop if self.accurate_stops else ask, is_take=False,
                                            is_long=True)

        if self._position.quantity < 0:
            if self.take and ask <= self.take:
                self.__exec_risk_management(quote_time, self.take, is_take=True, is_long=False)

            if self.stop and bid >= self.stop:
                self.__exec_risk_management(quote_time, self.stop if self.accurate_stops else bid, is_take=False,
                                            is_long=False)

        # call super method
        super().update_market_data(instrument, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs)

    def statistics(self) -> Dict:
        return {
            'takes': self.n_takes,
            'stops': self.n_stops,
            'average_time_to_take': np.mean(self.times_to_take) if self.times_to_take else np.nan,
            'average_time_to_stop': np.mean(self.times_to_stop) if self.times_to_stop else np.nan,
        }


class MultiTakeStopTracker(Tracker):
    """
    More sophisticated stop/take tracker logic. It can track multiple take targets and maintain partial
    closes of position at different levels.
    """

    def __init__(self, debug=False, take_by_limit_orders=False):
        # take config {price: (fraction, user_data)}
        self.part_takes: Dict[float, Tuple[float, Any]] = {}

        # some helper to calculate average take price
        self.average_take_price = np.nan
        self._part_closed_values = 0.0
        self._part_closed_cumsize = 0.0

        # stop config
        self.stop = None
        self._stop_user_data = None

        # stats
        self.n_stops = 0
        self.n_takes = 0
        self.times_to_take = []
        self.times_to_stop = []

        # if set to true it will execute stops exactly at set level otherwise at current quote
        # second case may introduce additional costs if being tested on OHLC bars
        self.accurate_stops = False

        # what is last triggered event: 'stop' or 'take' (None if nothing was triggered yet)
        self.last_triggered_event = None
        # if we use limit order for take profit ?
        self.take_by_limit_orders = take_by_limit_orders

        if debug:
            self.debug = print

    def debug(self, *args, **kwargs):
        pass

    def is_take_full(self):
        return 1.0 in self.part_takes.values()

    def stop_at(self, trade_time, stop_price: float, user_data=None):
        self.stop = stop_price
        self._stop_user_data = user_data

    def take_at(self, trade_time, take_price: float, user_data=None):
        self.partial_take_at(trade_time, take_price, 1, user_data)

    def partial_take_at(self, trade_time, take_price: float, close_fraction: float, user_data=None):
        """
        Partial close position at price take_price. It uses close_fraction to get position size to close.
        if close_fraction == 1 it will close whole position.

        Eample: If we want to close 1/3 of position at 3 targets we need to call:
        > tracker.partial_take_at(time, 100, 1/3)
        > tracker.partial_take_at(time, 120, 1/2)
        > tracker.partial_take_at(time, 140, 1)
        """
        if close_fraction > 1.0 or close_fraction <= 0:
            raise ValueError(f" >>> close_fraction must be in range [0...1] but got {close_fraction} !")

        # check the price
        if not (np.isfinite(take_price) and take_price > 0):
            raise ValueError(f" >>> take_price must be positive number but got {take_price} !")

        # check if it's still possible to set next part take
        if self.is_take_full():
            raise ValueError(f" >>> partial_take_at can't be configured anymore")

        # add part stop
        self.part_takes[take_price] = (close_fraction, user_data)

    def trade(self, trade_time, quantity, comment='', exact_price=None, market_order=True):
        if quantity == 0:
            self._cleanup()

        # call super method
        super().trade(trade_time, quantity, comment, exact_price=exact_price, market_order=market_order)

    def on_take(self, timestamp, price, is_partial, closed_amount, user_data=None):
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
        self.part_takes = {}
        self.stop = None
        self.average_take_price = np.nan
        self._part_closed_values = 0.0
        self._part_closed_cumsize = 0.0
        self._stop_user_data = None

    def __process_take_risk_management(self, timestamp, exec_price, fraction_to_close, user_data, is_long):
        pos_dir = 'long' if is_long else 'short'

        # new position after part or full closing
        current_pos = self._position.quantity
        new_pos = int(current_pos * (1 - fraction_to_close))
        pos_delta_closed = new_pos - current_pos
        is_part_take = new_pos != 0

        # calculate average take price
        self._part_closed_values += abs(pos_delta_closed) * exec_price
        self._part_closed_cumsize += abs(pos_delta_closed)
        self.average_take_price = self._part_closed_values / self._part_closed_cumsize

        # for log record
        evt = 'part_take' if is_part_take else 'take'

        self.debug(f' -> [{timestamp}] {evt} {pos_dir} [{self._instrument}] at {exec_price:.5f}')
        self.times_to_take.append(timestamp - self._service.last_trade_time)

        super().trade(
            timestamp, new_pos,
            f'{evt} {pos_dir} at {exec_price} by {"LIMIT" if self.take_by_limit_orders else "MARKET"} -> [{new_pos}]',
            exact_price=exec_price,
            market_order=not self.take_by_limit_orders
        )

        self.last_triggered_event = evt
        self.n_takes += 1
        self.on_take(timestamp, exec_price, is_part_take, pos_delta_closed, user_data)

        # remove processed record
        del self.part_takes[exec_price]

        # here we need to clean up stops/takes because position is closed
        if new_pos == 0:
            self._cleanup()

    def __process_stop_risk_management(self, timestamp, exec_price, is_long):
        pos_dir = 'long' if is_long else 'short'
        self.debug(f' -> [{timestamp}] stop {pos_dir} [{self._instrument}] at {exec_price:.5f}')
        self.times_to_stop.append(timestamp - self._service.last_trade_time)
        super().trade(timestamp, 0, f'stop {pos_dir} at {exec_price}', exact_price=exec_price)
        self.n_stops += 1
        self.last_triggered_event = 'stop'
        self.on_stop(timestamp, exec_price, self._stop_user_data)

        # here we need to clean up stops/takes because position is closed
        self._cleanup()

    def update_market_data(self, instrument: str, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs):
        # check active stop/take
        # self.debug(f' ~~> {self._position.quantity} [{quote_time}] {bid}|{ask}({"S" if is_service_quote else "O"})')

        # - process long -
        if self._position.quantity > 0:
            # process take config
            if self.part_takes:
                for tprc in sorted(self.part_takes.keys()):
                    if bid >= tprc:
                        t_data = self.part_takes[tprc]
                        self.__process_take_risk_management(quote_time, tprc, t_data[0], t_data[1], is_long=True)

            # process stop config
            if self.stop and ask <= self.stop:
                self.__process_stop_risk_management(quote_time, self.stop if self.accurate_stops else ask, is_long=True)

        # - process short -
        if self._position.quantity < 0:
            # process take config
            if self.part_takes:
                for tprc in sorted(self.part_takes.keys(), reverse=True):
                    if ask <= tprc:
                        t_data = self.part_takes[tprc]
                        self.__process_take_risk_management(quote_time, tprc, t_data[0], t_data[1], is_long=False)

            # process stop config
            if self.stop and bid >= self.stop:
                self.__process_stop_risk_management(quote_time, self.stop if self.accurate_stops else bid,
                                                    is_long=False)

        # call super method
        super().update_market_data(instrument, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs)

    def statistics(self) -> Dict:
        return {
            'takes': self.n_takes,
            'stops': self.n_stops,
            'average_time_to_take': np.mean(self.times_to_take) if self.times_to_take else np.nan,
            'average_time_to_stop': np.mean(self.times_to_stop) if self.times_to_stop else np.nan,
        }


class TriggerOrderTypes(Enum):
    STOP_ORDER = 'StopOrder'
    LIMIT_ORDER = 'LimitOrder'


@dataclass
class TriggerOrder:
    """
    Generic triggered order (both for limit and stop orders) with risk management (stop | take)
    """
    # trigger price
    price: float

    # position to open
    quantity: int

    # stop level (if None not used)
    stop: float

    # takes level: may be None(not used) or dict {price: position fraction to close}
    take: Union[None, Dict[float, float]]

    # order type
    order_type: TriggerOrderTypes

    # user comment
    comment: str = ''

    # any user data
    user_data: Any = None

    # true if order was triggered (executed)
    fired: bool = False

    def __str__(self):
        takes = ','.join([f"{f} x {p}" for p, f in self.take.items()]) if self.take is not None else '---'
        return f"[{'FIRED' if self.fired else 'ACTIVE'}] {self.order_type.value} for {'buy' if self.quantity > 0 else 'sell'} " \
               f"of {self.quantity} @ {self.price} (T|S: ({takes}) | {self.stop})"


class TriggeredOrdersTracker(MultiTakeStopTracker):
    """
    Buy/Sell Stop trigger orders tracker implementation

    # 28-Mar-2022: basic class is MultiTakeStopTracker now
    """

    def __init__(self, debug=False,
                 accurate_stop_execution=True,
                 take_by_limit_orders=True,
                 open_by_limit_orders=False):
        """
        :param accurate_stop_execution: if true it emulates execution at exact stop level
        """
        super().__init__(debug, take_by_limit_orders=take_by_limit_orders)
        self.last_quote = Quote(np.nan, np.nan, np.nan, np.nan, np.nan)
        self.orders: List[TriggerOrder] = list()
        self.fired: List[TriggerOrder] = list()
        self.accurate_stops = accurate_stop_execution
        self.open_by_limit_orders = open_by_limit_orders
        if accurate_stop_execution:
            self.debug(f' > TriggeredOrdersTracker accurate_stops parameter is set')
        if open_by_limit_orders:
            self.debug(f' > open_by_limit_orders is set !')

    def trade(self, trade_time, quantity, comment='', exact_price=None, market_order=True):
        if quantity == 0:
            self._cleanup()

        pnl = 0
        if np.isfinite(quantity):
            pnl = self._position.update_position_bid_ask(
                trade_time, quantity, self.last_quote.bid, self.last_quote.ask, exec_price=exact_price,
                **self._service.get_aux_quote(), comment=comment, crossed_market=market_order)

            # set last trade time
            self._service.last_trade_time = trade_time
        return pnl

    def _setup_part_takes(self, time, takes: Dict[float, float], user_data):
        for p, f in takes.items():
            self.partial_take_at(time, p, f, user_data)

    def _process_stop_order(self, o: TriggerOrder, last_quote: Quote, quote_time, bid, ask, bid_size, ask_size) -> bool:
        # check if stop order can be triggered
        if (o.quantity > 0 and last_quote.ask < o.price <= ask) or (o.quantity < 0 and last_quote.bid > o.price >= bid):
            o.fired = True
            is_mkt_open = not self.open_by_limit_orders
            self.trade(quote_time, o.quantity, comment=o.comment, exact_price=o.price, market_order=is_mkt_open)
            self.stop_at(quote_time, o.stop, o.user_data)

            # setup takes targets
            if o.take is not None:
                self._setup_part_takes(quote_time, o.take, o.user_data)

            self.fired.append(o)
            # call callback on trigger fire
            self.on_trigger_fired(quote_time, o)
            return False

        # order was not fired and needed to be tracked
        return True

    def _process_limit_order(self, o: TriggerOrder, last_quote: Quote, quote_time, bid, ask, bid_size, ask_size) -> bool:
        # check if limit order can be executed
        if (o.quantity > 0 and last_quote.ask > o.price >= ask) or (o.quantity < 0 and last_quote.bid < o.price <= bid):
            o.fired = True
            self.trade(quote_time, o.quantity, comment=o.comment, exact_price=o.price, market_order=False)
            self.stop_at(quote_time, o.stop, o.user_data)

            # setup takes targets
            if o.take is not None:
                self._setup_part_takes(quote_time, o.take, o.user_data)

            self.fired.append(o)
            # call callback on trigger fire
            self.on_trigger_fired(quote_time, o)
            return False

        # order was not fired and needed to be tracked
        return True

    def update_market_data(self, instrument: str, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs):
        # store data
        q = self.last_quote

        # check active orders
        if np.isfinite(q.ask) and self.orders:
            n_orders = []

            for o in self.orders:
                keep_tracking = False

                if o.order_type == TriggerOrderTypes.STOP_ORDER:
                    keep_tracking = self._process_stop_order(o, q, quote_time, bid, ask, bid_size, ask_size)

                elif o.order_type == TriggerOrderTypes.LIMIT_ORDER:
                    keep_tracking = self._process_limit_order(o, q, quote_time, bid, ask, bid_size, ask_size)

                if keep_tracking:
                    n_orders.append(o)

            self.orders = n_orders

        # update last quote and series
        q.time, q.bid, q.ask, q.bid_ask, q.ask_size = quote_time, bid, ask, bid_size, ask_size
        super().update_market_data(instrument, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs)

    def on_trigger_fired(self, timestamp, order: TriggerOrder):
        pass

    def cancel(self, order: TriggerOrder):
        """
        Cancel order if it's active (not fired)
        """
        if order in self.orders:
            self.orders.remove(order)

    def cancel_all(self):
        """
        Cancel all orders
        """
        self.orders.clear()

    def _create_new_trigger_order(self, order_type: TriggerOrderTypes, price, quantity, stop, take, comment, user_data):
        """
        Create new trigger order at specified price. Order may have stop and multiple take targets.

        :param order_type: order type
        :param price: price where order should be triggered
        :param quantity: position size to executed (positive - long, negative - short)
        :param stop: stop price (if none - no stop)
        :param take: take price (if none - no stop), or dict {price: fraction_to_close}
                     fraction_to_close <= 1 and prod(1-fraction_to_close) == 0
        :param comment: user comment for this order
        :param user_data: custom user data for this order
        :return: order's object
        """
        is_buy = quantity > 0

        # for compatibility - if just price we want to close whole position
        takes = {take: 1.0} if isinstance(take, float) else take
        if 1.0 not in takes.values():  # faster check
            raise ValueError("If fractional take targets are passed prod of 1-f must be equal to 0.0 !!!")

        closer_take = min(takes.keys()) if is_buy else max(takes.keys())
        if (is_buy and (stop >= price or closer_take <= price)) or (
                quantity < 0 and (stop <= price or closer_take >= price)):
            raise ValueError(
                f"Wrong stop/take ({stop}/{closer_take}) for {'buy' if is_buy else 'sell'} stop order at {price}")

        to = TriggerOrder(price, quantity, stop, takes, order_type, comment, user_data)
        self.orders.append(to)

        return to

    def limit_order(self, price, quantity,
                    stop=None, take=Union[None, Dict[float, float], float],
                    comment='', user_data=None):
        """
        Create new limit order at specified price. Order may have stop and multiple take targets.

        :param price: price where order should be triggered
        :param quantity: position size to executed (positive - long, negative - short)
        :param stop: stop price (if none - no stop)
        :param take: take price (if none - no stop), or dict {price: fraction_to_close}
                     fraction_to_close <= 1 and prod(1-fraction_to_close) == 0
        :param comment: user comment for this order
        :param user_data: custom user data for this order
        :return: order's object
        """
        is_buy = quantity > 0
        if (is_buy and price > self.last_quote.bid) or (not is_buy and price < self.last_quote.ask):
            raise ValueError(f"Can't send {'buy' if is_buy else 'sell'} limit order at price {price}"
                             f" market now is {self.last_quote} | {comment}")

        return self._create_new_trigger_order(TriggerOrderTypes.LIMIT_ORDER, price, quantity, stop, take, comment,
                                              user_data)

    def stop_order(self, price, quantity,
                   stop=None, take=Union[None, Dict[float, float], float],
                   comment='', user_data=None):
        """
        Create new stop order at specified price. Order may have stop and multiple take targets.
        # 28-mar-2022: added multiple takes target (Lviv / in the shelter under air raid alert)
        
        :param price: price where order should be triggered
        :param quantity: position size to executed (positive - long, negative - short)
        :param stop: stop price (if none - no stop)
        :param take: take price (if none - no stop), or dict {price: fraction_to_close}
                     fraction_to_close <= 1 and prod(1-fraction_to_close) == 0
        :param comment: user comment for this order
        :param user_data: custom user data for this order
        :return: order's object
        """
        is_buy = quantity > 0
        if (is_buy and price < self.last_quote.ask) or (not is_buy and price > self.last_quote.bid):
            raise ValueError(f"Can't send {'buy' if is_buy else 'sell'} stop order at price {price}"
                             f" market now is {self.last_quote} | {comment}")

        return self._create_new_trigger_order(TriggerOrderTypes.STOP_ORDER, price, quantity, stop, take, comment,
                                              user_data)

    def statistics(self) -> Dict:
        return {'triggers': len(self.fired) + len(self.orders), 'fired': len(self.fired), **super().statistics()}


class FixedTrader(TakeStopTracker):
    """
    Fixed trader tracker:
     fixed position size, fixed stop and take
    """

    def __init__(self, size, take, stop, tick_size=1, debug=False, take_by_limit_orders=True):
        super().__init__(debug, take_by_limit_orders=take_by_limit_orders)
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

    def __init__(self, size, take, stop, debug=False, take_by_limit_orders=True):
        super().__init__(debug, take_by_limit_orders=take_by_limit_orders)
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

    def __init__(self, size, timeframe, period, take_target, stop_risk, atr_smoother='sma',
                 debug=False, take_by_limit_orders=True):
        super().__init__(debug, take_by_limit_orders=take_by_limit_orders)
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


class SignalBarTracker(TriggeredOrdersTracker):
    """
    Tracker with delay execution
    """

    def __init__(self, timeframe, tick_size=1e-5,
                 entry_factor=0, stop_factor=0, impr='initial', risk_reward=1,
                 debug=False,
                 accurate_stop_execution=True,
                 take_by_limit_orders=True,
                 open_by_limit_orders=False):
        """
        :param accurate_stop_execution: if true it will emulates execution at exact stop level
        """
        self.timeframe = timeframe
        self.tick_size = tick_size
        self.entry_factor = entry_factor
        self.stop_factor = stop_factor
        self.impr = impr
        if not self.impr in ['initial', 'improve']: raise ValueError("impr must be 'initial or 'improve")
        self.risk_reward = risk_reward
        super().__init__(debug, accurate_stop_execution, take_by_limit_orders, open_by_limit_orders)

    def initialize(self):
        self.ohlc = self.get_ohlc_series(self.timeframe)

    def update_market_data(self, instrument: str, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs):
        super().update_market_data(instrument, quote_time, bid, ask, bid_size, ask_size, is_service_quote, **kwargs)
        q = self.last_quote
        cur_ohlc = self.get_ohlc_series(self.timeframe)

        if np.isfinite(q.ask) and len(self.orders) > 0:
            # check to cancel order if price worse than stop
            qty = self.orders[0].quantity
            stop = self.orders[0].stop
            high = cur_ohlc.highs()[-1]
            low = cur_ohlc.lows()[-1]
            if (qty > 0 and low < stop) or (qty < 0 and high > stop):
                self.cancel(self.orders[0])

            if cur_ohlc.is_new_bar == True and self.impr == 'improve':
                # check improvement policy
                high_old = cur_ohlc.highs()[-2]
                low_old = cur_ohlc.lows()[-2]
                entry, _, take = self.deal_param(qty, high_old, low_old)
                self.cancel(self.orders[0])
                if (qty > 0 and entry <= stop) or (qty < 0 and entry >= stop):
                    return None
                self.to = self.stop_order(entry, qty, stop, take,
                                          comment='DelayTracker; ' +
                                                  # 'time:'+str(signal_time)+'; '+
                                                  'take=' + str(round(take, 5)) + '; ' +
                                                  'stop=' + str(round(stop, 5))
                                          , user_data=mstruct(time=str(quote_time))
                                          )
                self.debug(f' -> [{str(quote_time)}] set order at {entry:.5f} with stop {stop:.5f} and take {take:.5f}')

    def deal_param(self, qty, high, low):
        if qty > 0:
            entry = high * (1 + self.entry_factor)
            stop = low * (1 - self.stop_factor)
        else:
            entry = low * (1 - self.entry_factor)
            stop = high * (1 + self.stop_factor)
        take = entry + self.risk_reward * (entry - stop)
        return entry, stop, take

    def on_signal(self, signal_time, signal_qty, quote_time, bid, ask, bid_size, ask_size):
        if not signal_qty == 0 and len(self.orders) == 0:
            cur_ohlc = self.get_ohlc_series(self.timeframe)
            high = cur_ohlc.highs()[-1]
            low = cur_ohlc.lows()[-1]
            entry, stop, take = self.deal_param(signal_qty, high, low)
            self.to = self.stop_order(entry, signal_qty, stop, take,
                                      comment='DelayTracker; ' +
                                              # 'time:'+str(signal_time)+'; '+
                                              'take=' + str(round(take, 5)) + '; ' +
                                              'stop=' + str(round(stop, 5))
                                      , user_data=mstruct(time=str(quote_time))
                                      )
            self.debug(f' -> [{str(signal_time)}] set order at {entry:.5f} with stop {stop:.5f} and take {take:.5f}')
        return 0
