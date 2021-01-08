import pandas as pd
import numpy as np

from numpy.lib.stride_tricks import as_strided as stride
from ira.experimental.ui_utils import blue, green, magenta, yellow, cyan, red, white
from ira.analysis.timeseries import infer_series_frequency
from ira.simulator.utils import shift_signals
from ira.utils.utils import mstruct


class TradingService:
    def push_signal(self, time, signal):
        raise ValueError('push_signal method must be implemented')

    def trade(self, time, signal):
        raise ValueError('trade method must be implemented')


class Tracker:
    def track_signal(self, service: TradingService, signal_time, signal, signal_price, times, data):
        service.trade(signal_time, signal)
        return signal_time
    
    def filter(self, signals, *args, **kwargs):
        """
        Signals filtering (pass everything by default)
        """
        return signals


class GeneralSignalProcessor(TradingService):
    """
    Basic signals tracker
    """
    def __init__(self, instrument, price_data, execution_price_name='close', actualize_execution_times=True):
        self.data = price_data
        self.instrument = instrument
        self.field = execution_price_name
        self.f_idx = self.data.columns.get_loc(execution_price_name)
        self._data = price_data.values
        # self._data_timeline = price_data.index.to_numpy()
        self._data_timeline = price_data.index
        self.data_size = len(price_data)
        self._d0, self._d1 = self._data.shape
        self._s0, self._s1 = self._data.strides
        self._final_signals = {}
        self._skiped_signals = []
        self._tframe = pd.Timedelta(infer_series_frequency(price_data[:20]))
        self._shift = actualize_execution_times
        # processing queues
        self._processed_times = []
        self._processed_sigs = []
        # workaround for simulator
        self.trade(self.data.index[0], 0)

    def track_signal(self, signal_time, signal, signal_price, times, data):
        self.trade(signal_time, signal)
        return signal_time

    def trade(self, time, position):
        self._final_signals[time] = position

    def push_signal(self, time, signal):
        self._processed_times.append(time)
        self._processed_sigs.append(signal)

    def _get_data_slice_at(self, time):
        s_index = self._data_timeline.get_loc(time)
        # s_index = np.where(self._data_timeline == time)[0][0]
        w = self.data_size - s_index
        a = stride(self._data, (self._d0 - (w - 1), w, self._d1), (self._s0, self._s0, self._s1))
        data = a[s_index]
        return mstruct(
            times=self._data_timeline[s_index + 1:],
            ohlc=data[1:, :],
            price=data[0, self.f_idx])

    def _init_processing_queues(self, ):
        self._processed_times = list(reversed(list(self._generated_signals.index.to_numpy())))
        self._processed_sigs = list(reversed(list(self._generated_signals.to_numpy())))

    def process(self, tracker: Tracker, signals, *args, **kwargs):
        _dbg = False
        if 'debug' in kwargs:
            _dbg = kwargs.pop('debug')

        # clean data
        self._final_signals = {}
        self._skiped_signals = []

        # call generator
        self._generated_signals = tracker.filter(signals, *args, **kwargs)
        self._init_processing_queues()
        last_processed_time = pd.Timestamp('1684-01-01 00:00:00')

        # go through all the signals
        while len(self._processed_sigs) > 0:
            sig_t, sig_s = self._processed_times.pop(-1), self._processed_sigs.pop(-1)

            if sig_t < last_processed_time:
                self._skiped_signals.append(sig_t)
                if _dbg: print(magenta('~> skip %s last processed time is %s' % (sig_t, last_processed_time)))
                continue

            # get data after signal
            s_loc = self._get_data_slice_at(sig_t)
            last_processed_time = tracker.track_signal(self, sig_t, sig_s, s_loc.price, s_loc.times, s_loc.ohlc)

        # final dataframe
        trade_signals = pd.DataFrame.from_dict(self._final_signals, orient='index', columns=[self.instrument])

        # if we use close prices we need to shift signals to future to force execution on closes
        if self._shift and self.field == 'close':
            trade_signals = shift_signals(trade_signals, seconds=(self._tframe - pd.Timedelta('1S')).seconds)

        return trade_signals


class BasicTracker(Tracker):

    def track(self, instrument, signals, prices, execution_price_name='close'):
        print(f'--(N signals : {len(signals)})----------------------------------------')
        processor = GeneralSignalProcessor(instrument, prices, execution_price_name, True)
        positions = processor.process(self, signals)
        # TODO: .....
        print(positions.head())
        print('------------------------------------------')


class SimpleSignalTracker(BasicTracker):

    def __init__(self, fixed_size=1):
        self.fixed_size = fixed_size

    def track_signal(self, service: TradingService, signal_time, signal, signal_price, times, data):
        service.trade(signal_time, self.fixed_size * signal)
        return signal_time

