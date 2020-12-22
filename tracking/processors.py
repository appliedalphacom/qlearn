import pandas as pd
import numpy as np

from numpy.lib.stride_tricks import as_strided as stride
from ira.experimental.ui_utils import blue, green, magenta, yellow, cyan, red, white
from ira.analysis.timeseries import infer_series_frequency
from ira.simulator.utils import shift_signals
from ira.utils.utils import mstruct


class BasicSignalTracker:
    """
    Basic continuation signals tracker
    
    13-Jul-2020: added filtering
    """
    def __init__(self, instrument, price_data, field='close', actualize_execution_times=True):
        self.data = price_data
        self.instrument = instrument
        self.field = 'close'
        self.f_idx = self.data.columns.get_loc(field)
        self._data = price_data.values
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
        
    def trade(self, time, position):
        self._final_signals[time] = position
            
    def push_signal(self, time, signal):
        self._processed_times.append(time)
        self._processed_sigs.append(signal)
    
    def signals(self, prices, *args, **kwargs):
        raise ValueError("Method 'signals' must be implemented !")
    
    def track_signal(self, signal_time, signal, signal_price, times, data):
        self.trade(signal_time, signal)
        return signal_time
    
    def _get_data_slice_at(self, time):
        s_index = self._data_timeline.get_loc(time)
        w = self.data_size - s_index
        a = stride(self._data, (self._d0 - (w - 1), w, self._d1), (self._s0, self._s0, self._s1))
        data = a[s_index]
        return mstruct(
            times=self._data_timeline[s_index+1:],
            ohlc=data[1:, :],
            price=data[0, self.f_idx])
    
    def setup(self, *args, **kwargs):
        return self
    
    def filter(self, signals, *args, **kwargs):
        """
        Signals filtering (pass everything by default)
        """
        return signals
    
    def _init_processing_queues(self,):
        self._processed_times = list(reversed(list(self._generated_signals.index.to_numpy())))
        self._processed_sigs = list(reversed(list(self._generated_signals.to_numpy())))
    
    def process(self, *args, **kwargs):
        _dbg = False
        if 'debug' in kwargs:
            _dbg = kwargs.pop('debug')
        
        # clean data
        self._final_signals = {}
        self._skiped_signals = []
        
        # custom setup
        self.setup(*args, **kwargs)
        
        # call generator
        self._generated_signals = self.filter(self.signals(self.data))
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
            last_processed_time = self.track_signal(sig_t, sig_s, s_loc.price, s_loc.times, s_loc.ohlc)
        
        # final dataframe
        trade_signals = pd.DataFrame.from_dict(self._final_signals, orient='index', columns=[self.instrument])
        
        # if we use close prices we need to shift signals to future to force execution on closes
        if self._shift and self.field == 'close':
            trade_signals = shift_signals(trade_signals, seconds=(self._tframe - pd.Timedelta('1S')).seconds)
            
        return trade_signals


class SimpleSignalTracker(BasicSignalTracker):

    def __init__(self, instrument, price_data, field, signals, actualize_execution_times=True):
        super(SimpleSignalTracker, self).__init__(instrument, price_data, field, actualize_execution_times)
        self.signals_ = signals

        def signals(self, prices):
            return self.signals_


