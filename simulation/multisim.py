from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ira.simulator.SignalTester import Tracker
from ira.utils.nb_functions import z_backtest
# Global for storing the data to be served
# g_data = {}
from qlearn import MarketDataComposer


def _type(obj):
    if obj is None:
        t = 'unknown'
    elif isinstance(obj, (list, tuple)):
        t = 'list'
    elif isinstance(obj, Tracker):
        t = 'tracker'
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        t = 'signal'
    elif isinstance(obj, Pipeline, BaseEstimator):
        t = 'estimator'
    else:
        t = 'unknown'
    return t


def start_stop_sigs(data, start=None, stop=None):
    """
    Generate stub signals (NaNs mainly for backtester progress)
    """
    r = None

    if stop is not None:
        try:
            stop = str(pd.Timestamp(start) + pd.Timedelta(stop))
        except:
            pass

    ss = {}
    for i, d in data.items():
        start = d.index[0] if start is None else start
        stop = d.index[-1] if stop is None else stop

        dx = len(d[start:stop]) // 100
        idx = d.index

        # split into 100 parts
        for j in range(0, 101):
            ss[idx[j * dx]] = np.nan

        # ss[idx[-1]] = np.nan
        r = pd.concat((r, pd.Series(ss, name=i)), axis=1)
        return r


class SimSetup:
    def __init__(self, signals, trackers, experiment_name=None):
        self.signals = signals
        self.signal_type = _type(signals)
        self.trackers = trackers
        self.name = experiment_name

    def get_signals(self, data, start, stop):
        sx = self.signals

        if sx is None or self.signal_type == 'unknown':
            return start_stop_sigs(data, start, stop)

        if self.signal_type == 'estimator':
            if isinstance(sx, MarketDataComposer):
                sx.selector.for_range(start, stop)
            return sx.fit(data, None).predict(data)

        _z = slice(start, stop) if start is not None and stop is not None else None
        return sx[_z] if _z is not None else sx

    def __repr__(self):
        return f'{self.name} : {self.signal_type} | {repr(self.trackers) if self.trackers is not None else "<no tracker>"}'


def _is_signals_generator(obj): return _type(obj) in ['signal', 'estimator']


def _is_tracker(obj): return _type(obj) == 'tracker'


def _recognize(setup, data, name) -> List[SimSetup]:
    r = list()

    if isinstance(setup, dict):
        for n, v in setup.items():
            r.extend(_recognize(v, data, name + '/' + n))

    elif isinstance(setup, (list, tuple)):
        if len(setup) == 2 and _is_signals_generator(setup[0]) and _is_tracker(setup[1]):
            r.append(SimSetup(setup[0], setup[1], name))
        else:
            for j, s in enumerate(setup):
                r.extend(_recognize(s, data, name + '/' + str(j)))

    elif _is_tracker(setup):
        r.append(SimSetup(None, setup, name))

    elif isinstance(setup, (pd.DataFrame, pd.Series)):
        r.append(SimSetup(setup, None, name))

    return r


def _proc_run(s: SimSetup, data, start, stop, broker, spreads):
    b = z_backtest(s.get_signals(data, start, stop), data, broker, spread=spreads,
                   name=s.name, execution_logger=True, trackers=s.trackers)
    return b


def simulation(setup, data, broker='', project='', start=None, stop=None, spreads=0):
    sims = _recognize(setup, data, project)
    results = []

    for i, s in enumerate(sims):
        print(s)
        if True:
            b = _proc_run(s, data, start, stop, broker, spreads)
            results.append(b)
    return results
