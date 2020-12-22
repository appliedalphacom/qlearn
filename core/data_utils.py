import pandas as pd
import numpy as np
import re
from datetime import timedelta as timedelta_t
from typing import Union, List, Set, Dict
from dataclasses import dataclass

from qlearn.core.utils import infer_series_frequency


@dataclass
class DataType:
    type: str
    symbols: List[str]
    freq: str
    subtypes: Set[str]
    

def time_delta_to_str(d: Union[int, timedelta_t, pd.Timedelta]):
    """
    Convert timedelta object to pretty print format

    :param d:
    :return:
    """
    seconds = d.seconds if isinstance(d, pd.Timedelta) else d.total_seconds() if isinstance(d, timedelta_t) else int(d)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    r = ''
    if days > 0:
        r += '%dD' % days
    if hours > 0:
        r += '%dH' % hours
    if minutes > 0:
        r += '%dMin' % minutes
    if seconds > 0:
        r += '%dS' % seconds
    return r


def timeseries_density(dx, period='1Min'):
    """
    Detect average records density per period
    :param dx:
    :param period:
    :return:
    """
    return dx.groupby(pd.Grouper(freq=period)).count().mean().mean()


def inner_join_and_split(d1, d2, dropna=True, keep='last'):
    """
    Joins two series (frames) and reindex on most dense time index

    :param d1: first frame
    :param d2: second frame
    :param dropna: if need frop nans
    :param keep: what to keep on same timestamps
    :return: tuple of reindexed frames (d1, d2)
    """
    extract_by_key = lambda x, sfx: x.filter(regex='.*_%s' % sfx).rename(columns=lambda y: y.split('_')[0])

    # period for density of records: we take more sparsed one
    dens_period = max((d1.index[-1] - d1.index[0]) / len(d1), (d2.index[-1] - d2.index[0]) / len(d2))
    if timeseries_density(d1, dens_period) > timeseries_density(d2, dens_period):
        m1 = pd.merge_asof(d1, d2, left_index=True, right_index=True, suffixes=['_X1', '_X2'])
    else:
        m1 = pd.merge_asof(d2, d1, left_index=True, right_index=True, suffixes=['_X2', '_X1'])

    if dropna:
        m1.dropna(inplace=True)

    if m1.index.has_duplicates:
        m1 = m1[~m1.index.duplicated(keep=keep)]

    return extract_by_key(m1, 'X1'), extract_by_key(m1, 'X2')


def merge_ticks_from_dict(data, instruments, dropna=True, keep='last'):
    """
    :param data:
    :param instruments:
    :param dropna:
    :param keep:
    :return:
    """
    if len(instruments) == 1:
        return pd.concat([data[instruments[0]], ], keys=instruments, axis=1)

    max_dens_period = max([(d.index[-1] - d.index[0]) / len(d) for s, d in data.items() if s in instruments])
    densitites = {s: timeseries_density(data[s], max_dens_period) for s in instruments}
    pass_dens = dict(sorted(densitites.items(), key=lambda x: x[1], reverse=True))
    ins = list(pass_dens.keys())
    mss = list(inner_join_and_split(data[ins[0]], data[ins[1]], dropna=False, keep=keep))

    for s in ins[2:]:
        mss.append(inner_join_and_split(mss[0], data[s], dropna=False, keep=keep)[1])

    r = pd.concat(mss, axis=1, keys=ins)
    if dropna:
        r.dropna(inplace=True)

    return r


def _get_top_names(cols):
    return list(set(cols.get_level_values(0).values))


def make_dataframe_from_dict(data: dict, data_type: str):
    """
    Produses dataframe from dictionary
    :param data:
    :param data_type:
    :return:
    """
    if isinstance(data, dict):
        if data_type in ['ohlc', 'frame']:
            return pd.concat(data.values(), keys=data.keys(), axis=1)
        elif data_type == 'ticks':
            return merge_ticks_from_dict(data, list(data.keys()))
        else:
            raise ValueError(f"Don't know how to merge '{data_type}'")
    return data


def do_columns_contain(cols, keys):
    return all([c in cols for c in keys])


def detect_data_type(data) -> DataType:
    """
    Finds info about data structure

    :param data:
    :return:
    """
    dtype = re.findall(".*\'(.*)\'.*", f'{type(data)}')[0]
    freq = None
    symbols = []
    subtypes = None

    if isinstance(data, pd.DataFrame):
        cols = data.columns

        if isinstance(cols, pd.MultiIndex):
            # multi index dataframe
            dtype = 'multi'
            symbols = _get_top_names(cols)
        else:
            # just dataframe
            dtype = 'frame'
            if do_columns_contain(cols, ['open', 'high', 'low', 'close']):
                symbols = ['OHLC1']
                dtype = 'ohlc'
            elif do_columns_contain(cols, ['bid', 'ask']):
                symbols = ['TICKS1']
                dtype = 'ticks'

    elif isinstance(data, pd.Series):
        dtype = 'series'
        symbols = [data.name if data.name is not None else 'SERIES1']

    elif isinstance(data, dict):
        dtype = 'dict'
        symbols = list(data.keys())
        subtypes = {detect_data_type(v).type for v in data.values()}

    if isinstance(data, (pd.DataFrame, pd.Series)):
        freq = time_delta_to_str(infer_series_frequency(data[:100]))

    return DataType(type=dtype, symbols=symbols, freq=freq, subtypes=subtypes)


