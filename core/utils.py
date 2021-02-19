import pandas as pd
import numpy as np
from typing import Union, List, Set, Dict
from datetime import timedelta
from itertools import product
import types


def _check_frame_columns(x, *args):
    if not (isinstance(x, pd.DataFrame) and sum(x.columns.isin(args)) == len(args)):
        raise ValueError(f"Input series must be DataFrame with {args} columns !")


def infer_series_frequency(series):
    """
    Infer frequency of given timeseries

    :param series: Series, DataFrame or DatetimeIndex object
    :return: timedelta for found frequency
    """

    if not isinstance(series, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
        raise ValueError("infer_series_frequency> Only DataFrame, Series of DatetimeIndex objects are allowed")

    times_index = (series if isinstance(series, pd.DatetimeIndex) else series.index).to_pydatetime()
    if times_index.shape[0] < 2:
        raise ValueError("Series must have at least 2 points to determ frequency")

    values = np.array(sorted([(x.total_seconds()) for x in np.diff(times_index)]))
    diff = np.concatenate(([1], np.diff(values)))
    idx = np.concatenate((np.where(diff)[0], [len(values)]))
    freqs = dict(zip(values[idx[:-1]], np.diff(idx)))
    return timedelta(seconds=max(freqs, key=freqs.get))


def _wrap_single_list(param_grid: Union[List, Dict]):
    """
    Wraps all non list values as single
    :param param_grid:
    :return:
    """
    as_list = lambda x: x if isinstance(x, (tuple, list, dict, np.ndarray)) else [x]
    if isinstance(param_grid, list):
        return [_wrap_single_list(ps) for ps in param_grid]
    return {k: as_list(v) for k, v in param_grid.items()}


def permutate_params(parameters: dict, conditions: Union[types.FunctionType, list, tuple] = None, wrap_as_list=True) -> \
List[Dict]:
    """
    Generate list of all permutations for given parameters and it's possible values

    Example:

    >>> def foo(par1, par2):
    >>>     print(par1)
    >>>     print(par2)
    >>>
    >>> # permutate all values and call function for every permutation
    >>> [foo(**z) for z in permutate_params({
    >>>                                       'par1' : [1,2,3],
    >>>                                       'par2' : [True, False]
    >>>                                     }, conditions=lambda par1, par2: par1<=2 and par2==True)]

    1
    True
    2
    True

    :param conditions: list of filtering functions
    :param parameters: dictionary
    :param wrap_as_list: if True (default) it wraps all non list values as single lists (required for sklearn)
    :return: list of permutations
    """
    if conditions is None:
        conditions = []
    elif isinstance(conditions, types.FunctionType):
        conditions = [conditions]
    elif isinstance(conditions, (tuple, list)):
        if not all([isinstance(e, types.FunctionType) for e in conditions]):
            raise ValueError('every condition must be a function')
    else:
        raise ValueError('conditions must be of type of function, list or tuple')

    args = []
    vals = []
    for (k, v) in parameters.items():
        args.append(k)
        vals.append([v] if not isinstance(v, (list, tuple)) else v)
    d = [dict(zip(args, p)) for p in product(*vals)]
    result = []
    for params_set in d:
        conditions_met = True
        for cond_func in conditions:
            func_param_args = cond_func.__code__.co_varnames
            func_param_values = [params_set[arg] for arg in func_param_args]
            if not cond_func(*func_param_values):
                conditions_met = False
                break
        if conditions_met:
            result.append(params_set)

    # if we need to follow sklearn rules we should wrap every non iterable as list
    return _wrap_single_list(result) if wrap_as_list else result


def debug_output(data, name, start=3, end=3):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        hdr = f'.-<{name} {len(data)} records>-' + ' -' * 50
        sep = ' -' * 50
        print(hdr[:len(sep)])
        print(data.head(start).to_string(header=True))
        print(' \t ...  ')
        print(data.tail(end).to_string(header=False))
        print(sep)
    else:
        print(data)
