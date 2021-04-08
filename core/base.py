import copy
import inspect
from typing import Union, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from qlearn.core.data_utils import make_dataframe_from_dict, pre_close_time_shift
from qlearn.core.metrics import ForwardReturnsCalculator
from qlearn.core.pickers import AbstractDataPicker, SingleInstrumentPicker, PortfolioPicker
from qlearn.core.structs import MarketInfo, _FIELD_MARKET_INFO, _FIELD_EXACT_TIME, _FIELD_FILTER_INDICATOR, \
    QLEARN_VERSION
from qlearn.core.utils import get_object_params


def predict_and_postprocess(class_predict_function):
    def wrapped_predict(obj, xp, *args, **kwargs):
        # run original predict method
        yh = class_predict_function(obj, xp, *args, **kwargs)

        # if this predictor doesn't provide tag and we operate with closes
        if not getattr(obj, _FIELD_EXACT_TIME, False) and obj.market_info_.column == 'close':
            yh = yh.shift(1, freq=pre_close_time_shift(xp))

        # if we want to filter out signals
        if hasattr(obj, _FIELD_FILTER_INDICATOR):
            filter_indicator_name = getattr(obj, _FIELD_FILTER_INDICATOR)

            # filter out predictions by filter's value (passed signals where filter > 0)
            if filter_indicator_name is not None and filter_indicator_name in xp.columns:
                ms = pd.merge_asof(yh.rename('S'), xp[filter_indicator_name].rename('F'), left_index=True,
                                   right_index=True)
                yh = ms[(ms.F > 0) & (ms.S != 0)].S

        return yh

    return wrapped_predict


def preprocess_fitargs_and_fit(class_fit_function):
    def wrapped_fit(obj, x, y, **fit_params):
        # intercept market_info_
        if _FIELD_MARKET_INFO in fit_params:
            obj.market_info_ = fit_params.pop(_FIELD_MARKET_INFO)

        return class_fit_function(obj, x, y, **fit_params)

    return wrapped_fit


def _decorate_class_method_if_exist(cls, method_name, decorator):
    m = inspect.getmembers(cls, lambda x: (inspect.isfunction(x) or inspect.ismethod(x)) and x.__name__ == method_name)
    if m:
        setattr(cls, m[0][0], decorator(m[0][1]))


def signal_generator(cls):
    cls.__qlearn__ = QLEARN_VERSION
    setattr(cls, _FIELD_MARKET_INFO, None)
    setattr(cls, _FIELD_EXACT_TIME, False)
    setattr(cls, _FIELD_FILTER_INDICATOR, None)
    _decorate_class_method_if_exist(cls, 'predict', predict_and_postprocess)
    _decorate_class_method_if_exist(cls, 'predict_proba', predict_and_postprocess)
    _decorate_class_method_if_exist(cls, 'fit', preprocess_fitargs_and_fit)
    return cls


def collect_qlearn_estimators(p, estimators_list, step=''):
    if isinstance(p, BaseEstimator) and hasattr(p, '__qlearn__'):
        estimators_list.append((step, p))
        return estimators_list

    if isinstance(p, Pipeline):
        for sn, se in p.steps:
            collect_qlearn_estimators(se, estimators_list, (step + '__' + sn) if step else sn)
        return estimators_list

    if isinstance(p, BaseSearchCV):
        return collect_qlearn_estimators(p.estimator, estimators_list, step)

    if isinstance(p, MarketDataComposer):
        return collect_qlearn_estimators(p.predictor, estimators_list, step)

    return estimators_list


class Filter(TransformerMixin):
    """
    Basic class for any signal filtering class
    """
    filter_indicator = None

    def get_filter(self, x: Union[pd.Series, pd.DataFrame]):
        """
        Filtering logic implementation. Method should return pandas Series or numpy array.
        """
        # default filter
        return pd.Series(1, x.index)

    def _filter_name(self):
        name = None
        if hasattr(self, _FIELD_FILTER_INDICATOR):
            name = getattr(self, _FIELD_FILTER_INDICATOR)
        if name is None:
            name = self.__class__.__name__
        return name

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x):
        f = self.get_filter(x)

        if not isinstance(f, pd.Series):
            if not isinstance(f, np.ndarray):
                raise ValueError(f'{self.__class__.__name__} get_filter must return Series or numpy array')
            else:
                ff = f.flatten()
                if len(ff) != len(x):
                    raise ValueError(
                        f'{self.__class__.__name__} get_filter returned wrong size ({len(ff)} but expected {len(x)})')
                f = pd.Series(ff, index=x.index)

        return pd.concat((x, f.rename(self._filter_name())), axis=1).ffill()

    def apply_to(self, preidictor):
        """
        Apply this filter to predictor
        """
        return signals_filtering_pipeline(self, preidictor)


def signals_filtering_pipeline(fltr: Filter, predictor):
    """
    Make pipeline from filter and predictor
    """
    if not isinstance(fltr, Filter):
        raise ValueError('First argument must be instance of Filter class')

    if not isinstance(predictor, BaseEstimator):
        raise ValueError('Second argument must be instance of BaseEstimator class')

    p_cls, f_cls = predictor.__class__, fltr.__class__
    filter_name = f_cls.__name__
    p_meths = {**p_cls.__dict__, **{_FIELD_FILTER_INDICATOR: filter_name}}
    f_meths = {**f_cls.__dict__, **{_FIELD_FILTER_INDICATOR: filter_name}}
    new_p_cls = type(p_cls.__name__, tuple(p_cls.mro()[1:]), p_meths)
    new_f_cls = type(f_cls.__name__, tuple(f_cls.mro()[1:]), f_meths)
    f_params = get_object_params(fltr)

    new_p_inst = new_p_cls(**predictor.get_params())
    new_f_inst = new_f_cls(**f_params)

    return Pipeline([(f_cls.__name__, new_f_inst), (p_cls.__name__, new_p_inst)])


class MarketDataComposer(BaseEstimator):
    """
    Market data composer for any predictors related to trading signals generation
    """

    def __init__(self, predictor, selector: AbstractDataPicker, column='close', debug=False):
        self.column = column
        self.predictor = predictor
        self.selector = selector
        self.fitted_predictors_ = {}
        self.best_params_ = None
        self.best_score_ = None
        self.estimators_ = collect_qlearn_estimators(predictor, list())
        self.debug = debug

    def __prepare_market_info_data(self, symbol, kwargs) -> dict:
        self.market_info_ = MarketInfo(symbol, self.column)
        new_kwargs = dict(**kwargs)
        for name, _ in self.estimators_:
            mi_name = f'{name}__{_FIELD_MARKET_INFO}' if name else _FIELD_MARKET_INFO
            new_kwargs[mi_name] = MarketInfo(symbol, self.column)
        return new_kwargs

    def for_interval(self, start, stop):
        """
        Setup dates interval for fitting/prediction
        """
        self.selector.for_range(start, stop)
        return self

    def take(self, data, nth: Union[str, int] = 0):
        """
        Helper method to take n-th iteration from data picker

        :param data: data to be iterated
        :param nth: if int it returns n-th iteration of data
        :return: data or none if not matched
        """
        return self.selector.take(data, nth)

    def as_datasource(self, data) -> Dict:
        """
        Return prepared data ready to be used in simulator

        :param data: input data
        :return: {symbol : preprocessed_data}
        """
        return self.selector.as_datasource(data)

    def fit(self, X, y, **fit_params):
        # reset fitted predictors
        self.fitted_predictors_ = {}
        self.best_params_ = {}
        self.best_score_ = {}

        for symbol, xp in self.selector.iterate(X):
            # propagate market info meta data to be passed to fit method of all qlearn estimators
            n_fit_params = self.__prepare_market_info_data(symbol, fit_params)

            # in case we still have nothing we need to feed it by some values
            # to avoid fit validation failure
            if y is None:
                y = np.zeros_like(xp)

            # process fitting on prepared data
            _f_p = self.predictor.fit(xp, y, **n_fit_params)

            # store best parameters for each symbol
            if hasattr(_f_p, 'best_params_') and hasattr(_f_p, 'best_score_'):
                self.best_params_[str(symbol)] = _f_p.best_params_
                self.best_score_[str(symbol)] = _f_p.best_score_

                # just some output on unclear situations
                if self.debug:
                    print(symbol, _f_p.best_params_)

            # here we need to keep a copy of fitted object
            self.fitted_predictors_[str(symbol)] = copy.deepcopy(_f_p)

        return self

    def __get_prediction(self, symbol, x):
        p_key = str(symbol)

        if p_key not in self.fitted_predictors_:
            raise ValueError(f"Can't find fitted predictor for '{p_key}' !")

        # run predictor
        predictor = self.fitted_predictors_[p_key]
        yh = predictor.predict(x)
        return yh

    def predict(self, x):
        """
        Get prediction on all market data from x
        """
        r = dict()

        for symbol, xp in self.selector.iterate(x):
            yh = self.__get_prediction(symbol, xp)
            if isinstance(symbol, str):
                r[symbol] = yh
            else:
                r = yh

        return make_dataframe_from_dict(r, 'frame')

    def estimated_portfolio(self, x, forwards_calculator: ForwardReturnsCalculator):
        """
        Get estimated portfolio based on forwards calculator
        """
        rets = {}
        if forwards_calculator is None or not hasattr(forwards_calculator, 'get_forward_returns'):
            raise ValueError(
                "forwards_calculator parameter doesn't have get_forward_returns(price, signals, market_info) method"
            )
        for symbol, xp in self.selector.iterate(x):
            yh = self.__get_prediction(symbol, xp)
            return_series = forwards_calculator.get_forward_returns(xp, yh, MarketInfo(symbol, self.column))
            rets[symbol] = forwards_calculator.get_forward_returns(xp, yh, MarketInfo(symbol, self.column))

        return make_dataframe_from_dict(rets, 'frame')


class SingleInstrumentComposer(MarketDataComposer):
    """
    Shortcut for MarketDataComposer(x, SingleInstrumentPicker(), ...)
    """
    def __init__(self, predictor, column='close', debug=False):
        super().__init__(predictor, SingleInstrumentPicker(), column, debug)


class PortfolioComposer(MarketDataComposer):
    """
    Shortcut for MarketDataComposer(x, PortfolioPicker(), ...)
    """
    def __init__(self, predictor, column='close', debug=False):
        super().__init__(predictor, PortfolioPicker(), column, debug)
