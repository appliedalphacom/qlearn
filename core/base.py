import numpy as np
import copy
import inspect
from typing import Union, List, Dict

import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, ClusterMixin, DensityMixin

from qlearn.core.data_utils import make_dataframe_from_dict
from qlearn.core.pickers import AbstractDataPicker
from qlearn.core.utils import debug_output


class MarketDataComposer(BaseEstimator):
    def __init__(self, predictor,
                 data_picker: AbstractDataPicker,
                 transformer=None,
                 column='close',
                 timeframe=None,
                 align_prediction_by=None,
                 debug=False,
                 ):
        self.column = column
        self.predictor = predictor
        self.transformer = transformer
        self.timeframe = timeframe
        self.picker = data_picker
        self.debug_ = debug
        self.fitted_predictors_ = {}
        self.symbols_ = []
        self.best_params_ = None
        self.best_score_ = None

        # how to deal with predictions
        self._propagate_prediction = False
        self._fill_by = None
        self._forward = False
        if align_prediction_by is not None:
            if align_prediction_by in ['ffill', 'forward']:
                self._propagate_prediction = True
                self._forward = True
            elif isinstance(align_prediction_by, (int, float)) or np.isnan(align_prediction_by):
                self._propagate_prediction = True
                self._fill_by = align_prediction_by
            else:
                print(f'WARN: not sure how to deal with passed align_prediction_by="{align_prediction_by}"')

    def take(self, data, nth: Union[str, int] = 0):
        """
        Helper method to take n-th iteration from data picker

        :param data: data to be iterated
        :param nth: if int it returns n-th iteration of data
        :return: data or none if not matched
        """
        return self.picker.take(data, nth)

    def as_datasource(self, data) -> Dict:
        """
        Return prepared data ready to be used in simulator

        :param data: input data
        :return: {symbol : preprocessed_data}
        """
        return self.picker.as_datasource(data)

    def fit(self, X, y, **fit_params):
        # reset fitted predictors
        self.fitted_predictors_ = {}
        self.best_params_ = {}
        self.best_score_ = {}

        # we setup timeframe
        self.picker.timeframe = self.timeframe
        for symbol, xp in self.picker.iterate(X):
            yp = y

            if self.transformer is not None:
                yp = self.transformer.transform(xp, y, **fit_params)
                if self.debug_:
                    debug_output(yp, 'Transformed data')

            # set meta data
            self.symbols_ = symbol
            _f_p = self.predictor.fit(xp, yp, **fit_params)

            # store best parameters for each symbol
            if hasattr(_f_p, 'best_params_') and hasattr(_f_p, 'best_score_'):
                self.best_params_[str(symbol)] = _f_p.best_params_
                self.best_score_[str(symbol)] = _f_p.best_score_

                # just some output on unclear situations
                if self.debug_:
                    print(symbol, _f_p.best_params_)

            # here we need to keep a copy of fitted object
            self.fitted_predictors_[str(symbol)] = copy.deepcopy(_f_p)

        return self

    def predict(self, X):
        r = dict()
        for symbol, xp in self.picker.iterate(X):
            # set meta data
            self.symbols_ = symbol
            p_key = str(symbol)

            if p_key not in self.fitted_predictors_:
                raise ValueError(f"It seems that predictor was not trained for '{p_key}' !")

            yh = self.fitted_predictors_[p_key].predict(xp)

            # if we use it for model fitting we need to have consistent prediction series
            if self._propagate_prediction:
                yh = yh.reindex(xp.index)
            if self._fill_by is not None:
                yh = yh.fillna(self._fill_by)
            if self._forward:
                yh = yh.ffill()

            if isinstance(symbol, str):
                r[symbol] = yh
            else:
                r = yh

        return make_dataframe_from_dict(r, 'frame')


@dataclass
class MetaInfo:
    symbols: Union[List[str], None]
    column: str
    timeframe: Union[str, None]
    tick_sizes: dict = None  # TODO: or move to some global context ???
    tick_prices: dict = None  # TODO: tick price for futures


class BasicMarketEstimator(BaseEstimator):

    @staticmethod
    def metadata():
        c_frame = inspect.currentframe().f_back
        while c_frame:
            for c, t in c_frame.f_locals.items():
                if isinstance(t, MarketDataComposer) or (
                        hasattr(t, 'symbols_') and hasattr(t, 'column') and hasattr(t, 'timeframe')):
                    return MetaInfo(symbols=t.symbols_,
                                    column=t.column,
                                    timeframe=t.timeframe,
                                    # ------------------------------
                                    tick_sizes={},
                                    tick_prices={},
                                    # ------------------------------
                                    )
            c_frame = c_frame.f_back
        return MetaInfo(symbols=None, column='close', timeframe=None)

    def mix_with(obj, clss):
        """
        Dynamically mix class of object with another class.

        :param clss:
        :return:
        """
        if clss not in obj.__class__.__bases__:
            obj.__class__.__bases__ = obj.__class__.__bases__ + (clss,)

        return obj

    def as_classifier(self):
        return BasicMarketEstimator.mix_with(self, ClassifierMixin)

    def as_regressor(self):
        return BasicMarketEstimator.mix_with(self, RegressorMixin)

    def as_cluster(self):
        return BasicMarketEstimator.mix_with(self, ClusterMixin)

    def as_density(self):
        return BasicMarketEstimator.mix_with(self, DensityMixin)
