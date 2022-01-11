import pandas as pd
import numpy as np

from qlearn.core.utils import debug_output


class ForwardReturns:
    """
    Forward returns calculator
    """

    def __init__(self, horizon=1, returns_type='pct', tick_size=1):
        if horizon <= 0:
            raise ValueError("horizon parameter must be positive for forward returns calculations !!!")

        self.horizon = horizon
        r_types = {
            'pct': 'pct',
            'percentage': 'pct',
            'logret': 'logret',
            'log': 'logret',
            'abs': 'abs',
            'absolute': 'abs',
            'usd': 'abs',
            'pips': 'pips',
            'ticks': 'pips',
        }
        self.tick_size_ = tick_size
        if returns_type in r_types:
            self.returns_type = r_types[returns_type]
        else:
            raise ValueError(f'Unknown type for returns presentation "{returns_type}" !')

    def _forward_returns(self, price: pd.Series):
        rets = None

        if self.returns_type == 'pct':
            rets = price.pct_change(-self.horizon)

        elif self.returns_type == 'logret':
            rets = np.log(price).diff(-self.horizon)

        elif self.returns_type == 'abs':
            rets = price.diff(-self.horizon)

        elif self.returns_type == 'pips':
            rets = price.diff(-self.horizon) / self.tick_size_

        return rets.fillna(0).rename('returns')

    def _get_series_for_returns(self, x_data, r_data, column_name, **kwargs):
        if isinstance(x_data, pd.DataFrame):

            # we got portfolio data here
            if isinstance(x_data.columns, pd.MultiIndex):
                if r_data is None:
                    raise ValueError('>>> For portfolio data second argument must be passed !!!')

                r_tmp = r_data
                if isinstance(r_data, pd.DataFrame):
                    if column_name in r_data.columns:
                        r_tmp = r_data[column_name]
                    else:
                        raise ValueError(f">>> Can't find '{column_name}' in returns data !!!")

                # we will use data from r_data reindexed on prices index
                return r_tmp.reindex(x_data.index).ffill()

            if column_name not in x_data.columns:
                raise ValueError(f'Data must contain "{column_name}" column !')

            return x_data[column_name]

        return None

    def transform(self, x_data, r_data, used_price='close', **kwargs):
        series_for_ret = self._get_series_for_returns(x_data, r_data, used_price, **kwargs)
        if series_for_ret is not None:
            return self._forward_returns(series_for_ret)
        return r_data


class ForwardReturnsDirection(ForwardReturns):
    """
    Forward return's direction classificator (only Up (+1) or Down (-1))
    """

    def __init__(self, horizon, debug=False):
        super(ForwardReturnsDirection, self).__init__(horizon, 'abs')
        self.__debug = debug

    def transform(self, x_data, r_data, used_price='close', **kwargs):
        series_for_ret = self._get_series_for_returns(x_data, r_data, used_price, **kwargs)

        if self.__debug:
            debug_output(x_data, 'X_data')
            debug_output(r_data, 'R_data')

        if series_for_ret is not None:
            fwd_rets = np.sign(self._forward_returns(series_for_ret))
            if self.__debug:
                debug_output(fwd_rets, 'fwd_rets')
            return fwd_rets

        raise ValueError("Can't transform returns !")


class ForwardReturns3Classes(ForwardReturns):
    """
    Forward return classificator. It produces 3 classes for forward directions:
       up (+1), down (-1) and flat (0)
    """

    def __init__(self, horizon, tick_size, tick_tolerance):
        super(ForwardReturns3Classes, self).__init__(horizon, 'ticks', tick_size=tick_size)
        self.tick_tolerance = tick_tolerance

    def identify_classes(self, returns: pd.Series, t_threshold):
        return (1 + ((returns > t_threshold) + 0) - ((returns < -t_threshold) + 0)) - 1

    def transform(self, x_data, r_data, used_price='close', **kwargs):
        series_for_ret = self._get_series_for_returns(x_data, r_data, used_price, **kwargs)
        if series_for_ret is not None:
            return self.identify_classes(self._forward_returns(series_for_ret), self.tick_tolerance)

        raise ValueError("Can't transform returns !")
