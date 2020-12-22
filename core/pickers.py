import re
import pandas as pd
from ira.analysis.tools import ohlc_resample
from qlearn.core.data_utils import _get_top_names, detect_data_type, make_dataframe_from_dict


class Resampler:

    def _resample(self, data, timeframe, tz):
        """
        Resample data to timeframe
        :param data:
        :return:
        """
        r = data
        if timeframe is not None:
            if isinstance(data, pd.DataFrame):
                cols = data.columns
                if isinstance(cols, pd.MultiIndex):
                    symbols = _get_top_names(cols)
                    return pd.concat([ohlc_resample(data[c], timeframe, resample_tz=tz) for c in symbols], axis=1, keys=symbols)

            # all the rest cases
            r = ohlc_resample(data, timeframe, resample_tz=tz)
        return r


class AbstractDataPicker(Resampler):
    """
    Generic abstract data picker
    """

    def __init__(self, rules=None, timeframe=None, tz='UTC'):
        rules = [] if rules is None else rules
        self.rules = rules if isinstance(rules, (list, tuple, set)) else [rules]
        self.timeframe = timeframe
        self.tz = tz

    def select_range(self, start, stop=None):
        """
        If we need to select range of data to operate on
        :param start: starting date
        :param stop: end date
        :return:
        """
        # TODO !!!!
        pass

    def _is_selected(self, s, rules):
        if not rules:
            return True
        for r in rules:
            if re.match(r, s):
                return True
        return False

    def iterdata(self, data, selected_symbols, data_type, symbols, entries_types):
        raise NotImplementedError('Method must be implemented !')

    def iterate(self, data):
        info = detect_data_type(data)
        selected = [s for s in info.symbols if self._is_selected(s, self.rules)]
        return self.iterdata(data, selected, info.type, info.symbols, info.subtypes)


class SingleInstrumentPicker(AbstractDataPicker):
    """
    Iterate symbol by symbol
    """

    def __init__(self, rules=None, timeframe=None, tz='UTC'):
        super().__init__(rules=rules, timeframe=timeframe, tz=tz)

    def iterdata(self, data, selected_symbols, data_type, symbols, entries_types):
        if data_type == 'dict' or data_type == 'multi':
            # iterate every dict entry or column from multi index dataframe
            for s in selected_symbols:
                yield s, self._resample(data[s], self.timeframe, self.tz)
        elif data_type == 'ohlc' or data_type == 'series' or data_type == 'ticks':
            # just single series
            yield symbols[0], self._resample(data, self.timeframe, self.tz)
        else:
            raise ValueError(f"Unknown data type '{data_type}'")


class PortfolioPicker(AbstractDataPicker):
    """
    Iterate whole portfolio
    """

    def __init__(self, rules=None, timeframe=None, tz='UTC'):
        super().__init__(rules=rules, timeframe=timeframe, tz=tz)

    def iterdata(self, data, selected_symbols, data_type, symbols, entries_types):
        if data_type == 'dict':
            if entries_types:
                if len(entries_types) > 1:
                    raise ValueError(
                        "Dictionary contains data with different types so not sure how to merge them into portfolio !")
            else:
                raise ValueError("Couldn't detect types of dictionary items so not sure how to deal with it !")

            subtype = list(entries_types)[0]
            yield selected_symbols, make_dataframe_from_dict({s: self._resample(data[s], self.timeframe, self.tz) for s in selected_symbols}, subtype)

        elif data_type == 'multi':
            yield selected_symbols, self._resample(data[selected_symbols], self.timeframe, self.tz)

        elif data_type == 'ohlc' or data_type == 'series' or data_type == 'ticks':
            yield symbols, self._resample(data, self.timeframe, self.tz)

        else:
            raise ValueError(f"Unknown data type '{data_type}'")
