from collections import defaultdict
import numpy as np
import pandas as pd

from ira.charting.lookinglass import LookingGlass
from ira.analysis.tools import scols, srows
from ira.utils.utils import mstruct
from ira.charting.plot_helpers import fig
from ira.utils.nb_functions import sbp

import matplotlib.pyplot as plt
import seaborn as sns

from qlearn.core.data_utils import ohlc_to_flat_price_series, infer_series_frequency


def plot_returns_statistics(model, data, scoring):
    rX = model.estimated_portfolio(data, scoring)
    
    sbp(13, 1, c=2)
    plt.plot(rX.cumsum());  plt.title('Equity')
    
    sbp(13, 3);     
    for c in rX.columns:
        sns.kdeplot(rX[c], label=c) 
        
    plt.axvline(0, ls='--', c='r'); 
    plt.title('Forward returns distribution estimation');