import numpy as np
import pandas as pd

def remove_extreme_value(factor, std=3, have_negative=True):
    '''
    input:
    ------
    factor: A series whose index is stock code, whose values are factor values;
    std: the multiple of standard deviation ;
    have_negative: bool value to check if there are negative values.

    output:
    -------
    Series.
    '''

    r = factor.dropna().copy()
    if have_negative == False:
        r = r[r >= 0]
    else:
        pass

    # remove extreme values
    edge_up = r.mean() + std * r.std()
    edge_low = r.mean() - std * r.std()
    r[r > edge_up] = edge_up
    r[r < edge_low] = edge_low

    return r


def standardize(s, win=50):
    '''
    Method:
    -------
    result = (data - data.mean())/data.std()

    Input:
    ------
    Series

    Output:
    -------
    Series

    '''

    rolling_mean = s.rolling(window=win, min_periods=0).mean()
    rolling_std = s.rolling(window=win, min_periods=0).std()
    result = (np.array(s) - np.array(rolling_mean)) / np.array(rolling_std)

    return pd.Series(result).fillna(0)


def normalize(s, win=50):
    '''
    Method:
    -------
    result = (data - data.min())/(data.max() - data.min())

    Input:
    ------
    Series

    Output:
    -------
    Series

    '''

    rolling_min = s.rolling(window=win, min_periods=0).min()
    rolling_max = s.rolling(window=win, min_periods=0).max()
    result = (np.array(s) - np.array(rolling_min)) / (np.array(rolling_max) - np.array(rolling_min))

    return pd.Series(result)


def maxabs(s):
    '''
    Method:
    -------
    result = data/10^np.ceil(log10(data.abs().max()))

    Input:
    ------
    Series

    Output:
    -------
    Series

    '''

    data = s.dropna().copy()
    re = data/10**np.ceil(np.log10(data.abs().max()))

    return re

def get_quantiles(x):
    '''
    Method:
    -------
    
    for each value in x, return which quantile it corresponds to

    Input:
    ------
    Series

    Output:
    -------
    Series
    
    '''

    re = [np.count_nonzero(x<x_i)/(len(x)-1) for x_i in x]

    return pd.Series(re)