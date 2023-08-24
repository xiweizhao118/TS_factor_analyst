import pandas as pd
import numpy as np

def ret_rate(data, duration):
    '''
    Calculate a stock's return rate in a certain period.

    Params:
    -------
    data (dataframe): stock, contains column 'date' and 'close';
    duration (int).

    Return:
    -------
    Dataframe.
    '''

    ret = pd.DataFrame({'date': data['date'],
                        'y': data['close'].pct_change(duration).fillna(0)})
    
    return ret

def price(data, label='close'):
    '''
    Return a stock's price.

    Params:
    -------
    data (dataframe): stock, contains column 'date' and 'close';
    label (str): stock's column name.

    Return:
    -------
    Dataframe.
    '''

    price = pd.DataFrame({'date': data['date'],
                          'y': data[label]})
    
    return price

def sharpe_ratio(data, period=5):
    '''
    Calculate a stock's daily sharpe ratio.

    Method:
    -------
    Benchmark is government bond.
    excess return mean/std is calculated in the future period;
    daily sharpe ratio = excess return mean / excess return std.

    Params:
    -------
    data (dataframe): stock, contains column 'date' and 'close';
    label (str): stock's column name.

    Return:
    -------
    Dataframe.
    '''
    
    from . import download as dl

    # calculate daily return rate
    daily_ret = ret_rate(data, duration=1)

    # benchmark: const
    const = 0.02
    benchmark = pd.DataFrame({'date': data['date'],
                              'y': const * len(data)})

    # data preparation
    excess_returns = daily_ret['y'] - benchmark['y']
    avg_excess_return = [excess_returns[i:i+period].mean() for i in range(len(data)-period)] + [excess_returns[i:].mean() for i in range(len(data)-period,len(data))]
    sd_excess_return = [excess_returns[i:i+period].std() for i in range(len(data)-period)] + [excess_returns[len(data)-period:len(data)].std()] * period

    # calculate the daily sharpe ratio
    daily_sharpe_ratio = np.array(avg_excess_return)/np.array(sd_excess_return)

    daily_sharpe_ratio = pd.DataFrame({'date': data['date'],
                                       'y': daily_sharpe_ratio})

    return daily_sharpe_ratio
