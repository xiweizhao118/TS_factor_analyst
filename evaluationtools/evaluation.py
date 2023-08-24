
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
import pandas as pd
import sys
sys.path.append("..")
import utils

def distribution(factor, plot=True):
    '''
    Plot the factors distribution.

    inputs:
    -------
    factor: dataframe: date, factor
    lb: string: plot label

    '''
    # delete rows where values are zero
    if factor.iloc[:,1][:100].mean() == factor.iloc[:,1][0]:
        factor = factor[factor.iloc[:,1]!=factor.iloc[:,1][0]]

    if plot:
        import matplotlib.pyplot as plt

        utils.plot(factor['date'], factor.iloc[:,1], [5, 20], 'factor')

        plt.figure().set_figheight(5)
        plt.figure().set_figwidth(20)
        plt.hist(factor.iloc[:,1], bins='auto')
        plt.grid()


def if_normal(factor, plot=True):
    '''
    Check if the factors distribution is normal.
    [KS test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
    
    '''
    from scipy import stats
    result = stats.kstest(factor['factor'], 'norm', args=(
        factor['factor'].mean(), factor['factor'].std()))
    print('--- K-S Test Results ---')
    print(f'Statistic: {result[0]:.4f}')
    print(f'p value: {result[1]:.4f}')

    if plot:
        import pylab
        import statsmodels.api as sm
        my_data = factor['factor']
        sm.qqplot(my_data, line='45')
        pylab.show()
    return result


def adf_test(factor, win, plot=False):
    '''
    Check if factor is stable. When ADF Statistics is more negative, that means factor is more stable.

    input:
    ------
    dataframe, colums: date, factor

    return:
    -------
    ADF statistics, p-value, critical values (dictionary).

    '''

    f = factor['factor'].diff().fillna(0)

    adf = []
    p_value = []
    critical_values = []
    for i in range(win, len(f)):
        df = f[(f.index >= i-win) & (f.index < i)]
        tmp = adfuller(df.values)
        adf.append(tmp[0])
        p_value.append(tmp[1])
        critical_values.append(tmp[4])

    if plot:
        # plot
        utils.plot(factor['date'].iloc[win:], adf, [5, 20], 'ADF statistics')
        utils.plot(factor['date'].iloc[win:], p_value, [5, 20], 'p values')

    return adf, p_value, critical_values


def kpss_test(factor, win, plot=False):
    '''
    The KPSS test figures out if a time series is stationary around a mean or linear trend, or is non-stationary due to a unit root.

    input:
    ------
    dataframe, colums: date, factor

    return:
    -------
    KPSS statistics, p-value, num lags, critical values (dictionary).

    '''
    f = factor['factor'].diff().fillna(0)

    kps = []
    p_value = []
    n_lags = []
    critical_values = []
    for i in range(win, len(f)):
        df = f[(f.index >= i-win) & (f.index < i)]
        tmp_kps, tmp_p_value, tmp_n_lags, tmp_critical_values = kpss(df.values)
        kps.append(tmp_kps)
        p_value.append(tmp_p_value)
        n_lags.append(tmp_n_lags)
        critical_values.append(tmp_critical_values)

    if plot:
        # plot
        utils.plot(factor['date'].iloc[win:], kps, [5, 20], 'KPSS statistics')
        utils.plot(factor['date'].iloc[win:], p_value, [5, 20], 'p values')

    return kps, p_value, n_lags, critical_values


def ic_test(df, f, win, plot=False, train_length=None):

    f = f['factor']

    # transfer dataframe to series
    ret = df['close'].pct_change(periods=252).fillna(0).squeeze()

    # calculate the coefficients between factors and returns, using spearman method
    def rolling_spearman(seqa, seqb, window):
        seqa = np.array(seqa)
        seqb = np.array(seqb)
        stridea = seqa.strides[0]
        ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
        strideb = seqb.strides[0]
        ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
        ar = pd.DataFrame(ssa)
        br = pd.DataFrame(ssb)
        ar = ar.rank(1)
        br = br.rank(1)
        corrs = ar.corrwith(br, 1)
        return pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)
    
    ic = pd.Series(rolling_spearman(ret, f, window=252)).fillna(0)

    if plot:
        utils.plot(df['date'].iloc[win:], ic.iloc[win:], [5, 20], 'IC', train_length)

    return ic


def grangers_causation_matrix(data, factor, test, maxlag, win, plot=False):
    '''
    If a given p-value is < significance level (0.05), for example, take the value 0.0, we can reject the null hypothesis and conclude that factors Granger causes return. 

    input:
    ------
    data: zz500 index.
    factor: dataframe, colums: date, factor.
    test: 'params_ftest', 'ssr_ftest' are based on F distribution; 'ssr_chi2test', 'lrtest' are based on chi-square distribution.
    maxlag: int, If an integer, computes the test for all lags up to maxlag. If an iterable, computes the tests only for the lags in maxlag.
    win: rolling window length.

    return:
    -------
    min_p_value: list, minimum p value

    '''

    f = factor['factor']
    ret = data['close'].pct_change(periods=25).fillna(0)
    df = pd.merge(ret, f, left_index=True, right_index=True)

    min_p_value = []
    for i in range(win, len(f)):
        ddf = df[(df.index >= i-win) & (df.index < i)]
        tmp = grangercausalitytests(ddf, maxlag=maxlag, verbose=False)
        p_tmp = [round(tmp[j+1][0][test][1], 4) for j in range(maxlag)]
        min_p_value.append(np.min(p_tmp))

    if plot:
        # plot
        utils.plot(factor['date'].iloc[win:], min_p_value,
                   [5, 20], 'granger causality')

    return min_p_value
