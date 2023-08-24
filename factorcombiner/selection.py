import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad

def select(factor_combined, y, train_size = 0.8, num=10):
    '''
    Select 10 factors in the factors pool.

    Method:
    -------
    Rank factors by a value:
    value = exp(-abs(corr_mutual).mean()) * (1+ic.mean())^2
    where: corr_mutual denotes correlation value between two factors;
           ic denotes ic value calculated by factor and daily return rate.
    
    Params:
    -------
    factor_combined (dataframe): factor pool;
    y (dataframe): a calculation method of return rate, the training target.

    Return:
    -------
    selected_col_list (list[str]): a list of selected factors' name.
    '''

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
    
    # init: define factor for train
    ic = []
    fname = factor_combined.columns
    train_len = int(len(factor_combined)*train_size)
    factor_train = factor_combined.iloc[:train_len,:]

    # calculate ic
    for i in range(len(fname)):
        f = factor_train.iloc[:,i]
        ret = y.iloc[:train_len,1]
        tmp = pd.Series(rolling_spearman(ret, f, window=252)).fillna(0)
        ic.append(tmp.mean())

    # calculate corr
    corr = factor_train.corr()
    corr_mutual = []
    for i in range(len(fname)):
        tmp = list(corr.iloc[:,i])
        tmp.pop(i)
        corr_mutual.append(abs(np.array(tmp).mean()))

    # calculate selection value
    selection_value = list(np.exp(-np.array(corr_mutual))*(1+np.array(ic))**2)
    selection_df = pd.DataFrame({'factor': fname,
                                'value': selection_value})
    
    # rank and select columns name
    selection_df = selection_df.sort_values(by="value",ascending=False)
    selection_df = selection_df.reset_index(drop=True)
    selected_col_list = list(selection_df.iloc[:num,0])
    
    return selected_col_list