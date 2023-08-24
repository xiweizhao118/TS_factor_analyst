import torch
from itertools import count
import math
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import utils

class calculator():
    def __init__(self, df, factor) -> None:
        self.ret= np.array(df.iloc[:,1])
        self.ret_mean = self.ret.mean()
        self.factor = factor
    
    def calc_single_IC_ret(self):
        single_ic = []
        for i in range(self.factor.shape[1]):
            f = np.array(self.factor.iloc[:,i])
            f_mean = f.mean()
            num = sum((self.ret-self.ret_mean)*(f-f_mean))
            den = math.sqrt(sum((self.ret-self.ret_mean)**2)*sum((f-f_mean)**2))
            single_ic.append(num/den)
        return single_ic
    
    def calc_mutual_IC(self,index):
        multi_ic = []
        f = np.array(self.factor.iloc[:,index])
        f_mean = f.mean()
        for i in range(self.factor.shape[1]):
            ff = np.array(self.factor.iloc[:,i])
            ff_mean = ff.mean()
            num = sum((f-f_mean)*(ff-ff_mean))
            den = math.sqrt(sum((f-f_mean)**2)*sum((ff-ff_mean)**2))
            multi_ic.append(num/den)
        return multi_ic

class LinearCombine(calculator):
    '''
    Method:
    -------
    Model: simple linear model, the coefficient of each factor is to be estimated;
    Loss function: IC value between combined factor and the real return rate in a future period;
    Optimization method: torch.optim.Adam

    Params:
    -------
    capacity: combined factor columns length;
    train_size: traning set's percentage.

    Return:
    -------
    weights (np.array): optimal results of combination weights;
    new_factor (pd.Series): a combined factor using weights and the old factors;
    bound (list): for backtesting, calculated using the training set.
    '''

    def __init__(
        self,
        capacity: int,
        device: torch.device = torch.device('cpu')
    ):

        self.size = capacity
        self.single_ics: np.ndarray = np.zeros(self.size)
        self.mutual_ics: np.ndarray = np.identity(self.size)
        self.weights: np.ndarray = np.zeros(self.size)
        self.best_ic_ret: float = -1
        self.device = device
        self.eval_cnt = 0

    def _optimize(self, lr: float, n_iter: int) -> np.ndarray:

        ics_ret = torch.from_numpy(np.array(self.single_ics)).to(self.device)
        ics_mut = torch.from_numpy(np.array(self.mutual_ics)).to(self.device)
        weights = torch.from_numpy(np.array(self.weights)).to(self.device).requires_grad_()
        optim = torch.optim.Adam([weights], lr=lr)

        loss_ic_min = 1e9 + 7  # An arbitrary big value
        best_weights = weights.cpu().detach().numpy()
        iter_cnt = 0
        for it in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()

            loss = loss_ic

            optim.zero_grad()
            loss.backward()
            optim.step()

            if loss_ic_min - loss_ic_curr > 1e-6:
                iter_cnt = 0
            else:
                iter_cnt += 1

            if loss_ic_curr < loss_ic_min:
                best_weights = weights.cpu().detach().numpy()
                loss_ic_min = loss_ic_curr

            if iter_cnt >= n_iter or it >= 10000:
                print("Iteration times: %d" %it)
                break

        return best_weights

    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(self.mutual_ics[:self.size, :self.size],self.single_ics[:self.size])[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights[:self.size]
        
    def _calc_ics(self, df, factor):
        Cal = calculator(df, factor)

        single_ic = Cal.calc_single_IC_ret()

        mutual_ics = []
        for i in range(self.size):
            mutual_ic = Cal.calc_mutual_IC(i)
            mutual_ics.append(mutual_ic)
        mutual_ics = np.array(mutual_ics)

        return single_ic, mutual_ics

        
    def train_new_expr(self, df, factor, train_size):
        train_size = int(len(df) * train_size)
        self.single_ics, self.mutual_ics = self._calc_ics(df.iloc[:train_size,:], factor.iloc[:train_size,:])
        
        new_weights = self._optimize(lr=5e-4, n_iter=500)

        new_factor = pd.Series([new_weights.dot(np.array(factor.iloc[i,:])) for i in range(len(factor))])

        return new_weights, new_factor
    
        

