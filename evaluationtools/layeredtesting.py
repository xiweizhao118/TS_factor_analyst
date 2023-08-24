import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nri_quant import utilities as nut
from nri_quant.utilities import nnumba as nnb


class LayeredBacktestingEngine:
    def __init__(self, df, factor, window, n_group, ret_period=1):
        deltalen = len(df)-len(factor)
        # preprocessing: combine dataframe factor and return rate, index is date
        self.factor = pd.DataFrame(
            {
                "factor": np.array(factor['factor']),
                "ret_1d": np.log(df['close'][deltalen:]).diff().shift(1).values,
                "ret_5d": np.log(df['close'][deltalen:]).diff(periods=5).shift(5).values,
                "ret_10d": np.log(df['close'][deltalen:]).diff(periods=10).shift(10).values,
                "ret_21d": np.log(df['close'][deltalen:]).diff(periods=21).shift(21).values,
                "ret_63d": np.log(df['close'][deltalen:]).diff(periods=63).shift(63).values,
            },
            index=factor['date'].values
        )
        # initialization: win is a const
        self.win = window
        self.n_group = n_group
        self.ret_period = ret_period

    def roll_group(self):
        def f(x): return nnb.split(x, self.n_group, 2)
        q_group_roll = nut.rolling_apply(
            f, self.factor['factor'].values, self.win).tolist()
        df_q_group_roll = pd.DataFrame(
            q_group_roll, index=self.factor.index[self.win-1:], columns=[f'D-{i}' for i in range(self.win-1, -1, -1)])
        self.factor['q_group_roll'] = df_q_group_roll.iloc[:, -1]
        return self.factor

    def cal_average_return_by_group(self, roll_seires):
        labels = roll_seires['q_group_roll']
        returns = roll_seires[f'ret_{self.ret_period}d']
        group_returns = {g: np.nanmean(list(returns[index] for (index, value) in enumerate(labels) if value == g))
                         for g in np.unique(labels.dropna())}

        return pd.Series(group_returns, name="group_return")

    def cal_roll_res_list(self):
        self.factor = self.roll_group()
        roll_res_list = [self.cal_average_return_by_group(self.factor.iloc[i:i+self.win]).rename(
            self.factor.index[i+self.win]) for i in range(len(self.factor)-self.win)]
        roll_res_df = pd.concat(roll_res_list, axis=1).T
        return roll_res_df.fillna(0).cumsum()

# code = "000905"
# start_time = '2018-01-01'
# end_time = '2023-05-31'
# data = download_domestic_index_data(code, start_time, end_time)
# factor = augur_0002(data)
# engine = LayeredBacktestingEngine(data, factor, 90, 5)
# cret_group = engine.cal_roll_res_list(ret_period=1)
# cret_group.plot.line()
# plt.show()
