import torch, math, random
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
from datetime import datetime as date
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

class LSTM(nn.Module):

        def __init__(self, param):
            super(LSTM, self).__init__()
            
            self.num_classes = param['num_classes']
            self.num_layers = param['num_layers']
            self.input_size = param['input_size']
            self.hidden_size = param['hidden_size']
            
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True)
            
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

        def forward(self, x):
            h_0 = Variable(torch.zeros(
                self.num_layers, x.size(0), self.hidden_size))
            
            c_0 = Variable(torch.zeros(
                self.num_layers, x.size(0), self.hidden_size))
            
            # Propagate input through LSTM
            ula, (h_out, _) = self.lstm(x, (h_0, c_0))
            
            h_out = h_out.view(-1, self.hidden_size)
            
            out = self.fc(h_out)
            
            return out

class LSTMDataPreprocess():
    def __init__(self, df) -> None:
        self.df = df

    def sliding_windows(self, data, seq_length):
        x = []
        y = []

        for i in range(len(data)-seq_length-1):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)
    
    def load_data(self, seq_length, training_data, test_size):
        # Set labels
        x, y = self.sliding_windows(training_data, seq_length)
        dataX = Variable(torch.Tensor(np.array(x)))
        dataY = Variable(torch.Tensor(np.array(y)))

        # train set
        train_size = int(len(y) * (1-test_size))
        trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
        trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

        testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
        testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

        return dataX, dataY, trainX, trainY, testX, testY
    
class LSTMEngine(LSTM):
    def __init__(self, df, seq_length, test_size) -> None:
        from sklearn.preprocessing import MinMaxScaler
        self.df = df
        self.seq_length = seq_length
        # load data and preprocessing
        dataEngine = LSTMDataPreprocess(self.df)

        # Scaler
        training_set = df.loc[:,['close']].values
        self.sc = MinMaxScaler()
        training_data = self.sc.fit_transform(training_set)
        self.dataX, self.dataY, self.trainX, self.trainY, self.testX, self.testY = dataEngine.load_data(seq_length, training_data, test_size)


    def train(self, LSTM_parameters, train_parameters):
        # initialize network
        lstm = LSTM(LSTM_parameters)
        # evaluation functions
        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(lstm.parameters(), lr=train_parameters['learning_rate'])
        #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

        # Train the model
        loss0 = 1000
        epoch = 0
        while True:
            outputs = lstm(self.trainX)
            optimizer.zero_grad()
            
            # obtain the loss function
            loss = criterion(outputs, self.trainY)
            
            loss.backward()
            
            optimizer.step()

            epoch += 1

            if abs(loss0-loss.item()) <= 0.0001 * loss0:
                break
            loss0 = loss.item()
            if epoch == train_parameters['num_epochs']:
                print('Warning: too many epochs')
                break
        
        return lstm
        
    def predict(self, lstm, evaluation):
        # load trained model
        train_predict = lstm(self.dataX)
        data_predict = train_predict.data.numpy()
        data_predict = self.sc.inverse_transform(data_predict)

        if evaluation:
            dataY_plot = self.dataY.data.numpy()
            dataY_plot = self.sc.inverse_transform(dataY_plot)
            self.evaluation(lstm, dataY_plot, data_predict)
        
        data_predict = self.df['close'][:self.seq_length+1].tolist() + data_predict.squeeze().tolist()

        return data_predict

    def evaluation(self, lstm, dataY_plot, data_predict):

        # result = pd.DataFrame({'real': sum(dataY_plot.tolist(),[]), 'predict': sum(data_predict.tolist(),[])})

        outputs = lstm(self.testX)
        # obtain the loss function
        loss = torch.nn.MSELoss()(outputs, self.testY)
        print('------------------------')
        print("Loss: %1.5f" % (loss.item()))

        print('------------------------')
        plt.figure().set_figheight(2)
        plt.figure().set_figwidth(10)
        plt.axvline(x=len(self.trainX), c='r', linestyle='--')
        plt.plot(dataY_plot, label='real')
        plt.plot(data_predict, label='predict')
        plt.suptitle('Time-Series Prediction')
        plt.legend()
        plt.show()


class XGBFuncModule():
    def __init__(self, df, N) -> None:
        self.df = df
        self.N = N

    def preprocess(self):
        # Change all column headings to be lower case, and remove spacing
        self.df.columns = [str(x).lower().replace(' ', '_') for x in self.df.columns]
        # Get month of each sample
        self.df['month'] = [int(str(self.df['date'][i])[5:7]) for i in range(len(self.df))]
        # Convert Date column to datetime
        self.df.loc[:, 'date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d')
        # Sort by datetime
        self.df.sort_values(by='date', inplace=True, ascending=True)

        return self.df
    
    def get_mape(self, y_true, y_pred):
        """
        Compute mean absolute percentage error (MAPE)
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def feature_engineer(self):

        self.df['range_hl'] = self.df['high'] - self.df['low']
        self.df['range_oc'] = self.df['open'] - self.df['close']

        lag_cols = ['close', 'range_hl', 'range_oc', 'volume']
        shift_range = [x + 1 for x in range(self.N)]

        for col in lag_cols:
            for i in shift_range:

                new_col='{}_lag_{}'.format(col, i)   # 格式化字符串
                self.df[new_col]=self.df[col].shift(i)

        return self.df[self.N:]

    def scale_row(self, row, feat_mean, feat_std):
        """
        Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
        Inputs
            row      : pandas series. Need to scale this.
            feat_mean: mean
            feat_std : standard deviation
        Outputs
            row_scaled : pandas series with same length as row, but scaled
        """
        # If feat_std = 0 (this happens if adj_close doesn't change over N days),
        # set it to a small number to avoid division by zero
        feat_std = 0.001 if feat_std == 0 else feat_std
        row_scaled = (row - feat_mean) / feat_std

        return row_scaled

    def get_mov_avg_std(self, col, df):
        """
        Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
        Inputs
            df         : dataframe. Can be of any length.
            col        : name of the column you want to calculate mean and std dev
            N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
        Outputs
            df_out     : same as df but with additional column containing mean and std dev
        """
        mean_list = df[col].rolling(window=self.N, min_periods=1).mean()  # len(mean_list) = len(df)
        std_list = df[col].rolling(window=self.N, min_periods=1).std()  # first value will be NaN, because normalized by N-1

        # Add one timestep to the predictions ,这里又shift了一步
        mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
        std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

        # Append mean_list to df
        df_out = df.copy()
        df_out[col + '_mean'] = mean_list
        df_out[col + '_std'] = std_list

        return df_out
        
class XGBEngine(XGBFuncModule):
     
    def __init__(self, data_df, test_size, N):
        from sklearn.preprocessing import StandardScaler
        self.data_df = data_df
        self.test_size = test_size
        self.N = N
        self.scaler = StandardScaler()
        self.Func = XGBFuncModule(self.data_df, self.N)
    
    def load_data(self):
        # 第一步：获取数据，整理数据
        self.Func.preprocess()
        # 第二步：特征工程
        df= self.Func.feature_engineer()
        # 第三步：数据标准化，先统一计算出标准化的数据，在对其进行数据切分。
        cols_list = [
            "close",
            "range_hl",
            "range_oc",
            "volume"
        ]
        for col in cols_list:
            df = self.Func.get_mov_avg_std(col, df)


        # 第四步：生成训练数据和测试数据。因训练数据和测试数据的标准化方式不同，因此需切分训练和测试数据。
        num_test = int(self.test_size * len(df))
        num_train = len(df) - num_test
        train = df[:num_train]
        test = df[num_train:]

        # 第五步：标签和特征的标准化，此步的目的是为了对在训练集不能代表总体的情况下，使树模型正确运行的一种取巧
        cols_to_scale = [
            "close"
        ]
        for i in range(1, self.N + 1):
            cols_to_scale.append("close_lag_" + str(i))
            cols_to_scale.append("range_hl_lag_" + str(i))
            cols_to_scale.append("range_oc_lag_" + str(i))
            cols_to_scale.append("volume_lag_" + str(i))

        # 启示三：标准化也不应带测试集，以避免信息泄漏
        train_scaled = self.scaler.fit_transform(train[cols_to_scale])
        # Convert the numpy array back into pandas dataframe
        train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
        train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]

        test_scaled = test[['date']]
        for col in cols_list:
            feat_list = [col + '_lag_' + str(shift) for shift in range(1, self.N + 1)]
            temp = test.apply(lambda row: self.Func.scale_row(row[feat_list], row[col + '_mean'], row[col + '_std']), axis=1)
            test_scaled = pd.concat([test_scaled, temp], axis=1)

        # 第六步：建立样本
        features = []
        for i in range(1, self.N + 1):
            features.append("close_lag_" + str(i))
            features.append("range_hl_lag_" + str(i))
            features.append("range_oc_lag_" + str(i))
            features.append("volume_lag_" + str(i))

        target = "close"

        X_train = train[features]
        X_train = train[target]
        X_sample = test[features]
        y_sample = test[target]
        X_train_scaled = train_scaled[features]
        y_train_scaled = train_scaled[target]
        X_sample_scaled = test_scaled[features]

        return features, train, test, X_train_scaled, y_train_scaled, X_sample_scaled, y_sample
    
    def XGBTrain(self, GridSearchCV_parameters, XGB_parameters, evaluation):

        features, train, test, X_train_scaled, y_train_scaled, X_sample_scaled, y_sample = self.load_data()

        # 第七步：开始训练
        from sklearn.model_selection import GridSearchCV
        from xgboost import XGBRegressor
        model=XGBRegressor(seed=XGB_parameters['seed'],
                            n_estimators=XGB_parameters['n_estimators'],
                            max_depth=XGB_parameters['max_depth'],
                            eval_metric=XGB_parameters['eval_metric'],
                            learning_rate=XGB_parameters['learning_rate'],
                            min_child_weight=XGB_parameters['min_child_weight'],
                            subsample=XGB_parameters['subsample'],
                            colsample_bytree=XGB_parameters['colsample_bytree'],
                            colsample_bylevel=XGB_parameters['colsample_bylevel'],
                            gamma=XGB_parameters['gamma'])
        gs=GridSearchCV(estimator= model,param_grid=GridSearchCV_parameters,cv=5,refit= True,scoring='neg_mean_squared_error')

        gs.fit(X_train_scaled,y_train_scaled)

        est_scaled = gs.predict(X_train_scaled)
        train['est'] = est_scaled * math.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]

        # 第八步：测试集
        pre_y_scaled = gs.predict(X_sample_scaled)
        test['pre_y_scaled'] = pre_y_scaled
        test['pre_y']=test['pre_y_scaled'] * test['close_std'] + test['close_mean']
        plot_y = pd.concat([self.data_df['close'][:self.N],train['est'],test['pre_y']])

        if evaluation:
            print('------------------------')
            print ('GridSearchCV最优参数: ' + str(gs.best_params_))
            imp = list(zip(train[features], gs.best_estimator_.feature_importances_))
            imp.sort(key=lambda tup: tup[1])
            print('------------------------')
            print('特征重要性：')
            for i in range(-1,-10,-1):
                print(imp[i])
            
            self.evaluation(plot_y, test, y_sample)

        return plot_y
    
    def evaluation(self, plot_y, test, y_sample):
        # 第九步：测试评估
        # 列举特征重要值和RMSE
        from sklearn.metrics import mean_squared_error
        rmse=math.sqrt(mean_squared_error(y_sample, test['pre_y']))
        print('------------------------')
        print("RMSE on dev set = %0.3f" % rmse)
        mape = self.Func.get_mape(y_sample, test['pre_y'])
        print("MAPE on dev set = %0.3f%%" % mape)
        print('------------------------')

        # 可视化
        plt.figure().set_figheight(10)
        plt.figure().set_figwidth(20)
        plt.axvline(self.data_df['date'][int(len(self.data_df)*(1-self.test_size))], c='r', linestyle='--')
        plt.plot(self.data_df['date'], self.data_df['close'], color = 'blue', linewidth=2, label = 'real')
        plt.plot(self.data_df['date'], plot_y, color='orange', linewidth=2, label = 'predict')
        plt.legend()
        plt.show()

class LPPLS(object):

    def __init__(self, observations):
        """
        Args:
            observations (np.array,pd.DataFrame): 2xM matrix with timestamp and observed value.
        """
        assert isinstance(observations, (np.ndarray, pd.DataFrame)), \
            f'Expected observations to be <pd.DataFrame> or <np.ndarray>, got :{type(observations)}'

        self.observations = observations
        self.coef_ = {}
        self.indicator_result = []
    
    from numba import njit
    @staticmethod
    @njit
    def lppls(t, tc, m, w, a, b, c1, c2):
        return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * np.log(tc - t))) + (c2 * np.sin(w * np.log(tc - t)))))

    def func_restricted(self, x, *args):
        """
        Finds the least square difference.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        Args:
            x(np.ndarray):  1-D array with shape (n,).
            args:           Tuple of the fixed parameters needed to completely specify the function.
        Returns:
            (float)
        """

        tc = x[0]
        m = x[1]
        w = x[2]
        observations = args[0]

        rM = self.matrix_equation(observations, tc, m, w)
        a, b, c1, c2 = rM[:, 0].tolist()
        # print('type', type(res))
        # print('func_restricted', res)

        delta = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in observations[0, :]]
        delta = np.subtract(delta, observations[1, :])
        delta = np.power(delta, 2)

        return np.sum(delta)

    @staticmethod
    @njit
    def matrix_equation(observations, tc, m, w):
        """
        Derive linear parameters in LPPLs from nonlinear ones.
        """
        T = observations[0]
        P = observations[1]
        N = len(T)

        # @TODO make taking tc - t or |tc - t| configurable
        dT = np.abs(tc - T)
        phase = np.log(dT)

        fi = np.power(dT, m)
        gi = fi * np.cos(w * phase)
        hi = fi * np.sin(w * phase)

        fi_pow_2 = np.power(fi, 2)
        gi_pow_2 = np.power(gi, 2)
        hi_pow_2 = np.power(hi, 2)

        figi = np.multiply(fi, gi)
        fihi = np.multiply(fi, hi)
        gihi = np.multiply(gi, hi)

        yi = P
        yifi = np.multiply(yi, fi)
        yigi = np.multiply(yi, gi)
        yihi = np.multiply(yi, hi)

        matrix_1 = np.array([
            [N,          np.sum(fi),       np.sum(gi),       np.sum(hi)],
            [np.sum(fi), np.sum(fi_pow_2), np.sum(figi),     np.sum(fihi)],
            [np.sum(gi), np.sum(figi),     np.sum(gi_pow_2), np.sum(gihi)],
            [np.sum(hi), np.sum(fihi),     np.sum(gihi),     np.sum(hi_pow_2)]
        ])

        matrix_2 = np.array([
            [np.sum(yi)],
            [np.sum(yifi)],
            [np.sum(yigi)],
            [np.sum(yihi)]
        ])

        return np.linalg.solve(matrix_1, matrix_2)

    def fit(self, max_searches, minimizer='Nelder-Mead', obs=None):
        """
        Args:
            max_searches (int): The maxi amount of searches to perform before giving up. The literature suggests 25.
            minimizer (str): See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            obs (Mx2 numpy array): the observed time-series data. Optional, if not included will use self.scaled_obs
        Returns:
            tc, m, w, a, b, c, c1, c2, O, D
        """
        if obs is None:
            obs = self.observations


        # print('obs',obs)
        search_count = 0
        # find bubble
        while search_count < max_searches:
            tc_init_min, tc_init_max = self._get_tc_bounds(obs, 0.50, 0.50)
            t1 = obs[0, 0]
            t2 = obs[0, -1]

            # @TODO make configurable
            # set random initialization limits for non-linear params
            init_limits = [
                (max(t2 - 60, t2 - 0.5 * (t2 - t1)), min(t2 + 252, t2 + 0.5 * (t2 - t1))),  # tc
                # (tc_init_min, tc_init_max),
                (0.0, 1.0),  # m
                (2.0, 15.0),  # ω
            ]

            # randomly choose vals within bounds for non-linear params
            non_lin_vals = [random.uniform(a[0], a[1]) for a in init_limits]

            tc = non_lin_vals[0]
            m = non_lin_vals[1]
            w = non_lin_vals[2]
            seed = np.array([tc, m, w])

            # Increment search count on SVD convergence error, but raise all other exceptions.
            try:
                tc, m, w, a, b, c, c1, c2 = self.estimate_params(obs, seed, minimizer)
                O = self.get_oscillations(w, tc, t1, t2)
                D = self.get_damping(m, w, b, c)
                return tc, m, w, a, b, c, c1, c2, O, D
            except Exception as e:
                # print(e)
                search_count += 1
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    def estimate_params(self, observations, seed, minimizer):
        """
        Args:
            observations (np.ndarray):  the observed time-series data.
            seed (list):  time-critical, omega, and m.
            minimizer (str):  See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        Returns:
            tc, m, w, a, b, c, c1, c2
        """
        from scipy.optimize import minimize

        cofs = minimize(
            args=observations,
            fun=self.func_restricted,
            x0=seed,
            method=minimizer
        )

        if cofs.success:
            tc = cofs.x[0]
            m = cofs.x[1]
            w = cofs.x[2]
            # r =
            # m_f =

            rM = self.matrix_equation(observations, tc, m, w)
            a, b, c1, c2 = rM[:, 0].tolist()

            c = self.get_c(c1, c2)

            # Use sklearn format for storing fit params
            # @TODO only save when running single fits.
            for coef in ['tc', 'm', 'w', 'a', 'b', 'c', 'c1', 'c2']:
                self.coef_[coef] = eval(coef)
            return tc, m, w, a, b, c, c1, c2
        else:
            raise UnboundLocalError

    def plot_fit(self, show_tc=False):
        """
        Args:
            observations (Mx2 numpy array): the observed data
        Returns:
            nothing, should plot the fit
        """
        tc, m, w, a, b, c, c1, c2 = self.coef_.values()
        time_ord = [pd.Timestamp.fromordinal(d) for d in self.observations[0, :].astype('int32')]
        t_obs = self.observations[0, :]
        # ts = pd.to_datetime(t_obs*10**9)
        # compatible_date = np.array(ts, dtype=np.datetime64)

        lppls_fit = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in t_obs]
        price = self.observations[1, :]

        first = t_obs[0]
        last = t_obs[-1]

        O = ((w / (2.0 * np.pi)) * np.log((tc - first) / (tc - last)))
        D = (m * np.abs(b)) / (w * np.abs(c))

        fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(14, 8))
        # fig.suptitle(
        #     'Single Fit\ntc: {:.2f}, m: {:.2f}, w: {:.2f}, a: {:.2f}, b: {:.2f}, c: {:.2f}, O: {:.2f}, D: {:.2f}'.format(tc, m, w, a, b, c, O, D),
        #     fontsize=16)

        ax1.plot(time_ord, price, label='price', color='black', linewidth=0.75)
        ax1.plot(time_ord, lppls_fit, label='lppls fit', color='blue', alpha=0.5)
        # if show_tc:
        #     ax1.axvline(x=np.array(tc_ts, dtype=np.datetime64), label='tc={}'.format(ts), color='red', alpha=0.5)
        # set grids
        ax1.grid(which='major', axis='both', linestyle='--')
        # set labels
        ax1.set_ylabel('ln(p)')
        ax1.legend(loc=2)

        plt.xticks(rotation=45)
        # ax1.xaxis.set_major_formatter(months)
        # # rotates and right aligns the x labels, and moves the bottom of the
        # # axes up to make room for them
        # fig.autofmt_xdate()

    def compute_indicators(self, res, filter_conditions_config=None):
        pos_lst = []
        neg_lst = []
        pos_conf_lst = []
        neg_conf_lst = []
        price = []
        ts = []
        _fits = []

        if filter_conditions_config is None:
            # TODO make configurable again!
            m_min, m_max = (0.0, 1.0)
            w_min, w_max = (2.0, 15.0)
            O_min = 2.5
            D_min = 0.5
        else:
            # TODO parse user provided conditions
            pass

        for r in res:
            ts.append(r['t2'])
            price.append(r['p2'])
            pos_qual_count = 0
            neg_qual_count = 0
            pos_count = 0
            neg_count = 0
            # _fits.append(r['res'])

            for idx, fits in enumerate(r['res']):
                t1 = fits['t1']
                t2 = fits['t2']
                tc = fits['tc']
                m = fits['m']
                w = fits['w']
                b = fits['b']
                c = fits['c']
                O = fits['O']
                D = fits['D']

                # t_delta = t2 - t1
                # pct_delta_min = t_delta * 0.5
                # pct_delta_max = t_delta * 0.5
                # tc_min = t2 - pct_delta_min
                # tc_max = t2 + pct_delta_max

                # [max(t2 - 60, t2 - 0.5 * (t2 - t1)), min(252, t2 + 0.5 * (t2 - t1))]

                # print('lb: max({}, {})={}'.format(t2 - 60, t2 - 0.5 * (t2 - t1), max(t2 - 60, t2 - 0.5 * (t2 - t1))))
                # print('ub: min({}, {})={}'.format(t2 + 252, t2 + 0.5 * (t2 - t1), min(t2 + 252, t2 + 0.5 * (t2 - t1))))
                #
                # print('{} < {} < {}'.format(max(t2 - 60, t2 - 0.5 * (t2 - t1)), tc, min(t2 + 252, t2 + 0.5 * (t2 - t1))))
                # print('______________')

                tc_in_range = max(t2 - 60, t2 - 0.5 * (t2 - t1)) < tc < min(t2 + 252, t2 + 0.5 * (t2 - t1))
                m_in_range = m_min < m < m_max
                w_in_range = w_min < w < w_max

                if b != 0 and c != 0:
                    O = O
                else:
                    O = np.inf

                O_in_range = O > O_min
                D_in_range = D > D_min  # if m > 0 and w > 0 else False

                if tc_in_range and m_in_range and w_in_range and O_in_range and D_in_range:
                    is_qualified = True
                else:
                    is_qualified = False

                if b < 0:
                    pos_count += 1
                    if is_qualified:
                        pos_qual_count += 1
                if b > 0:
                    neg_count += 1
                    if is_qualified:
                        neg_qual_count += 1
                # add this to res to make life easier
                r['res'][idx]['is_qualified'] = is_qualified

            _fits.append(r['res'])

            pos_conf = pos_qual_count / pos_count if pos_count > 0 else 0
            neg_conf = neg_qual_count / neg_count if neg_count > 0 else 0
            pos_conf_lst.append(pos_conf)
            neg_conf_lst.append(neg_conf)

            # pos_lst.append(pos_count / (pos_count + neg_count))
            # neg_lst.append(neg_count / (pos_count + neg_count))

            # tc_lst.append(tc_cnt)
            # m_lst.append(m_cnt)
            # w_lst.append(w_cnt)
            # O_lst.append(O_cnt)
            # D_lst.append(D_cnt)

        res_df = pd.DataFrame({
            'time': ts,
            'price': price,
            'pos_conf': pos_conf_lst,
            'neg_conf': neg_conf_lst,
            '_fits': _fits,
        })
        return res_df
        # return ts, price, pos_lst, neg_lst, pos_conf_lst, neg_conf_lst, #tc_lst, m_lst, w_lst, O_lst, D_lst

    def plot_confidence_indicators(self, res_df):
        """
        Args:
            res (list): result from mp_compute_indicator
            condition_name (str): the name you assigned to the filter condition in your config
            title (str): super title for both subplots
        Returns:
            nothing, should plot the indicator
        """

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18, 10))

        ord = res_df['time'].astype('int32')
        ts = [pd.Timestamp.fromordinal(d) for d in ord]

        # plot pos bubbles
        ax1_0 = ax1.twinx()
        ax1.plot(ts, res_df['price'], color='black', linewidth=0.75)
        # ax1_0.plot(compatible_date, pos_lst, label='pos bubbles', color='gray', alpha=0.5)
        ax1_0.plot(ts, res_df['pos_conf'], label='bubble indicator (pos)', color='red', alpha=0.5)

        # plot neg bubbles
        ax2_0 = ax2.twinx()
        ax2.plot(ts, res_df['price'], color='black', linewidth=0.75)
        # ax2_0.plot(compatible_date, neg_lst, label='neg bubbles', color='gray', alpha=0.5)
        ax2_0.plot(ts, res_df['neg_conf'], label='bubble indicator (neg)', color='green', alpha=0.5)

        # if debug:
        #     ax3.plot(ts, tc_lst, label='tc count')
        #     ax3.plot(ts, m_lst, label='m count')
        #     ax3.plot(ts, w_lst, label='w count')
        #     ax3.plot(ts, O_lst, label='O count')
        #     ax3.plot(ts, D_lst, label='D count')

        # set grids
        ax1.grid(which='major', axis='both', linestyle='--')
        ax2.grid(which='major', axis='both', linestyle='--')

        # set labels
        ax1.set_ylabel('ln(p)')
        ax2.set_ylabel('ln(p)')

        ax1_0.set_ylabel('bubble indicator (pos)')
        ax2_0.set_ylabel('bubble indicator (neg)')

        ax1_0.legend(loc=2)
        ax2_0.legend(loc=2)

        plt.xticks(rotation=45)
        # format the ticks
        # ax1.xaxis.set_major_locator(years)
        # ax2.xaxis.set_major_locator(years)
        # ax1.xaxis.set_major_formatter(years_fmt)
        # ax2.xaxis.set_major_formatter(years_fmt)
        # ax1.xaxis.set_minor_locator(months)
        # ax2.xaxis.set_minor_locator(months)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()
        return res_df

    def mp_compute_nested_fits(self, workers, window_size=80, smallest_window_size=20, outer_increment=5, inner_increment=2, max_searches=25, filter_conditions_config={}):
        from multiprocessing import Pool
        from tqdm import tqdm

        obs_copy = self.observations
        obs_opy_len = len(obs_copy[0]) - window_size
        func = self._func_compute_nested_fits

        # print('obs_copy', obs_copy)
        # print('obs_opy_len', obs_opy_len)

        func_arg_map = [(
            obs_copy[:, i:window_size + i],
            window_size,
            i,
            smallest_window_size,
            outer_increment,
            inner_increment,
            max_searches,
        ) for i in range(0, obs_opy_len+1, outer_increment)]

        with Pool(processes=workers) as pool:
            self.indicator_result = list(tqdm(pool.imap(func, func_arg_map), total=len(func_arg_map)))

        return self.indicator_result

    def compute_nested_fits(self, window_size=80, smallest_window_size=20, outer_increment=5, inner_increment=2,
                            max_searches=25):
        import xarray as xr
        obs_copy = self.observations
        obs_copy_len = len(obs_copy[0]) - window_size
        window_delta = window_size - smallest_window_size
        res = []
        i_idx = 0
        for i in range(0, obs_copy_len + 1, outer_increment):
            j_idx = 0
            obs = obs_copy[:, i:window_size + i]
            t1 = obs[0][0]
            t2 = obs[0][-1]
            res.append([])
            i_idx += 1
            for j in range(0, window_delta, inner_increment):
                obs_shrinking_slice = obs[:, j:window_size]
                tc, m, w, a, b, c, c1, c2, O, D = self.fit(max_searches, obs=obs_shrinking_slice)
                res[i_idx-1].append([])
                j_idx += 1
                for k in [t2, t1, a, b, c, m, 0, tc]:
                    res[i_idx-1][j_idx-1].append(k)
        return xr.DataArray(
            data=res,
            dims=('t2', 'windowsizes', 'params'),
            coords=dict(
                        t2=obs_copy[0][(window_size-1):],
                        windowsizes=range(smallest_window_size, window_size, inner_increment),
                        params=['t2', 't1', 'a', 'b', 'c', 'm', '0', 'tc'],
                        )
        )

    def _func_compute_nested_fits(self, args):

        obs, window_size, n_iter, smallest_window_size, outer_increment, inner_increment, max_searches = args

        window_delta = window_size - smallest_window_size

        res = []

        # print('obs', obs)
        t1 = obs[0][0]
        t2 = obs[0][-1]
        p1 = obs[1][0]
        p2 = obs[1][-1]

        # if self.scale_obs:
        #     t1 = self.inverse_transform_observations([[t1, p1]])[0, 0]
        #     t2 = self.inverse_transform_observations([[t2, p2]])[0, 0]
        #     p1 = self.inverse_transform_observations([[t1, p1]])[0, 1]
        #     p2 = self.inverse_transform_observations([[t2, p2]])[0, 1]

                    # tc_init_min, tc_init_max = self._get_tc_bounds(obs_shrinking_slice, tc_min, tc_max)
                    #
                    # tc_in_range = last - tc_init_min < tc < last + tc_init_max
                    # m_in_range = m_min < m < m_max
                    # w_in_range = w_min < w < w_max
                    # O_in_range = self._is_O_in_range(tc, w, last, O_min)
                    # D_in_range = self._is_D_in_range(m, w, b, c, D_min)
                    #
                    # qualified[value] = tc_in_range and m_in_range and w_in_range and O_in_range and D_in_range

        # run n fits on the observation slice.
        for j in range(0, window_delta, inner_increment):
            obs_shrinking_slice = obs[:, j:window_size]

            # fit the model to the data and get back the params
            if self.__class__.__name__ == 'LPPLSCMAES':
                # print('cmaes fit is running!')
                tc, m, w, a, b, c, c1, c2, O, D = self.fit(max_iteration=2500, pop_size=4, obs=obs_shrinking_slice)
            else:
                tc, m, w, a, b, c, c1, c2, O, D = self.fit(max_searches, obs=obs_shrinking_slice)

            nested_t1 = obs_shrinking_slice[0][0]
            nested_t2 = obs_shrinking_slice[0][-1]
            nested_p1 = obs_shrinking_slice[1][0]
            nested_p2 = obs_shrinking_slice[1][-1]

            # TODO consider rescaling data to be ∈ [0, 1] for perf?
            # if self.scale_obs:
            #     sub_t1 = self.inverse_transform_observations([[sub_t1, sub_p1]])[0, 0]
            #     sub_t2 = self.inverse_transform_observations([[sub_t2, sub_p2]])[0, 0]
            #     tc = self.inverse_transform_observations([[tc, 0]])[0, 0]

            res.append({
                'tc_d': self.ordinal_to_date(tc),
                'tc': tc,
                'm': m,
                'w': w,
                'a': a,
                'b': b,
                'c': c,
                'c1': c1,
                'c2': c2,
                't1_d': self.ordinal_to_date(nested_t1),
                't2_d': self.ordinal_to_date(nested_t2),
                't1': nested_t1,
                't2': nested_t2,
                'O': O,
                'D': D,
            })

        # return {'t1': self.ordinal_to_date(t1), 't2': self.ordinal_to_date(t2), 'p2': p2, 'res': res}
        return {'t1': t1, 't2': t2, 'p2': p2, 'res': res}

    def _get_tc_bounds(self, obs, lower_bound_pct, upper_bound_pct):
        """
        Args:
            obs (Mx2 numpy array): the observed data
            lower_bound_pct (float): percent of (t_2 - t_1) to use as the LOWER bound initial value for the optimization
            upper_bound_pct (float): percent of (t_2 - t_1) to use as the UPPER bound initial value for the optimization
        Returns:
            tc_init_min, tc_init_max
        """
        t_first = obs[0][0]
        t_last = obs[0][-1]
        t_delta = t_last - t_first
        pct_delta_min = t_delta * lower_bound_pct
        pct_delta_max = t_delta * upper_bound_pct
        tc_init_min = t_last - pct_delta_min
        tc_init_max = t_last + pct_delta_max
        return tc_init_min, tc_init_max

    def _is_O_in_range(self, tc, w, last, O_min):
        return ((w / (2 * np.pi)) * np.log(abs(tc / (tc - last)))) > O_min

    def _is_D_in_range(self, m, w, b, c, D_min):
        return False if m <= 0 or w <= 0 else abs((m * b) / (w * c)) > D_min

    def get_oscillations(self, w, tc, t1, t2):
        return ((w / (2.0 * np.pi)) * np.log((tc - t1) / (tc - t2)))

    def get_damping(self, m, w, b, c):
        return (m * np.abs(b)) / (w * np.abs(c))

    def get_c(self, c1, c2):
        if c1 and c2:
            # c = (c1 ** 2 + c2 ** 2) ** 0.5
            return c1 / np.cos(np.arctan(c2 / c1))
        else:
            return 0
    
    def ordinal_to_date(self, ordinal):
        # Since pandas represents timestamps in nanosecond resolution,
        # the time span that can be represented using a 64-bit integer
        # is limited to approximately 584 years
        try:
            return date.fromordinal(int(ordinal)).strftime('%Y-%m-%d')
        except (ValueError, OutOfBoundsDatetime):
            return str(pd.NaT)
