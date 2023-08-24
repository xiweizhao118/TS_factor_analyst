import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


class BacktestingEngine:
    def __init__(self, data, factor):
        self.data = data
        self.portfolio = {'cash': 1000000, 'positions': {}, 'total': [1000000]}
        self.marketvalue = [1000000]
        self.trades = []
        self.factor = factor

    def run_backtest(self):
        # all in at per trade
        risk_per_trade = 1  # Risk 2% of portfolio per trade

        for index, row in self.data.iterrows():
            # Implement your trading logic for each data point
            close_price = row['close']
            signal = row['signal']

            if signal == -1:
                # Buy signal
                available_cash = self.portfolio['cash']
                position_size = available_cash * risk_per_trade / close_price
                self.execute_trade(symbol=self.data.name,
                                   quantity=position_size, price=close_price)

                positions_value = sum(self.portfolio['positions'].get(
                    symbol, 0) * close_price for symbol in self.portfolio['positions'])
                self.portfolio['total'].append(
                    self.portfolio['cash'] + positions_value)
                self.marketvalue.append(
                    self.portfolio['cash'] + positions_value)
            elif signal == 1:
                # Sell signal
                position_size = self.portfolio['positions'].get(
                    self.data.name, 0)
                if position_size > 0:
                    self.execute_trade(
                        symbol=self.data.name, quantity=-position_size, price=close_price)
                    positions_value = sum(self.portfolio['positions'].get(
                        symbol, 0) * close_price for symbol in self.portfolio['positions'])
                    self.portfolio['total'].append(
                        self.portfolio['cash'] + positions_value)
                    self.marketvalue.append(
                        self.portfolio['cash'] + positions_value)
            else:
                # No trade
                value = self.portfolio['total'][-1]
                self.portfolio['total'].append(value)
                position_size = self.portfolio['positions'].get(
                    self.data.name, 0)
                if position_size > 0:
                    # when cash equals 0, update the market value based on the latest close price
                    positions_value = sum(self.portfolio['positions'].get(
                        symbol, 0) * close_price for symbol in self.portfolio['positions'])
                    self.marketvalue.append(
                        self.portfolio['cash'] + positions_value)
                else:
                    # when that is all of the cash, the market value remains still
                    self.marketvalue.append(value)

    def calculate_performance(self):
        # Calculate performance metrics based on executed trades
        trade_prices = np.array([trade['price'] for trade in self.trades])
        trade_quantities = np.array([trade['quantity']
                                    for trade in self.trades])

        trade_returns = np.diff(trade_prices) / trade_prices[:-1]
        trade_pnl = trade_returns * trade_quantities[:-1]

        total_pnl = np.sum(trade_pnl)
        average_trade_return = np.mean(trade_returns)
        win_ratio = np.sum(trade_pnl > 0) / len(trade_pnl)

        return total_pnl, average_trade_return, win_ratio

    def execute_trade(self, symbol, quantity, price):
        # Update portfolio and execute trade
        self.portfolio['cash'] -= quantity * price

        if symbol in self.portfolio['positions']:
            self.portfolio['positions'][symbol] += quantity
        else:
            self.portfolio['positions'][symbol] = quantity

        self.trades.append(
            {'symbol': symbol, 'quantity': quantity, 'price': price})

    def get_portfolio_value(self):
        # Portfolio = Cash + Price * Close_daily
        portfolio_value = self.portfolio['total']

        return portfolio_value

    def get_portfolio_returns(self):
        # Calculate the daily portfolio returns
        portfolio_value = self.get_portfolio_value()
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        return returns

    def print_portfolio_summary(self):
        print('--- Portfolio Summary ---')
        print('Cash:', self.portfolio['cash'])
        print('Positions:')
        for symbol, quantity in self.portfolio['positions'].items():
            print(symbol + ':', quantity)
        # print('Trades:')
        # for i in self.trades:
        #     print(i)

    def plot_portfolio_value(self, train_size=None):

        portfolio_value = self.portfolio['total']
        dates = self.data['date']
        market_value = self.marketvalue

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot portfolio value
        if len(dates) != len(portfolio_value):
            portfolio_value = portfolio_value[1:]
            market_value = market_value[1:]
        ax1.plot(dates, portfolio_value, 'royalblue', label='Portfolio Value')
        ax1.plot(dates, market_value, 'lightcoral', label='Market Value')
        ax1.set_xlabel('Date')
        ax1.set_title('Portfolio Value and Market Value Over Time')
        ax1.grid()
        if train_size:
            plt.axvline(x=dates[int(len(dates)*train_size)], c='orange', linestyle='--')

        fig.tight_layout()
        plt.legend()
        plt.show()


class SetTradingSignalEngine:
    def __init__(self, factor, bound):
        self.date = factor.date.unique()
        self.factor = factor.set_index('date')
        self.bound = self.cal_bound(
            lb=bound.iloc[:,0], ub=bound.iloc[:,1]).set_index('date')

    def check_trend(self, date):
        # Check factor tendency, last 2 days
        hist = self.factor[self.factor.index ==
                           date-datetime.timedelta(days=2)].values
        cur = self.factor[self.factor.index ==
                          date-datetime.timedelta(days=1)].values
        if hist < cur:
            tend = True
        else:
            tend = False

        return tend

    def cal_bound(self, ub, lb):
        # Calculate signal trigger points, dependent on factors' self distribution

        bound = pd.DataFrame({'date': self.date,
                              'upper_bound': ub,
                              'lower_bound': lb})

        bound.set_index('date')

        return bound

    def set_signal_init(self, upperbound):
        # Calculate initial signals:
        # Here is the setting:
        # When factor value touches the upper bound with an increasing trend, it is a 'buy' signal.
        # When factor value touches the lower bound with an decreasing trend, it is a 'sell' signal.

        signal = [0] * 2
        ub = self.bound['upper_bound']
        lb = self.bound['lower_bound']

        for i in self.date[2:]:
            tend = self.check_trend(i)
            ub_abs = self.factor[self.factor.index == i-datetime.timedelta(
                days=1)]['factor'].values - ub[ub.index == i].values
            lb_abs = self.factor[self.factor.index == i-datetime.timedelta(
                days=1)]['factor'].values - lb[lb.index == i].values
            if upperbound == 'sell':
                if (tend & (ub_abs >= 0)):
                    tmp = 1
                elif ((not tend) & (lb_abs <= 0)):
                    tmp = -1
                else:
                    tmp = 0
            elif upperbound == 'buy':
                if (tend & (ub_abs >= 0)):
                    tmp = -1
                elif ((not tend) & (lb_abs <= 0)):
                    tmp = 1
                else:
                    tmp = 0
            else:
                print('upperbound error: not exist.')
            signal.append(tmp)

        # check if the first trading signal is buy, or change it to zero
        if signal[0] == 1:
            signal[0] = 0
        
        return signal

    def set_signal(self, upperbound='sell'):
        # Clean the initial signals based on full-position trading strategy.
        signal = self.set_signal_init(upperbound)
        tag = signal[np.nonzero(signal)[0][0]]
        for i in range(np.nonzero(signal)[0][0]+1, len(signal)):
            if signal[i] == tag:
                signal[i] = 0
            elif signal[i] == -tag:
                tag = signal[i]

        return signal

    def plot_signal(self, signal, train_size=None):

        factor = self.factor['factor']
        ub = self.bound['upper_bound']
        lb = self.bound['lower_bound']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot factors value
        ax1.plot(self.date, factor, 'royalblue', label='factor')
        ax1.plot(self.date, ub, 'lightcoral', label='upper bound')
        ax1.plot(self.date, lb, 'lightcoral', label='lower bound')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('factor Value')
        ax1.set_title('factor Value Over Time')
        if train_size:
            plt.axvline(x=self.date[int(len(self.date)*train_size)], c='orange', linestyle='--')

        # Plot buy/sell signals
        ax2 = ax1.twinx()
        ax2.plot(self.date, signal, 'darkgrey', label='Buy/Sell Signal')
        ax2.grid(None)

        fig.tight_layout()
        plt.show()

def cal_bound(factor, switch='constant', train_size=0.8):

    f = factor['factor']

    if switch == 'rolling':
        lb = f.rolling(window=50,min_periods=0).min() + 1.2 * f.rolling(window=50,min_periods=0).std()
        ub = f.rolling(window=50,min_periods=0).max() - 1.2* f.rolling(window=50,min_periods=0).std()
    elif switch == 'constant':
        lb = f[:int(len(f)*train_size)].mean() - 1.5 * f[:int(len(f)*train_size)].std()
        ub = f[:int(len(f)*train_size)].mean() + 1.5 * f[:int(len(f)*train_size)].std()
        lb = [lb] * len(f)
        ub = [ub] * len(f)
    else:
        print('Switch error: not exist.')
    
    bound = pd.DataFrame({'lb': lb,
                          'ub': ub})
    
    return bound


def backtest_main(data, factor, bound, upperbound = 'sell', plot=False, train_size=None):

    # Define Trading Strategies
    signalengine = SetTradingSignalEngine(factor, bound)
    signal = signalengine.set_signal(upperbound)
    data['signal'] = [0] * (len(data)-len(factor)) + signal
    if plot:
        signalengine.plot_signal(pd.Series(signal), train_size)
    # Set the name attribute for the data DataFrame
    data.name = data['symbol'][0]

    # Start back testing
    engine = BacktestingEngine(data, factor)
    initial_portfolio_value = engine.get_portfolio_value()[0]
    engine.run_backtest()

    # Evaluate Performance
    final_portfolio_value = engine.get_portfolio_value()[-1]
    returns = engine.get_portfolio_returns()
    total_returns = (final_portfolio_value -
                     initial_portfolio_value) / initial_portfolio_value
    # Assuming 252 trading days in a year
    annualized_returns = (1 + total_returns) ** (252 / len(data)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    if volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (annualized_returns - 0.02) / \
            volatility  # Assuming risk-free rate of 100%

    # Calculate and print performance metrics
    total_pnl, average_trade_return, win_ratio = engine.calculate_performance()

    # Visualize Results
    if plot:
        engine.plot_portfolio_value(train_size)
        engine.print_portfolio_summary()
        print('--- Performance Metrics ---')
        print(f'Total Returns: {total_returns:.2%}')
        print(f'Annualized Returns: {annualized_returns:.2%}')
        print(f'Volatility: {volatility:.2%}')
        print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
        print(f'Total P&L: {total_pnl:.2f}')
        print(f'Average Trade Return: {average_trade_return:.2%}')
        print(f'Win Ratio: {win_ratio:.2%}')
    
    return annualized_returns
