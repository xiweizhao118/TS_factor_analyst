import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd


def plot(x, y, size, lb, train_length=None):
    plt.figure().set_figheight(size[0])
    plt.figure().set_figwidth(size[1])
    plt.plot(x, y, color='blue', linewidth=2, label=lb)
    plt.legend()
    plt.grid()
    if train_length:
        plt.axvline(x=x[train_length], c='r', linestyle='--')


def roughly_round_to_yearmonth(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index) - datetime.timedelta(days=10)
    return df.resample("1m").last()


def save_factor(path, name, factor):
    if not os.path.isdir(path):
        os.makedirs(path)
    factor.to_csv(f'{path}/{name}.csv')

def select_date(df, start, end):
    df = df[(df['date']>=start) & (df['date']<=end)]
    df = df.reset_index().iloc[:,1:]

    return df
