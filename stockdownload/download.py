import pandas as pd
import akshare as ak
import numpy as np
import yfinance as yf

def download_domestic_index_data(symbol, start_time, end_time):

    df = ak.index_zh_a_hist(symbol = symbol, period='daily', 
                                        start_date=start_time, end_date=end_time)
    
    # add the index code at the first
    duration = len(df)
    df['symbol'] = np.array([symbol]*duration)

    # translate
    df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "turnover","振幅": "amplitude", "涨跌幅": "quote change", "涨跌额": "change amount", "换手率": "turnover rate"}, inplace=True)

    df['date'] = pd.to_datetime(df['date'])

    return df

def download_domestic_industry_data(symbol, start_time, end_time):
    
    df = ak.stock_board_industry_hist_em(symbol = symbol, period='日k', 
                                        start_date=start_time, end_date=end_time)
    
    # add the index code at the first
    duration = len(df)
    df['symbol'] = np.array([symbol]*duration)

    # translate
    df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "turnover","振幅": "amplitude", "涨跌幅": "quote change", "涨跌额": "change amount", "换手率": "turnover rate"}, inplace=True)

    df['date'] = pd.to_datetime(df['date'])

    return df

def download_abroad_index_data(symbol, start_time, end_time):
    
    # Get data on this ticker
    df = yf.Ticker(symbol)

    # Get the historical prices for this ticker
    # Set the start date to Jan 1st, 2015
    df = df.history(start=start_time, end=end_time)
    
    # Restructure date format
    df['date'] = df.index
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    
    # translate
    df.rename(columns={"Open": "open", "Close": "close", "High": "high", "Low": "low", "Volume": "volume", "Dividends": "dividends","Stock Splits": "stock splits"}, inplace=True)

    # add the index code at the first
    duration = len(df)
    df['symbol'] = np.array([symbol]*duration)

    df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    
    return df
