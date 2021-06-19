# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 09:42:48 2021

@author: Ikuo
"""
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpl


def readData(filepath):
    df0 = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = df0.rename(columns={'Date':'date', 'Open':'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume':'volume'})
    series_data = {}
    times = df.index.to_pydatetime()
    series_data['date'] = list(times)
    for column in df.columns:
        series = list(df[column].values.tolist())
        series_data[column] = series
    return (df, series_data)

def plotGraph(filepath):
    (df, series_data) = readData(filepath)
    mpl.plot(df, type='candle')
    print(df)

    
if __name__ == '__main__':
    plotGraph('./stock_data/AAPL.csv')