# -*- coding: utf-8 -*-
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append('./db')
sys.path.append('./utility')
sys.path.append('./setting')
sys.path.append('./datatype')
sys.path.append('./view')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from pylab import rcParams
rcParams['figure.figsize'] = 10, 6

from TimeseriesGraph import TimeseriesGraph, MDHM

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

from PriceDatabase import PriceDatabase
from Timeframe import Timeframe
from TimeUtility import TimeUtility
from Timeseries import Timeseries, DATA_TYPE_ARRAYS, DATA_TYPE_XM, OHLCV, TIME, OPEN, LOW, HIGH, CLOSE, VOLUME
import Setting

class TSA(object):

    def __init__(self, data):
        self.data = data
        self.time = data[TIME]
        self.length = len(self.time)


    def draw(self, timeseries:Timeseries):
        close = timeseries.data(CLOSE)
        mean = ARIMA.mean(close, 60)
        std = ARIMA.stdev(close, 60)
        adft = ARIMA.adft(close)
        print(adft)
        lags = adft['number of lags']
        result = seasonal_decompose(close, model='mutiplicative', freq=lags)
        dic = {'std': std, 'seasonal': result.seasonal, 'residual': result.resid}
        self.drawTimeseries({'close':close, 'mean':mean, 'trend': result.trend}, dic, adft)

    def evaluate(self, train:Timeseries, test:Timeseries):
        d = train.data(CLOSE)
        fig, ax = TimeseriesGraph.makeFig(1, 1, (15, 5))
        graph = TimeseriesGraph(ax, train.time, MDHM)
        model = auto_arima(train.data(CLOSE), start_p=0, start_q=0,
                           test='adf',
                           max_p=3,
                           max_q=3,
                           m=1,
                           d=None,
                           seasonal=False,
                           start_P=0,
                           D=0,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
        model.plot_diagnostics(figsize=(15,8))
        plt.show()

    def learn(self, train:Timeseries, order):
        model = ARIMA(train.data[CLOSE], order=order)
        fitted = model.fit(disp=-1)
        print(fitted.summary())
        return fitted

    @classmethod
    def mean(cls, data, window):
        s = pd.Series(data)
        r = s.rolling(window)
        m = r.mean()
        return list(m)

    @classmethod
    def stdev(cls, data, window):
        s = pd.Series(data)
        r = s.rolling(window)
        m = r.std()
        return list(m)

    @classmethod
    def adft(cls, data):
        adf = adfuller(data, autolag='AIC')
        out = {}
        out['Test Statics'] = adf[0]
        out['p-value'] = adf[1]
        out['number of lags'] = adf[2]
        out['critical value'] = adf[4]
        return out

    def drawTimeseries(self, original, component, adft):
        m = len(original.keys())
        n = len(component.keys())
        heights = [5]
        for i in range(n):
            heights.append(2)
        fig, axes = TimeseriesGraph.gridFig(15, heights)
        graph0 = TimeseriesGraph(axes[0], self.time, MDHM)
        i = 0
        legend = []
        for key, value in original.items():
            graph0.plot(value, i, 1)
            legend.append({'color':i, 'label':key})
            i += 1
        graph0.drawLegend(legend, None)
        graph0.grid()
        text = 'p-value:' + str(adft['p-value']) + '  lag:' + str(adft['number of lags'])
        x = axes[0].get_xlim()
        y = axes[0].get_ylim()
        graph0.text((x[1] - x[0]) / 2 + x[0], y[1], text, 'black', 10)
        i = 1
        for key, value in component.items():
            graph = TimeseriesGraph(axes[i], self.time, MDHM)
            graph.plot(value, i, 1)
            graph.setTitle('', '', key)
            graph.grid()
            i += 1
        plt.show()


def stock_analysis():
    stock = 'US30Cash'
    timeframe = Timeframe('M1')
    db = PriceDatabase()
    t0 = TimeUtility.jstTime(2020, 10, 10, 21, 30)
    t1 = TimeUtility.jstTime(2020, 10, 17, 5, 30)
    dic, values = db.priceRange(stock, timeframe, t0, t1)
    ts = Timeseries(values, DATA_TYPE_XM, names=OHLCV)
    arima = ARIMA(dic)
    arima.draw(ts)

    time = ts.time
    close = ts.data(CLOSE)
    close_log = np.log(close)
    ts_log = Timeseries([time, close_log], DATA_TYPE_ARRAYS, names=[CLOSE])
    train, test = ts_log.split(0.9)
    arima.evaluate(train, test)


    #fit = arima.learn(train, (2, 1, 0))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stock_analysis()

