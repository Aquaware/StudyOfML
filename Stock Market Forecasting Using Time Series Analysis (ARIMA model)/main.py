# -*- coding: utf-8 -*-
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append('./db')
sys.path.append('./utility')
sys.path.append('./setting')
sys.path.append('./model')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from pylab import rcParams
rcParams['figure.figsize'] = 10, 6

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

from PriceDatabase import PriceDatabase
from Timeframe import Timeframe
from TimeUtility import TimeUtility
from Timeseries import TIME, OPEN, LOW, HIGH, CLOSE, VOLUME
import Setting

class PredictWithARIMA(object):

    def __init__(self, data):
        self.data = data
        self.time = data[TIME]
        self.length = len(self.time)

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




    def drawTimeseries(self, data_list):
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')
        for data in data_list:
            plt.plot(range(self.length), data)
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stock = 'US30Cash'
    timeframe = Timeframe('M1')
    db = PriceDatabase()
    t0 = TimeUtility.jstTime(2020, 10, 1, 21, 30)
    t1 = TimeUtility.jstTime(2020, 10, 13, 5, 30)
    dic, values = db.priceRange(stock, timeframe, t0, t1)
    predict = PredictWithARIMA(dic)
    mean = PredictWithARIMA.mean(dic[CLOSE], 20)
    std = PredictWithARIMA.stdev(dic[CLOSE], 20)
    predict.drawTimeseries([dic[CLOSE], mean])