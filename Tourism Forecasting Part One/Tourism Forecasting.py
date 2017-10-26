

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm 
from statsmodels.tsa.arima_model import ARIMA

# this is the custom evaluation method used in this game
def MASE(train, test, forecast):
    MAE0 = mean_absolute_error(train[1:], train[:-1])
    MAE1 = mean_absolute_error(test, forecast)
    return MAE0 / MAE1


# read the train dataset and test dataset
train_data = pd.read_csv("tourism_data.csv", dtype = np.float64)
train_data.index = pd.date_range('2001-1-1', periods=43, freq='A')

result = []

# there are a few hundreds of time series to forecast, do it one by one
for i in range(len(train_data.columns)):
    print(i)

    ts = train_data.ix[:, i]
    ts = ts[ts.notnull()]

    # for each time series, use 1/3 of them to do the test, rest for the train
    to_test = len(ts) // 3
    train = ts[:-to_test]
    test = ts[-to_test:]


    # we only try a few types of ARIMA models 
    order_list = [(1, 0, 0)
                    , (1, 1, 0)
                    , (1, 1, 1)
                    , (0, 1, 1)
                    , (0, 0, 1)
                    ]

    try_models = []

    for j in order_list:
        try:
            m = ARIMA(train, j).fit(disp = 0)
            try_models.append(m)

        except:
            pass

    # to see whether all the models failed to fit the data
    if len(try_models) == 0:
        print("column {} has no model".format(i))
    

    # for all the model fitted, we use the evaluation to pick the best one
    error = float("inf")
    best = None

    for i, model in enumerate(try_models):

        f = model.predict(str(min(test.index).year), str(max(test.index).year))
    
        if sum(f.isnull()) > 0:
            pass
        else:
            temp = MASE(train, test, f)
            if temp < error:
                best = model

    # use the best one to forecast the test data
    chosen_model = ARIMA(ts, [best.k_ar, best.k_diff if hasattr(best, 'k_diff') else 0, best.k_ma]).fit(disp = 0)
    result.append(chosen_model.predict('2043', '2046'))

# after finished all the forecasting, save to file
final = pd.concat(result, axis = 1)
submit = pd.read_csv("example.csv", dtype = np.float64)
final.columns = submit.columns
final.to_csv("Tourism Forecast Part One.csv", index = 0)

































def objfunc(order, exog, endog):
    from statsmodels.tsa.arima_model import ARIMA
    fit = ARIMA(endog, order, exog).fit()
    return fit.aic()

from scipy.optimize import brute
grid = (slice(1, 0, 1), slice(0, 0, 1), slice(1, 0, 0))
brute(objfunc, grid, args=(exog, endog), finish=None)





res = sm.tsa.seasonal_decompose(ts[ts.notnull()]) 


# -*- coding:utf-8 -*-
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# �ƶ�ƽ��ͼ
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # ��size�����ݽ����ƶ�ƽ��
    rol_mean = timeSeries.rolling(window=size).mean()
    # ��size�����ݽ��м�Ȩ�ƶ�ƽ��
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rolmean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()

def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()

'''
����Unit Root Test
   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
   root, with the alternative that there is no unit root. That is to say the
   bigger the p-value the more reason we assert that there is a unit root
'''
def testStationarity(ts):
    dftest = adfuller(ts)
    # ������������õ�ֵ������������
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

# ����غ�ƫ���ͼ��Ĭ�Ͻ���Ϊ31��
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

