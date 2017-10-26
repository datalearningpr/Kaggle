

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn import preprocessing
from sklearn.model_selection import KFold


# read the train dataset and test dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
features_data = pd.read_csv("features.csv")
stores_data = pd.read_csv("stores.csv")

# use seasonal ARIMA in this game
from statsmodels.tsa.statespace.sarimax import SARIMAX


import time
start_time = time.time()

forecast_list = []
for i in test_data.Store.unique():
    for j in test_data.Dept.unique():
        print('{0}_{1}'.format(i, j))
        # get the time sereis of the data
        sales = train_data.ix[(train_data.Store == i) & (train_data.Dept == j),'Weekly_Sales']
        date_list = train_data.ix[(train_data.Store == i) & (train_data.Dept == j),'Date']
        sales.index = date_list.map(pd.to_datetime)

        # if the time series is long enough to conduct the forecast
        if len(sales) > 17:

            sales = sales.reindex(pd.date_range('2010-02-05', '2012-10-26', freq = 'W-FRI'))
            # fill in the missing data using interpolation
            sales.interpolate(method = 'linear', inplace = True)

            if sales.isnull().sum() > 0:
                sales.fillna(0, inplace = True)
                print('-----------------')
                print('{0}_{1} filled na'.format(i, j))
                print('-----------------')

            # due to the fact that sales data shall be seasonal, and we are dealing with weekly data
            # only 2 models we will try
            try:
                model = SARIMAX(sales, order = (1, 1, 1), seasonal_order = (0, 1, 0, 52)).fit(disp = 0)
            except:
                model = SARIMAX(sales, order = (1, 1, 0), seasonal_order = (0, 1, 0, 52)).fit(disp = 0)
            forecast = model.predict(start = len(sales), end = len(sales) + 38)
            temp = pd.DataFrame({'Id': forecast.index.map(lambda x: str(i) + '_' + str(j) + '_' + x.strftime('%Y-%m-%d')),
                                 'Weekly_Sales': list(forecast)})
        else:
            temp = pd.DataFrame({'Id': pd.date_range('2012-11-02', '2013-07-26', freq = 'W-FRI').map(lambda x: str(i) + '_' + str(j) + '_' + x.strftime('%Y-%m-%d')),
                                 'Weekly_Sales': [0] * 39})
        forecast_list.append(temp)
print("--- %s seconds ---" % (time.time() - start_time))

# save forecast to the file
pre_final = pd.concat(forecast_list)
pre_final.to_csv('pre_final.csv', index = 0)


sample = pd.read_csv("sampleSubmission.csv")

final = sample.merge(pre_final, left_on = ['Id'], right_on = ['Id'], how = 'left', suffixes=('', '_y'))
final["Weekly_Sales"] = final["Weekly_Sales_y"]
del final["Weekly_Sales_y"]
final.to_csv('Walmart Store Sales Forecasting.csv', index = 0)



