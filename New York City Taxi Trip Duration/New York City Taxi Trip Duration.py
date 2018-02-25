
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy.stats import boxcox

from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.linear_model import SGDRegressor
import xgboost as xgb

from sklearn.cluster import MiniBatchKMeans
from math import radians, cos, sin, asin, sqrt  

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor



# read the train dataset and test dataset
# deal with NA and missing value if possible

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# this is to get the smallest time of the pickup_datetime
# will be used later
min_time = min(pd.to_datetime(train_data['pickup_datetime']).min(), pd.to_datetime(test_data['pickup_datetime']).min())


# this is the formula to get the distance with lon and lat
def haversine(lon1, lat1, lon2, lat2):
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """  
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371  
    return c * r * 1000

# this is to get the manhattan distance
def manhattan(lon1, lat1, lon2, lat2):
    return haversine(lon1, lat1, lon1, lat2) + haversine(lon1, lat1, lon2, lat1)

# clustering will help to divide the location into different category
# so that we can have more feature
cluster_data = train_data.sample(500000, random_state = 2018)

# data size is too large, cannot use 1.5m, just use small sample of 50k
train_data = train_data.sample(50000, random_state = 2018)




# below is all about feature engineering, that is the main purpose of playing this game 


train_data['distance'] = train_data.apply(lambda x: haversine(x.pickup_longitude, x.pickup_latitude,
                                            x.dropoff_longitude, x.dropoff_latitude), axis=1)
test_data['distance'] = test_data.apply(lambda x: haversine(x.pickup_longitude, x.pickup_latitude,
                                            x.dropoff_longitude, x.dropoff_latitude), axis=1)

train_data['log_distance'] = np.log(train_data['distance']+1)
test_data['log_distance'] = np.log(test_data['distance']+1)


train_data['manhattan'] = train_data.apply(lambda x: manhattan(x.pickup_longitude, x.pickup_latitude,
                                            x.dropoff_longitude, x.dropoff_latitude), axis=1)
test_data['manhattan'] = test_data.apply(lambda x: manhattan(x.pickup_longitude, x.pickup_latitude,
                                            x.dropoff_longitude, x.dropoff_latitude), axis=1)

train_data['log_manhattan'] = np.log(train_data['manhattan']+1)
test_data['log_manhattan'] = np.log(test_data['manhattan']+1)

train_data['month'] = train_data.pickup_datetime.apply(lambda x: int(x[5:7]))
test_data['month'] = test_data.pickup_datetime.apply(lambda x: int(x[5:7]))

train_data['hour'] = train_data.pickup_datetime.apply(lambda x: int(x[11:13]))
test_data['hour'] = test_data.pickup_datetime.apply(lambda x: int(x[11:13]))

train_data['weekday'] = pd.to_datetime(train_data['pickup_datetime']).dt.weekday
test_data['weekday'] = pd.to_datetime(test_data['pickup_datetime']).dt.weekday

train_data['weekofyear'] = pd.to_datetime(train_data['pickup_datetime']).dt.weekofyear
test_data['weekofyear'] = pd.to_datetime(test_data['pickup_datetime']).dt.weekofyear

train_data['pickup_moment'] = (pd.to_datetime(train_data['pickup_datetime']) - min_time).dt.total_seconds()
test_data['pickup_moment'] = (pd.to_datetime(test_data['pickup_datetime']) - min_time).dt.total_seconds()



locations = np.vstack((cluster_data[['pickup_latitude', 'pickup_longitude']].values,
                    cluster_data[['dropoff_latitude', 'dropoff_longitude']].values,
                    test_data[['pickup_latitude', 'pickup_longitude']].values,
                    test_data[['dropoff_latitude', 'dropoff_longitude']].values))

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(locations)


train_data['pickup_cluster'] = kmeans.predict(train_data[['pickup_latitude', 'pickup_longitude']])
train_data['dropoff_cluster'] = kmeans.predict(train_data[['dropoff_latitude', 'dropoff_longitude']])

test_data['pickup_cluster'] = kmeans.predict(test_data[['pickup_latitude', 'pickup_longitude']])
test_data['dropoff_cluster'] = kmeans.predict(test_data[['dropoff_latitude', 'dropoff_longitude']])


train_data['busy_hour'] = train_data.pickup_datetime.apply(lambda x: 1 if int(x[11:13])>7 else 0)
test_data['busy_hour'] = test_data.pickup_datetime.apply(lambda x: 1 if int(x[11:13])>7 else 0)

train_data['busy_month'] = train_data.pickup_datetime.apply(lambda x: 1 if int(x[5:7])>2 else 0)
test_data['busy_month'] = test_data.pickup_datetime.apply(lambda x: 1 if int(x[5:7])>2 else 0)


train_data['vendor1'] = train_data.vendor_id.apply(lambda x: 1 if x ==1 else 0)
test_data['vendor1'] = test_data.vendor_id.apply(lambda x: 1 if x ==1 else 0)

train_data['store_and_fwd_flag_num'] = train_data.vendor_id.apply(lambda x: 1 if x =='Y' else 0)
test_data['store_and_fwd_flag_num'] = test_data.vendor_id.apply(lambda x: 1 if x =='Y' else 0)

train_data['pickup_datetime_round'] = pd.to_datetime(train_data['pickup_datetime']).dt.round('60min')
test_data['pickup_datetime_round'] = pd.to_datetime(test_data['pickup_datetime']).dt.round('60min')

a = train_data[['id', 'pickup_datetime_round', 'pickup_cluster', 'dropoff_cluster']]
b = test_data[['id', 'pickup_datetime_round', 'pickup_cluster', 'dropoff_cluster']]

df_all = pd.concat((a, b))

go_out = df_all.groupby(['pickup_datetime_round', 'pickup_cluster']).agg({'id': 'count'}).reset_index()
go_in = df_all.groupby(['pickup_datetime_round', 'dropoff_cluster']).agg({'id': 'count'}).reset_index()

train_data['dropoff_cluster_count'] = np.array(train_data[['pickup_datetime_round', 'dropoff_cluster']].merge(go_in, on=['pickup_datetime_round', 'dropoff_cluster'], how='left')['id'].fillna(0))
train_data['pickup_cluster_count'] = np.array(train_data[['pickup_datetime_round', 'pickup_cluster']].merge(go_out, on=['pickup_datetime_round', 'pickup_cluster'], how='left')['id'].fillna(0))

test_data['dropoff_cluster_count'] = np.array(test_data[['pickup_datetime_round', 'dropoff_cluster']].merge(go_in, on=['pickup_datetime_round', 'dropoff_cluster'], how='left')['id'].fillna(0))
test_data['pickup_cluster_count'] = np.array(test_data[['pickup_datetime_round', 'pickup_cluster']].merge(go_out, on=['pickup_datetime_round', 'pickup_cluster'], how='left')['id'].fillna(0))



feature_list = [
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude',
       'log_distance', 'log_manhattan', 'month',
       'hour', 'weekday', 'weekofyear', 'busy_hour', 'busy_month', 'vendor1',
       'pickup_cluster', 'dropoff_cluster','store_and_fwd_flag_num','pickup_moment',
       'dropoff_cluster_count', 'pickup_cluster_count']

sc = preprocessing.StandardScaler()
x_train = sc.fit_transform(train_data[feature_list])
x_test = sc.transform(test_data[feature_list])
y_train = train_data['trip_duration']


# finally, we use stacking to gain power of several models

class SimpleStacking():
    def __init__(self, models, stacking_model):
        self.models = models
        self.stacking_model = stacking_model

    def fit_predict(self, k, x_train, x_test, y_train, seed = 20161218):
        k_fold = KFold(k, shuffle = True, random_state = seed)

        stacking_train = np.zeros((x_train.shape[0], len(self.models)))
        stacking_test = np.zeros((x_test.shape[0], len(self.models)))
 
        for i, model in enumerate(self.models):

            stacking_test_i = np.zeros((x_test.shape[0], k))
        
            for j, index in enumerate(k_fold.split(x_train)):
                print("status: {}, {}".format(i, j))
                train_index = index[0]
                test_index = index[1]
                self.models[i].fit(x_train[train_index], y_train[train_index])
                stacking_train[test_index, i] = self.models[i].predict(x_train[test_index])
                stacking_test_i[:, j] = self.models[i].predict(x_test)

            stacking_test[:, i] = stacking_test_i.mean(1)

        parameter_grid = {  'learning_rate': [0.03, 0.05, 0.1],
                        'loss': ['huber'],
                        'max_depth': [3, 4, 5],
                        'max_features': ['sqrt', 'log2']}

        grid_search = GridSearchCV(self.stacking_model, param_grid = parameter_grid, cv = 4, n_jobs = -1, scoring = 'neg_mean_squared_error')
        grid_search.fit(stacking_train, y_train)

        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))

        output = grid_search.predict(stacking_test)
        return output


########################################################################

# use 4 powerful models to do stacking so that we can get better result than just using one of them


parameter_xgb = {  'reg_alpha': 0,
                    'reg_lambda': 1,
                    'max_depth': 7,
                    'learning_rate': 0.2}

parameter_sgd = {  'loss': 'squared_epsilon_insensitive',
                    'penalty': 'l2',
                    'alpha': 0.002,
                    'l1_ratio' : 0.5}

parameter_gb = {  'learning_rate': 0.1,
                    'loss': 'lad',
                    'max_depth': 7,
                    'max_features': 'auto'}

parameter_rf = {  'n_estimators': 500,
                    'criterion': 'mse',
                    'max_depth': 26}

model = GradientBoostingRegressor(n_estimators = 300)
model1 = xgb.XGBRegressor(n_estimators = 300, objective = 'reg:linear')
model2 = SGDRegressor(n_iter = 1000)
model3 = RandomForestRegressor(max_features = 'sqrt')

model.set_params(**parameter_gb)
model1.set_params(**parameter_xgb)
model2.set_params(**parameter_sgd)
model3.set_params(**parameter_rf)

s_model = GradientBoostingRegressor(n_estimators = 300)
model_list = [model, model1, model2, model3]

ss = SimpleStacking(model_list, s_model)
output = ss.fit_predict(5, x_train, x_test, np.log(np.array(y_train)))

final_output = np.exp(output)
submit_data = pd.read_csv("sample_submission.csv")
submit_data['trip_duration'] = final_output
submit_data.to_csv("Taxi_Trip_Duration.csv", index = 0)

