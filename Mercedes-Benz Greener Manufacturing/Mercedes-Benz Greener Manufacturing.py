
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

from sklearn.cluster import MiniBatchKMeans
from math import radians, cos, sin, asin, sqrt  

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor


from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD



# read the train dataset and test dataset
# deal with NA and missing value if possible

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


from sklearn import preprocessing


# for categorical data, just use simple label encoding, can try one hot coding
le = preprocessing.LabelEncoder()
le.fit(np.hstack((train_data['X0'], test_data['X0'])))
train_data['X0'] = le.transform(train_data['X0'])
test_data['X0'] = le.transform(test_data['X0'])

le.fit(np.hstack((train_data['X1'], test_data['X1'])))
train_data['X1'] = le.transform(train_data['X1'])
test_data['X1'] = le.transform(test_data['X1'])

le.fit(np.hstack((train_data['X2'], test_data['X2'])))
train_data['X2'] = le.transform(train_data['X2'])
test_data['X2'] = le.transform(test_data['X2'])

le.fit(np.hstack((train_data['X3'], test_data['X3'])))
train_data['X3'] = le.transform(train_data['X3'])
test_data['X3'] = le.transform(test_data['X3'])

le.fit(np.hstack((train_data['X4'], test_data['X4'])))
train_data['X4'] = le.transform(train_data['X4'])
test_data['X4'] = le.transform(test_data['X4'])

le.fit(np.hstack((train_data['X5'], test_data['X5'])))
train_data['X5'] = le.transform(train_data['X5'])
test_data['X5'] = le.transform(test_data['X5'])

le.fit(np.hstack((train_data['X6'], test_data['X6'])))
train_data['X6'] = le.transform(train_data['X6'])
test_data['X6'] = le.transform(test_data['X6'])

le.fit(np.hstack((train_data['X8'], test_data['X8'])))
train_data['X8'] = le.transform(train_data['X8'])
test_data['X8'] = le.transform(test_data['X8'])


# interaction of features, can try more combinations
train_data['X314_315'] = preprocessing.PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(train_data[['X314', 'X315']])[:,2:]
train_data['X118_119'] = preprocessing.PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(train_data[['X118', 'X119']])[:,2:]
train_data['X47_48'] = preprocessing.PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(train_data[['X47', 'X48']])[:,2:]


test_data['X314_315'] = preprocessing.PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(test_data[['X314', 'X315']])[:,2:]
test_data['X118_119'] = preprocessing.PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(test_data[['X118', 'X119']])[:,2:]
test_data['X47_48'] = preprocessing.PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(test_data[['X47', 'X48']])[:,2:]




feature_list = list(train_data.columns)
feature_list.remove('ID')
feature_list.remove('y')



# different ways of dimension reduction, can provide more new features

random_state = 2018
feature_num = 12

# tSVD
tsvd = TruncatedSVD(n_components=feature_num, random_state=random_state)
tsvd_results_train = tsvd.fit_transform(train_data[feature_list])
tsvd_results_test = tsvd.transform(test_data[feature_list])

# PCA
pca = PCA(n_components=feature_num, random_state=random_state)
pca2_results_train = pca.fit_transform(train_data[feature_list])
pca2_results_test = pca.transform(test_data[feature_list])

# ICA
ica = FastICA(n_components=feature_num, random_state=random_state)
ica2_results_train = ica.fit_transform(train_data[feature_list])
ica2_results_test = ica.transform(test_data[feature_list])

# GRP
grp = GaussianRandomProjection(n_components=feature_num, random_state=random_state)
grp_results_train = grp.fit_transform(train_data[feature_list])
grp_results_test = grp.transform(test_data[feature_list])

# SRP
srp = SparseRandomProjection(n_components=feature_num, dense_output=True, random_state=random_state)
srp_results_train = srp.fit_transform(train_data[feature_list])
srp_results_test = srp.transform(test_data[feature_list])


final_feature_list = []

for i in range(1, feature_num + 1):
    train_data['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test_data['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train_data['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test_data['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train_data['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test_data['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train_data['grp_' + str(i)] = grp_results_train[:, i - 1]
    test_data['grp_' + str(i)] = grp_results_test[:, i - 1]

    train_data['srp_' + str(i)] = srp_results_train[:, i - 1]
    test_data['srp_' + str(i)] = srp_results_test[:, i - 1]

    final_feature_list.append('srp_' + str(i))
    



sc = preprocessing.StandardScaler()
x_train_more = sc.fit_transform(train_data[final_feature_list + feature_list])
x_test_more = sc.transform(test_data[final_feature_list + feature_list])
y_train = train_data['y']



sc = preprocessing.StandardScaler()
x_train = sc.fit_transform(train_data[feature_list])
x_test = sc.transform(test_data[feature_list])





from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators = 300)
parameter_grid = {  'learning_rate': [0.01],
                    'loss': ['huber'],
                    'max_depth': [3],
                    'max_features': ['auto']}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 4, n_jobs = -1, scoring='r2')
grid_search.fit(x_train_more, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



output1 = grid_search.predict(x_test_more)



import xgboost as xgb

model = xgb.XGBRegressor(n_estimators = 300, objective = 'reg:linear')
parameter_grid = {  'reg_alpha': [0],
                    'reg_lambda': [1],
                    'max_depth': [3],
                    'learning_rate': [0.03]}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 4, n_jobs = -1, scoring='r2')
grid_search.fit(x_train_more, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

output2 = grid_search.predict(x_test_more)



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

        parameter_grid = {  'learning_rate': [0.03],
                        'loss': ['huber'],
                        'max_depth': [3, 4, 5],
                        'max_features': ['sqrt', 'log2']}

        grid_search = GridSearchCV(self.stacking_model, param_grid = parameter_grid, cv = 4, n_jobs = -1, scoring = 'r2')
        grid_search.fit(stacking_train, y_train)

        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))

        output = grid_search.predict(stacking_test)
        return output


########################################################################

# use powerful models to do stacking so that we can get better result than just using one of them


parameter_xgb = {  'reg_alpha': 0,
                    'reg_lambda': 1,
                    'max_depth': 3,
                    'learning_rate': 0.03}

parameter_sgd = {  'loss': 'squared_epsilon_insensitive',
                    'penalty': 'l2',
                    'alpha': 0.002,
                    'l1_ratio' : 0.5}

parameter_gb = {  'learning_rate': 0.01,
                    'loss': 'huber',
                    'max_depth': 3,
                    'max_features': 'auto'}

parameter_rf = {  'n_estimators': 500,
                    'criterion': 'mse',
                    'max_depth': 11}

model = GradientBoostingRegressor(n_estimators = 300)
model1 = xgb.XGBRegressor(n_estimators = 300, objective = 'reg:linear')
model2 = SGDRegressor(n_iter = 1000)
model3 = RandomForestRegressor(max_features = 'sqrt')

model.set_params(**parameter_gb)
model1.set_params(**parameter_xgb)
model2.set_params(**parameter_sgd)
model3.set_params(**parameter_rf)

s_model = GradientBoostingRegressor(n_estimators = 300)
model_list = [model, model1, model3]

ss = SimpleStacking(model_list, s_model)
output = ss.fit_predict(5, x_train, x_test, np.array(y_train))



def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor

model = Sequential()

model.add(Dense(1024, input_dim=x_train_more.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse',
                optimizer='adam',
                metrics=[coeff_determination] 
                )

model.fit(x_train_more, y_train, batch_size=128, epochs=100)

output3 = np.squeeze(model.predict(x_test_more))


# final result is combination of 4 different models
final_output = output3 * 0.1 + output * 0.15 + output1 * 0.35 + output2 * 0.4

submit_data = pd.read_csv("sample_submission.csv")
submit_data['y'] = final_output
submit_data.to_csv("Mercedes-Benz_Greener_Manufacturing_final.csv", index = 0)

