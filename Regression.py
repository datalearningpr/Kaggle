
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
from sklearn import datasets


boston = datasets.load_boston()
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,
                                                    random_state=100)

sc = preprocessing.StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)


# 1. LinearRegression, no need to GridSearch, just set n_jobs

from sklearn.linear_model import LinearRegression

model = LinearRegression(n_jobs = -1)
model.fit(x_train, np.log(y_train))

model.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), model.predict(x_train))
model.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), model.predict(sc.transform(x_test)))



# 2. Lasso

from sklearn.linear_model import Lasso

model = Lasso()
parameter_grid = {'alpha' : np.logspace(-4,0,10)}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))




# 3. Ridge

from sklearn.linear_model import Ridge

model = Ridge()
parameter_grid = {'alpha' : np.logspace(-4,0,10)}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))




# 4. ElasticNet

from sklearn.linear_model import ElasticNet

model = ElasticNet()
parameter_grid = {  'alpha': np.logspace(-4,0,10),
                    'l1_ratio' : [0.25, 0.5, 0.75]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))




# 5. LogisticRegression

from sklearn.linear_model import LogisticRegression

breast_cancer = datasets.load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,
                                                    random_state=100)

sc = preprocessing.StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)

model = LogisticRegression()
parameter_grid = {  'C': np.linspace(0.01, 1, 10)}
#[0.0001, 0.001, 0.01, 1, 10, 100, 1000]
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

grid_search.score(x_train, y_train)
grid_search.score(sc.transform(x_test), y_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, grid_search.predict(sc.transform(x_test)), 
                            target_names = breast_cancer.target_names))




# 6. SGDRegressor

from sklearn.linear_model import SGDRegressor

model = SGDRegressor(max_iter = 1000)
parameter_grid = {  'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': np.logspace(-4,0,10),
                    'l1_ratio' : [0.25, 0.5, 0.75]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))





# 7. KNNRegressor

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_jobs = -1)
parameter_grid = {  'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski'],
                    'p': [1, 2, 3, 10]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))





# 8. DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
parameter_grid = {  'criterion': ['mse', 'mae'],
                    'max_features': ['auto', 'sqrt'],
                    'max_depth': [3, 4, 5, 6, 7, 8]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))




# 9. RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_features = 'sqrt', n_jobs = -1)
parameter_grid = {  'n_estimators': [500],
                    'criterion': ['mse', 'mae'],
                    'min_samples_leaf': [1, 10, 50]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))





# 10. SVR

from sklearn.svm import SVR

model = SVR()
parameter_grid = {  'C': [0.001, 0.01, 1, 10, 100, 1000],
                    'kernel': ['linear', 'rbf', 'sigmoid'],
                    'gamma': np.logspace(-3, 2, 6)}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))





# 11. AdaBoost

from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(n_estimators = 300)
parameter_grid = {  'learning_rate': [0.9, 0.95, 1, 1.05, 1.1],
                    'loss': ['linear', 'square', 'exponential']}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))





# 12. Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators = 300)
parameter_grid = {  'learning_rate': [0.1],
                    'loss': ['ls', 'lad', 'huber', 'quantile'],
                    'max_depth': [3, 6, 8, 9],
                    'max_features': ['auto', 'sqrt', 'log2']}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))




# 13. xgboost

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators = 300, objective = 'reg:linear')
parameter_grid = {  'colsample_bytree': [0.4, 0.5 0.6],
                    'max_depth': [3, 6, 8, 9],
                    'learning_rate': [0.1]}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


grid_search.score(x_train, np.log(y_train))
metrics.r2_score(np.log(y_train), grid_search.predict(x_train))
grid_search.score(sc.transform(x_test), np.log(y_test))
metrics.r2_score(np.log(y_test), grid_search.predict(sc.transform(x_test)))


