
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
from sklearn.metrics import classification_report

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state=100)

sc = preprocessing.StandardScaler()
sc.fit(x_train)

mc = preprocessing.MinMaxScaler()
mc.fit(x_train)

x_train = mc.transform(x_train)

# 1. KNN

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_jobs = -1)
parameter_grid = {  'n_neighbors': [4, 5, 6, 7],
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski'],
                    'p': [1, 2, 3, 10]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(sc.transform(x_test)), 
                            target_names = iris.target_names))



# 2. Naive Bayes
    
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

model0 = GaussianNB()
model = MultinomialNB()
parameter_grid = {'alpha' : np.linspace(0.1, 1, 5)}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

model0.fit(x_train, y_train)

print(classification_report(y_test, grid_search.predict(mc.transform(x_test)), 
                            target_names = iris.target_names))

print(classification_report(y_test, model0.predict(sc.transform(x_test)), 
                            target_names = iris.target_names))


# 3. SVM

from sklearn.svm import SVC, LinearSVC, NuSVC

model = SVC()
parameter_grid = {'C': [0.001, 0.01, 1, 10, 100, 1000],
                    'kernel': ['linear', 'rbf', 'sigmoid'],
                    'gamma': np.logspace(-3, 2, 6)}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(mc.transform(x_test)), 
                            target_names = iris.target_names))

model = LinearSVC()
parameter_grid = {  'C': [0.001, 0.01, 1, 10, 100, 1000],
                    'loss': ['hinge', 'squared_hinge'],
                    'penalty': ['l2']}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(mc.transform(x_test)), 
                            target_names = iris.target_names))


model = NuSVC()
parameter_grid = {  'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'kernel': ['linear', 'rbf', 'sigmoid'],
                    'gamma': np.logspace(-3, 2, 6)}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(mc.transform(x_test)), 
                            target_names = iris.target_names))




# 4. Decision Tree

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=100)
parameter_grid = {  'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', None],
                    'max_depth': [2, 3, 4]}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(mc.transform(x_test)), 
                            target_names = iris.target_names))



################################################################################################
# below methods are frequetly used in the games
################################################################################################


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
parameter_grid = {  'C': np.linspace(0.01, 1, 10),
                    'dual': [True, False]}
#[0.0001, 0.001, 0.01, 1, 10, 100, 1000]
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

grid_search.score(x_train, y_train)
grid_search.score(sc.transform(x_test), y_test)

print(classification_report(y_test, grid_search.predict(sc.transform(x_test)), 
                            target_names = breast_cancer.target_names))




# 6. SGD

from sklearn.linear_model import SGDClassifier

model = SGDClassifier(max_iter = 1000)
parameter_grid = {  'loss': ['hinge', 'log', 'squared_hinge', 'perceptron'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': np.logspace(-4,0,10),
                    'l1_ratio' : [0.25, 0.5, 0.75]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(sc.transform(x_test)), 
                            target_names = iris.target_names))



# 7. RandomForest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid = {  'n_estimators': [500],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_leaf': [1, 10, 50]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(sc.transform(x_test)), 
                            target_names = iris.target_names))



# 8. AdaBoost

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators = 300)
parameter_grid = {  'learning_rate': [0.9, 0.95, 1, 1.05, 1.1]}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(sc.transform(x_test)), 
                            target_names = iris.target_names))





# 9. Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators = 300)
parameter_grid = {  'learning_rate': [0.1],
                    'loss': ['deviance'],  # 'exponential'
                    'max_depth': [3, 6, 8, 9],
                    'max_features': ['auto', 'sqrt', 'log2']}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(sc.transform(x_test)), 
                            target_names = iris.target_names))




# 10. xgboost

import xgboost as xgb

model = xgb.XGBClassifier(n_estimators = 300, objective = 'multi:softmax') # "binary:logistic"
parameter_grid = {  'colsample_bytree': [0.4, 0.5 0.6],
                    'max_depth': [3, 6, 8, 9],
                    'learning_rate': [0.1]}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 3, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

print(classification_report(y_test, grid_search.predict(sc.transform(x_test)), 
                            target_names = iris.target_names))




