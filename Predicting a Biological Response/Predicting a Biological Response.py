

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation

# read the train dataset and test dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


feature_list = train_data.columns.tolist()
feature_list.remove("Activity")

sc = preprocessing.StandardScaler()

x_train = sc.fit_transform(train_data[feature_list])
y_train = train_data['Activity']
x_test = sc.transform(test_data[feature_list])


from sklearn.model_selection import KFold
def cross_validation_gen(k, model, x, y, x2):
    stacking_train = np.zeros(x.shape[0])
    stacking_test = np.zeros(x2.shape[0])

    kf = KFold(n_splits = k, shuffle = True, random_state = 20170101)
    for train, test in kf.split(x):     
        model.fit(x[train], y[train])
        stacking_train[test] = model.predict_proba(x[test])[:, 1]
        stacking_test = stacking_test + model.predict_proba(x2)[:, 1] / k 

    return stacking_train, stacking_test


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
parameter_grid = {  'n_estimators': 300,
                    'learning_rate': 0.05,
                    'loss': 'deviance', 
                    'max_depth': 6,
                    'max_features': 'sqrt'}

model.set_params(**parameter_grid)



import xgboost as xgb

model1 = xgb.XGBClassifier(n_estimators = 300, objective = 'binary:logistic')

parameter_grid1 = {  'reg_alpha': 0,
                    'reg_lambda': 0.001,
                    'max_depth': 3,
                    'learning_rate': 0.1}

model1.set_params(**parameter_grid1)



from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid2 = {  'n_estimators': 500,
                    'criterion': 'entropy',
                    'max_depth': 60}


model2.set_params(**parameter_grid2)


from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid3 = {  'n_estimators': 500,
                    'criterion': 'gini',
                    'max_depth': 60}


model3.set_params(**parameter_grid3)


from sklearn.ensemble import ExtraTreesClassifier

model4 = ExtraTreesClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid4 = {  'n_estimators': 500,
                    'criterion': 'entropy',
                    'max_depth': 60}
model4.set_params(**parameter_grid4)


from sklearn.ensemble import ExtraTreesClassifier

model5 = ExtraTreesClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid5 = {  'n_estimators': 500,
                    'criterion': 'gini',
                    'max_depth': 60}
model5.set_params(**parameter_grid5)


stack_train = []
stack_test =[]
model_list = [model, model1, model2, model3, model4, model5]

for m in model_list:
    print(m)
    a,b = cross_validation_gen(10, m, x_train,y_train, x_test)
    stack_train.append(a)
    stack_test.append(b)

final_train = np.vstack(stack_train).T
final_test = np.vstack(stack_test).T



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

parameter_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000]} 
grid_search = GridSearchCV(model, param_grid = parameter_grid, scoring = 'neg_log_loss', cv = 10, n_jobs = -1)
grid_search.fit(final_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

pre_final = grid_search.predict_proba(final_test)[:, 1]


submit_data = pd.read_csv("svm_benchmark.csv")
submit_data['PredictedProbability'] = pre_final
submit_data.to_csv('Predicting a Biological Response_final.csv', index = 0)







from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid = {  'n_estimators': [500],
                    'criterion': ['gini'],
                    'max_depth': [3]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, scoring = 'neg_log_loss', cv = 5, n_jobs = -1)
grid_search.fit(final_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))






