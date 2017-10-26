

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
import json
from sklearn.decomposition import TruncatedSVD, IncrementalPCA

# read the train dataset and test dataset
train_data = pd.read_table("train.tsv")
test_data = pd.read_table("test.tsv")

# to process the json field of the data
def process(x, field):
    j = json.loads(x)
    if field in j:
        return j[field]
    else:
        return " "

# get the json field of data, and it has 3 keys, get each one of them out

train_data['title'] = train_data["boilerplate"].apply(lambda x: process(x, "title")).fillna("")
train_data['body'] = train_data["boilerplate"].apply(lambda x: process(x, "body")).fillna("")
train_data['url'] = train_data["boilerplate"].apply(lambda x: process(x, "url")).fillna("")

test_data['title'] = test_data["boilerplate"].apply(lambda x: process(x, "title")).fillna("")
test_data['body'] = test_data["boilerplate"].apply(lambda x: process(x, "body")).fillna("")
test_data['url'] = test_data["boilerplate"].apply(lambda x: process(x, "url")).fillna("")

# and then join them together to get a new field

train_data["all_text"] = train_data['title'] + " " + train_data['body'] + " " + train_data['url']
test_data["all_text"] = test_data['title'] + " " + test_data['body'] + " " + test_data['url']


train_text = list(np.array(train_data['all_text']))
test_text = list(np.array(test_data['all_text']))

# use tf-ifd to deal with the text field
tfv = TfidfVectorizer(min_df = 3,  max_features = None, strip_accents = 'unicode',  
        analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 2), use_idf = 1, smooth_idf = 1, sublinear_tf = 1)


transformed_text = tfv.fit_transform(train_text + test_text)


# since the output of if-idf is way to big, use svd to reduce it to 150 columns
svd = TruncatedSVD(150, n_iter = 7, random_state = 2017)
svd_text = svd.fit_transform(transformed_text)
svd.explained_variance_ratio_.sum()

x_train = svd_text[:len(train_text)]
x_test = svd_text[len(train_text):]
y_train = train_data['label']



#################################################################################################################
# first method, directly use the tf-idf output to do the LR model, output is very promising
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

parameter_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000],
                  'dual':[True, False]} 
grid_search = GridSearchCV(model, param_grid = parameter_grid, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

pre_final = grid_search.predict_proba(x_test)[:, 1]

submit_data = pd.read_csv("sampleSubmission.csv")
submit_data['label'] = pre_final
submit_data.to_csv('StumbleUpon Evergreen Classification Challenge_LR.csv', index = 0)






##################################################################################################################
# then we try to ensemble using stacking to see whether we can get better result

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
                    'max_depth': 19,
                    'max_features': 'sqrt'}
model.set_params(**parameter_grid)



import xgboost as xgb
model1 = xgb.XGBClassifier(n_estimators = 300, objective = 'binary:logistic')
parameter_grid1 = {  'reg_alpha': 0,
                    'reg_lambda': 0.001,
                    'max_depth': 8,
                    'learning_rate': 0.1}
model1.set_params(**parameter_grid1)



from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid2 = {  'n_estimators': 300,
                    'criterion': 'entropy',
                    'max_depth': 19}
model2.set_params(**parameter_grid2)


model3 = RandomForestClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid3 = {  'n_estimators': 300,
                    'criterion': 'gini',
                    'max_depth': 19}
model3.set_params(**parameter_grid3)


from sklearn.ensemble import ExtraTreesClassifier
model4 = ExtraTreesClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid4 = {  'n_estimators': 300,
                    'criterion': 'entropy',
                    'max_depth': 19}
model4.set_params(**parameter_grid4)
from sklearn.ensemble import ExtraTreesClassifier

model5 = ExtraTreesClassifier(max_features = 'sqrt', n_jobs = -1)
parameter_grid5 = {  'n_estimators': 300,
                    'criterion': 'gini',
                    'max_depth': 19}
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


# stacking model is LR
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(dual=True)

parameter_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000]} 
grid_search = GridSearchCV(model, param_grid = parameter_grid, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search.fit(final_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

pre_final = grid_search.predict_proba(final_test)[:, 1]

submit_data = pd.read_csv("sampleSubmission.csv")
submit_data['label'] = pre_final
submit_data.to_csv('StumbleUpon Evergreen Classification Challenge_Ensemble.csv', index = 0)

