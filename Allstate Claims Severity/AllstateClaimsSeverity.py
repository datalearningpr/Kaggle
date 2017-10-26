
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn import preprocessing
import itertools


# read the train dataset and test dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# combine train dataset with test dataset
all_data = pd.concat([train_data, test_data])
all_data.reset_index(inplace = True)
all_data.drop('index', axis = 1, inplace = True)
all_data = all_data.reindex_axis(all_data.columns, axis = 1)

# first take a look at the data
all_data.info()
all_data.head()

# check whether there is missing values
all_data.isnull().sum()


feature_list = all_data.columns.tolist()
feature_list.remove("id")
feature_list.remove("loss")

corr = all_data[feature_list].corr()

indices = np.where(corr > 0.8)
pairs = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices) if x != y and x < y]


cat_list = ['cat1', 'cat10', 'cat100', 'cat101', 'cat102', 'cat103', 'cat104', 'cat105', 'cat106', 'cat107', 'cat108', 'cat109', 'cat11', 'cat110', 'cat111', 'cat112', 'cat113', 'cat114', 'cat115', 'cat116', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19', 'cat2', 'cat20', 'cat21', 'cat22', 'cat23', 'cat24', 'cat25', 'cat26', 'cat27', 'cat28', 'cat29', 'cat3', 'cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat4', 'cat40', 'cat41', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46', 'cat47', 'cat48', 'cat49', 'cat5', 'cat50', 'cat51', 'cat52', 'cat53', 'cat54', 'cat55', 'cat56', 'cat57', 'cat58', 'cat59', 'cat6', 'cat60', 'cat61', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66', 'cat67', 'cat68', 'cat69', 'cat7', 'cat70', 'cat71', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76', 'cat77', 'cat78', 'cat79', 'cat8', 'cat80', 'cat81', 'cat82', 'cat83', 'cat84', 'cat85', 'cat86', 'cat87', 'cat88', 'cat89', 'cat9', 'cat90', 'cat91', 'cat92', 'cat93', 'cat94', 'cat95', 'cat96', 'cat97', 'cat98', 'cat99']
rest_list = ['cont1', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'loss']

# this features are tested and selected accordingly
# due to limitation of machine, you cannot test all combinations
COMB_FEATURE = ['cat3',
                'cat4',
                'cat5',
                'cat7',
                'cat10',
                'cat11',
                'cat12',
                'cat13',
                'cat14',
                'cat16',
                'cat23',
                'cat28',
                'cat36',
                'cat40',
                'cat57',
                'cat72',
                'cat79',
                'cat80',
                'cat81',
                'cat87',
                'cat89']

for i, comb in enumerate(itertools.combinations(COMB_FEATURE, 2)):
    print(i)
    feat = comb[0] + "_" + comb[1]
    cat_list.append(feat)
    all_data[feat] = all_data[comb[0]] + all_data[comb[1]]


for i, col in enumerate(cat_list):
    print(i)
    all_data[col] = pd.factorize(all_data[col])[0]

feature_list = all_data.columns.tolist()
feature_list.remove("id")
feature_list.remove("loss")

# we need to scale the features
x_train = all_data.loc[-all_data["loss"].isnull(), feature_list]
y_train = all_data.loc[-all_data["loss"].isnull(), "loss"]
x_test = all_data.loc[all_data["loss"].isnull(), feature_list]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# select the best 15 features to use
k_best = SelectKBest(f_regression, k = 30).fit(x_train, np.array(y_train))
selected_features = list(x_train.columns[k_best.get_support()])

x_train_selected = x_train.ix[:, selected_features]
x_test_selected = x_test.ix[:, selected_features]


std_scaler = preprocessing.StandardScaler()
x_train = std_scaler.fit_transform(x_train_selected)
x_test = std_scaler.transform(x_test_selected)


import xgboost as xgb

model = xgb.XGBRegressor(n_estimators = 300, objective = 'reg:linear')
parameter_grid = {  'reg_alpha': [0],
                    'reg_lambda': [0.001],
                    'max_depth': [7],
                    'learning_rate': [0.2]}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 4, n_jobs = -1, scoring = 'neg_mean_absolute_error')
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# predict the classification
output = grid_search.predict(x_test)


final_output = pd.read_csv("sample_submission.csv")
final_output["loss"] = np.exp(output)
final_output.to_csv("Allsate_Claims_Severity_final.csv", index = 0)
