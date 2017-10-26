

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


# read the train dataset and test dataset
# deal with NA and missing value if possible

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")



test_data.ix[666, 'GarageType'] = np.NaN
test_data.ix[666, 'GarageCars'] = 0
test_data.ix[666, 'GarageArea'] = 0


test_data.ix[1116, 'GarageYrBlt'] = test_data.ix[1116, 'YearRemodAdd']
test_data.ix[1116, 'GarageFinish'] = 'Unf'
test_data.ix[1116, 'GarageQual'] = 'TA'
test_data.ix[1116, 'GarageCond'] = 'TA'

test_data.Exterior1st = test_data.Exterior1st.fillna("VinylSd") 
test_data.Exterior2nd = test_data.Exterior2nd.fillna("VinylSd") 

####################################################################

from scipy.stats import boxcox

qual_number = {'Ex': 5, 
               'Gd': 4,
               'TA': 3,
               'Fa': 2,
               'Po': 1,
               None: 0}

neighborhood_number = {'MeadowV': 0,
                        'IDOTRR': 0.01,
                        'BrDale': 0.05,
                        'BrkSide': 0.19,
                        'Edwards': 0.21,
                        'OldTown': 0.21,
                        'Sawyer': 0.27,
                        'Blueste': 0.27,
                        'SWISU': 0.3,
                        'NPkVill': 0.3,
                        'NAmes': 0.32,
                        'Mitchel': 0.38,
                        'SawyerW': 0.52,
                        'NWAmes': 0.53,
                        'Gilbert': 0.55,
                        'Blmngtn': 0.56,
                        'CollgCr': 0.57,
                        'Crawfor': 0.62,
                        'ClearCr': 0.63,
                        'Somerst': 0.68,
                        'Veenker': 0.72,
                        'Timber': 0.73,
                        'StoneBr': 0.94,
                        'NridgHt': 0.95,
                        'NoRidge': 1}


Condition1_number = {'Artery': 0,
                        'RRAe': 0.05,
                        'Feedr': 0.1,
                        'RRAn': 0.61,
                        'Norm': 0.61,
                        'RRNe': 0.67,
                        'RRNn': 0.88,
                        'PosN': 0.91,
                        'PosA': 1}


Condition2_number = {'RRNn': 0,
                        'Artery': 0.08,
                        'Feedr': 0.19,
                        'RRAn': 0.29,
                        'Norm': 0.52,
                        'RRAe': 0.56,
                        'PosN': 0.89,
                        'PosA': 1}

SaleCondition_number = {'AdjLand': 0,
                        'Abnorml': 0.36,
                        'Family': 0.38,
                        'Alloca': 0.49,
                        'Normal': 0.54,
                        'Partial': 1}

RoofStyle_number = {'Gambrel': 0,
                        'Gable': 0.34,
                        'Mansard': 0.47,
                        'Flat': 0.65,
                        'Hip': 0.93,
                        'Shed': 1}

RoofMatl_number = {'Roll': 0,
                    'ClyTile': 0.15,
                    'CompShg': 0.26,
                    'Metal': 0.26,
                    'Tar&Grv': 0.29,
                    'WdShake': 0.54,
                    'Membran': 0.54,
                    'WdShngl': 1}



Heating_number = {'Floor': 0,
                    'Grav': 0.04,
                    'Wall': 0.26,
                    'OthW': 0.6,
                    'GasW': 0.9,
                    'GasA': 1}


Electrical_number = {'Mix': 0,
                    'FuseP': 0.36,
                    'FuseF': 0.46,
                    'FuseA': 0.59,
                    'SBrkr': 1}



Foundation_number = {'Slab': 0,
                    'BrkTil': 0.28,
                    'CBlock': 0.45,
                    'Stone': 0.59,
                    'Wood': 0.74,
                    'PConc': 1}

GarageType_number = {None: 0,
                    'CarPort': 0.07,
                    'Detchd': 0.29,
                    '2Types': 0.42,
                    'Basment': 0.49,
                    'Attchd': 0.75,
                    'BuiltIn': 1}


MSSubClass_number = {30: 0,
                    180: 0.07,
                    45: 0.14,
                    190: 0.33,
                    90: 0.36,
                    160: 0.4,
                    50: 0.44,
                    85: 0.47,
                    150: 0.47,
                    40: 0.53,
                    70: 0.6,
                    80: 0.62,
                    20: 0.72,
                    75: 0.76,
                    120: 0.81,
                    60: 1}


MSZoning_number = {'C (all)': 0,
                    'RM': 0.5,
                    'RH': 0.54,
                    'RL': 0.89,
                    'FV': 1}


BldgType_number = {'2fmCon': 0,
                    'Duplex': 0.11,
                    'Twnhs': 0.15,
                    'TwnhsE': 0.94,
                    '1Fam': 1}

HouseStyle_number = {'1.5Unf': 0,
                    'SFoyer': 0.29,
                    '1.5Fin': 0.38,
                    '2.5Unf': 0.52,
                    'SLvl': 0.6,
                    '1Story': 0.68,
                    '2Story': 0.93,
                    '2.5Fin': 1}

Functional_number = {'Maj2': 0,
                    'Sev': 0.54,
                    'Min2': 0.68,
                    'Min1': 0.7,
                    'Maj1': 0.77,
                    'Mod': 0.89,
                    'Typ': 1}


LandSlope_number = {'Gtl': 0,
                    'Mod': 0.7,
                    'Sev': 1}




transformed_data = train_data.ix[:,['SalePrice']]



transformed_data["age"] = train_data.YrSold - train_data.YearBuilt
transformed_data["modage"] = train_data.YrSold - train_data.YearRemodAdd


transformed_data['LotArea'] = np.log(train_data['LotArea'])
transformed_data = transformed_data.join(pd.get_dummies(train_data["LotShape"], prefix = "LotShape").ix[:, :-1])
transformed_data = transformed_data.join(pd.get_dummies(train_data["LandContour"], prefix = "LandContour").ix[:, :-1])
transformed_data['LandSlope_Gtl'] = train_data['LandSlope'] == 'Gtl'



transformed_data['Neighborhood'] = train_data['Neighborhood'].map(neighborhood_number)

transformed_data['Condition1'] = train_data['Condition1'].map(Condition1_number)
transformed_data['Condition2'] = train_data['Condition2'].map(Condition2_number)


transformed_data = transformed_data.join(pd.get_dummies(train_data["Alley"], prefix = "Alley"))

transformed_data['OverallQual'] = train_data['OverallQual']
transformed_data['OverallCond'] = train_data['OverallCond']





transformed_data['SaleCondition'] = (train_data['SaleCondition'].map(SaleCondition_number))
transformed_data['RoofStyle'] = (train_data['RoofStyle'].map(RoofStyle_number))
transformed_data['RoofMatl'] = (train_data['RoofMatl'].map(RoofMatl_number))
transformed_data['Heating'] = (train_data['Heating'].map(Heating_number))
transformed_data['Electrical'] = (train_data['Electrical'].fillna('SBrkr').map(Electrical_number))
transformed_data['Foundation'] = (train_data['Foundation'].map(Foundation_number))

transformed_data['GarageType'] = (train_data['GarageType'].map(GarageType_number))
# test data has 150
transformed_data['MSSubClass'] = (train_data['MSSubClass'].map(MSSubClass_number))
# test has 4 missing
transformed_data['MSZoning'] = (train_data['MSZoning'].map(MSZoning_number))

transformed_data['BldgType'] = (train_data['BldgType'].map(BldgType_number))
transformed_data['HouseStyle'] = (train_data['HouseStyle'].map(HouseStyle_number))
transformed_data['Functional'] = (train_data['Functional'].map(Functional_number))


transformed_data['CentralAir'] = (train_data['CentralAir'] == 'Y')




transformed_data['TotRmsAbvGrd'] = train_data['TotRmsAbvGrd']


transformed_data['PavedDrive'] = (train_data['PavedDrive'] == 'Y')

transformed_data["area"] = np.log(train_data['OpenPorchSF'] + \
                           train_data['EnclosedPorch'] + \
                           train_data['3SsnPorch'] + \
                           train_data['ScreenPorch'] + \
                           train_data['GrLivArea'] + \
                           train_data['WoodDeckSF'] + \
                           train_data['TotalBsmtSF'] + \
                           train_data['PoolArea'])




transformed_data['HeatingQC_Number'] = train_data['HeatingQC'].map(qual_number)
transformed_data['FireplaceQu_Number'] = train_data['FireplaceQu'].map(qual_number) 
transformed_data['PoolQC_Number'] = train_data['PoolQC'].map(qual_number)
transformed_data['GarageQual_Number'] = train_data['GarageQual'].map(qual_number)



#-----------------------------------------------------------------------------------------------


transformed_test_data = test_data.ix[:,['SalePrice']]

transformed_test_data["age"] = test_data.YrSold - test_data.YearBuilt
transformed_test_data["modage"] = test_data.YrSold - test_data.YearRemodAdd



transformed_test_data['LotArea'] = np.log(test_data['LotArea'])

transformed_test_data = transformed_test_data.join(pd.get_dummies(test_data["LotShape"], prefix = "LotShape").ix[:, :-1])
transformed_test_data = transformed_test_data.join(pd.get_dummies(test_data["LandContour"], prefix = "LandContour").ix[:, :-1])
transformed_test_data['LandSlope_Gtl'] = test_data['LandSlope'] == 'Gtl'


transformed_test_data['Neighborhood'] = test_data['Neighborhood'].map(neighborhood_number)

transformed_test_data['Condition1'] = test_data['Condition1'].map(Condition1_number)
transformed_test_data['Condition2'] = test_data['Condition2'].map(Condition2_number)

transformed_test_data = transformed_test_data.join(pd.get_dummies(test_data["Alley"], prefix = "Alley"))

transformed_test_data['OverallQual'] = test_data['OverallQual']
transformed_test_data['OverallCond'] = test_data['OverallCond']

transformed_test_data['SaleCondition'] = (test_data['SaleCondition'].map(SaleCondition_number))
transformed_test_data['RoofStyle'] = (test_data['RoofStyle'].map(RoofStyle_number))
transformed_test_data['RoofMatl'] = (test_data['RoofMatl'].map(RoofMatl_number))
transformed_test_data['Heating'] = (test_data['Heating'].map(Heating_number))
transformed_test_data['Electrical'] = (test_data['Electrical'].fillna('SBrkr').map(Electrical_number))
transformed_test_data['Foundation'] = (test_data['Foundation'].map(Foundation_number))



transformed_test_data['GarageType'] = (test_data['GarageType'].map(GarageType_number))
# test data has 150
transformed_test_data['MSSubClass'] = (test_data['MSSubClass'].map(MSSubClass_number))
# test has 4 missing
# fill NA of MSZoning with logical guessing

test_data.ix[test_data.MSZoning.isnull() & (test_data.Neighborhood == 'Mitchel') &
             (test_data.Condition1 == 'Artery'), 'MSZoning'] = 'RL'
test_data.ix[test_data.MSZoning.isnull() & (test_data.Neighborhood == 'IDOTRR') &
             (test_data.Condition1 == 'Norm'), 'MSZoning'] = 'RM'

transformed_test_data['MSZoning'] = (test_data['MSZoning'].map(MSZoning_number))

transformed_test_data['BldgType'] = (test_data['BldgType'].map(BldgType_number))
transformed_test_data['HouseStyle'] = (test_data['HouseStyle'].map(HouseStyle_number))
# test has 2 missing
transformed_test_data['Functional'] = (test_data['Functional'].fillna('Typ').map(Functional_number))

transformed_test_data['CentralAir'] = (test_data['CentralAir'] == 'Y')
transformed_test_data['TotRmsAbvGrd'] = test_data['TotRmsAbvGrd']
transformed_test_data['PavedDrive'] = (test_data['PavedDrive'] == 'Y')




transformed_test_data["area"] = np.log(test_data['OpenPorchSF'] + \
                           test_data['EnclosedPorch'] + \
                           test_data['3SsnPorch'] + \
                           test_data['ScreenPorch'] + \
                           test_data['GrLivArea'] + \
                           test_data['WoodDeckSF'] + \
                           test_data['TotalBsmtSF'].fillna(0.0) + \
                           test_data['PoolArea'])


transformed_test_data['HeatingQC_Number'] = test_data['HeatingQC'].map(qual_number)
transformed_test_data['FireplaceQu_Number'] = test_data['FireplaceQu'].map(qual_number)
transformed_test_data['PoolQC_Number'] = test_data['PoolQC'].map(qual_number)
transformed_test_data['GarageQual_Number'] = test_data['GarageQual'].map(qual_number)


transformed_test_data['Exterior1st_ImStucc'] = 0
transformed_test_data['Exterior1st_Stone'] = 0 




transformed_data.drop(pd.Int64Index([523, 1298]), inplace = True)


transformed_data.RoofMatl = np.log1p(transformed_data.RoofMatl)
transformed_test_data.RoofMatl = np.log1p(transformed_test_data.RoofMatl)




skew_list=['age',
'modage',
'LotArea',
'LotShape_IR1', 
'Neighborhood',
'OverallQual', 
'OverallCond', 
'Foundation',
'GarageType',
'MSSubClass', 
'HouseStyle',  
'TotRmsAbvGrd', 
'area', 
'HeatingQC_Number',
'FireplaceQu_Number']


from scipy.stats import skew
skewed_feats = transformed_data[skew_list].apply(lambda x: skew(x))
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
skewed_feats = skewed_feats.index

transformed_data[skewed_feats] = transformed_data[skewed_feats].apply(lambda x: np.log(x+1))
transformed_test_data[skewed_feats] = transformed_data[skewed_feats].apply(lambda x: np.log(x+1))



#------------------------------------------------------------------------



# now, all the data has been transformed, so we can get the feature we want and scale it before throwing them to the model

feature_list = transformed_data.columns.tolist()
feature_list.remove('SalePrice')
sc = preprocessing.StandardScaler()


x_train = sc.fit_transform(transformed_data[feature_list])
x_test = sc.transform(transformed_test_data[feature_list])
y_train = transformed_data['SalePrice']




########################################################################
# this part will help find out which features are more helpful
########################################################################
from sklearn.linear_model import Lasso

model = Lasso()
parameter_grid = {'alpha' : [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.1, 0.5, 1]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 4, n_jobs = -1)
grid_search.fit(x_train, np.log(y_train))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

data = list(zip(grid_search.best_estimator_.coef_, feature_list))
sorted(data, key=lambda tup: np.abs(tup[0]))
########################################################################


# now we are going to use the simple stacking skill to do the ensembling

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

        grid_search = GridSearchCV(self.stacking_model, param_grid = parameter_grid, cv = 4, n_jobs = -1)
        grid_search.fit(stacking_train, y_train)

        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))

        output = grid_search.predict(stacking_test)
        return output

########################################################################

# use 3 powerful models to do stacking so that we can get better result than just using one of them


parameter_xgb = {  'reg_alpha': 0,
                    'reg_lambda': 0.1,
                    'max_depth': 3,
                    'learning_rate': 0.2}

parameter_sgd = {  'loss': 'epsilon_insensitive',
                    'penalty': 'l1',
                    'alpha': 0.12915496650148828,
                    'l1_ratio' : 0.5}

parameter_gb = {  'learning_rate': 0.03,
                    'loss': 'huber',
                    'max_depth': 5,
                    'max_features': 'sqrt'}

model = GradientBoostingRegressor(n_estimators = 300)
model1 = xgb.XGBRegressor(n_estimators = 300, objective = 'reg:linear')
model2 = SGDRegressor(n_iter = 1000)

model.set_params(**parameter_gb)
model1.set_params(**parameter_xgb)
model2.set_params(**parameter_sgd)


s_model = GradientBoostingRegressor(n_estimators = 300)
model_list = [model, model1, model2]

ss = SimpleStacking(model_list, s_model)
output = ss.fit_predict(4, x_train, x_test, np.log(np.array(y_train)))

final_output = np.exp(output)
submit_data = pd.read_csv("sample_submission.csv")
submit_data['SalePrice'] = final_output
submit_data.to_csv("House_Prices_final.csv", index = 0)

