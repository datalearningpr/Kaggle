
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



# read the train dataset and test dataset
# deal with NA and missing value if possible

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


train_data["id"] = -100
train_data["group"] = "train"
test_data["group"] = "test"
test_data["ACTION"] = -1

# combine the train and test data
data = pd.concat([train_data, test_data], ignore_index=True)




# this is using all the fields related to role to come up with a new feature 
count_df = data.groupby(['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
       'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
       'ROLE_CODE']).size().reset_index(name='counts')
data["role_id"] = pd.merge(data, count_df, how='left', on=['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
       'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
       'ROLE_CODE'])["counts"]



# this is not using the desc and role_code
count_df = data.groupby(['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
       'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY']).size().reset_index(name='counts')
data["type_id"] = pd.merge(data, count_df, how='left', on=['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
       'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY'])["counts"]


names = ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
       'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
       'ROLE_CODE',"role_id"]


# one way counting
for i in names:
    count_df = data.groupby([i]).size().reset_index(name='counts')
    data[i+"__cnt"] = pd.merge(data, count_df, how='left', on=[i])["counts"]


# funtions to do two-way and three-way counting
def two_way_count(data, name1, name2):
    count_df = data.groupby([name1, name2]).size().reset_index(name='counts')
    return pd.merge(data, count_df, how='left', on=[name1, name2])["counts"]

def three_way_count(data, name1, name2, name3):
    count_df = data.groupby([name1, name2, name3]).size().reset_index(name='counts')
    return pd.merge(data, count_df, how='left', on=[name1, name2, name3])["counts"]


# conduct the two-way counting
data["mgr_res_cnt"] = two_way_count(data, "MGR_ID", "RESOURCE")
data["mgr_rid_cnt"] = two_way_count(data, "MGR_ID", "role_id")
data["rid_res_cnt"] = two_way_count(data, "role_id", "RESOURCE")


data["mgr_rocd_cnt"] = two_way_count(data, "MGR_ID", "ROLE_CODE")
data["mgr_rd_cnt"] = two_way_count(data, "MGR_ID", "ROLE_DEPTNAME")
data["mgr_rf_cnt"] = two_way_count(data, "MGR_ID", "ROLE_FAMILY")
data["mgr_rfd_cnt"] = two_way_count(data, "MGR_ID", "ROLE_FAMILY_DESC")
data["mgr_rt_cnt"] = two_way_count(data, "MGR_ID", "ROLE_TITLE")
data["mgr_rr1_cnt"] = two_way_count(data, "MGR_ID", "ROLE_ROLLUP_1")
data["mgr_rr2_cnt"] = two_way_count(data, "MGR_ID", "ROLE_ROLLUP_2")

data["res_rf_cnt"] = two_way_count(data, "RESOURCE", "ROLE_FAMILY")
data["res_rfd_cnt"] = two_way_count(data, "RESOURCE", "ROLE_FAMILY_DESC")
data["res_rt_cnt"] = two_way_count(data, "RESOURCE", "ROLE_TITLE")
data["res_rd_cnt"] = two_way_count(data, "RESOURCE", "ROLE_DEPTNAME")
data["res_rocd_cnt"] = two_way_count(data, "RESOURCE", "ROLE_CODE")
data["res_rr1_cnt"] = two_way_count(data, "RESOURCE", "ROLE_ROLLUP_1")
data["res_rr2_cnt"] = two_way_count(data, "RESOURCE", "ROLE_ROLLUP_2")

data["rr1_rr2_cnt"] = two_way_count(data, "ROLE_ROLLUP_1", "ROLE_ROLLUP_2")
data["rr1_rf_cnt"] = two_way_count(data, "ROLE_ROLLUP_1", "ROLE_FAMILY")
data["rr1_rfd_cnt"] = two_way_count(data, "ROLE_ROLLUP_1", "ROLE_FAMILY_DESC")
data["rr1_rt_cnt"] = two_way_count(data, "ROLE_ROLLUP_1", "ROLE_TITLE")
data["rr1_rd_cnt"] = two_way_count(data, "ROLE_ROLLUP_1", "ROLE_DEPTNAME")
data["rr1_rocd_cnt"] = two_way_count(data, "ROLE_ROLLUP_1", "ROLE_CODE")

data["rr2_rf_cnt"] = two_way_count(data, "ROLE_ROLLUP_2", "ROLE_FAMILY")
data["rr2_rfd_cnt"] = two_way_count(data, "ROLE_ROLLUP_2", "ROLE_FAMILY_DESC")
data["rr2_rt_cnt"] = two_way_count(data, "ROLE_ROLLUP_2", "ROLE_TITLE")
data["rr2_rd_cnt"] = two_way_count(data, "ROLE_ROLLUP_2", "ROLE_DEPTNAME")
data["rr2_rocd_cnt"] = two_way_count(data, "ROLE_ROLLUP_2", "ROLE_CODE")

data["rf_rfd_cnt"] = two_way_count(data, "ROLE_FAMILY", "ROLE_FAMILY_DESC")
data["rf_rt_cnt"] = two_way_count(data, "ROLE_FAMILY", "ROLE_TITLE")
data["rf_rd_cnt"] = two_way_count(data, "ROLE_FAMILY", "ROLE_DEPTNAME")
data["rf_rocd_cnt"] = two_way_count(data, "ROLE_FAMILY", "ROLE_CODE")

data["rfd_rt_cnt"] = two_way_count(data, "ROLE_FAMILY_DESC", "ROLE_TITLE")
data["rfd_rd_cnt"] = two_way_count(data, "ROLE_FAMILY_DESC", "ROLE_DEPTNAME")
data["rfd_rocd_cnt"] = two_way_count(data, "ROLE_FAMILY_DESC", "ROLE_CODE")

data["rt_rd_cnt"] = two_way_count(data, "ROLE_TITLE", "ROLE_DEPTNAME")
data["rt_rocd_cnt"] = two_way_count(data, "ROLE_TITLE", "ROLE_CODE")

data["rd_rocd_cnt"] = two_way_count(data, "ROLE_DEPTNAME", "ROLE_CODE")


# conduct the three-way counting

data["res_mgr_rr2_cnt"] = three_way_count(data, "RESOURCE", "MGR_ID", "ROLE_ROLLUP_2")
data["res_mgr_rd_cnt"] = three_way_count(data, "RESOURCE", "MGR_ID", "ROLE_DEPTNAME")
data["res_mgr_rocd_cnt"] = three_way_count(data, "RESOURCE", "MGR_ID", "ROLE_CODE")
data["res_mgr_rdf_cnt"] = three_way_count(data, "RESOURCE", "MGR_ID", "ROLE_FAMILY_DESC")
data["res_mgr_rf_cnt"] = three_way_count(data, "RESOURCE", "MGR_ID", "ROLE_FAMILY")
data["rf_mgr_rd_cnt"] = three_way_count(data, "ROLE_FAMILY", "MGR_ID", "ROLE_DEPTNAME")
data["rf_mgr_rr2_cnt"] = three_way_count(data, "ROLE_FAMILY", "MGR_ID", "ROLE_ROLLUP_2")
data["res_rd_rf_cnt"] = three_way_count(data, "RESOURCE", "ROLE_DEPTNAME", "ROLE_FAMILY")
data["res_rd_rr2_cnt"] = three_way_count(data, "RESOURCE", "ROLE_DEPTNAME", "ROLE_ROLLUP_2")


# using label encoding to make the id from 0 to 1, 2, etc

le = preprocessing.LabelEncoder()
data["resource_id"] = le.fit_transform(data["RESOURCE"])
le = preprocessing.LabelEncoder()
data["manager_id"] = le.fit_transform(data["MGR_ID"])
le = preprocessing.LabelEncoder()
data["r_id"] = le.fit_transform(data["ROLE_CODE"])



# this feature generation is a bit special
# it is using the eigen value from the svd decomposition
# svd is used in recommendation system, which can provide transformation method given
# two sets of features

# therefore, here, we use svd to get the value for each element of a given feature

from scipy import sparse
from numpy import linalg

tmp = data[["r_id", "manager_id"]].groupby(["r_id", "manager_id"]).size().reset_index(name='counts')

A = tmp["r_id"]
B = tmp["manager_id"]

sparse_m = sparse.coo_matrix((np.ones(len(A)), (A, B)), shape = (len(A.unique()), len(B.unique()))).tocsc().todense()
U, sigma, VT = linalg.svd(sparse_m)
V = VT.T

data["r_manager_ev0"] = U[data["r_id"], 0]
data["r_manager_ev1"] = U[data["r_id"], 1]
data["r_manager_ev2"] = U[data["r_id"], 2]
data["r_manager_ev3"] = U[data["r_id"], 3]
data["r_manager_ev4"] = U[data["r_id"], 4]

data["manager_r_ev0"] = V[data["manager_id"], 0]
data["manager_r_ev1"] = V[data["manager_id"], 1]
data["manager_r_ev2"] = V[data["manager_id"], 2]
data["manager_r_ev3"] = V[data["manager_id"], 3]
data["manager_r_ev4"] = V[data["manager_id"], 4]




tmp = data[["r_id", "resource_id"]].groupby(["r_id", "resource_id"]).size().reset_index(name='counts')

A = tmp["r_id"]
B = tmp["resource_id"]

sparse_m = sparse.coo_matrix((np.ones(len(A)), (A, B)), shape = (len(A.unique()), len(B.unique()))).tocsc().todense()
U, sigma, VT = linalg.svd(sparse_m)
V = VT.T

data["r_resource_ev0"] = U[data["r_id"], 0]
data["r_resource_ev1"] = U[data["r_id"], 1]
data["r_resource_ev2"] = U[data["r_id"], 2]
data["r_resource_ev3"] = U[data["r_id"], 3]
data["r_resource_ev4"] = U[data["r_id"], 4]

data["resource_r_ev0"] = V[data["resource_id"], 0]
data["resource_r_ev1"] = V[data["resource_id"], 1]
data["resource_r_ev2"] = V[data["resource_id"], 2]
data["resource_r_ev3"] = V[data["resource_id"], 3]
data["resource_r_ev4"] = V[data["resource_id"], 4]




tmp = data[["resource_id", "manager_id"]].groupby(["resource_id", "manager_id"]).size().reset_index(name='counts')

A = tmp["resource_id"]
B = tmp["manager_id"]

sparse_m = sparse.coo_matrix((np.ones(len(A)), (A, B)), shape = (len(A.unique()), len(B.unique()))).tocsc().todense()
U, sigma, VT = linalg.svd(sparse_m)
V = VT.T

data["resource_manager_ev0"] = U[data["resource_id"], 0]
data["resource_manager_ev1"] = U[data["resource_id"], 1]
data["resource_manager_ev2"] = U[data["resource_id"], 2]
data["resource_manager_ev3"] = U[data["resource_id"], 3]
data["resource_manager_ev4"] = U[data["resource_id"], 4]

data["manager_resource_ev0"] = V[data["manager_id"], 0]
data["manager_resource_ev1"] = V[data["manager_id"], 1]
data["manager_resource_ev2"] = V[data["manager_id"], 2]
data["manager_resource_ev3"] = V[data["manager_id"], 3]
data["manager_resource_ev4"] = V[data["manager_id"], 4]


# finally generate the train and test datasets

feature_list = list(data.columns)
feature_list.remove('ACTION')
feature_list.remove('id')
feature_list.remove('group')
feature_list.remove('resource_id')
feature_list.remove('manager_id')
feature_list.remove('r_id')

x_train = data.loc[data.group=="train", feature_list]
x_test = data.loc[data.group=="test", feature_list]
y_train = data.loc[data.group=="train", 'ACTION']



#################################################################################

# the models are all tree based models, they all performing quite well with the new features
# we generated above

# simple average blending is already giving very good score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


parameter_xgb = {   'colsample_bytree': 0.4,
                    'max_depth': 9,
                    'learning_rate': 0.1}

parameter_gb = {  'learning_rate': 0.1,
                    'max_depth': 9,
                    'max_features': 'sqrt'}

parameter_rf1 = {  'n_estimators': 500,
                    'criterion': 'gini'}
parameter_rf2 = {  'n_estimators': 500,
                    'criterion': 'entropy'}
parameter_et = {  'n_estimators': 500,
                    'criterion': 'entropy'}


model = RandomForestClassifier(max_features = 'sqrt')
model1 = RandomForestClassifier(max_features = 'sqrt')
model2 = ExtraTreesClassifier(max_features = 'sqrt')
model3 = GradientBoostingClassifier(n_estimators = 300)
model4 = xgb.XGBClassifier(n_estimators = 300, objective = 'binary:logistic')

model.set_params(**parameter_rf1)
model1.set_params(**parameter_rf2)
model2.set_params(**parameter_et)
model3.set_params(**parameter_gb)
model4.set_params(**parameter_xgb)



result = []
for m in [model, model1, model2, model3, model4]:
    m.fit(x_train, y_train)
    result.append(m.predict_proba(x_test)[:, 1])


final_output = np.mean(result, axis = 0)

submit_data = pd.read_csv("sampleSubmission.csv")
submit_data['Action'] = final_output
submit_data.to_csv("Employee_Access_Challenge_final.csv", index = 0)






