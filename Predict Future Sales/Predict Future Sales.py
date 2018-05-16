
import re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import gc
from collections import Counter


# read the train dataset and test dataset
# deal with NA and missing value if possible

train_data = pd.read_csv("sales_train_v2.csv")
test_data = pd.read_csv("test.csv")
item_categories = pd.read_csv("item_categories.csv")
items = pd.read_csv("items.csv")
shops = pd.read_csv("shops.csv")

# remove the shop_id does not appear in the test data
filtered_train = train_data.merge(test_data[['shop_id']].drop_duplicates(), how = 'inner')
sales_count = Counter(filtered_train["item_cnt_day"])

# this is the most tricky part of this game
# most of the sales value are very low, and there are some abnormal big values
# gives good result if clip all the sales to be within a small number
# can try more values other than 20
filtered_train["item_cnt_day"] = filtered_train["item_cnt_day"].clip(0, 20)
filtered_train["date_1"] = filtered_train.date.apply(lambda x: datetime.date(int(x[6:10]),int(x[3:5]), 1))

# we need to predict monthly sales, not daily
columns = ['date_1', 'shop_id', 'item_id', 'date_block_num']
group_data = filtered_train.groupby(columns)['item_cnt_day'].agg('sum').reset_index().rename(columns = {"item_cnt_day": "item_cnt_month"})
group_data["item_cnt_month"] = group_data["item_cnt_month"].clip(0, 20)

# we need to generate the missing rows(which will have sales to be 0)
from itertools import product
grid = []
for block_num in group_data['date_block_num'].unique():
    cur_shops = group_data[group_data['date_block_num']==block_num]['shop_id'].unique()
    cur_items = group_data[group_data['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
grid = pd.merge(grid,group_data[["date_1","date_block_num"]].drop_duplicates(), on = ["date_block_num"], how = "left")
grid_data = pd.merge(grid,group_data[['shop_id', 'item_id', 'date_block_num', 'item_cnt_month']],how='left',on=index_cols).fillna(0)

# make the test data to be the same format to concat
test_data["date_block_num"] = 34
test_data["date_1"] = datetime.date(2015, 11 ,1)
test_data["item_cnt_month"] = 0
test_data.drop('ID', axis=1, inplace=True)

# combine train and test data, get the category id
all_data = pd.concat([grid_data, test_data])
merge_data = pd.merge(all_data, items[['item_id', 'item_category_id']], how='left', on=['item_id'])

del train_data
del test_data
del item_categories
del items
del shops
del filtered_train
del sales_count
del columns
del group_data
del all_data
gc.collect()


# get the time lag sales features, can try to get more months if needed
index_cols = ['shop_id', 'item_id', 'date_block_num']
for i in [1, 2, 3, 4, 6, 9, 12]:
    lag_sales = merge_data.loc[:, ['shop_id', 'item_id', "date_block_num", "item_cnt_month"]]
    lag_sales["date_block_num"] += i
    lag_sales.rename(columns = {"item_cnt_month": "item_cnt_month_{}".format(i)}, inplace = True)
    merge_data = pd.merge(merge_data, lag_sales, how='left', on=index_cols).fillna(0)

# get the time related features
merge_data["year"] = pd.to_datetime(merge_data["date_1"]).dt.year - 2013
merge_data["month"] = pd.to_datetime(merge_data["date_1"]).dt.month
merge_data["days_month"] = pd.to_datetime(merge_data["date_1"]).dt.days_in_month



# generate the target encoding features
# was planing to do for item_id, or even interactions like ship_id + item_id
# but the speed is too slow, so only use shop_id and item_category_id
import category_encoders as ce

encoder = ce.TargetEncoder(cols=['shop_id'])
merge_data["shop_id_encode"] = 0
merge_data.loc[merge_data.date_block_num!=34, "shop_id_encode"] = np.array(encoder.fit_transform(merge_data.loc[merge_data.date_block_num!=34, ["shop_id"]],merge_data.loc[merge_data.date_block_num!=34, "item_cnt_month"] ))
merge_data.loc[merge_data.date_block_num==34, "shop_id_encode"] = np.array(encoder.transform(merge_data.loc[merge_data.date_block_num==34, ["shop_id"]]))

encoder = ce.TargetEncoder(cols=['item_category_id'])
merge_data["item_category_id_encode"] = 0
merge_data.loc[merge_data.date_block_num!=34, "item_category_id_encode"] = np.array(encoder.fit_transform(merge_data.loc[merge_data.date_block_num!=34, ["item_category_id"]],merge_data.loc[merge_data.date_block_num!=34, "item_cnt_month"] ))
merge_data.loc[merge_data.date_block_num==34, "item_category_id_encode"] = np.array(encoder.transform(merge_data.loc[merge_data.date_block_num==34, ["item_category_id"]]))


train_features = ['item_id', 'shop_id', 'item_category_id', 'year', 'month', 'days_month',
'shop_id_encode', 'item_category_id_encode',
'item_cnt_month_1', 'item_cnt_month_2', 'item_cnt_month_3', 'item_cnt_month_4', 
'item_cnt_month_6', "item_cnt_month_9", 'item_cnt_month_12']

train_mask = (merge_data.date_block_num < 34) & ((merge_data.date_block_num > 27))
test_mask = (merge_data.date_block_num == 34)

# do not use the old data which cannot get lag sales features
merge_data = merge_data.loc[merge_data.date_block_num>11,:]

# scaling all features
sc = preprocessing.StandardScaler()
merge_data.loc[merge_data.date_block_num!=34, train_features] = sc.fit_transform(merge_data.loc[merge_data.date_block_num!=34, train_features])

x_train = merge_data.loc[train_mask, train_features]
y_train = merge_data.loc[train_mask, "item_cnt_month"]

x_test = sc.transform(merge_data.loc[test_mask, train_features])
y_test = merge_data.loc[test_mask, "item_cnt_month"]


# will use 5 different models
# sgd-linear, lightGBM, catBoost, xgBoost, NN

print("sgd")
from sklearn.linear_model import SGDRegressor

model_sgd = SGDRegressor(penalty = 'l2', random_state = 0)
model_sgd.fit(x_train, y_train)
pred_sgd_1 = model_sgd.predict(x_train)
pred_sgd_2 = model_sgd.predict(x_test)
print(np.sqrt(mean_squared_error(y_train, pred_sgd_1)))
print(np.sqrt(mean_squared_error(y_test, pred_sgd_2)))


print("lightgbm")
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor()
model_lgb.fit(x_train, y_train)
pred_lgb_1 = model_lgb.predict(x_train)
pred_lgb_2 = model_lgb.predict(x_test)
print(np.sqrt(mean_squared_error(y_train, pred_lgb_1)))
print(np.sqrt(mean_squared_error(y_test, pred_lgb_2)))


print("catboost")
from catboost import CatBoostRegressor

model_cat = CatBoostRegressor(iterations=300, learning_rate=0.1,
                              l2_leaf_reg=1, depth=8, verbose = False)
model_cat.fit(x_train, y_train)
pred_cat_1 = model_cat.predict(x_train)
pred_cat_2 = model_cat.predict(x_test)
print(np.sqrt(mean_squared_error(y_train, pred_cat_1)))
print(np.sqrt(mean_squared_error(y_test, pred_cat_2)))


print("xgboost")
import xgboost as xgb

model_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1,
 min_child_weight=1, max_depth=8)
model_xgb.fit(np.array(x_train), y_train)
pred_xgb_1 = model_xgb.predict(np.array(x_train))
pred_xgb_2 = model_xgb.predict(np.array(x_test))
print(np.sqrt(mean_squared_error(y_train, pred_xgb_1)))
print(np.sqrt(mean_squared_error(y_test, pred_xgb_2)))


print("keras")
from keras.models import Sequential
from keras.layers import Dense
def keras_model():
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

model_nn = keras_model()
model_nn.fit(x=x_train, y=y_train, batch_size=512, epochs=3, verbose = False)
pred_nn_1 = model_nn.predict(x_train)
pred_nn_2 = model_nn.predict(x_test)
print(np.sqrt(mean_squared_error(y_train, pred_nn_1)))
print(np.sqrt(mean_squared_error(y_test, pred_nn_2)))



# take simple average of 5 models and generate submit file
test = pd.read_csv("test.csv")
to_submit = merge_data.loc[test_mask, ["shop_id", "item_id"]]
to_submit["item_cnt_month"] = (pred_sgd_2.clip(0 ,20) + 
                               pred_lgb_2.clip(0, 20) + 
                               pred_cat_2.clip(0, 20) +
                               pred_xgb_2.clip(0, 20) +
                               pred_nn_2.flatten().clip(0,20)) / 5

final = pd.merge(test, to_submit, on = ["shop_id", "item_id"], how = "left")
final[["ID", "item_cnt_month"]].to_csv("Predict Future Sales.csv", index = False)

