
import re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from sklearn import preprocessing
from sklearn import metrics

# read the train dataset and test dataset
# deal with NA and missing value if possible

train_data = pd.read_csv("train.csv", dtype={"StateHoliday" : np.str}, parse_dates=['Date'])
test_data = pd.read_csv("test.csv", parse_dates=['Date'])
store = pd.read_csv("store.csv")
store_states = pd.read_csv("store_states.csv")

test_data["Open"].fillna(1, inplace = True)

filtered_data = pd.merge(train_data, test_data[["Store"]].drop_duplicates(), how = "inner", on = ["Store"])

filtered_data["test"] = 0
test_data["test"] = 1
test_data["Sales"] = 0

# remove outlier sales
mean_std = filtered_data.groupby(["Store"])["Sales"].agg(["mean", "std"]).reset_index()
merged_data = pd.merge(filtered_data, mean_std, on = ["Store"], how = "left")
merged_data["temp"] = np.abs(merged_data.Sales - merged_data["mean"]) <= 3 * merged_data["std"]

columns = list(merged_data.columns)
columns.remove("temp")
t_data = merged_data.loc[merged_data.temp == True, columns]

# only use sales > 0, which means "Open" feature is not used
t_data = t_data.loc[(t_data.Sales !=0) & (t_data.Open == 1), :]



all_data = pd.concat([t_data, test_data], axis=0)
store["CompetitionDistance"] = store["CompetitionDistance"].fillna(store["CompetitionDistance"].mean())

all_data = pd.merge(all_data, store[["Store", "StoreType", "Assortment", "Promo2"]], on = ["Store"], how = "left")

all_data["year"] = all_data.Date.dt.year
all_data["month"] = all_data.Date.dt.month
all_data["day"] = all_data.Date.dt.day

# this external data
all_data = pd.merge(all_data, store_states, on = ["Store"], how = "left")

# transfrom text to numeric label
le = preprocessing.LabelEncoder()
all_data["s_holiday"] = le.fit_transform(all_data["StateHoliday"])
all_data["StoreType_n"] = le.fit_transform(all_data["StoreType"])
all_data["Assortment_n"] = le.fit_transform(all_data["Assortment"])
all_data["State_n"] = le.fit_transform(all_data["State"])


# use embedding skill with NN for categorical features
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding

input_store = Input(shape=(1,))
output_store = Embedding(1115, 10, name='store_embedding')(input_store)
output_store = Reshape(target_shape=(10,))(output_store)

input_DayOfWeek = Input(shape=(1,))
output_DayOfWeek = Embedding(7, 6, name='DayOfWeek_embedding')(input_DayOfWeek)
output_DayOfWeek = Reshape(target_shape=(6,))(output_DayOfWeek)

input_promo = Input(shape=(1,))
output_promo = Dense(1)(input_promo)

input_year = Input(shape=(1,))
output_year = Embedding(3, 2, name='year_embedding')(input_year)
output_year = Reshape(target_shape=(2,))(output_year)

input_month = Input(shape=(1,))
output_month = Embedding(12, 6, name='month_embedding')(input_month)
output_month = Reshape(target_shape=(6,))(output_month)

input_day = Input(shape=(1,))
output_day = Embedding(31, 10, name='day_embedding')(input_day)
output_day = Reshape(target_shape=(10,))(output_day)

input_SchoolHoliday = Input(shape=(1,))
output_SchoolHoliday = Dense(1)(input_SchoolHoliday)

input_Promo2 = Input(shape=(1,))
output_Promo2 = Dense(1)(input_Promo2)

input_StateHoliday = Input(shape=(1,))
output_StateHoliday = Embedding(4, 3, name='StateHoliday_embedding')(input_StateHoliday)
output_StateHoliday = Reshape(target_shape=(3,))(output_StateHoliday)

input_StoreType = Input(shape=(1,))
output_StoreType = Embedding(4, 3, name='StoreType_embedding')(input_StoreType)
output_StoreType = Reshape(target_shape=(3,))(output_StoreType)


input_Assortment = Input(shape=(1,))
output_Assortment = Embedding(3, 2, name='Assortment_embedding')(input_Assortment)
output_Assortment = Reshape(target_shape=(2,))(output_Assortment)


input_State = Input(shape=(1,))
output_State = Embedding(8, 7, name='State_embedding')(input_State)
output_State = Reshape(target_shape=(7,))(output_State)

input_model = [input_store, input_DayOfWeek, input_promo,
            input_year, input_month, input_day
            ,input_SchoolHoliday, input_Promo2, input_StateHoliday, input_StoreType, input_Assortment
            , input_State]

output_embeddings = [output_store, output_DayOfWeek, output_promo,
                    output_year, output_month, output_day
                    ,output_SchoolHoliday, output_Promo2, output_StateHoliday, output_StoreType, output_Assortment
                    ,output_State]

output_model = Concatenate()(output_embeddings)
output_model = Dense(1024, kernel_initializer="uniform")(output_model)
output_model = Activation('relu')(output_model)
output_model = Dense(512, kernel_initializer="uniform")(output_model)
output_model = Activation('relu')(output_model)
output_model = Dense(1)(output_model)

model = Model(inputs=input_model, outputs=output_model)
model.compile(loss="mse", optimizer='adam')

features = ['Store', 'DayOfWeek', 'Promo', 'Open', 'year', 'month', 'day', 'test', "Sales", 'SchoolHoliday','Promo2', 's_holiday', 'StoreType_n', 'Assortment_n', "State_n"]
train_features = ['Store', 'DayOfWeek', 'Promo', 'year', 'month', 'day', 'SchoolHoliday','Promo2', 's_holiday', 'StoreType_n', 'Assortment_n', "State_n"]

nn_data = all_data.loc[:, features]

# for embedding, the label should be starting from 0
nn_data.loc[:, "day"] = nn_data.loc[:, "day"] - 1
nn_data.loc[:, "DayOfWeek"] = nn_data.loc[:, "DayOfWeek"] - 1
nn_data.loc[:, "month"] = nn_data.loc[:, "month"] - 1
nn_data.loc[:, "year"] = nn_data.loc[:, "year"] - 2013
nn_data.loc[:, "Store"] = nn_data.loc[:, "Store"] - 1


x_train = nn_data.loc[nn_data.test == 0, train_features]
x_test = nn_data.loc[nn_data.test == 1, train_features]
y_train = nn_data.loc[nn_data.test == 0, "Sales"]

# format to fit the NN input
x_train_list = list(np.array(x_train).T)
x_test_list = list(np.array(x_test).T)


# this is the valuation method used in this game
def rmspe(yhat, y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

# training
model.fit(x_train_list, y_train, epochs= 15, batch_size=512)
nn_pred = np.ravel(model.predict(x_train_list, batch_size=512))
nn_pred[nn_data[nn_data.test==0].Open == 0] = 0
print(rmspe(y_train, nn_pred.clip(0, np.max(y_train))))


# testing
nn_pred_2 = np.ravel(model.predict(x_test_list, batch_size=512))
nn_pred_2[nn_data[nn_data.test==1].Open == 0] = 0

# prepare submission
to_submit = all_data.loc[all_data.test==1, ["Id"]]
to_submit["Id"] = to_submit["Id"].astype("int")
to_submit["Sales"] = nn_pred_2.clip(0, np.max(y_train))
to_submit.to_csv("Rossmann Store Sales.csv", index = False)
