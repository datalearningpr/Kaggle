
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.decomposition import PCA



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


feature_list=list(all_data.columns)
feature_list.remove("label")

std_scaler = preprocessing.StandardScaler()
all_data_scaled = std_scaler.fit_transform(all_data[feature_list])

pca = PCA(n_components = 90)
pca.fit(all_data_scaled)
print(sum(pca.explained_variance_ratio_))

all_data_pca = pca.transform(all_data_scaled)


x_train = all_data_pca[np.where(-all_data["label"].isnull())[0]]
y_train = all_data.loc[-all_data["label"].isnull(), "label"]
x_test = all_data_pca[np.where(all_data["label"].isnull())[0]]


# frist method is to use the random forest, since the data is big, gridsearchcv is slow

model = RandomForestClassifier(max_features = 'sqrt')

# we use GridSearchCV to get the best parameters for model
parameter_grid = {'max_depth' : [7],
                  'n_estimators': [300, 500],
                  'criterion': ['entropy', 'gini']}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 4, n_jobs = -1)

# train the model with dataset after setting the parameters
grid_search.fit(x_train[0:10000], y_train[0:10000])

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

output = model.predict_classes(np.asmatrix(x_test))

final_output = pd.read_csv("sample_submission.csv")
final_output["Label"]=output

final_output.to_csv("DigitRecognizer_predicted_randomforest.csv", index = 0)

####################################################################################################


# second method is to use KNN

model = neighbors.KNeighborsClassifier(5)
model.fit(x_train, y_train)

output = model.predict(x_test).astype(int)

final_output = pd.read_csv("sample_submission.csv")
final_output["Label"]=output

final_output.to_csv("DigitRecognizer_predicted_KNN.csv", index = 0)



####################################################################################################

# third, use the NN, all connected type

x_train = all_data.loc[-all_data["label"].isnull(), feature_list]
y_train_pre = all_data.loc[-all_data["label"].isnull(), "label"]
x_test = all_data.loc[all_data["label"].isnull(), feature_list]

y_train = pd.get_dummies(y_train_pre.astype(int), prefix = 'num')
x_train = x_train / 255
x_test = x_test / 255

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))   
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy',metrics=["accuracy"], optimizer='adam') 

model.fit(np.asmatrix(x_train), np.asmatrix(y_train),
          batch_size=128, nb_epoch=7,
          verbose=1)

output = model.predict_classes(np.asmatrix(x_test))

final_output = pd.read_csv("sample_submission.csv")
final_output["Label"]=output

final_output.to_csv("DigitRecognizer_predicted_NN.csv", index = 0)


####################################################################################################

# fourth, use the CNN, best for image processing

x_train = all_data.loc[-all_data["label"].isnull(), feature_list]
y_train_pre = all_data.loc[-all_data["label"].isnull(), "label"]
x_test = all_data.loc[all_data["label"].isnull(), feature_list]

y_train = pd.get_dummies(y_train_pre.astype(int), prefix = 'num')
x_train = x_train / 255
x_test = x_test / 255

x_train_shaped = np.reshape(np.asarray(x_train), (len(x_train), 28, 28, 1))
x_test_shaped = np.reshape(np.asarray(x_test), (len(x_test), 28, 28, 1))

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

nb_conv = 5 
nb_pool = 2

model = Sequential()

model.add(Conv2D(32, nb_conv, nb_conv, input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy',metrics=["accuracy"], optimizer='adam') 

model.fit(x_train_shaped, y_train,
          batch_size=256, epochs=4,
          verbose=1)

output = model.predict_classes(x_test_shaped)

final_output = pd.read_csv("sample_submission.csv")
final_output["Label"]=output

final_output.to_csv("DigitRecognizer_predicted_CNN.csv", index = 0)


