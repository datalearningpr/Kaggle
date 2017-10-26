

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
from sklearn.model_selection import KFold, StratifiedKFold

from scipy.stats import boxcox
from scipy.stats import skew

from sklearn.metrics import log_loss
from sklearn.metrics import classification_report


# read the train dataset and test dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")



feature_list = train_data.columns.tolist()
feature_list.remove("id")
feature_list.remove("species")

sc = preprocessing.StandardScaler()
le = preprocessing.LabelEncoder()

x_train = sc.fit_transform(train_data[feature_list])
y_train = le.fit_transform(train_data['species'])
x_test = sc.transform(test_data[feature_list])



############################################################################################

# first method is purely using the feature data from the train dataset, without using the images
# use the most basic one LR

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')

parameter_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000]} 
grid_search = GridSearchCV(model, param_grid = parameter_grid, scoring = 'neg_log_loss', cv = 5, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

output = grid_search.predict_proba(x_test)

submit_data = pd.read_csv("sample_submission.csv")
column_list = submit_data.columns.tolist()
column_list.remove('id')
submit_data.ix[:, column_list] = output

submit_data.to_csv("Leaf_Classification_LR.csv", index = 0)

############################################################################################




# second method is using the NN to apply on the feature only dataset without using any images

y_train_NN = pd.get_dummies(train_data['species'])

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

model = Sequential()

model.add(Dense(1024, input_shape=(192,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(99))
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy',metrics=["accuracy"], optimizer='rmsprop') 

model.fit(np.asmatrix(x_train), np.asmatrix(y_train_NN),
          batch_size=32, epochs=12,
          verbose=1)


output = model.predict_proba(x_test)

submit_data = pd.read_csv("sample_submission.csv")
column_list = submit_data.columns.tolist()
column_list.remove('id')
submit_data.ix[:, column_list] = output

submit_data.to_csv("Leaf_Classification_NN.csv", index = 0)

