

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
import os, sys
from PIL import Image, ImageOps


# read the train dataset and test dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


size = 96, 96

new_img = []
for i in range(1, 1585):
    im = Image.open('images\\{}.jpg'.format(str(i)))

    img = im.resize(size, Image.ANTIALIAS)
    new_img.append(np.array(img)/255)
    

train_pic = np.asarray(new_img)[train_data["id"] - 1]
test_pic = np.asarray(new_img)[test_data["id"] - 1]


x_train = np.reshape(train_pic, (len(train_pic), 96, 96, 1))
x_test = np.reshape(test_pic, (len(test_pic), 96, 96, 1))
y_train = pd.get_dummies(train_data['species'])
        
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import np_utils

nb_conv = 5 
nb_pool = 2

model = Sequential()

model.add(Conv2D(32, nb_conv, nb_conv, input_shape=(96, 96, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(99))
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy',metrics=["accuracy"], optimizer='rmsprop') 

model.fit(x_train, np.asmatrix(y_train),
          batch_size=32, epochs=15,
          verbose=1, )

output = model.predict_proba(x_test)

submit_data = pd.read_csv("sample_submission.csv")
column_list = submit_data.columns.tolist()
column_list.remove('id')
submit_data.ix[:, column_list] = output

submit_data.to_csv("Leaf_Classification_CNN.csv", index = 0)
