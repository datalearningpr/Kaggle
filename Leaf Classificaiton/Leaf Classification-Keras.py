

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
        




feature_list = train_data.columns.tolist()
feature_list.remove("id")
feature_list.remove("species")

sc = preprocessing.StandardScaler()
le = preprocessing.LabelEncoder()

x_train_num = sc.fit_transform(train_data[feature_list])
x_test_num = sc.transform(test_data[feature_list])





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

image = Input(shape=(96, 96, 1), name='image')

net = Conv2D(8, nb_conv, nb_conv, input_shape=(96, 96, 1))(image)
net = Activation('relu')(net)
net = Conv2D(32, nb_conv, nb_conv)(net)
net = Activation('relu')(net)
net = MaxPooling2D(pool_size=(nb_pool, nb_pool))(net)
net = Dropout(0.2)(net)

net = Flatten()(net)

net = Dense(512)(net)
net = Activation('relu')(net)
net = Dropout(0.2)(net)

net = Dense(128)(net)
net = Activation('relu')(net)
net = Dropout(0.2)(net)

net = Dense(16)(net)
net = Activation('relu')(net)
net = Dropout(0.2)(net)

numerical = Input(shape=(192,), name='numerical')
concatenated = merge([net, numerical], mode='concat')

net = Dense(1024)(concatenated)
net = Activation('relu')(net)
net = Dropout(0.2)(net)
net = Dense(512)(net)
net = Activation('sigmoid')(net)
net = Dropout(0.2)(net)
net = Dense(99)(net)
out = Activation('softmax')(net)

model = Model(inputs=[image, numerical], outputs=out)
model.compile(loss='categorical_crossentropy',metrics=["accuracy"], optimizer='rmsprop') 

model.fit([x_train, x_train_num], np.asmatrix(y_train),
          batch_size=32, epochs=6,
          verbose=1, )

output = model.predict([x_test, x_test_num])

submit_data = pd.read_csv("sample_submission.csv")
column_list = submit_data.columns.tolist()
column_list.remove('id')
submit_data.ix[:, column_list] = output

submit_data.to_csv("Leaf_Classification_Keras.csv", index = 0)

