
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

from sklearn.cluster import MiniBatchKMeans
from math import radians, cos, sin, asin, sqrt  

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from bs4 import BeautifulSoup  
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk.data

# read the train dataset and test dataset
# deal with NA and missing value if possible

train_data = pd.read_csv("train.tsv", sep='\t')
test_data = pd.read_csv("test.tsv", sep='\t')


# directy use the DeepNLP method
# use word embedding + CNN
# it will give very good result already
# can consider use it as base model and apply ensemble methods

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

from keras.layers import Conv1D, MaxPooling1D, Embedding


num_words = 20000       # unique number of words used to vectorize the text
sequence_len = 50       # to make each sentence to same length
embedding_dims = 30     # word embedding size
filters = 250           # how many CONV
kernel_size = 3         # CONV size
hidden_dims = 256       # how many hidden units in the fully connected network



tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_data["Phrase"])
sequences = tokenizer.texts_to_sequences(train_data["Phrase"])

test_sequences = tokenizer.texts_to_sequences(test_data["Phrase"])


def vec_y(y):
    if y == 0:
        return np.array([1, 0, 0, 0, 0])
    elif y == 1:
        return np.array([0, 1, 0, 0, 0])
    elif y == 2:
        return np.array([0, 0, 1, 0, 0])
    elif y == 3:
        return np.array([0, 0, 0, 1, 0])
    elif y == 4:
        return np.array([0, 0, 0, 0, 1])


x_train = pad_sequences(sequences, maxlen=sequence_len)
x_test = pad_sequences(test_sequences, maxlen=sequence_len)
y_train = np.vstack(train_data['Sentiment'].apply(vec_y))


model = Sequential()
model.add(Embedding(num_words,
                    embedding_dims,
                    input_length=sequence_len))
model.add(Dropout(0.2))

model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))

model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=3)


pred = model.predict(np.asmatrix(x_test))
final_output = pred[:,1]

submit_data = pd.read_csv("sampleSubmission.csv")
submit_data['sentiment'] = final_output
submit_data.to_csv("Bag_of_Words_Meets_Bags_of_Popcorn_final.csv", index = 0)

