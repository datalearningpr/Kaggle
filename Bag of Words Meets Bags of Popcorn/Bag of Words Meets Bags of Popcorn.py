
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

train_data = pd.read_csv("labeledTrainData.tsv", sep='\t')
unlabeled_train_data = pd.read_csv("unlabeledTrainData.tsv", sep='\t', quoting=3)
test_data = pd.read_csv("testData.tsv", sep='\t')



######################################################################
# this part is the tutorial of this game, which will give you an idea 
# of how to do NLP, bags of wrods, word2vec methods
# tf-idf method also included here, although it is not in the tutorial
# it is also a very popular method dealing with text

def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review, "lxml").get_text()

    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)


train_data['clean_review'] = train_data.apply(lambda x: review_to_wordlist(x.review, True), axis=1)
test_data['clean_review'] = test_data.apply(lambda x: review_to_wordlist(x.review, True), axis=1)



vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(train_data['clean_review'])
train_data_features = train_data_features.toarray()


test_data_features = vectorizer.transform(test_data['clean_review'])
test_data_features = test_data_features.toarray()


vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print(count, tag)



# this is the tf-idf method
train_text = list(np.array(train_data['review']))
test_text = list(np.array(test_data['review']))

tfv = TfidfVectorizer(min_df = 3,  max_features = None, strip_accents = 'unicode',  
        analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 2), use_idf = 1, smooth_idf = 1, sublinear_tf = 1)

train_data_features = tfv.fit_transform(train_text)
test_data_features = tfv.transform(test_text)



x_train = train_data_features
x_test = test_data_features
y_train = train_data['sentiment']


# tow simple methods are tested, LR, NB
# although the tutorial is using RF, but it is too slow

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
parameter_grid = {  'C': [0.1],
                    'dual': [True]}

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 5, n_jobs = -1, scoring="roc_auc")
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

model = MultinomialNB()
parameter_grid = {'alpha' : [0.1]}
grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = 5, n_jobs = -1, scoring="roc_auc")
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



# predict the output

output = grid_search.predict_proba(x_test)[:,1]
final_output = output

submit_data = pd.read_csv("sampleSubmission.csv")
submit_data['sentiment'] = final_output
submit_data.to_csv("Bag_of_Words_Meets_Bags_of_Popcorn_final.csv", index = 0)









# this part is using the word2vec method

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

sentences = []  
for review in train_data["review"]:
    sentences += review_to_sentences(review, tokenizer)

for review in unlabeled_train_data["review"]:
    sentences += review_to_sentences(review, tokenizer)


num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 0.001   # Downsample setting for frequent words

from gensim.models import word2vec

# model = word2vec.Word2Vec.load("300features_40minwords_10context")

model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)


model.most_similar("awful")
model.most_similar("man")
model.doesnt_match("france england germany berlin".split())

import numpy as np  # Make sure that numpy is imported

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec



def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs


train_data_features = getAvgFeatureVecs( train_data["clean_review"], model, num_features )
test_data_features = getAvgFeatureVecs( test_data["clean_review"], model, num_features )

x_train = train_data_features
x_test = test_data_features
y_train = train_data['sentiment']


###############################################################
# for model fit and prediction, can reuse the code above------
###############################################################


# this part is using the wrod2vec clustering
from sklearn.cluster import KMeans
import time

start = time.time()

word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] // 5

kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

word_centroid_map = dict(zip( model.wv.index2word, idx ))


def create_bag_of_centroids( wordlist, word_centroid_map ):

    num_centroids = max( word_centroid_map.values() ) + 1

    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids


train_centroids = np.zeros( (train_data["review"].size, num_clusters), \
    dtype="float32" )


counter = 0
for review in train_data["clean_review"]:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( test_data["review"].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in test_data["clean_review"]:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

x_train = train_centroids
x_test = test_centroids
y_train = train_data['sentiment']

###############################################################
# for model fit and prediction, can reuse the code above------
###############################################################



#-----------------------------------------------------------------------
# below is using NN methods
# two methods are tesed, simple word embedding, word embedding + CNN




# simple word embedding method
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

num_words = 35000       # unique number of words used to vectorize the text
sequence_len = 800      # to make each sentence to same length
embedding_dims = 100     # word embedding size


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_data["review"])
sequences = tokenizer.texts_to_sequences(train_data["review"])

test_sequences = tokenizer.texts_to_sequences(test_data["review"])


def vec_y(y):
    if y == 0:
        return np.array([1, 0])
    elif y == 1:
        return np.array([0, 1])

x_train = pad_sequences(sequences, maxlen=sequence_len)
x_test = pad_sequences(test_sequences, maxlen=sequence_len)
y_train = np.vstack(train_data['sentiment'].apply(vec_y))

model = Sequential()
model.add(Embedding(num_words,
                    embedding_dims,
                    input_length=sequence_len))
model.add(GlobalAveragePooling1D())

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=15)

model.save('Fast_text_model.h5')

pred = model.predict(np.asmatrix(x_test))
index = pred[:,1]
final_output = index

submit_data = pd.read_csv("sampleSubmission.csv")
submit_data['sentiment'] = final_output
submit_data.to_csv("Bag_of_Words_Meets_Bags_of_Popcorn_final.csv", index = 0)




# word embedding + CNN

from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding

num_words = 35000       # unique number of words used to vectorize the text
sequence_len = 800      # to make each sentence to same length
embedding_dims = 100    # word embedding size
filters = 250           # how many CONV
kernel_size = 3         # CONV size
hidden_dims = 256       # how many hidden units in the fully connected network

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
model.add(Dense(2))
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


