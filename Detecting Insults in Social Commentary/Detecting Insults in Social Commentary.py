
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, IncrementalPCA

# read the train dataset and test dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_text = list(np.array(train_data.ix[:, 2]))
test_text = list(np.array(test_data.ix[:, 2]))

tfv = TfidfVectorizer(min_df = 3,  max_features = None, strip_accents = 'unicode',  
        analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 2), use_idf = 1, smooth_idf = 1, sublinear_tf = 1)

transformed_text = tfv.fit_transform(train_text + test_text)


# pca = IncrementalPCA(5000)
# pca_text = pca.fit_transform(transformed_text.toarray())

# x_train = pca_text[:len(train_text)]
# x_test = pca_text[len(train_text):]
# y_train = train_data['Insult']


# pca and svd are both fine here to reduce the Dimensionality 
svd = TruncatedSVD(5000, n_iter = 7, random_state = 2017)
svd_text = svd.fit_transform(transformed_text)
svd.explained_variance_ratio_.sum()

x_train = svd_text[:len(train_text)]
x_test = svd_text[len(train_text):]
y_train = train_data['Insult']



# LR is the best sigle model in this case
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
parameter_grid = {
        'C': np.linspace(0.01, 1, 10)
        }
#[0.0001, 0.001, 0.01, 1, 10, 100, 1000]
grid_search = GridSearchCV(model, param_grid = parameter_grid, scoring = 'roc_auc', cv = 5, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# test out with the real test data
predict = grid_search.predict_proba(x_test)
solution_data = pd.read_csv("impermium_verification_labels.csv")
metrics.roc_auc_score(solution_data['Insult'], predict[:, 1])



