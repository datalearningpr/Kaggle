
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



# grouping features
datain["Category"] = pd.cut(datain['age'], bins=[0,25,70,100], labels=["young", "middle", "old"])

# preprocessing, transform non numeric features to numeric features
from sklearn import preprocessing

x = pd.DataFrame()  
y = pd.DataFrame()
le = preprocessing.LabelEncoder()

le.fit(pd.unique(datain["age"]))
x["age"] = le.transform(datain["age"])

# one hot coding for category data
x_train = x_train.join(pd.get_dummies(train_data["feature"], prefix = "feature").ix[:, :-1])

#####################################################################################################

# split the dataset into train dataset and test dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)  

#####################################################################################################

# summary of predictions and expected results
from sklearn import metrics

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# ROC and AUC
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(expected, predicted)
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

plt.plot(false_positive_rate,true_positive_rate)
plt.show() 

#####################################################################################################

# K Means method
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(x_train)
predicted = model.predict(x_train)

plt.scatter(x_train.ix[:, 3], x_train.ix[:, 4], c = predicted)
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1], marker='+',s=300)
plt.show()


# PCA method
from sklearn.decomposition import PCA  

pca = PCA(n_components = 5) 
pca.fit(x_train)
pca.explained_variance_
newX = pca.transform(x_train)

# SVD method
from sklearn.decomposition import TruncatedSVD  

svd = TruncatedSVD(150, n_iter = 7, random_state = 2017)
svd.fit(x_train)
svd.explained_variance_ratio_.sum()
newX = svd.transform(x_train)




####################################################################################################
# stacking method to ensemble differnt models

from sklearn.model_selection import KFold
def cross_validation_gen(k, model, x, y, x2):
    stacking_train = np.zeros(x.shape[0])
    stacking_test = np.zeros(x2.shape[0])

    kf = KFold(n_splits = k, shuffle = True, random_state = 20170101)
    for train, test in kf.split(x):     
        model.fit(x[train], y[train])
        stacking_train[test] = model.predict_proba(x[test])[:, 1]
        stacking_test = stacking_test + model.predict_proba(x2)[:, 1] / k 

    return stacking_train, stacking_test

#####  here to set all the models you want to stack #####

stack_train = []
stack_test =[]
model_list = [model, model1, model2, model3, model4, model5]

for m in model_list:
    print(m)
    a,b = cross_validation_gen(10, m, x_train,y_train, x_test)
    stack_train.append(a)
    stack_test.append(b)

final_train = np.vstack(stack_train).T
final_test = np.vstack(stack_test).T



# stacking model is LR as example, can be others
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

parameter_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000],
                  'dual': [True, False]} 
grid_search = GridSearchCV(model, param_grid = parameter_grid, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search.fit(final_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

pre_final = grid_search.predict_proba(final_test)[:, 1]


