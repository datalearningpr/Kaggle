
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from nameparser import HumanName 

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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

# we found out there are 
# 263 missing values for Age
# 1014 missing values for Cabin
# 2 missing values for Embarked
# 1 missing values for Fare


# fill the NA values
# since Embarked and Fare have so few missing values, we just fill in with mode or median
all_data.ix[all_data["Embarked"].isnull(), "Embarked"] = all_data["Embarked"].dropna().mode().values
all_data.ix[all_data["Fare"].isnull(), "Fare"] = all_data["Fare"].dropna().median()


# Cabin has too many missing values, we are not going to estimate them
# Age has certain amount of missing values as well, so cannot just fill them all with median


# we can explore the existing features to see how useful they are towards Survived

explore_data = all_data[-all_data["Survived"].isnull()]

# first, we can explore the surviving status
explore_data.Survived.value_counts().plot(kind = 'bar')
plt.show()

# then we can explore the Embarked
explore_data.Embarked.value_counts().sort_index().plot(kind = "bar")
plt.show()
survived_0 = explore_data.Embarked[explore_data.Survived == 0].value_counts()
survived_1 = explore_data.Embarked[explore_data.Survived == 1].value_counts()
df=pd.DataFrame({'live': survived_1, 'dead': survived_0})
df.plot(kind='bar', stacked = True)
plt.show()

# then we can explore the Sex
explore_data.Sex.value_counts().sort_index().plot(kind = "bar")
plt.show()
survived_0 = explore_data.Sex[explore_data.Survived == 0].value_counts()
survived_1 = explore_data.Sex[explore_data.Survived == 1].value_counts()
df=pd.DataFrame({'live': survived_1, 'dead': survived_0})
df.plot(kind='bar', stacked = True)
plt.show()



# the rest of features exploration requires feature engineering first

# for feature name, we can get the title of the name so that it might indicate the age or social status
all_data["title"] = all_data["Name"].apply(lambda x: re.split("[,.]", x)[1].strip())
all_data.ix[all_data["title"].apply(lambda x: x.strip() in ["Mme", "Mlle"]),"title"] = 'Mlle'
all_data.ix[all_data["title"].apply(lambda x: x.strip() in ['Capt', 'Don', 'Major', 'Sir']),"title"] = 'Sir'
all_data.ix[all_data["title"].apply(lambda x: x.strip() in ['Dona', 'Lady', 'the Countess', 'Jonkheer']), "title"] = 'Lady'

# with the new feature title, we can fill age NAs based on titles
age_by_title = all_data[-all_data["Age"].isnull()].groupby(["title"])["Age"].median()
all_data.loc[all_data["Age"].isnull(), "Age"] = all_data.loc[all_data["Age"].isnull(), "title"].apply(lambda x: age_by_title[x])

# we can get familysize with Parch and SibSp
all_data["family_size"] = all_data["Parch"] + all_data["SibSp"] + 1


# for Age feature, we can group them into bins so that they can represent different age groups
# age to divide age group is up to users
all_data["age_0-5"] = (all_data["Age"] <= 5)
all_data["age_6-18"] = (all_data["Age"] > 5) & (all_data["Age"] <= 18)
all_data["age_19-60"] = (all_data["Age"] > 18) & (all_data["Age"] <= 60)
all_data["age_61"] = (all_data["Age"] > 60)


explore_data = all_data[-all_data["Survived"].isnull()]

# we can explore the family size further
explore_data.family_size.value_counts().sort_index().plot(kind = "bar")
plt.show()
survived_0 = explore_data.family_size[explore_data.Survived == 0].value_counts()
survived_1 = explore_data.family_size[explore_data.Survived == 1].value_counts()
df=pd.DataFrame({'live': survived_1, 'dead': survived_0})
df.plot(kind='bar', stacked = True)
plt.show()

# we can see that family size with 2, 3, 4 have higer surviving rate, but travel alone is very low

all_data["travel_alone"]=all_data["family_size"]==1
all_data["travel_smallgroup"]=(all_data["family_size"]>1) & (all_data["family_size"]<5)
all_data["travel_largegroup"]=(all_data["family_size"]>=5)

all_data["has_cabin"]=-all_data["Cabin"].isnull()
all_data["cabin_num"]=all_data["Cabin"].map(lambda x: '' if x is np.nan else x[0])
all_data["cabin_highchance"]=(all_data["cabin_num"] == 'B') | (all_data["cabin_num"] == 'D') | (all_data["cabin_num"] == 'E')

# we can make categorical data into binary data 
all_data = all_data.join(pd.get_dummies(all_data["Embarked"], prefix = "Embarked"))
all_data = all_data.join(pd.get_dummies(all_data["title"], prefix = "title"))
all_data = all_data.join(pd.get_dummies(all_data["Pclass"], prefix = "Pclass"))

# we need to make string type data into number
all_data["sex_num"]=pd.factorize(all_data["Sex"])[0]

# we can make the fare to bins as well
all_data["fare_num"]=pd.qcut(all_data["Fare"],3,labels=[1,2,3])
all_data=all_data.join(pd.get_dummies(all_data["fare_num"], prefix="fare_num"))


# feature list we came up with

feature_list=['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'family_size', 'age_0-5', 'age_6-18', 
             'age_19-60', 'age_61', 'travel_alone', 'travel_smallgroup', 'travel_largegroup', 'has_cabin',
             'cabin_highchance', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'title_Col', 'title_Dr', 'title_Lady',
             'title_Master', 'title_Miss', 'title_Mlle', 'title_Mr', 'title_Mrs', 'title_Ms', 'title_Rev', 'title_Sir',
             'Pclass_1', 'Pclass_2', 'Pclass_3', 'sex_num', 'fare_num', 'fare_num_1', 'fare_num_2', 'fare_num_3']

# make boolean to be int
for col in feature_list:
    if all_data[col].dtype=='bool':
        all_data[col]=all_data[col].astype(int)

# we need to scale the features
x_train = all_data.loc[-all_data["Survived"].isnull(), feature_list].apply(lambda x: x / x.max(), axis = 0)
y_train = all_data.loc[-all_data["Survived"].isnull(), "Survived"]
x_test = all_data.loc[all_data["Survived"].isnull(), feature_list].apply(lambda x: x / x.max(), axis = 0)


# select the best 15 features to use
k_best = SelectKBest(k = 15).fit(x_train, y_train)
selected_features = list(x_train.columns[k_best.get_support()])

x_train_selected = x_train.ix[:, selected_features]
x_test_selected = x_test.ix[:, selected_features]


# we are using Random Forest to classify
model = RandomForestClassifier(max_features = 'sqrt')


# we use GridSearchCV to get the best parameters for model
parameter_grid = {'max_depth' : [4,5,6,7,8],
                  'n_estimators': [200, 240, 250, 350],
                  'criterion': ['gini','entropy']}

cross_validation = StratifiedKFold(y_train, n_folds=5)

grid_search = GridSearchCV(model, param_grid = parameter_grid, cv = cross_validation)

# train the model with dataset after setting the parameters
grid_search.fit(x_train_selected, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# predict the classification
output = grid_search.predict(x_test_selected).astype(int)

# write output in csv
final_output = pd.DataFrame(list(zip(test_data["PassengerId"],output)), columns = ["PassengerId","Survived"])
final_output.to_csv("Titanic_predicted_final.csv", index = 0)

# *****************************************************************
# As it is "Random" Forest, each time the model might be different 
# even running on the same dataset, so best of outcome have seen is
# 0.81340
# *****************************************************************
