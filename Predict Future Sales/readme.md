# Predict Future Sales
##### Regression
This one can be a good game for learning **feature engineering** in regression.

The following basic skills shall be picked up after playing this game:
* **Target Encoding**, this is one of the most powerful encoding skills which can be applied to categorical features, it gives more relationship information between the feature and target. Unfortunately, the dataset in this game is very large, some of the target encodings are very slow so that they are not tested in this game. But only using shop_id and category_id already gives good improvements.
* Most of the time, we treat time series as time series. But in this game, we need to predict tons of time series. The forecast we are using is finding each time point its features to predict the sales using historical sales as part of the features.
* Due to large volume of data, try to use batch training methods like sgd and NN, so that memory and speed will not be an issue.
* Although some of the time point, there might not be any sales, we still want to have that record in the dataset with sales to be 0. So, when doing target encoding or scaling, it will tell more information between the features and target.
* There are skills not used in this game that might help as well, such as 2 level of ensembling. In this game, you can use first layer model to predict each time point with some historical data, then use the predicted data to train second layer model to the final target.
* Simple average of 5 models with good features can get the loss lower than 1.00 already, which will give you high position already.
###### Hope this script will be helpful
