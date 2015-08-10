# -*- coding: utf-8 -*-
"""
Created on aug 2, 2015

using LassoCV 32 the gini score is only .19 to .21 so it must want all the columns or feature engineering

best AdaBoost is only 0.188 gini
making the base estimator RandomForestRegressor made it worse than default DecisionTreeRegressor
@author: brian
"""

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import RandomForestRegressor
import giniscore as giniscore

# update both files with features
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    #one-hot the categoricals
    return pd.get_dummies(df)

# assume you're in code or data directory
# apply function to both training and testing files
train = make_features("../data/train.csv")
test = make_features("../data/test.csv")

feature_cols = train.columns
# exclude the response
feature_cols = feature_cols.drop('Hazard')

trn2=train.sample(10000).copy()
#trn2=train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trn2[feature_cols], trn2['Hazard'], random_state=1)


#fit the model and predict
# model = AdaBoostRegressor(base_estimator=RandomForestRegressor())
model = AdaBoostRegressor()
model.fit(X_train,y_train)
y_pred =model.predict(X_test)

coef = giniscore.Gini(y_pred,y_test)

print 'Gini coefficient is ', coef

model.score(X_train,y_train)
# score with 100 rows RF estimator is .92

# gini with default columns, default estimator is 0.12 
# gini with 1000 rows all columns, default estimater is 0.188
# gini with 10000 rows all columns, default estimater is 0.1802
# gini with all rows all columns, default estimater is 0.12759

# gini with 100 rows RF esimator is .098
# gini with 1000 rows RF estimator is .0876

# ugh, using LassoCV 32 the score is only .19 to 21 so it must want all the columns

# benchmark is .20 , Kaggle public LB says .263387
# < 14 97.5% benchmark is .172
# < 10 90.5% benchmark is .1472
# < 30 99.88% benchmark is .228 but Kaggle public LB says 0.262842, worse


featureImportances = zip(train.columns, model.feature_importances_)

featureImportancesSorted =sorted(featureImportances,key=lambda x: x[1], reverse=True)

featureImportancesSorted


