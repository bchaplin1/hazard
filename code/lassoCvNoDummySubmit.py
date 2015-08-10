# -*- coding: utf-8 -*-
"""
Created on Sat Aug 01 17:45:55 2015
run LassoCV on whole training set to get the feature variables it picks
Then use these features for further analysis, looking at correlations
Assume non-dummy variables is the best way to encode
score of .05468 dont' bother, much less than get_dummies


@author: brian
"""
import pandas as pd
from sklearn import preprocessing


# update both files with features
def make_features2(filename):
    df = pd.read_csv(filename, index_col=0)
    columns = df.dtypes[train.dtypes == 'object'].keys()
    for column in columns:
        lbl = preprocessing.LabelEncoder()
        df[column] = lbl.fit_transform(train[column])
    return df;

# assume you're in code or data directory
# apply function to both training and testing files
train = make_features2("../data/train.csv")
test = make_features2("../data/test.csv")


from sklearn import linear_model
feature_cols = train.columns
# exclude the response
feature_cols = feature_cols.drop('Hazard')


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train['Hazard'], random_state=1)

lassoModel = linear_model.LassoCV()
#fit the model and predict
lassoModel.fit(X_train,y_train)
# print the coefficients
lassoModel.coef_.size

lassoModel.score(X_test,y_test)
#.05468 dont' bother, much less than get_dummies

