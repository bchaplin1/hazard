# -*- coding: utf-8 -*-
"""
Created on Sat Aug 01 17:45:55 2015
run LassoCV on whole training set to get the feature variables it picks
Then use these features for further analysis, looking at correlations
Assume dummy variables is the best way to encode

public leader board 0.325017
# 0.27497 gini on train

['T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N','T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E','T1_V9_F','T1_V11_E','T1_V11_H','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C','T1_V15_W','T1_V16_J','T1_V16_K','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D']
@author: brian
"""
import pandas as pd


# update both files with features
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    #one-hot the categoricals
    return pd.get_dummies(df)

# assume you're in code or data directory
# apply function to both training and testing files
train = make_features("../data/train.csv")
test = make_features("../data/test.csv")
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

lassoModel.score(X_test,y_test)
#.0974

y_pred =lassoModel.predict(X_test)

coef = Gini(y_pred,y_test)

print 'Gini coefficient is ', coef
# 0.273


lassoModel = linear_model.LassoCV()
lassoModel.fit(train[feature_cols],train['Hazard'])
lassoModel.coef_



y_pred2 =lassoModel.predict(train[feature_cols])

coef2 = Gini(y_pred2,train['Hazard'])

print 'Gini coefficient is ', coef2
# 0.27497
coeff = zip(feature_cols,lassoModel.coef_)

feats = [feature[0] for feature in coeff if feature[1] >0 ]
# save these for further analysis and feature evaluation
feats


X_oos = test[feature_cols]
oos_pred = lassoModel.predict(X_oos)

# create submission file
sub = pd.DataFrame({'Id':test.index, 'Hazard':oos_pred}).set_index('Id')
sub.to_csv('linLassoCv.csv')  # 0.864
