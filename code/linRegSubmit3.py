# -*- coding: utf-8 -*-
"""
Created on Aug 1, 2015
run LassoCV then removed 2 that had high collinearity
 ['T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N','T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E','T1_V9_F','T1_V11_E','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C','T1_V15_W','T1_V16_J','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D']
train/test split  score .896021, gini on .263 
 public LB of 0.305399
@author: brian
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


# update both files with features
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    #one-hot the categoricals
    return pd.get_dummies(df)

# assume you're in code or data directory
# apply function to both training and testing files
train = make_features("../data/train.csv")
test = make_features("../data/test.csv")


#fit the model and predict
feature_cols =  ['T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N','T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E','T1_V9_F','T1_V11_E','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C','T1_V15_W','T1_V16_J','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D']



# instantiate and fit
model = LinearRegression()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train['Hazard'], random_state=1)


#fit the model and predict
model.fit(X_train,y_train)
# print the coefficients

model.score(X_test,y_test)
#.896021

y_pred =model.predict(X_test)

coef = Gini(y_pred,y_test)

print 'Gini coefficient is ', coef
# 0.263

#re-fit the model
model.fit(train[feature_cols],train['Hazard'])

X_oos = test[feature_cols]
oos_pred = model.predict(X_oos)

# create submission file
sub = pd.DataFrame({'Id':test.index, 'Hazard':oos_pred}).set_index('Id')
sub.to_csv('linreg3.csv')  



