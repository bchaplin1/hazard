# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:41:27 2015

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
feature_cols  = ['T1_V5_E','T1_V8_C','T1_V15_C','T1_V4_N','T2_V15','T1_V12_C','T1_V11_H','T1_V8_C','T1_V5_K','T1_V2','T1_V9_E','T2_V1','T2_V2','T2_V9','T1_V2']
# score 0.08, gini .253 public LB of .286345

# instantiate and fit
model = LinearRegression()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train['Hazard'], random_state=1)


#fit the model and predict
model.fit(X_train,y_train)
# print the coefficients
print model.intercept_
print model.coef_

model.score(X_test,y_test)

y_pred =model.predict(X_test)

coef = Gini(y_pred,y_test)

print 'Gini coefficient is ', coef


X_oos = test[feature_cols]
oos_pred = model.predict(X_oos)

# create submission file
sub = pd.DataFrame({'Id':test.index, 'Hazard':oos_pred}).set_index('Id')
sub.to_csv('linreg1.csv')  # 0.864



