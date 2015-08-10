# -*- coding: utf-8 -*-
"""
Created on Aug 1, 2015
run LassoCV then removed 4 that had high p-values
[u'T1_V1', u'T1_V2', u'T1_V3', u'T2_V14', u'T2_V15', u'T1_V4_H',
       u'T1_V4_N', u'T1_V4_W', u'T1_V5_B', u'T1_V5_D', u'T1_V5_K', u'T1_V7_B',
       u'T1_V7_C', u'T1_V8_C', u'T1_V9_E', u'T1_V9_F', u'T1_V11_E',
       u'T1_V11_M', u'T1_V12_C', u'T1_V15_A', u'T1_V15_C', u'T1_V15_W',
       u'T1_V17_N', u'T2_V5_D', u'T2_V5_E', u'T2_V11_N', u'T2_V13_B',
       u'T2_V13_D']

train/test split  score 0.08868, gini on .2618 
 public LB of 0.305564
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
feature_cols = [u'T1_V1', u'T1_V2', u'T1_V3', u'T2_V14', u'T2_V15', u'T1_V4_H',
       u'T1_V4_N', u'T1_V4_W', u'T1_V5_B', u'T1_V5_D', u'T1_V5_K', u'T1_V7_B',
       u'T1_V7_C', u'T1_V8_C', u'T1_V9_E', u'T1_V9_F', u'T1_V11_E',
       u'T1_V11_M', u'T1_V12_C', u'T1_V15_A', u'T1_V15_C', u'T1_V15_W',
       u'T1_V17_N', u'T2_V5_D', u'T2_V5_E', u'T2_V11_N', u'T2_V13_B',
       u'T2_V13_D']


# instantiate and fit
model = LinearRegression()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train['Hazard'], random_state=1)


#fit the model and predict
model.fit(X_train,y_train)
# print the coefficients

model.score(X_test,y_test)

y_pred =model.predict(X_test)

coef = Gini(y_pred,y_test)

print 'Gini coefficient is ', coef

#re-fit the model
model.fit(train[feature_cols],train['Hazard'])

X_oos = test[feature_cols]
oos_pred = model.predict(X_oos)

# create submission file
sub = pd.DataFrame({'Id':test.index, 'Hazard':oos_pred}).set_index('Id')
sub.to_csv('linreg2.csv')  # 0.864



