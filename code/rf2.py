# -*- coding: utf-8 -*-
"""
Created on aug 2, 2015

using LassoCV 32 the gini score is only .19 to .21 so it must want all the columns or feature engineering
@author: brian
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
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

feature_cols = ['T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N',
'T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E',
'T1_V9_F','T1_V11_E','T1_V11_H','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C',
'T1_V15_W','T1_V16_J','T1_V16_K','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D']


#we'll modify the sample submission file to make our submission
# submission = pd.read_csv('../input/sample_submission.csv')





from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train['Hazard'], random_state=1)


#fit the model and predict
model = RF(n_jobs=-1)
model.fit(X_train,y_train)
y_pred =model.predict(X_test)

coef = giniscore.Gini(y_pred,y_test)

print 'Gini coefficient is ', coef
# ugh, using LassoCV 32 the score is only .19 to 21 so it must want all the columns

# benchmark is .20 , Kaggle public LB says .263387
# < 14 97.5% benchmark is .172
# < 10 90.5% benchmark is .1472
# < 30 99.88% benchmark is .228 but Kaggle public LB says 0.262842, worse


featureImportances = zip(train.columns, model.feature_importances_)

featureImportancesSorted =sorted(featureImportances,key=lambda x: x[1], reverse=True)

featureImportancesSorted


