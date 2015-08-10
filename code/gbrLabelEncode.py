# -*- coding: utf-8 -*-
"""
Created on aug 2, 2015

use label encoding

public LB gini of 0.366189
gini on training set of 0.34274

200 estimators public LB 0.372387, train gini 0.363734, 98 variables used
500 estimators public LB 0.374926, train gini 0.3996455, all variables used
huber method better than ls or lad

@author: brian
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
import giniscore as giniscore

# update both files with features
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    #one-hot the categoricals
    le1 = preprocessing.LabelEncoder()
    le1.fit(df.T1_V11)
    
    
    return pd.get_dummies(df)


T1_V11
T1_V16
T1_V4
T1_V5
T1_V9


# assume you're in code or data directory
# apply function to both training and testing files
train = make_features("../data/train.csv")
test = make_features("../data/test.csv")

feature_cols = train.columns
# exclude the response
feature_cols = feature_cols.drop('Hazard')

# trn2=train.sample(10000).copy()
trn2=train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trn2[feature_cols], trn2['Hazard'], random_state=1)


#fit the model and predict
# model = AdaBoostRegressor(base_estimator=RandomForestRegressor())
model = GradientBoostingRegressor(loss='huber',n_estimators=200)
model.fit(X_train,y_train)
y_pred =model.predict(X_test)

coef = giniscore.Gini(y_pred,y_test)

print 'Gini coefficient is ', coef
# gini 0.3098633 with all rows and loss ls
# gini 0.302133 with all rows and loss lad
# gini 0.310395 will all rows and loss huber
# gini 0.3155986 will all rows and loss huber


model.score(X_train,y_train)
# score with 10000 rows  is .193
# score with all rows is 0.13




model2 = GradientBoostingRegressor(loss='huber',n_estimators=500)
model2.fit(train[feature_cols],train['Hazard'])



y_pred2 =model2.predict(train[feature_cols])

coef2 = giniscore.Gini(y_pred2,train['Hazard'])

print 'Gini coefficient is ', coef2
# 0.34274
# 0.36373 for 200 estimators
# 0.3996455 for 500 estimators

featureImportances = zip(train.columns, model2.feature_importances_)

featureImportancesSorted =sorted(featureImportances,key=lambda x: x[1], reverse=True)

featureImportancesSorted

featUsed = [feature[0] for feature in featureImportancesSorted if feature[1] >0 ]

featUsed
# 70 features used are (in order of importance)
# 'T1_V1', 'T2_V1', 'T1_V14', 'T1_V8_B', 'T2_V14', 'T1_V12_B', 'T2_V2', 'T1_V2', 'T2_V8', 'T1_V7_B', 'T1_V15_A', 'T1_V9_D', 'T1_V16_A', 'T1_V4_H', 'T2_V10', 'T1_V13', 'T2_V12_Y', 'T1_V11_A', 'T1_V15_W', 'T2_V4', 'T1_V9_E', 'T2_V5_A', 'T2_V15', 'T1_V5_A', 'T1_V15_H', 'T1_V5_J', 'T1_V9_G', 'T2_V5_C', 'T1_V12_D', 'T1_V15_F', 'T1_V16_R', 'T2_V9', 'T2_V12_N', 'T1_V4_W', 'T1_V16_H', 'T1_V5_C', 'T1_V4_B', 'T2_V3_Y', 'T1_V4_G', 'T1_V15_S', 'T1_V15_C', 'T1_V11_F', 'T1_V8_D', 'T2_V13_A', 'T1_V4_E', 'T1_V17_N', 'T1_V11_L', 'T2_V5_D', 'T2_V13_C', 'T2_V6', 'T2_V11_Y', 'T2_V7', 'T2_V5_F', 'T1_V5_L', 'T1_V3', 'T1_V16_J', 'T1_V9_C', 'T1_V4_S', 'T1_V16_I', 'T1_V7_D', 'T1_V5_B', 'T1_V16_D', 'T1_V10', 'T1_V16_O', 'T1_V4_C', 'T2_V13_D', 'T1_V7_A', 'T1_V11_H', 'T1_V4_N'
# with 200 estimators, 98 used
# with 500 estimators, all used


X_oos = test[feature_cols]
oos_pred = model2.predict(X_oos)

# create submission file
sub = pd.DataFrame({'Id':test.index, 'Hazard':oos_pred}).set_index('Id')
sub.to_csv('gradientBoostRegrHuber500estimators.csv')  

