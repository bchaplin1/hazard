# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:41:27 2015

@author: brian
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF

# assume you're in code or data directory
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

#we'll modify the sample submission file to make our submission
submission = pd.read_csv('../input/sample_submission.csv')

#prep the data for sklearn by separating predictors and response
X = train.drop('Hazard', axis = 1)
y = train['Hazard']

#one-hot the categoricals
num_X = pd.get_dummies(X)
num_Xt = pd.get_dummies(test)

#fit the model and predict
model = RF(n_jobs=-1)
model.fit(num_X,y)
prediction = model.predict(num_Xt)

#write the submission file
submission['Hazard'] = prediction
submission.to_csv('basic_RF.csv', index = False)


