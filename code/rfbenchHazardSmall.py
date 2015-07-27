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

threshold = 30
# < 15 is 97.5% of the training rows
# < 10 is 90.75% of the training rows
# < 30 is 99.88% of the training rows
train[train.Hazard < threshold].shape[0]/float(train.shape[0])
trainHazardSmall = train[train.Hazard < threshold]



#we'll modify the sample submission file to make our submission
submission = pd.read_csv('../input/sample_submission.csv')

#prep the data for sklearn by separating predictors and response
X = trainHazardSmall.drop('Hazard', axis = 1)
y = trainHazardSmall['Hazard']

#one-hot the categoricals
num_X = pd.get_dummies(X)
num_Xt = pd.get_dummies(test)

#fit the model and predict
model = RF(n_jobs=-1)
model.fit(num_X,y)
prediction = model.predict(num_Xt)

#write the submission file
submission['Hazard'] = prediction
submission.to_csv('basic_RFLt30.csv', index = False)


