# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:41:27 2015

@author: brian
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF


# update both files with features
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    #one-hot the categoricals
    return pd.get_dummies(df)

# assume you're in code or data directory
# apply function to both training and testing files
train = make_features("../data/train.csv")
test = make_features("../data/test.csv")


#we'll modify the sample submission file to make our submission
# submission = pd.read_csv('../input/sample_submission.csv')



#prep the data for sklearn by separating predictors and response
# discard the outliers since scoring is Gini coefficient
threshold = 30
# < 15 is 97.5% of the training rows
# < 10 is 90.75% of the training rows
# < 30 is 99.88% of the training rows
train[train.Hazard < threshold].shape[0]/float(train.shape[0])
trainHazardSmall = train[train.Hazard < threshold]

X = trainHazardSmall.drop('Hazard', axis = 1)
y = trainHazardSmall['Hazard']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


#fit the model and predict
model = RF(n_jobs=-1)
model.fit(X_train,y_train)
y_pred =model.predict(X_test)

coef = Gini(y_pred,y_test)

print 'Gini coefficient is ', coef

# benchmark is .20 , Kaggle public LB says .263387
# < 14 97.5% benchmark is .172
# < 10 90.5% benchmark is .1472
# < 30 99.88% benchmark is .228 but Kaggle public LB says 0.262842, worse


featureImportances = zip(X.columns, model.feature_importances_)

featureImportancesSorted =sorted(featureImportances,key=lambda x: x[1], reverse=True)

featureImportancesSorted


train.T2_V1.value_counts()

train[['T2_V1','T2_V2','T1_V2','Hazard']].corr()

C= train.corr()

type(C)

C.Hazard.sort()
C.head()
C.sort('Hazard').tail(10)['Hazard']

# add transparency
train.plot(kind='scatter', x='T1_V8_C', y='Hazard', alpha=0.3)
train.plot(kind='scatter', x='T1_V2', y='Hazard', alpha=0.3)

# share the x and y axes
train.Hazard.hist(by=train.T1_V8_C, sharex=True, sharey=True)
train.Hazard.hist(by=train.T1_V8_C)

train[train.Hazard>20].Hazard.hist(by=train.T1_V8_C)

train.boxplot(by='Hazard')


train.groupby('T1_V8_C').Hazard.agg(['count', 'mean','std'])
train.groupby('T1_V9_E').Hazard.agg(['count', 'mean','std'])
train.groupby('T1_V5_K').Hazard.agg(['count', 'mean','std'])

train.groupby('T1_V2').Hazard.agg(['count', 'mean','std'])


C.describe()






train






