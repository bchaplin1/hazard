# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:34:14 2015

@author: brian
"""


#feature_cols  = ['T1_V8_C','T1_V9_E','T1_V5_K']

# feature_cols  = ['T1_V5_E','T1_V8_C','T1_V15_C','T1_V4_N','T2_V15','T1_V12_C','T1_V11_H','T1_V8_C','T1_V5_K','T1_V2','T1_V9_E','T2_V1','T2_V2','T2_V9','T1_V2']
# T1_V5_E mean
# T1_V8_C mean
# T1_V15_C    0.060321
# T1_V4_N     0.065470
# T2_V15      0.066527
# T1_V12_C    0.074483
# T1_V11_H    0.079414
# T1_V8_C     0.086275
# T1_V5_K     0.094186
# T1_V2       0.104895
# T1_V9_E     0.108297
#[('T2_V1', 0.09010408822686175),
# ('T2_V2', 0.062291219000935469),
# ('T2_V9', 0.061440143097315582),
# ('T1_V2', 0.060688115376786603),



feature_cols = train.columns
# exclude the response
feature_cols = feature_cols.drop('Hazard')

X = train[feature_cols]
y = train.Hazard


# TASK 3: split the data into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# create a fitted model
import statsmodels.api as sm
mod = sm.OLS(y_train,X_train)
res = mod.fit_regularized(alpha=1)
res.summary()

y_pred = res.predict(X_test)

import numpy as np

print metrics.r2_score(y_test, y_pred)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

from sklearn import linear_model
ridgeModel = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
ridgeModel.fit(X_train,y_train)
ridgeModel.coef_

y_pred = ridgeModel.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

lassoModel = linear_model.LassoCV()
lassoModel.fit(X_train,y_train)
lassoModel.coef_

y_pred = lassoModel.predict(X_test)

coeff = zip(feature_cols,lassoModel.coef_)

type(coeff[0])

feats = [feature[0] for feature in coeff if feature[1] >0 ]

X = train[feats]
y = train.Hazard

# TASK 3: split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression
# instantiate and fit
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# TASK 5: make predictions on testing set and calculate accuracy
y_pred = linreg.predict(X_test)
print metrics.r2_score(y_test, y_pred)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
# RMSE of 3.8688 for LASSO 32 features


# print the coefficients
print linreg.intercept_
print linreg.coef_

linreg.score(X,y)

mod2 = sm.OLS(y_train,X_train)
res = mod2.fit()
res.summary()

# remove column numbers 12, 19, 25,27
import pandas.core.index

featIndex = pandas.core.index.Index(feats)
feat2 = featIndex.drop(featIndex[[11,18,24,26]])

X = train[feat2]
y = train.Hazard

# TASK 3: split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression
# instantiate and fit
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# TASK 5: make predictions on testing set and calculate accuracy
y_pred = linreg.predict(X_test)

print metrics.r2_score(y_test, y_pred)
#0.08868
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
# 3.869008 with 28 after removing 4 with high p values

#print the summary           
mod2 = sm.OLS(y_train,X_train)
res = mod2.fit()
res.summary()

feat2            
 
