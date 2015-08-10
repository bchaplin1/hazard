# -*- coding: utf-8 -*-
"""
Created on Sat Aug 08 14:29:45 2015

normalized gini 0.657669926619
public lb of 0.371129

public lb of 0.364498 with GBR 1000 estimators which was worse than dummies 0.49424 norm gini for 1000, test1,2,3
running again gets different training estimate but same model at 0.364498 on public LB
train1,2,3 is [1210]  train-rmse:2.993954     val-rmse:3.771549 0.66871784649 for all three

0.363681 with GBR 1000 with test1,2,3

combining selected dummies and label encoding
[896] train-rmse:3.253069     val-rmse:3.690047 gini 0.567087759262

        # gini 0.542122229878 and 0.382608 on LB combining all dummies with label


@author: brian
"""

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import giniscore as giniscore
from sklearn.ensemble import GradientBoostingRegressor

def gini(y_true, y_pred):
    """ Simple implementation of the (normalized) gini score in numpy. 
        Fully vectorized, no python loops, zips, etc. Significantly
        (>30x) faster than previous implementions
        
        Credit: https://www.kaggle.com/jpopham91/
    """

    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true
    
    
def normalized_gini(y_true, y_pred):
    ng = gini(y_true, y_pred)/gini(y_true, y_true)
    return ng


def xgb_predict(train, target, test, max_rounds=10000):
    
    # set params
    params = dict(
        objective='reg:linear',
        eta=0.005,
        min_child_weight=6,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=1,
        silent=1,
        max_depth=9
        )
    plist = list(params.items())
    
    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2)
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgval = xgb.DMatrix(X_val, label=y_val)
    
    # set up test
    xgtest = xgb.DMatrix(test)
    
    # train using early stopping
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plist, xgtrain, max_rounds, watchlist, early_stopping_rounds=120)
    preds_train = model.predict(xgb.DMatrix(train), ntree_limit=model.best_iteration)
    preds = model.predict(xgtest, ntree_limit=model.best_iteration)
    
    return preds_train, preds, model
    
    
    
    
def prep_data1(train, target, test):
    # handle categorical data
    dummyFeatCols=['T1_V1','T1_V10','T1_V12','T1_V13','T1_V14','T1_V15','T1_V17','T1_V2','T1_V3','T1_V6','T1_V7','T1_V8','T2_V1','T2_V10','T2_V11','T2_V12','T2_V13','T2_V14','T2_V15','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9']
    data = pd.get_dummies(pd.concat([train, test]))
    X_train = data.loc[train.index].values
    y_train = target.values
    X_test = data.loc[test.index].values
    
    return X_train, y_train, X_test


def prep_data2(train, target, test):
    train = train.T.to_dict().values()
    test = test.T.to_dict().values()
    
    vec = DictVectorizer(sparse=False)
    train = vec.fit_transform(train)
    test = vec.transform(test)
    
    return train, target, test


def prep_data3(train, target, test):
    labelFeatCols=['T1_V11','T1_V16','T1_V4','T1_V5','T1_V9']
    train_s = np.array(train[labelFeatCols])
    test_s = np.array(test[labelFeatCols])
    
    # label encode the categorical variables
    for i in range(train_s.shape[1]):
        lbl = LabelEncoder()
        lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
        train_s[:,i] = lbl.transform(train_s[:,i])
        test_s[:,i] = lbl.transform(test_s[:,i])
    
    train_s = train_s.astype(float)
    test_s = test_s.astype(float)
    
    return train_s, target, test_s
    

if __name__ == '__main__':
    
    # load data
    train = pd.read_csv('train.csv', index_col='Id')
    test = pd.read_csv('test.csv', index_col='Id')
    
    # randomize data
    train = train.reindex(np.random.permutation(train.index))

    # split off target and indices
    target = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)
    test_ind = test.index
    
    # prepare data
    train1, target1, test1 = prep_data1(train, target, test)

    train2, target2, test2 = prep_data2(train, target, test)

    train3, target3, test3 = prep_data3(train, target, test)
    
    # combine all data prep methods
    train = np.hstack([train1,train3])
    test = np.hstack([test1,test3])
#    
# train1,2,3 is [1210]  train-rmse:2.993954     val-rmse:3.771549 0.66871784649 for all three
# train1,2,3 is [1284]  train-rmse:2.980390     val-rmse:3.682356  0.646624084163
# train1,2,3 is [1463]  train-rmse:2.875965     val-rmse:3.819517    
# train1 is [1457]  train-rmse:3.083193     val-rmse:3.718195, 0.608113416633  
# train2 is [685]   train-rmse:3.732203     val-rmse:4.061310,0.596904349295
# train3 is         train-rmse:3.087413     val-rmse:3.838870,  .6249
# train1,3 is [1008]  train-rmse:3.155898     val-rmse:3.744326, 0.589301647343
# selected train1, train3 [896] train-rmse:3.253069     val-rmse:3.690047 gini 0.567087759262
    
# train1, selected 3 is [1223]  train-rmse:3.133013     val-rmse:3.770925  gini .59495   
    
#    train = np.hstack([train1,train3])
#    test = np.hstack([test1,test3])

   
    # fit model and predict
    preds_train, preds, modl = xgb_predict(train, target, test, max_rounds=5000)
    
    scores = modl.get_fscore() #249 features
    
    
    
    
    
    
    model = GradientBoostingRegressor(loss='huber',n_estimators=500)
    model.fit(train,target)
    preds_train2 = model.predict(train)
    preds2 =model.predict(test)
    # 0.398579 for 100
    # 0.3975 for test1, test3
    # 0.39472
    # 0.486607192584 for 1000
    # 0.49424 norm gini for 1000, test1,2,3
    # 0.4422346 for exclusive combo of dummies and labelCreator
    # 0.442582848124 for dummies and 4 labelCreator 0.377088 LB
    
    
#    coef = giniscore.Gini(y_pred,y_test)

featureImportances = zip(range(0,254), model.feature_importances_)

featureImportancesSorted =sorted(featureImportances,key=lambda x: x[1], reverse=True)

featureImportancesSorted

featUsed = [feature[0] for feature in featureImportancesSorted if feature[1] >0 ]

featUsed
    
    # evaluate on training set
    print(normalized_gini(target, preds_train))
        print(normalized_gini(target, preds_train2))
        print (normalized_gini(target, (preds_train + preds_train2)/2.0))
        # 0.542122229878 and 0.382608 on LB
    # 0.657669926619 for all three

 #   print 'gini score',giniscore.Gini(target, preds_train)    
    
    # output predictions
    preds_all = (preds2 + preds)/2.0    
    
    
    
    out = pd.DataFrame({'Hazard': preds_all}, index=test_ind)
    out.to_csv('XgbGbr500SelectedDummies4LabelEncode.csv')

                