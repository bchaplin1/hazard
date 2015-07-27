# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:08:49 2015

@author: brian
"""

# pre-process the file
# what is this? preprocessing.LabelEncoder()
# need to create dummy variables

# use get_dummies or label classifier
# pd.get_dummies(train.T1_V8, prefix='T1_V8').head(10)
import pandas as pd

# assume you're in code or data directory
dtrain = pd.read_csv("c:/ml/hazard/data/train.csv")
dtest = pd.read_csv("c:/ml/hazard/data/test.csv")



