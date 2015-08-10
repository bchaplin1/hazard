# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:34:14 2015

@author: brian
"""

from sklearn.linear_model import LinearRegression
import pandas as pd


# update both files with features
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    #one-hot the categoricals
    return pd.get_dummies(df)

# assume you're in code or data directory
# apply function to both training and testing files
train = make_features("../data/train.csv")
test = make_features("../data/test.csv")

#feature_cols  = ['T1_V8_C','T1_V9_E','T1_V5_K']

feature_cols  = ['T1_V5_E','T1_V8_C','T1_V15_C','T1_V4_N','T2_V15','T1_V12_C','T1_V11_H','T1_V8_C','T1_V5_K','T1_V2','T1_V9_E','T2_V1','T2_V2','T2_V9','T1_V2']
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


# instantiate and fit
linreg = LinearRegression()
threshold = 15
trainSmall=train[train.Hazard< threshold]
X = trainSmall[feature_cols]
y = trainSmall.Hazard
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_

linreg.score(X,y)

C= trainSmall.corr()

import seaborn as sns
# display correlation matrix in Seaborn using a heatmap
sns.heatmap(C)

# find collinearity
for row in range(0,C.shape[0]):
    for col in range(0,C.shape[1]):
        correl = C.iloc[row,col]
        if (correl > .5) & (row < col):
            print C.columns[row],C.columns[col],correl



C.columns

C.Hazard.sort()
C.head()
C.sort('Hazard').tail(10)['Hazard']
# discover the columns that affect the mean

trainRows = train[train.Hazard<threshold]
meanvalue = trainRows.Hazard.mean()
print meanvalue
for col in trainRows.columns[1:]:
    aggs =trainRows.groupby([col]).Hazard.agg(['mean','median','count','std','max','min'])
    for idx in range(0,aggs.shape[0]):
        if abs(aggs.iloc[idx,0] - meanvalue) > 1.5:
            print 'mean',col,aggs.iloc[idx].name,aggs.iloc[idx,0],aggs.iloc[idx,1],aggs.iloc[idx,2],aggs.iloc[idx,3],aggs.iloc[idx,4]

trainRows.T1_V5_E.describe()
trainRows.groupby(['T1_V5_E']).Hazard.agg(['mean','median','count','std','max','min'])
trainRows.groupby(['T1_V8_C']).Hazard.agg(['mean','median','count','std','max','min'])

trainRows.Hazard.hist(bins=14)

train[train.Hazard>=14].hist()

train[(train.Hazard>=14) & (train.Hazard<40)].Hazard.hist(bins=14)

# get the cumulative density of the Hazard score
train.groupby('Hazard').Hazard.count().cumsum().plot()

train[(train.Hazard>=40)].Hazard.hist(bins=14)


type(trainRows.Hazard.value_counts()[1:1])

import matplotlib.pyplot as plt
train.Hazard.value_counts().plot(kind='bar', title='Top 1000 Movies by Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('Number of Movies')

train.Hazard.value_counts(ascending=True).cumsum().plot(kind='line')

train.Hazard.value_counts().cumsum()

type(trainRows)


train.Hazard[train.Hazard<threshold].shape

train.groupby(['T1_V3']).Hazard.agg(['mean','median','count','std','max','min'])
# T1_V3 is number 8 on RF < 30
train.groupby(['T2_V2']).Hazard.agg(['mean','median','count','std','max','min'])
# T2_V2 is number 2 on RF < 30
train.groupby(['T1_V7_C']).Hazard.agg(['mean','median','count','std','max','min'])
train.groupby(['T1_V8_C']).Hazard.agg(['mean','median','count','std','max','min'])
train.groupby(['T1_V12_C']).Hazard.agg(['mean','median','count','std','max','min'])

train.groupby(['T2_V1']).Hazard.agg(['mean','median','count','std','max','min'])
# T2_V2 is number 1 on 

train.plot(kind='scatter', x='T1_V3', y='Hazard', alpha=0.3)
train.plot(kind='scatter', x='T2_V2', y='Hazard', alpha=0.3)
trainRows.plot(kind='scatter', x='T2_V1', y='Hazard', alpha=0.3)

train.Hazard[train.Hazard<15].mean()

trainRows[['T2_V1', 'Hazard']].corr()

gb.iloc[8].name

aggs.loc[0,0]

gb.loc[8,'mean']
gb.shape[0]
type(gb)
aggs
gb

gb.loc[8,'mean']
aggs.loc[0,'mean']
aggs.iloc[0,0]
range
aggs

train.Hazard.groupby(['T1_V3']).mean()

            
train.T1_V3.describe() 

train.T1_V7_C.describe()           
            
    print col,mean,median,abs(mean - 4.022),abs(median - 3)


meanMedian    
    print col,mean,median,abs(mean - 4.022),abs(median - 3)   
    
    
train.groupby(level=0,axis=1).Hazard.mean()
    
    

for row in range(0,train.columns.shape[0]):

            

train.plot(kind='scatter', x='T2_V4', y='T2_V15', alpha=0.3)

train.plot(kind='scatter', x='T2_V4', y='Hazard', alpha=0.3)
train.plot(kind='scatter', x='T2_V15', y='Hazard', alpha=0.3)



train.groupby(['T1_V17_Y','T2_V12_Y']).Hazard.agg(['count', 'mean','std'])
train.groupby(['T2_V4','T2_V15']).Hazard.agg(['count', 'mean','std'])

train[['T1_V17_Y','T2_V12_Y']].groupby(['T1_V17_Y','T2_V12_Y']).sum()

train[['T1_V17_Y','T2_V12_Y']].groupby(['T1_V17_Y']).sum()

grouped = train[['T1_V17_Y','T2_V12_Y']].groupby(['T1_V17_Y','T2_V12_Y'])

train.Hazard[(train.T1_V17_Y == 0) & (train.T2_V12_Y == 0)].count() 
agg(['count', 'mean','std'])
 
           
            
        
 
