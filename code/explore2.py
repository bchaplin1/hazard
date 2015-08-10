# -*- coding: utf-8 -*-
"""
Created on Sat Aug 01 20:47:51 2015

explore the 32 from lasso cv

['T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N','T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E','T1_V9_F','T1_V11_E','T1_V11_H','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C','T1_V15_W','T1_V16_J','T1_V16_K','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D']


@author: brian
"""

import pandas as pd


# update both files with features
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    #one-hot the categoricals
   # return df
    return pd.get_dummies(df)

# assume you're in code or data directory
# apply function to both training and testing files
train = make_features("../data/train.csv")
test = make_features("../data/test.csv")



#fit the model and predict
feature_cols = ['T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N',
'T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E',
'T1_V9_F','T1_V11_E','T1_V11_H','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C',
'T1_V15_W','T1_V16_J','T1_V16_K','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D']

feature_cols_resp = ['Hazard','T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N','T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E','T1_V9_F','T1_V11_E','T1_V11_H','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C','T1_V15_W','T1_V16_J','T1_V16_K','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D']



C =train[feature_cols_resp].corr()
sorted(C['Hazard'])

import seaborn as sns
trn2 = train[feature_cols_resp].sample(10)

trn2.shape
sns.pairplot(train.sample(10))

sns.jointplot(['T1_V1','T1_V2'],'Hazard',train.sample(10),kind = "regr")


g = (sns.jointplot("T1_V1", "Hazard",
                    data=train.sample(1000), color="k")
         .plot_joint(sns.kdeplot, zorder=0, n_levels=6))
g

sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)

sns.pairplot(train[['T1_V1','T1_V2','Hazard']].sample(200),kind='reg')


sns.pairplot(train[['T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N','Hazard']].sample(100),kind='reg')
from pylab import savefig
savefig("plot1.png")
sns.pairplot(train[['T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E','Hazard']].sample(100),kind='reg')
savefig("plot2.png")

sns.pairplot(train[['T1_V9_F','T1_V11_E','T1_V11_H','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C','Hazard']].sample(100),kind='reg')
savefig("plot3.png")
sns.pairplot(train[['T1_V15_W','T1_V16_J','T1_V16_K','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D','Hazard']].sample(100),kind='reg')
savefig("plot4.png")
# display correlation matrix in Seaborn using a heatmap
sns.heatmap(C) 
# find collinearity
for row in range(0,C.shape[0]):
    for col in range(0,C.shape[1]):
        correl = C.iloc[row,col]
        if (correl > .4) & (row < col):
            print C.columns[row],C.columns[col],correl
            
"""
T1_V5_K T1_V9_E 0.588541580337
T1_V5_K T1_V11_H 0.455091738224
T1_V9_E T1_V11_H 0.625865917874
T1_V9_E T1_V16_K 0.414854375564     
"""            
import statsmodels.api as sm


mod2 = sm.OLS(train['Hazard'],train[feature_cols])
res = mod2.fit()
res.summary()   

# remove  T1_V11_H T1_V16_K 

feat2 =  ['T1_V1','T1_V2','T1_V3','T2_V7','T2_V14','T2_V15','T1_V4_H','T1_V4_N','T1_V4_W','T1_V5_B','T1_V5_D','T1_V5_K','T1_V6_N','T1_V7_C','T1_V8_C','T1_V9_E','T1_V9_F','T1_V11_E','T1_V11_M','T1_V12_C','T1_V15_A','T1_V15_C','T1_V15_W','T1_V16_J','T1_V17_N','T2_V3_N','T2_V5_D','T2_V5_E','T2_V11_N','T2_V13_D']


mod3 = sm.OLS(train['Hazard'],train[feat2])
res3 = mod3.fit()
res3.summary()   


# display correlation matrix in Seaborn using a heatmap
sns.heatmap(train[feat2].corr()) 
            