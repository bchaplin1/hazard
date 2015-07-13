# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:38:16 2015

@author: brian
"""

import pandas as pd

# assume you're in code or data directory
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# Write summaries of the train and test sets to the log
print('\nSummary of train dataset:\n')
print(train.describe())
train.shape[1]
train.columns[1]
train.iloc[1,1]
type(train.iloc[:,2])
train.iloc[:,2].value_counts().size

types = []
for i in range(train.shape[1]):
    types.append(type(train.iloc[1,i]))
    
train.shape
test.shape

for i in range(train.shape[1]):
    print train.columns[i],type(train.iloc[1,i]),train.iloc[:,i].value_counts().size

zip(train.columns,types)

train.describe()

cols1 = ['Hazard','T1_V8','T1_V1','T1_V2','T2_V1','T1_V15','T1_V4'] 

# what do the best and worst look like?
data =dtrain
data.sort('Hazard').head(5)[cols1]
data.sort('Hazard').tail(5)[cols1]


types
    print type(train.iloc[1,i])


# do we have any missing values?
train.isnull().sum()  # no missing values

# data types

train.T1_V15.describe()
train.T1_V15.value_counts() # so we have a lot of categorical variables

# what does the response look like?
train.Hazard.describe()
train.Hazard.hist() # very left-skewed, a lot have no hazard or hazard = 1
train.Hazard.value_counts()


train.corr()
import seaborn as sns
# display correlation matrix in Seaborn using a heatmap
sns.heatmap(train.corr()) 
#T2_V4 and T2_V15  and T2_V6 and T2_V14, maybe LDA or principal components

# get the features 
X = train.iloc[:,2:]
# get the response 
y = train.iloc[:,1]

# top random forest feature
train.T1_V8.describe()
train.T1_V8.value_counts()
train.T1_V1.describe()
train.T1_V1.value_counts()
train.T1_V1.hist()

cols = ['Hazard','T1_V8','T1_V1']
train[cols].head()
sns.pairplot( )

# histogram of beer servings grouped by continent
train.Hazard.hist(by=train.T1_V8)

# compare with bar plot
train.T1_V1.plot(kind='bar')

# use get_dummies or label classifier
pd.get_dummies(train.T1_V8, prefix='T1_V8').head(10)


train.head()

    # scatter matrix in Pandas
pd.scatter_matrix(train[['Hazard','T1_V1']])

# calculate the average duration for each genre
train.groupby('T1_V8').Hazard.agg(['count', 'mean'])


train[cols].corr()

# 3 booleans and 6 categorical columns, 
# convert to numbers and regress?  Probably not a good idea.

data = train
def visualize(data):
    # visualization
    import seaborn as sns
    import matplotlib.pyplot as plt

    # scatter matrix in Seaborn
    sns.pairplot(data)

    # scatter matrix in Pandas
    pd.scatter_matrix(data, figsize=(12, 10))

    # Use a **correlation matrix** to visualize the correlation between all numerical variables.

    # compute correlation matrix
    data.corr()

    # display correlation matrix in Seaborn using a heatmap
    sns.heatmap(data.corr())

visualize(train)


    



