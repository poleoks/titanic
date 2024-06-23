#%%
# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
import re
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder

#%%
#read data
gender_df=pd.read_csv("https://github.com/poleoks/titanic/blob/main/gender_submission.csv?raw=true" #,index_col=0
                      )
train_df=pd.read_csv("https://github.com/poleoks/titanic/blob/main/train.csv?raw=true")
test_df=pd.read_csv("https://github.com/poleoks/titanic/blob/main/test.csv?raw=true")
#%%
y_train=train_df['Survived']

X_train=train_df[test_df.columns.values]

# %%
#data split if
# df_test2, df_train2=tts(X_train,test_size=0.2,random_state=42, shuffle=True)
# print(X_train.shape,df_train2.shape, df_test2.shape)
# %%
#Explore data
#show all dtype per attr
test_df.info()
#%%
#show summary of numeric data
X_train.describe()
#%%
#show summary of categorical data
X_train.describe(include=['O'])


#%%
#Analyse by Pivoting
#pivot by attr to check survival
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#%%
#Analyse by visualization

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age')

#%%


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
#%%
#corr categorical var

grd = sns.FacetGrid(train_df, row='Embarked')
grd.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grd.add_legend()

#%%
#  Correlating categorical and numerical features

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived',aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
#%%
#show missing values for all data
X_train.isna().sum()
#%%
# fill na for numeric data

gen={'male':1,'female':0}
X_train['Sex']=X_train['Sex'].map(gen)
test_df['Sex']=test_df['Sex'].map(gen)
#numeric cols
numeric_cols=X_train.select_dtypes('number').iloc[:,1:].columns.tolist()
numeric_cols
#%%
from sklearn.impute  import SimpleImputer
si=SimpleImputer(strategy='mean')
si.fit(X_train[numeric_cols])
X_train[numeric_cols]=si.transform(X_train[numeric_cols])
test_df[numeric_cols]=si.transform(test_df[numeric_cols])
X_train.isna().sum()
#%%
#categorical cols

X_train[['Cabin', 'Embarked']]=X_train[['Cabin', 'Embarked']].fillna('unknown')

X_train.isna().sum()
#%%

#get summary for categorical var
X_train['Ticket'].value_counts()

#%%
# categorical_cols['Cabin_ext']=categorical_cols['Cabin'].replace(r'(d+[.\d])')
# extract only letters 
def rep(d):
    if isinstance(d, str):
        return re.sub(r'[^a-z.]', '', d.lower())
    else:
        return ''
#     return re.sub(r'[^a-zA-Z]','',d ).lower()

X_train['Cabin_ext']=X_train['Cabin'].apply(rep)
X_train['Ticket_ext']=X_train['Ticket'].apply(rep)

test_df['Ticket_ext']=test_df['Ticket'].apply(rep)
#%%
test_df['Cabin_ext']=test_df['Cabin'].apply(rep)

test_df['Cabin'].value_counts()
# %%
X_train.isna().sum()
#Encode the categorical vars
categorical_cols=['Embarked','Cabin_ext','Ticket_ext']
X_train[categorical_cols].info()
#%%

ohe=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(X_train[categorical_cols])

encoded_cols=ohe.get_feature_names_out().tolist()
encoded_cols
X_train[encoded_cols]=ohe.transform(X_train[categorical_cols])
test_df[encoded_cols]=ohe.transform(test_df[categorical_cols])

all_cols=encoded_cols + numeric_cols

# %%
#standardize data
X_train[all_cols].head()
#%%%
from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()
mms.fit(X_train[numeric_cols])

X_train[numeric_cols]=mms.transform(X_train[numeric_cols])
test_df[numeric_cols]=mms.transform(test_df[numeric_cols])
# X_train=mms.transform(X_train[all_cols])

X_train[all_cols].head()
# %%

#%%
