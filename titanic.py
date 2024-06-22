#%%
# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
import re

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder

#%%
#read data
gender_df=pd.read_csv("https://github.com/poleoks/titanic/blob/main/gender_submission.csv?raw=true" #,index_col=0
                      )
train_df=pd.read_csv("https://github.com/poleoks/titanic/blob/main/train.csv?raw=true")
test_df=pd.read_csv("https://github.com/poleoks/titanic/blob/main/test.csv?raw=true")
print(train_df.columns)
#%%
train_target=train_df['Survived']
train_df=train_df[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]

# %%
#data split if
# df_test2, df_train2=tts(train_df,test_size=0.2,random_state=42, shuffle=True)
# print(train_df.shape,df_train2.shape, df_test2.shape)
# %%
#Explore data
#show all dtype per attr
test_df.info()
#%%
#show summary of numeric data
train_df.describe()

#show missing values for all data
train_df.isna().sum()
#%%
# fill na for numeric data

gen={'male':1,'female':0}
train_df['Sex']=train_df['Sex'].map(gen)
test_df['Sex']=test_df['Sex'].map(gen)
#numeric cols
numeric_cols=train_df.select_dtypes('number').iloc[:,1:].columns.tolist()
numeric_cols
#%%
from sklearn.impute  import SimpleImputer
si=SimpleImputer(strategy='mean')
si.fit(train_df[numeric_cols])
train_df[numeric_cols]=si.transform(train_df[numeric_cols])
test_df[numeric_cols]=si.transform(test_df[numeric_cols])
train_df.isna().sum()
#%%
#categorical cols

train_df[['Cabin', 'Embarked']]=train_df[['Cabin', 'Embarked']].fillna('unknown')

train_df.isna().sum()
#%%

#get summary for categorical var
train_df['Ticket'].value_counts()

#%%
# categorical_cols['Cabin_ext']=categorical_cols['Cabin'].replace(r'(d+[.\d])')
# extract only letters 
def rep(d):
    if isinstance(d, str):
        return re.sub(r'[^a-z.]', '', d.lower())
    else:
        return ''
#     return re.sub(r'[^a-zA-Z]','',d ).lower()

train_df['Cabin_ext']=train_df['Cabin'].apply(rep)
train_df['Ticket_ext']=train_df['Ticket'].apply(rep)

test_df['Ticket_ext']=test_df['Ticket'].apply(rep)
#%%
test_df['Cabin_ext']=test_df['Cabin'].apply(rep)

test_df['Cabin'].value_counts()
# %%
train_df.isna().sum()
#Encode the categorical vars
categorical_cols=['Embarked','Cabin_ext','Ticket_ext']
train_df[categorical_cols].info()
#%%

ohe=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(train_df[categorical_cols])

encoded_cols=ohe.get_feature_names_out().tolist()
encoded_cols
train_df[encoded_cols]=ohe.transform(train_df[categorical_cols])
test_df[encoded_cols]=ohe.transform(test_df[categorical_cols])

all_cols=encoded_cols + numeric_cols

# %%
#standardize data
train_df[all_cols].head()
#%%%
from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()
mms.fit(train_df[numeric_cols])

train_df[numeric_cols]=mms.transform(train_df[numeric_cols])
test_df[numeric_cols]=mms.transform(test_df[numeric_cols])
# train_df=mms.transform(train_df[all_cols])

train_df[all_cols].head()
# %%

#%%
