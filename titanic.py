#%%
#import modules

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
# import sklearn
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder

#change dir
os.chdir('P:/Pertinent Files/Python/scripts')

#%%
#read data
gender_df=pd.read_csv("P:/Domestic/CI/Kaggle Submission/titanic/gender_submission.csv")
train_df=pd.read_csv("P:/Domestic/CI/Kaggle Submission/titanic/train.csv")
test_df=pd.read_csv("P:/Domestic/CI/Kaggle Submission/titanic/test.csv")

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
train_df.info()

#show summary of numeric data
train_df.describe()

#show missing values for all data
train_df.isna().sum()
# %%
#select numeric cols
numeric_cols=train_df.select_dtypes('number').iloc[:,1:].columns
#select_object cols
categorical_cols=train_df.select_dtypes('object').iloc[:,1:].columns

#get summary for categorical var
# categorical_cols['Cabin_ext']=categorical_cols['Cabin'].replace(r'(d+[.\d])')
train_df['Cabin_ext']=train_df['Cabin'].str.extract(r'([a-zA-Z]+)')
train_df['Ticket_ext']=train_df['Ticket'].str.extract(r'([a-zA-Z])')

test_df['Cabin_ext']=test_df['Cabin'].str.extract(r'([a-zA-Z]+)')
test_df['Ticket_ext']=test_df['Ticket'].str.extract(r'([a-zA-Z])')
# print(categorical_cols['Ticket_ext'].value_counts())
# %%
#Data cleaning


#%%
#Encode the categorical vars
gen={'male':1,'female':0}
train_df['Sex']=train_df['Sex'].map(gen)

#%%
categorical_cols=['Embarked','Cabin_ext','Ticket_ext']
ohe=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(train_df[categorical_cols])

encoded_cols=ohe.get_feature_names_out().tolist()

train_df[encoded_cols]=ohe.transform(train_df[categorical_cols])
test_df[encoded_cols]=ohe.transform(test_df[categorical_cols])


# %%
train_df.columns
# %%
