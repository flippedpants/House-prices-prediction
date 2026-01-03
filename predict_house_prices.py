#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib


# In[2]:


data = fetch_openml(data_id=42165, as_frame=True) 
data.keys()


# In[3]:


X = pd.DataFrame(data.data , columns=data.feature_names)
y = data.target


# In[4]:


X.head(10)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[20]:


imputer_cat = SimpleImputer(strategy='most_frequent')
X_train_imputed = pd.DataFrame(imputer_cat.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer_cat.transform(X_test), columns=X_test.columns)
X_train_imputed.head(10)


# In[25]:


s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
# X_train_imputed[object_cols].nunique()

for col in object_cols:
    X_train_imputed[col] = X_train_imputed[col].astype(str).str.strip()
    X_test_imputed[col] = X_test_imputed[col].astype(str).str.strip()


# In[29]:


ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value" , unknown_value=-1)

X_train_imputed[object_cols] = ordinal_encoder.fit_transform(X_train_imputed[object_cols])
X_test_imputed[object_cols] = ordinal_encoder.transform(X_test_imputed[object_cols])

final_X_train = X_train_imputed
final_X_test = X_test_imputed
final_X_train.head()


# In[31]:


print(final_X_train.dtypes.value_counts())


# In[30]:


xgb_base = XGBRegressor()
xgb_base.fit(final_X_train, y_train)


# In[ ]:




