# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:30:51 2020

@author: jsyi
"""

import cx_Oracle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def getDataframe():
    dsn_tns = cx_Oracle.makedsn('10.188.217.220', '1521', service_name='xe') # if needed, place an 'r' before any parameter in order to address special characters such as '\'.
    conn = cx_Oracle.connect(user=r'jsyi', password='cj123', dsn=dsn_tns) # if needed, place an 'r' before any parameter in order to address special characters such as '\'. For example, if your user name contains '\', you'll need to place 'r' before the user name: user=r'User Name'
    df = pd.read_sql_query("select * from aminoacid_price_vw order by day", conn)
    conn.close()
    
    return df

df = getDataframe()
df = df.interpolate()
df = df.dropna()
df.corr(method ='pearson')
sns.heatmap(df.corr(method ='pearson'), annot=True)
sns.pairplot(df)
df.head()
df = df[[
    'DAY',
    'LYSINE_PRICE',
    'TRYPTOPHAN_PRICE',
    'THREONINE_PRICE',
    'WHEAT',
    'CORN',
    'SOYBEAN',
    'WTI'
]]
df = df.set_index('DAY')
y_data = df['LYSINE_PRICE'].values
x_data = df.iloc[:,1:].values
#x_data = df[['x1','x9','x10','x11']].values

df.columns
df['LYSINE_PRICE'].plot()
df['CORN'].plot()

#fig, ax = plt.subplots(figsize=(20, 16))

plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['CORN']), color='black')
#fig.savefig('data/lysine-corn.png')

#LSTM
train_data = df[df.index<'2017-01-01']
test_data = df[(df.index<'2020-01-01')&(df.index>'2018-01-01')]
train_data.plot()
test_data.plot()
y_train = train_data['LYSINE_PRICE'].values
x_train = train_data.iloc[:,1:].values
y_test = test_data['LYSINE_PRICE'].values
x_test = test_data.iloc[:,1:].values

plt.plot(y_train)
plt.plot(y_test)
plt.plot(x_train)
plt.plot(x_test)


