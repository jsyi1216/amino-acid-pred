# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:56:32 2020

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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def getDataframe():
    dsn_tns = cx_Oracle.makedsn('10.188.217.220', '1521', service_name='xe') # if needed, place an 'r' before any parameter in order to address special characters such as '\'.
    conn = cx_Oracle.connect(user=r'jsyi', password='cj123', dsn=dsn_tns) # if needed, place an 'r' before any parameter in order to address special characters such as '\'. For example, if your user name contains '\', you'll need to place 'r' before the user name: user=r'User Name'
    df = pd.read_sql_query("select * from AMINO_PRICE_VW", conn)
    conn.close()
    
    return df

df = getDataframe()
df = df.dropna()
df.corr(method ='pearson')
sns.heatmap(df.corr(method ='pearson'), annot=True)
sns.pairplot(df)



# split by day
#year = df["DAY"].dt.to_period("Y")
#agg = df.groupby([year])
#for group in agg:
#    print(group)

df = df.set_index('DAY')
y_data = df['LYSINE_PRICE'].values
#x_data = df.iloc[:,1:].values
x_data = df[['CORN','WHEAT','SOYBEAN']].values
n_features = len(x_data[0])

df.columns
df['LYSINE_PRICE'].plot()
df['CORN'].plot()
df['WHEAT'].plot()
df['SOYBEAN'].plot()
df['CM_PRICE'].plot()
df['RM_PRICE'].plot()
df['FISHMEAL_PRICE'].plot()
df['PIGLET_PRICE'].plot()
df['SOW_PRICE'].plot()
df['PORK_PRICE'].plot()
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), color='black')
plt.plot(preprocessing.scale(df['CORN']), '.', color='red', markersize=5)
fig.savefig('data/lysine-corn.png')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), color='black')
plt.plot(preprocessing.scale(df['WHEAT']), '.', color='red', markersize=5)
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), color='red', markersize=5)
plt.plot(preprocessing.scale(df['SOYBEAN']), '.', color='black')
fig.savefig('data/lysine-sbm.png')
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['CM_PRICE']), color='black')
fig.savefig('data/lysine-cm.png')
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['RM_PRICE']), color='black')
fig.savefig('data/lysine-rm.png')
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['FISHMEAL_PRICE']), color='black')
fig.savefig('data/lysine-fishmeal.png')
fig, ax = plt.subplots(figsize=(20, 16))
fig.savefig('data/lysine-wheat.png')
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['PIGLET_PRICE']), color='black')
fig.savefig('data/lysine-piglet.png')
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['SOW_PRICE']), color='black')
fig.savefig('data/lysine-sow.png')
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['PORK_PRICE']), color='black')
fig.savefig('data/lysine-pork.png')


#Lasso Regression
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=1)
clf = linear_model.Lasso(alpha=1)
clf.fit(x_train, y_train)
clf.coef_
clf.intercept_
clf.predict(x_test)
y_test
np.corrcoef(clf.predict(x_test),y_test)**2
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(y_train)
plt.plot(clf.predict(x_train))
mean_absolute_error(y_train, clf.predict(x_train))
fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(y_test)
plt.plot(clf.predict(x_test))
mean_absolute_error(y_test, clf.predict(x_test))
mean_absolute_percentage_error(y_test, clf.predict(x_test))
df.describe()
#LSTM
df.head()
df[df.index<'2020-01-01']
