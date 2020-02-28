# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:51:49 2020

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
    df = pd.read_sql_query("select * from aminoacid_price_month_vw order by day", conn)
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
    'WHEAT_PRICE',
    'CORN_PRICE',
    'SOYBEAN_PRICE',
    'WTI_PRICE',
    'PPI',
    'CPI',
    'EXPORT_QTY',
    'EXPORT_PRICE'
]]
df = df.set_index('DAY')
y_data = df['LYSINE_PRICE'].values
x_data = df.iloc[:,1:].values
#x_data = df[['x1','x9','x10','x11']].values
n_features = len(x_data[0])

df.columns
df['LYSINE_PRICE'].plot()
df['WHEAT_PRICE'].plot()
df['CORN_PRICE'].plot()
df['SOYBEAN_PRICE'].plot()
df['EXCHANGE'].plot()
df['WTI_PRICE'].plot()
df['PPI'].plot()
df['CPI'].plot()
df['EXPORT_PRICE'].plot()

#fig, ax = plt.subplots(figsize=(20, 16))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['WHEAT_PRICE']), color='black')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['CORN_PRICE']), color='black')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['SOYBEAN_PRICE']), color='black')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['EXCHANGE']), color='black')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['WTI_PRICE']), color='black')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['PPI']), color='black')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['CPI']), color='black')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['EXPORT_PRICE']), color='black')
#fig.savefig('data/lysine-corn.png')



#Lasso Regression
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=1)
clf = linear_model.Lasso(alpha=1)
clf.fit(x_train, y_train)
clf.coef_
clf.intercept_
mean_absolute_error(y_test,clf.predict(x_test))
mean_absolute_percentage_error(y_test,clf.predict(x_test))

clf.predict(x_test)
y_test
np.corrcoef(clf.predict(x_train),y_train)**2
plt.plot(y_train)
plt.plot(clf.predict(x_train))
np.corrcoef(clf.predict(x_test),y_test)**2
plt.plot(y_test)
plt.plot(clf.predict(x_test))

resultDf=pd.DataFrame({'observed':clf.predict(x_test),'predicted':y_test})
resultDf.corr()
sns.pairplot(resultDf)
resultDf.corr(method ='pearson')**2
sns.heatmap(resultDf.corr(method ='pearson'), annot=True)

#LSTM
train_data = df[df.index<'2017-01-01']
test_data = df[(df.index<'2020-01-01')&(df.index>'2018-01-01')]
y_train = train_data['LYSINE_PRICE'].values
x_train = train_data.iloc[:,1:].values
y_test = test_data['LYSINE_PRICE'].values
x_test = test_data.iloc[:,1:].values

plt.plot(y_train)
plt.plot(y_test)
plt.plot(x_train)
plt.plot(x_test)

clf = linear_model.Lasso(alpha=1)
clf.fit(x_train, y_train)
clf.coef_
clf.intercept_

clf.predict(x_test)
y_test
mean_absolute_error(y_test,clf.predict(x_test))
mean_absolute_percentage_error(y_test,clf.predict(x_test))

np.corrcoef(clf.predict(x_train),y_train)**2
plt.plot(y_train)
plt.plot(clf.predict(x_train))
np.corrcoef(clf.predict(x_test),y_test)**2
plt.plot(y_test)
plt.plot(clf.predict(x_test))