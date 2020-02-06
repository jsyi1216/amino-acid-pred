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

def getDataframe():
    dsn_tns = cx_Oracle.makedsn('10.188.217.220', '1521', service_name='xe') # if needed, place an 'r' before any parameter in order to address special characters such as '\'.
    conn = cx_Oracle.connect(user=r'jsyi', password='cj123', dsn=dsn_tns) # if needed, place an 'r' before any parameter in order to address special characters such as '\'. For example, if your user name contains '\', you'll need to place 'r' before the user name: user=r'User Name'
    df = pd.read_sql_query("select * from lysine_price", conn)
    conn.close()
    
    return df

df = getDataframe()
df.corr()
sns.pairplot(df)
sns.heatmap(df.corr(), annot=True)



# split by day
year = df["DAY"].dt.to_period("Y")
agg = df.groupby([year])
for group in agg:
    print(group)

df = df.set_index('DAY')
y_data = df['LYSINE_PRICE'].values
x_data = df.iloc[:,1:].values
#x_data = df[['x1','x9','x10','x11']].values
n_features = len(x_data[0])

df.columns
df['LYSINE_PRICE'].plot()
df['CORN_PRICE'].plot()
df['SBM_PRICE'].plot()
df['CM_PRICE'].plot()
df['RM_PRICE'].plot()
df['FISHMEAL_PRICE'].plot()
df['WHEAT_PRICE'].plot()
df['PIGLET_PRICE'].plot()
df['SOW_PRICE'].plot()
df['PORK_PRICE'].plot()
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=1)
plt.plot(preprocessing.scale(df['CORN_PRICE']), color='black')
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='black', markersize=1)
plt.plot(preprocessing.scale(df['SBM_PRICE']))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='black', markersize=1)
plt.plot(preprocessing.scale(df['CM_PRICE']))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='black', markersize=1)
plt.plot(preprocessing.scale(df['RM_PRICE']))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='black', markersize=1)
plt.plot(preprocessing.scale(df['FISHMEAL_PRICE']))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='black', markersize=1)
plt.plot(preprocessing.scale(df['WHEAT_PRICE']))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='black', markersize=1)
plt.plot(preprocessing.scale(df['PIGLET_PRICE']))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='black', markersize=1)
plt.plot(preprocessing.scale(df['SOW_PRICE']))
plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='black', markersize=1)
plt.plot(preprocessing.scale(df['PORK_PRICE']))

plt.plot(preprocessing.scale(df['SBM_PRICE']), color='blue')
plt.plot(preprocessing.scale(df['CORN_PRICE']), color='black')
plt.plot(preprocessing.scale(df['WHEAT_PRICE']), color='red')


#Lasso Regression
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=1)
clf = linear_model.Lasso(alpha=1)
clf.fit(x_train, y_train)
clf.coef_
clf.intercept_
clf.predict(x_test)
y_test
np.corrcoef(clf.predict(x_test),y_test)**2
plt.plot(y_train)
plt.plot(clf.predict(x_train))
plt.plot(y_test)
plt.plot(clf.predict(x_test))

#LSTM
df.head()
df[df.index<'2020-01-01']
