# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:41:45 2020

@author: jsyi
"""


import cx_Oracle
import pandas as pd
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
from keras import backend as K
import itertools
import seaborn as sns
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

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_steps]

        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

df = getDataframe()
df = df.dropna()
df.corr(method ='pearson')
sns.heatmap(df.corr(method ='pearson'), annot=True)
#sns.pairplot(df)
df.head()

df = df.set_index('DAY')
y_data = df['LYSINE_PRICE_M'].values
x_data = df.iloc[:,1:].values

train_data = df[df.index<'2017-01-01']
test_data = df[(df.index<'2020-01-01')&(df.index>'2018-01-01')]
y_train = train_data['LYSINE_PRICE_M'].values
x_train = train_data.iloc[:,1:].values
y_test = test_data['LYSINE_PRICE_M'].values
x_test = test_data.iloc[:,1:].values

plt.plot(y_train)
plt.plot(y_test)
plt.plot(x_train)
plt.plot(x_test)

# define input sequence
raw_seq = df.values
plt.plot(raw_seq)

n_steps = 10
n_seq = 1
n_features = 1
# split into samples
X, y = split_sequence(raw_seq, n_steps)
y[:20]
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

# split train and test set
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=[r_squared])
model.summary()
len(X)
# fit model
history = model.fit(x_train, y_train, epochs=10000, verbose=1)

# demonstrate prediction
tr_pred = list(itertools.chain(*model.predict(x_train)))
plt.plot(y_train[100:200])
plt.plot(tr_pred[100:200])

print(np.corrcoef(y_train,tr_pred)**2)

te_pred = list(itertools.chain(*model.predict(x_test)))
plt.plot(y_test)
plt.plot(te_pred)

print(np.corrcoef(y_test,te_pred)**2)
