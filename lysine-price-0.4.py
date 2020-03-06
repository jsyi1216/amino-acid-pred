# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:39:07 2020

@author: jsyi
"""

import cx_Oracle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from keras import backend as K
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.models import model_from_json
import itertools

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
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)
        y.append(seq_y[0])
    return np.array(X), np.array(y)

df = getDataframe()
df.to_csv('dataset_200306.csv', index=False)
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
df = df.interpolate()
df = df.dropna()
df.corr(method ='pearson')
plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(method ='pearson'), annot=True)
#sns.pairplot(df)

df = df.set_index('DAY')
with pd.option_context('display.max_columns', len(df.columns)):
    print(df.describe(include='all'))

df['LYSINE_PRICE'].plot()
df['EXPORT_PRICE'].plot()
df['LYSINE_PRICE'].hist()
df['EXPORT_PRICE'].hist()
np.log1p(df['LYSINE_PRICE']).hist()
np.log1p(df['EXPORT_PRICE']).hist()
#fig, ax = plt.subplots(figsize=(20, 16))

plt.plot(preprocessing.scale(df['LYSINE_PRICE']), '.', color='red', markersize=5)
plt.plot(preprocessing.scale(df['EXPORT_PRICE']), color='black')
#fig.savefig('data/lysine-corn.png')


# define input sequence
raw_seq = df.values
plt.plot(raw_seq)

n_steps = 10
n_seq = 1
n_features = len(df.columns)
# split into samples
X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
raw_seq.shape
X.shape
y.shape

# split train and test set
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=100, kernel_size=4, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dropout(0.25)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mae', metrics=[r_squared])
model.summary()
# fit model
history = model.fit(x_train, y_train, epochs=10000, verbose=1)
X.shape
y.shape

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['r_squared'])
plt.title('model accuracy')
plt.ylabel('r_squared')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# demonstrate prediction
tr_pred = list(itertools.chain(*model.predict(x_train)))
plt.plot(y_train)
plt.plot(tr_pred)
print(np.corrcoef(y_train,tr_pred)**2)
print(mean_absolute_error(y_train, tr_pred))
print(mean_absolute_percentage_error(y_train, tr_pred))

te_pred = list(itertools.chain(*model.predict(x_test)))
plt.plot(y_test)
plt.plot(te_pred)
print(np.corrcoef(y_test,te_pred)**2)
print(mean_absolute_error(y_test, te_pred))
print(mean_absolute_percentage_error(y_test, te_pred))
np.mean(te_pred)

model.predict(x_test)

# serialize model to JSON
model_json = model.to_json()
with open("model/model_20200306.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/model_20200306.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model/model_20200306.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model/model_20200306.h5")
print("Loaded model from disk")



df[(df.index<'2020-01-01')&(df.index>'2018-01-01')]
X[110].shape
x_test.shape

x_eval=[]
x_eval.append(X[110])
x_eval=np.array(x_eval)
model.predict(x_eval)
