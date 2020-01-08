# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:50:13 2019

@author: jsyi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:22:58 2019

@author: jsyi
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras import backend as K
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import itertools
from sklearn import linear_model

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# hyperparameters
batch_size = 128
epochs = 30000
n_features = 4
df = pd.read_csv('data/dataset.csv')

y_data = df['y'].values
#x_data = df.iloc[:,1:].values
x_data = df[['x1','x9','x10','x11']].values
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=1)

#class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
x_train = x_train.reshape(x_train.shape[0], n_features, 1)
x_val = x_val.reshape(x_val.shape[0], n_features, 1)
x_test = x_test.reshape(x_test.shape[0], n_features, 1)
input_shape = (n_features, 1)
print('the shape of train X: ',x_train.shape)
print('the shape of validation X: ',x_val.shape)
print('the shape of test X: ',x_test.shape)

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

model = Sequential()
model.add(Conv1D(filters=50, kernel_size=2, activation='relu', input_shape=input_shape))
model.add(Conv1D(filters=50, kernel_size=2, activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r_squared])
model.summary()

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, verbose=0)

y_pred = list(itertools.chain(*model.predict(x_test)))
y_test
y_pred
np.corrcoef(y_pred,y_test)**2

print('Test loss:', score[0])
print('Test r squared value:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['r_squared'])
plt.plot(history.history['val_r_squared'])
plt.title('model correlation')
plt.ylabel('r squared value')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


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
