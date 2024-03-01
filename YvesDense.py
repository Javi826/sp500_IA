#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""

import os
import numpy as np
import pandas as pd
from pylab import plt, mpl
plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.set_option('display.precision', 4)
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'

url = 'http://hilpisch.com/aiif_eikon_id_eur_usd.csv'

symbol = 'EUR_USD'

raw = pd.read_csv(url, index_col=0, parse_dates=True)

raw.head()

raw.info()

data = pd.DataFrame(raw['CLOSE'].loc[:])
data.columns = [symbol]

data = data.resample('1h', label='right').last().ffill()

data.info()
data.plot(figsize=(10, 6));

lags = 5

def add_lags(data, symbol, lags, window=20):
    cols = []
    df = data.copy()
    df.dropna(inplace=True)
    df['r'] = np.log(df / df.shift())
    df['sma'] = df[symbol].rolling(window).mean()
    df['min'] = df[symbol].rolling(window).min()
    df['max'] = df[symbol].rolling(window).max()
    df['mom'] = df['r'].rolling(window).mean()
    df['vol'] = df['r'].rolling(window).std()
    df.dropna(inplace=True)
    df['d'] = np.where(df['r'] > 0, 1, 0)
    features = [symbol, 'r', 'd', 'sma', 'min', 'max', 'mom', 'vol']
    for f in features:
        for lag in range(1, lags + 1):
            col = f'{f}_lag_{lag}'
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols

data, cols = add_lags(data, symbol, lags)
print(len(cols))
print(cols)
c = data['d'].value_counts()

def cw(df):
    c0, c1 = np.bincount(df['d'])
    w0 = (1 / c0) * (len(df)) / 2
    w1 = (1 / c1) * (len(df)) / 2
    return {0: w0, 1: w1}

class_weight = cw(data)
class_weight
class_weight[0] * c[0]
class_weight[1] * c[1]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import random
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)

def create_model(hl=1, hu=128, optimizer=optimizer):
    model = Sequential()
    model.add(Dense(hu, input_dim=len(cols),
                    activation='relu'))
    for _ in range(hl):
        model.add(Dense(hu, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

set_seeds()
model = create_model(hl=1, hu=128)

model.fit(data[cols], data['d'], epochs=50,
          verbose=False, class_weight=cw(data))

model.evaluate(data[cols], data['d'])

data['p'] = np.where(model.predict(data[cols]) > 0.5, 1, 0)

data['p'].value_counts()

split = int(len(data) * 0.8)
train = data.iloc[:split].copy()
test = data.iloc[split:].copy()
set_seeds()
model = create_model(hl=1, hu=128)

hist = model.fit(train[cols], train['d'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False,
          class_weight=cw(train))

model.evaluate(train[cols], train['d'])
model.evaluate(test[cols], test['d'])
test['p'] = np.where(model.predict(test[cols]) > 0.5, 1, 0)
test['p'].value_counts()
res = pd.DataFrame(hist.history)
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--');