#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from pprint import pprint
from pylab import plt, mpl

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense,Dropout
from sklearn.metrics import accuracy_score

plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.set_option('display.precision', 4)
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seeds()

#fINANCIAL EXAMPLE
url = 'http://hilpisch.com/aiif_eikon_id_eur_usd.csv'

symbol = 'EUR_USD'
raw = pd.read_csv(url, index_col=0, parse_dates=True)
def generate_data():
    data = pd.DataFrame(raw['CLOSE'])
    data.columns = [symbol]
    data = data.resample('30min', label='right').last().ffill()
    return data

data = generate_data()
data = (data - data.mean()) / data.std()
p = data[symbol].values
p = p.reshape((len(p), -1))

lags = 5    
g = TimeseriesGenerator(p, p, length=lags, batch_size=5)

def create_rnn_model(hu=100, lags=lags, layer='SimpleRNN',
                           features=1, algorithm='estimation'):
    model = Sequential()
    if layer == 'SimpleRNN':
        model.add(SimpleRNN(hu, activation='relu',
                            input_shape=(lags, features)))
    else:
        model.add(LSTM(hu, activation='relu',
                       input_shape=(lags, features)))
    if algorithm == 'estimation':
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

model = create_rnn_model()

model.fit(g, epochs=500, steps_per_epoch=10,
          verbose=False)

y = model.predict(g, verbose=False)

data['pred'] = np.nan
data['pred'].iloc[lags:] = y.flatten()

data[[symbol, 'pred']].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75);

data[[symbol, 'pred']].iloc[50:100].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75);

data = generate_data()

data['r'] = np.log(data / data.shift(1))
data.dropna(inplace=True)
data = (data - data.mean()) / data.std()
r = data['r'].values
r = r.reshape((len(r), -1))
g = TimeseriesGenerator(r, r, length=lags, batch_size=5)
model = create_rnn_model()

model.fit(g, epochs=500, steps_per_epoch=10,
          verbose=False)
y = model.predict(g, verbose=False)

data['pred'] = np.nan
data['pred'].iloc[lags:] = y.flatten()
data.dropna(inplace=True)

data[['r', 'pred']].iloc[50:100].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75);
plt.axhline(0, c='grey', ls='--');

accuracy_score(np.sign(data['r']), np.sign(data['pred']))


split = int(len(r) * 0.8)
train = r[:split]
test = r[split:]
g = TimeseriesGenerator(train, train, length=lags, batch_size=5)

set_seeds()
model = create_rnn_model(hu=100)
model.fit(g, epochs=100, steps_per_epoch=10, verbose=False)
g_ = TimeseriesGenerator(test, test, length=lags, batch_size=5)
y = model.predict(g_)
accuracy_score(np.sign(test[lags:]), np.sign(y))

#FINNANCIAL FEATURES

data = generate_data()
data['r'] = np.log(data / data.shift(1))
window = 20
data['mom'] = data['r'].rolling(window).mean()
data['vol'] = data['r'].rolling(window).std()
data.dropna(inplace=True)

#ESTIMATION
split = int(len(data) * 0.8)
train = data.iloc[:split].copy()
mu, std = train.mean(), train.std()
train = (train - mu) / std
test = data.iloc[split:].copy()
test = (test - mu) / std
g = TimeseriesGenerator(train.values, train['r'].values,
                        length=lags, batch_size=5)

set_seeds()
model = create_rnn_model(hu=100, features=len(data.columns),
                         layer='SimpleRNN')

model.fit(g, epochs=100, steps_per_epoch=10,
                verbose=False)

g_ = TimeseriesGenerator(test.values, test['r'].values,
                         length=lags, batch_size=5)

y = model.predict(g_).flatten()

accuracy_score(np.sign(test['r'].iloc[lags:]), np.sign(y))

#CLASSIFICATION

set_seeds()
model = create_rnn_model(hu=50,
            features=len(data.columns),
            layer='LSTM',
            algorithm='classification')

train_y = np.where(train['r'] > 0, 1, 0)

np.bincount(train_y)

def cw(a):
    c0, c1 = np.bincount(a)
    w0 = (1 / c0) * (len(a)) / 2
    w1 = (1 / c1) * (len(a)) / 2
    return {0: w0, 1: w1}

g = TimeseriesGenerator(train.values, train_y,
                        length=lags, batch_size=5)

model.fit(g, epochs=5, steps_per_epoch=10,
          verbose=False, class_weight=cw(train_y))

test_y = np.where(test['r'] > 0, 1, 0)

g_ = TimeseriesGenerator(test.values, test_y,
                         length=lags, batch_size=5)

y = np.where(model.predict(g_, batch_size=None) > 0.5,
             1, 0).flatten()

np.bincount(y)

accuracy_score(test_y[lags:], y)

#DEEP NUERAL

def create_deep_rnn_model(hl=2, hu=100, layer='SimpleRNN',
                          optimizer='rmsprop', features=1,
                          dropout=False, rate=0.3, seed=100):
    if hl <= 2: hl = 2
    if layer == 'SimpleRNN':
        layer = SimpleRNN
    else:
        layer = LSTM
    model = Sequential()
    model.add(layer(hu, input_shape=(lags, features),
                     return_sequences=True,
                    ))
    if dropout:
        model.add(Dropout(rate, seed=seed))
    for _ in range(2, hl):
        model.add(layer(hu, return_sequences=True))
        if dropout:
            model.add(Dropout(rate, seed=seed))
    model.add(layer(hu))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

set_seeds()
model = create_deep_rnn_model(
            hl=2, hu=50, layer='SimpleRNN',
            features=len(data.columns),
            dropout=True, rate=0.3)

model.summary()

model.fit(g, epochs=200, steps_per_epoch=10,
          verbose=False, class_weight=cw(train_y))

y = np.where(model.predict(g_, batch_size=None) > 0.5,
             1, 0).flatten()

np.bincount(y)

accuracy_score(test_y[lags:], y)