# MAIN
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""

import yfinance as yf
import pandas as pd
from pathlib import Path


from modules.mod_init import *
from paths.paths import file_df_data,folder_csv,path_file_csv
from columns.columns import columns_csv_yahoo,columns_clean_order
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from sklearn.preprocessing import FunctionTransformer, StandardScaler,MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Embedding, Reshape, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score


results_path = Path('results', 'lstm_embeddings')

print(f'START MAIN')

# YAHOO CALL + SAVE + READING file
#------------------------------------------------------------------------------
symbol = "^GSPC"
start_date = "1980-01-01"
endin_date = "2023-12-31"
sp500_data = yf.download(symbol, start=start_date, end=endin_date)
sp500_data.to_csv(path_file_csv)
df_data = pd.read_csv(path_file_csv, header=None, skiprows=1, names=columns_csv_yahoo)

#print(f"The data has been saved to: {path_file_csv}")

#CALL module Datacleaning
#------------------------------------------------------------------------------

df_data_clean = mod_dtset_clean(df_data,start_date,endin_date)

#CALL PREPROCESSING
#------------------------------------------------------------------------------

filter_start_date = '2000-01-01'
filter_endin_date = '2018-12-31'
df_preprocessing = mod_preprocessing(df_data_clean,filter_start_date,filter_endin_date)
#df_preprocessing.info()

#DATA NORMALIZATION
#------------------------------------------------------------------------------
scaler = StandardScaler()
norm_columns = df_preprocessing.columns[df_preprocessing.columns.str.startswith('lag')]
df_preprocessing[norm_columns] = scaler.fit_transform(df_preprocessing[norm_columns])

print(df_preprocessing)

#TRAINING & TEST DATA
#------------------------------------------------------------------------------

cutoff = '2017-12-31'

train_data = df_preprocessing[df_preprocessing.date < cutoff].copy()
tests_data = df_preprocessing[df_preprocessing.date > cutoff].copy()

window_size = 5
n_features = 1
n_day_weeks = train_data.day_week.nunique()
lag_columns = df_preprocessing.columns[df_preprocessing.columns.str.startswith('lag')]

#X_train y_train
#------------------------------------------------------------------------------
lag_sequences_tr = df_preprocessing[df_preprocessing.date < cutoff][lag_columns].values.reshape(-1, len(lag_columns), 1)

X_train = [lag_sequences_tr]
y_train = df_preprocessing[df_preprocessing.date < cutoff]['direction']

#X_tests & y_tests
#------------------------------------------------------------------------------
lag_sequences_ts = df_preprocessing[df_preprocessing.date >= cutoff][lag_columns].values.reshape(-1, len(lag_columns), 1)

X_tests = [lag_sequences_ts]
y_tests = df_preprocessing[df_preprocessing.date >= cutoff]['direction']

#INPUTS
#------------------------------------------------------------------------------

r_lags = Input(shape=(window_size, n_features),name='r_lags')
print("Shape de r_lags:", r_lags.shape)

#LSTM LAYERS
#------------------------------------------------------------------------------
lstm1_units = 25
lstm2_units = 10

lstm1 = LSTM(units=lstm1_units,input_shape=(window_size,n_features),dropout=0.2,
             name='LSTM1',return_sequences=True)(r_lags) 

lstm_model = LSTM(units=lstm2_units,dropout=0.2,name='LSTM2')(lstm1)

# DENSE LAYER
#------------------------------------------------------------------------------
bn = BatchNormalization()(lstm_model)
hidden_dense = Dense(10, name='FC1')(bn)

# OUTPUT LAYER
#------------------------------------------------------------------------------
output = Dense(1, name='Output', activation='sigmoid')(hidden_dense)

# MODEL DEFINITION
#------------------------------------------------------------------------------
rnn = Model(inputs=[r_lags], outputs=output)
rnn.summary()

#TRAIN MODEL
#------------------------------------------------------------------------------
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1, write_grads=True)

rnn.compile(loss='binary_crossentropy',optimizer=optimizer,
            metrics=['accuracy',tf.keras.metrics.AUC(name='AUC')])


lstm_path = (results_path / 'lstm.classification.h5').as_posix()

checkpointer = ModelCheckpoint(filepath=lstm_path,
                               verbose=1,
                               monitor='val_AUC',
                               mode='max',
                               save_best_only=True)

early_stopping = EarlyStopping(monitor='val_AUC', 
                              patience=10,
                              restore_best_weights=True,
                              mode='max')

training = rnn.fit(X_train,
                   y_train,
                   epochs=24,
                   batch_size=32,
                   validation_data=(X_tests, y_tests),
                   callbacks=[early_stopping, checkpointer, tensorboard_callback],
                   verbose=1)

#PREDICTIONS
#------------------------------------------------------------------------------
evaluation = rnn.evaluate(X_tests, y_tests)
print("Evaluation Loss:", evaluation[0])
print("Evaluation Accuracy:", evaluation[1])
print("Evaluation AUC:", evaluation[2])

predictions = rnn.predict(X_tests).squeeze()
print(predictions)
predicted_labels = (predictions > 0.5).astype(int)
#print(predicted_labels)

# Crear un DataFrame con las predicciones y los valores reales
df_results = pd.DataFrame({'y_tests': y_tests, 'Predicted': predicted_labels})

df_results.to_excel('y_tests vs Predicted.xlsx')#, index=False)
print("Saved y_tests vs Predicted.xlsx")

print(f'ENDIN MAIN')
