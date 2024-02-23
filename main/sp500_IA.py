# MAIN
"""
Created on Mon Jan  8 22:54:48 2024
@author: javier
"""
from modules.mod_init import *
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing
from paths.paths import *
from columns.columns import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Embedding, Reshape, BatchNormalization
import pandas as pd

from pathlib import Path
results_path = Path('results', 'lstm_embeddings')

print(f'START MAIN')

# YAHOO Call
symbol = "^GSPC"

start_date = "1980-01-01"
endin_date = "2023-12-31"
sp500_data = yf.download(symbol, start=start_date, end=endin_date)

# SAVE yahoo file
sp500_data.to_csv(path_file_csv)
#print(f"The data has been saved to: {path_file_csv}")

#READING yahoo file
df_data = pd.read_csv(path_file_csv, header=None, skiprows=1, names=columns_csv_yahoo)

#CALL module Datacleaning
df_data_clean = mod_dtset_clean(df_data,start_date,endin_date)

#CALL module Preprocessing-Range
filter_start_date = '2000-01-01'
filter_endin_date = '2019-12-31'
df_preprocessing = mod_preprocessing(df_data_clean,filter_start_date,filter_endin_date)

#X_train y_train X_test y_test

cutoff = '2017-12-31'

train_data = df_preprocessing[df_preprocessing.date < cutoff].copy()
tests_data = df_preprocessing[df_preprocessing.date > cutoff].copy()
#print(train_data.head())

window_size = 5
n_features = 1

n_day_weeks = train_data.day_week.nunique()
lag_columns = df_preprocessing.columns[df_preprocessing.columns.str.startswith('lag')]

#X_train & y_train
#------------------------------------------------------------------------------
X_train = df_preprocessing[df_preprocessing.date < cutoff][lag_columns].values.reshape(-1, len(lag_columns), 1)
y_train = df_preprocessing[df_preprocessing.date < cutoff]['direction']

X_test = df_preprocessing[df_preprocessing.date >= cutoff][lag_columns].values.reshape(-1, len(lag_columns), 1)
y_test = df_preprocessing[df_preprocessing.date >= cutoff]['direction']

#------------------------------------------------------------------------------

#INPUT LAYERS
#------------------------------------------------------------------------------

r_lags = Input(shape=(window_size, n_features),
                name='r_lags')

day_week = Input(shape=(1,),
                name='day_week')

#------------------------------------------------------------------------------

#LSTM LAYERS
#------------------------------------------------------------------------------
lstm1_units = 25
lstm2_units = 10

lstm1 = LSTM(units=lstm1_units, 
             input_shape=(window_size, 
                          n_features), 
             name='LSTM1', 
             dropout=.2,
             return_sequences=True)(r_lags)

lstm_model = LSTM(units=lstm2_units, 
             dropout=.2,
             name='LSTM2')(lstm1)

#EMBEDDINGS LAYER
#------------------------------------------------------------------------------
day_week_embedding = Embedding(input_dim=n_day_weeks, 
                             output_dim=5, 
                             input_length=1)(day_week)
day_week_embedding = Reshape(target_shape=(5,))(day_week_embedding)


#CONCATENATE MODEL COMPONENTS
#------------------------------------------------------------------------------
merged = concatenate([lstm_model, 
                      day_week_embedding], name='Merged')
                      
bn = BatchNormalization()(merged)
hidden_dense = Dense(10, name='FC1')(bn)

output = Dense(1, name='Output', activation='sigmoid')(hidden_dense)

rnn = Model(inputs=[r_lags, day_week], outputs=output)
rnn.summary()

#TRAIN MODEL
#------------------------------------------------------------------------------
optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

rnn.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 
                     tf.keras.metrics.AUC(name='AUC')])

lstm_path = (results_path / 'lstm.classification.h5').as_posix()

checkpointer = ModelCheckpoint(filepath=lstm_path,
                               verbose=1,
                               monitor='val_AUC',
                               mode='max',
                               save_best_only=True)

early_stopping = EarlyStopping(monitor='val_AUC', 
                              patience=5,
                              restore_best_weights=True,
                              mode='max')

training = rnn.fit(X_train,
                   y_train,
                   epochs=50,
                   batch_size=32,
                   validation_data=(X_test, y_test),
                   callbacks=[early_stopping, checkpointer],
                   verbose=1)

print(f'ENDIN MAIN')