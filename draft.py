#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:07:22 2024

@author: javi
"""
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
merged = concatenate([lstm_model,day_week_embedding], name='Merged')
                      
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
                   validation_data=(X_tests, y_tests),
                   callbacks=[early_stopping, checkpointer],
                   verbose=1)

#X_train y_train
#------------------------------------------------------------------------------
lag_sequences_tr = df_preprocessing[df_preprocessing.date < cutoff][lag_columns].values.reshape(-1, len(lag_columns), 1)
day_week_data_tr = df_preprocessing[df_preprocessing.date < cutoff]['day_week'].values.reshape(-1, 1)

X_train = [lag_sequences_tr, day_week_data_tr]
y_train = df_preprocessing[df_preprocessing.date < cutoff]['direction']

#X_tests & y_tests
#------------------------------------------------------------------------------
lag_sequences_ts = df_preprocessing[df_preprocessing.date >= cutoff][lag_columns].values.reshape(-1, len(lag_columns), 1)
day_week_data_ts = df_preprocessing[df_preprocessing.date >= cutoff]['day_week'].values.reshape(-1, 1)
X_tests = [lag_sequences_ts, day_week_data_ts]
y_tests = df_preprocessing[df_preprocessing.date >= cutoff]['direction']