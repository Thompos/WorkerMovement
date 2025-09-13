# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:01:15 2024

@author: 
"""

#%%
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
from tensorflow.keras import layers, Sequential
import time
from sklearn.model_selection import GridSearchCV, ParameterGrid
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers, callbacks 
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model
import joblib
from keras.layers import Bidirectional, LSTM, Dense
from keras.initializers import GlorotUniform, HeUniform, RandomUniform
from keras.layers import Dense, Dropout
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from sklearn.metrics import make_scorer, accuracy_score, f1_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

# Run a script from another script
exec(open(f'{path_stem}UWB functions.py').read())

#%%
I3_model = load_model('best_move_modI3b_callback.h5')

Train_set = pd.read_pickle('Train_set.pkl')

scaler = joblib.load('I2_move_mod_scaler.pkl')

import pickle

with open('UWB_datasets_9pred.pkl', 'rb') as f:
    datasets = pickle.load(f)

#%%
I3_model.summary()
#%%
#%%    
#%%
#%%
selected_columns = ['difX', 'difY', 'dif_time', 'ed_dif']  # 'difSp',
#%%
#%%
X = Train_set[selected_columns].reset_index(drop=True)
y = Train_set['move_cat'].reset_index(drop=True)
#%%
#%%
temp = np.where(pd.isnull(X))
nanX_positions = list(set(temp[0]) )
X = X.drop(nanX_positions).reset_index(drop=True)
y = y.drop(nanX_positions).reset_index(drop=True)

#%%
window_size = 32
padding = pd.DataFrame({'difX': [0]*(window_size-1), 'difY': [0]*(window_size-1),
                        'dif_time': [0]*(window_size-1), 'ed_dif': [0]*(window_size-1) })
#%%
X_padded = pd.concat([padding, X], ignore_index=True)
X_padded = pd.concat([X_padded, padding], ignore_index=True)

#%%
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_padded)
#%%
#%%
#%%
Xpast = []
Xfuture = []

n_features = X_sc.shape[1]

for i in range( len(y) ):
        Xpast.append(X_sc[i:(i+window_size) ])
        Xfuture.append(X_sc[(i+2*window_size-2):(i+window_size-2):-1 ])
       
      
Xpast, Xfuture = np.array(Xpast), np.array(Xfuture)

#%%
#%%
temp = np.where(pd.isnull(y))
nanY_positions = list(set(temp[0]) )
#%%
#%%
TV_y = y.drop(nanY_positions).reset_index(drop=True)
TV_past = np.delete(Xpast, nanY_positions, axis=0)
TV_past = TV_past[np.arange(TV_past.shape[0])]

TV_future = np.delete(Xfuture, nanY_positions, axis=0)
TV_future = TV_future[np.arange(TV_future.shape[0])]

#%%
if len(TV_y) == len(TV_past) == len(TV_future) and np.all(TV_y.index == np.arange(len(TV_past))) and np.all(TV_y.index == np.arange(len(TV_future))):
    print("Indices match after removing instances.")
else:
    print("Indices do not match after removing instances.")

#%%
#%%
#%%
trainsize = int(0.8*len(TV_y))
validsize = len(TV_y) - trainsize
#%%
trainXpast = TV_past[:trainsize]
trainXfuture = TV_future[:trainsize]
validXpast = TV_past[trainsize:]
validXfuture = TV_future[trainsize:]

trainY = np.array(TV_y[:trainsize] )
validY = np.array(TV_y[trainsize:] )
TV_y = np.array(TV_y)
#%%
#%%
def create_model(features, timesteps, # past_array, future_array, 
                 hidden_layers=1, dense_units=12, dense_activation='relu',
                 dense_initializer='he_uniform',
                 lstm_units=6, lstm_activation = 'tanh', lstm_initializer='orthogonal',
                 dropout_rate=0):
    
    past_input = Input(shape=(timesteps, features), name='past_input')
    future_input = Input(shape=(timesteps, features), name='future_input')
            
    lstm_past = LSTM(units=lstm_units, name='lstm_past',
                     activation = lstm_activation, kernel_initializer=lstm_initializer )(past_input)
    lstm_past = Dropout(dropout_rate)(lstm_past)
    
    lstm_future = LSTM(units=lstm_units, name='lstm_future',
                     activation = lstm_activation, kernel_initializer=lstm_initializer )(future_input)
    lstm_future = Dropout(dropout_rate)(lstm_future)
    
    concat = Concatenate()([lstm_past, lstm_future])
    
    x = Dense(dense_units, activation=dense_activation, kernel_initializer=dense_initializer)(concat)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
          
    for _ in range(hidden_layers - 1):
        x = Dense(dense_units, activation=dense_activation, kernel_initializer=dense_initializer)(x)
        x = Dropout(dropout_rate)(x)
        
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[past_input, future_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
#%%
#%%
#%%
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
#%%
#%%
keras_classifier = KerasClassifier(build_fn=create_model, verbose=0)
 
param_grid = [{
    'hidden_layers': [1], # 1
    'dense_units': [15,35], # lower than 50
    'lstm_units': [25,40], # higher than 15
    'dropout_rate': [0], 
    'dense_activation': ['relu'],
    'dense_initializer': ['he_uniform']
}]

manual_param_grid = list(ParameterGrid(param_grid) )

scorer = make_scorer(accuracy_score) 
#%%
#%%
#%%
accuracies = []
f1_scores = []
validation_losses = []

for perm in manual_param_grid:
        
        model = create_model(trainXpast.shape[2], trainXpast.shape[1],
                             hidden_layers = perm['hidden_layers'],
                             dense_units = perm['dense_units'],
                             lstm_units = perm['lstm_units'],
                             dropout_rate = perm['dropout_rate'], 
                             dense_activation = perm['dense_activation'],
                             dense_initializer = perm['dense_initializer'] )
        
        history = model.fit(x=[trainXpast, trainXfuture], y=trainY,
                                validation_data=([validXpast, validXfuture], validY),
                             #class_weight=class_weights_dict,
                              epochs=80, batch_size=32, verbose=1, callbacks=[early_stopping])
           
        y_pred = (model.predict([validXpast, validXfuture]) > 0.5).astype(int) 
        accuracy = accuracy_score(validY, y_pred)
        f1 = f1_score(validY, y_pred, average='weighted')
        val_loss = history.history['val_loss'][-1]

        accuracies.append(accuracy)
        f1_scores.append(f1)
        validation_losses.append(val_loss)
        print(perm)

#%%
best_index = np.argmax(f1_scores)
print(max(f1_scores))
best_params = manual_param_grid[best_index]
print(best_params)
print(max(accuracies))
#%%
#  0.9886
#%%
checkpoint_filepath = 'lstm_move_modI4_callback.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

#%%
model = create_model(TV_past.shape[2], TV_past.shape[1],
                     hidden_layers = best_params['hidden_layers'],
                     dense_units = best_params['dense_units'],
                     lstm_units = best_params['lstm_units'],
                     dropout_rate = best_params['dropout_rate'], 
                     dense_activation = best_params['dense_activation'],
                     dense_initializer = best_params['dense_initializer'] )

history = model.fit(x=[TV_past, TV_future], y=TV_y,
                        validation_data=([TV_past, TV_future], TV_y),
                    #class_weight=class_weights_dict,
                      epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping, checkpoint])
#%%   
y_pred = (model.predict([TV_past, TV_future]) > 0.5 ).astype(int)
accuracy = accuracy_score(TV_y,  y_pred)
f1 = f1_score(TV_y, y_pred, average='weighted')
val_loss = history.history['val_loss'][-1]
print('Train set performance measures:')
print(f1)
print(val_loss)
print(accuracy)
    
#%%
#%%
I4_model = load_model('lstm_move_modI4_callback.h5')
#%%
#%% ###########################################################################
#################Making predictions for all workers ###########################################
###############################################################################
#
# Tri_dat = pd.read_pickle('UWB_dat_post_I3pred.pkl')

# Tri_dat_index = Tri_dat.reset_index(drop= True)

# Tri_dat['pred_lstm'] = np.nan

#%%
for r in range(1,7):
    
    dat = datasets[f'rig_data{r}'].reset_index(drop=True)
    dat = dat[selected_columns]
    
    temp = np.where(pd.isnull(dat))
    nanX_positions = list(set(temp[0]) )
    dat_rm = dat.drop(nanX_positions).reset_index(drop=True)
    dat_len = len(dat_rm)

    dat_padded = pd.concat([padding, dat_rm], ignore_index=True)
    dat_padded = pd.concat([dat_padded, padding], ignore_index=True) 

    dat_sc = scaler.transform(dat_padded)
    
    datpast = []
    datfuture = []

    for i in range( dat_len ):
        datpast.append(dat_sc[i:(i+window_size) ])
        datfuture.append(dat_sc[(i+2*window_size-2):(i+window_size-2):-1 ])
       
    datpast, datfuture = np.array(datpast), np.array(datfuture)
    
    pred = (I4_model.predict([datpast, datfuture]) > 0.5 ).astype(int)
    dat['pred_lstm'] = np.nan
    dat.loc[~dat.index.isin(nanX_positions), 'pred_lstm'] = pred
    
    lstmpreds =  dat['pred_lstm'].values
    
    rig_data = datasets[f'rig_data{r}']
    rig_data['pred_lstm'] = lstmpreds
    
    datasets[f'rig_data{r}'] = rig_data

#%%
#%%
#%%
with open('UWB_datasets_10lstm_pred.pkl', 'wb') as f:
    pickle.dump(datasets, f)
#%%
#%%
%matplotlib qt 
animation_fun_pred(datasets['rig_data1'], col = 'pred_lstm', pace = 0.1, testing = False)
#%%
animation_fun_pred(datasets['rig_data2'], col = 'pred_lstm', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data3'], col = 'pred_lstm', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data4'], col = 'pred_lstm', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data5'], col = 'pred_lstm', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data6'], col = 'pred_lstm', pace = 0.01, testing = False, recent = 0.2, atatime = 0.1)
#%%
#%%
#%%