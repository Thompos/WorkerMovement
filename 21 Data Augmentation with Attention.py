# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:01:15 2024

@author: 
"""

#%%
# train_valid_dict = {
#        'Xtrain_past_Alltime': 'Xtrain_future_Alltime'  'Ytrain_Alltime': ,      
#        'Xvalid_past_Alltime'  'Xvalid_future_Alltime'  'Yvalid_Alltime': ,
#        'Xtrain_past_Alltime'  'Xtrain_future_Alltime'  'Ytrain_Alltime': ,
   #     'XTV_past_Alltime'     'XTV_future_Alltime'     'yTV_Alltime'
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
from attention import Attention

import tensorflow as tf
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
#%%
#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

exec(open(f'{path_stem}UWB functions.py').read())
#%%
#%%
Train_set = pd.read_pickle('Train_set.pkl')

import pickle
with open('UWB_datasets_20lstm_time_pred.pkl', 'rb') as f:
    datasets = pickle.load(f)

with open('train_valid_dict.pkl', 'rb') as f:
    train_valid_dict = pickle.load(f)
    
with open('y_TV_dict.pkl', 'rb') as f:
    y_TV_dict = pickle.load(f)

with open('XpastTV_dict.pkl', 'rb') as f:
    XpastTV_dict = pickle.load(f)

with open('XfutureTV_dict.pkl', 'rb') as f:
    XfutureTV_dict = pickle.load(f)

#%%
#%%
Tri_dat = pd.concat( datasets.values() )   
#%%
window_size = 40
padding = pd.DataFrame({'difX': [0]*(window_size-1), 'difY': [0]*(window_size-1),
                        'dif_time': [0]*(window_size-1), 'ed_dif': [0]*(window_size-1),
                        'difSp': [0]*(window_size-1), 'dif_cos': [0]*(window_size-1), 'dif_sin': [0]*(window_size-1)})
#%%
selected_columns = ['difX', 'difY', 'dif_time', 'ed_dif', 'difSp', 'dif_cos', 'dif_sin' ]
#%%
keys = 
    'Train_set_actual': Train_set_actual.reset_index(drop = True).copy(),
    'Train_set_fast': Train_set_fast_alt.reset_index(drop = True).copy(),
    'Train_set_slow': Train_set_slow_alt.reset_index(drop = True).copy()
}
#%%
def create_model(features, timesteps, # past_array, future_array, 
                 hidden_layers=1, dense_units=12, dense_activation='relu',
                 dense_initializer='he_uniform',
                 lstm_units=4, lstm_activation = 'tanh', lstm_initializer='orthogonal',
                 dropout_rate=0,
                 attention_units = 32):
    
    past_input = Input(shape=(timesteps, features), name='past_input')
    future_input = Input(shape=(timesteps, features), name='future_input')
            
    lstm_past = LSTM(units=lstm_units, name='lstm_past', return_sequences = True, 
                     activation = lstm_activation, kernel_initializer=lstm_initializer )(past_input)
    lstm_past = Dropout(dropout_rate)(lstm_past)
    
    lstm_future = LSTM(units=lstm_units, name='lstm_future', return_sequences = True,
                     activation = lstm_activation, kernel_initializer=lstm_initializer )(future_input)
    lstm_future = Dropout(dropout_rate)(lstm_future)
    
    concat_lstm = Concatenate()([lstm_past, lstm_future])
    
    last_lstms = concat_lstm[:, -1, :]
    
    lstm_attention = Attention(units=attention_units)(concat_lstm)
    
    concat_lstm_attention = layers.Concatenate()([last_lstms, lstm_attention])
    
    x = Dense(dense_units, activation=dense_activation, kernel_initializer=dense_initializer)(concat_lstm_attention)
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
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
#%%
#%%
keras_classifier = KerasClassifier(build_fn=create_model, verbose=0)
 
param_grid = [{
    'hidden_layers': [2],
    'dense_units': [40], # lower than 50, greater than 20
    'lstm_units': [30], # lower than 40, greater than 20
    'dropout_rate': [0], 
    'dense_activation': ['relu'],
    'dense_initializer': ['he_uniform'],
    'attention_units' : [40] # above 32, less than 60
}]

manual_param_grid = list(ParameterGrid(param_grid) )

scorer = make_scorer(accuracy_score) 
#%%
Xtrain_past_Alltime = train_valid_dict['Xtrain_past_Alltime']
Xtrain_future_Alltime = train_valid_dict['Xtrain_future_Alltime']
Ytrain_Alltime = train_valid_dict['Ytrain_Alltime']

Xvalid_past_Alltime = train_valid_dict['Xvalid_past_Alltime']
Xvalid_future_Alltime = train_valid_dict['Xvalid_future_Alltime']
Yvalid_Alltime = train_valid_dict['Yvalid_Alltime']

Xtrain_past_Alltime = train_valid_dict['Xtrain_past_Alltime']
Xtrain_future_Alltime = train_valid_dict['Xtrain_future_Alltime']
Ytrain_Alltime = train_valid_dict['Ytrain_Alltime']

XTV_past_Alltime = train_valid_dict['XTV_past_Alltime']
XTV_future_Alltime = train_valid_dict['XTV_future_Alltime']
yTV_Alltime = train_valid_dict['yTV_Alltime']

#%%
accuracies = []
f1_scores = []
validation_losses = []

for perm in manual_param_grid:
        
        model = create_model(Xtrain_past_Alltime.shape[2], Xtrain_past_Alltime.shape[1],
                             hidden_layers = perm['hidden_layers'],
                             dense_units =  perm['dense_units'],
                             lstm_units =  perm['lstm_units'],
                             dropout_rate =  perm['dropout_rate'], 
                             dense_activation = perm['dense_activation'],
                             dense_initializer = perm['dense_initializer'],
                             attention_units = perm['attention_units'])
        
        history = model.fit(x=[Xtrain_past_Alltime, Xtrain_future_Alltime], y=Ytrain_Alltime,
                                validation_data=([Xvalid_past_Alltime, Xvalid_future_Alltime], Yvalid_Alltime),
                             #class_weight=class_weights_dict,
                             shuffle = True, 
                              epochs=80, batch_size=32, verbose=1, callbacks=[early_stopping])
           
        y_pred = (model.predict([Xvalid_past_Alltime, Xvalid_future_Alltime]) > 0.5).astype(int) 
        accuracy = accuracy_score(Yvalid_Alltime, y_pred)
        f1 = f1_score(Yvalid_Alltime, y_pred, average='weighted')
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
#  0.979
#%%
checkpoint_filepath = 'lstmAtt_modI21Time_callback.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

#%%
model = create_model(XTV_past_Alltime.shape[2], XTV_past_Alltime.shape[1],
                     hidden_layers = best_params['hidden_layers'],
                     dense_units = best_params['dense_units'],
                     lstm_units = best_params['lstm_units'],
                     dropout_rate = best_params['dropout_rate'], 
                     dense_activation = best_params['dense_activation'],
                     dense_initializer = best_params['dense_initializer'] )

history = model.fit(x=[XTV_past_Alltime, XTV_future_Alltime], y=yTV_Alltime,
                        validation_data=([XTV_past_Alltime, XTV_future_Alltime], yTV_Alltime), shuffle = True,
                    #class_weight=class_weights_dict,
                      epochs=75, batch_size=32, verbose=1, callbacks=[early_stopping, checkpoint])
#%%   
y_pred = (model.predict([XTV_past_Alltime, XTV_future_Alltime]) > 0.5 ).astype(int)

accuracy = accuracy_score(yTV_Alltime,  y_pred)
f1 = f1_score(yTV_Alltime, y_pred, average='weighted')
val_loss = history.history['val_loss'][-1]
print('Train set performance measures:')
print(f1)
print(val_loss)
print(accuracy)
    
#%%
#%%
I21Time_model = load_model('lstmAtt_modI21Time_callback.h5', custom_objects={'Attention': Attention})
#%%
y_pred = (I21Time_model.predict([XTV_past_Alltime, XTV_future_Alltime]) > 0.5 ).astype(int)
accuracy = accuracy_score(yTV_Alltime, y_pred)
f1 = f1_score(yTV_Alltime, y_pred, average='weighted')
# val_loss = history.history['val_loss'][-1]
print('Train set performance measures:')
print(f1)
#print(val_loss)
print(accuracy)
#%% 
# let's check the differences between the three time types
ypred_TV_dict = {}
count = 0

for ind, key in enumerate( y_TV_dict.keys() ):
     
    size = len( y_TV_dict[key]  )
    temp_pred = y_pred[count:(count+size)]
    count += size
    ypred_TV_dict[key] = temp_pred

    print(len(temp_pred))

###############################################################################
#################Making predictions for all workers #############################
###############################################################################
#%%
import joblib

# Load the scaler object from the pickle file
scaler = joblib.load('I20n_timealt_scaler.pkl')

#%%
for r in range(1,7):
    
    dat = datasets[f'rig_data{r}'].reset_index(drop=True).copy()
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
    
    pred = (I21Time_model.predict([datpast, datfuture]) > 0.5 ).astype(int)
    dat['pred_lstmAtt_time_alt'] = np.nan
    dat.loc[~dat.index.isin(nanX_positions), 'pred_lstmAtt_time_alt'] = pred
    
    lstmpreds =  dat['pred_lstmAtt_time_alt'].values
    
    rig_data = datasets[f'rig_data{r}'].copy()
    rig_data['pred_lstmAtt_time_alt'] = lstmpreds
    
    datasets[f'rig_data{r}'] = rig_data

#%%
#%%
#%%
with open('UWB_datasets_21lstmAtt_time_pred.pkl', 'wb') as f:
    pickle.dump(datasets, f)
#%%
#%%
%matplotlib qt 
animation_fun_pred(datasets['rig_data1'], col = 'pred_lstmAtt_time_alt', pace = 0.5, testing = False)
#%%
animation_fun_pred(datasets['rig_data2'], col = 'pred_lstmAtt_time_alt', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data3'], col = 'pred_lstmAtt_time_alt', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data4'], col = 'pred_lstmAtt_time_alt', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data5'], col = 'pred_lstmAtt_time_alt', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data6'], col = 'pred_lstmAtt_time_alt', pace = 0.01, testing = False, recent = 0.2, atatime = 0.1)
#%%
#%%


#%%
amount_to_use = int(0.5*len(Xvalid_past_Alltime) )
valid_y_pred = (I21Time_model.predict([Xvalid_past_Alltime[:amount_to_use], Xvalid_future_Alltime[:amount_to_use]]) > 0.5 ).astype(int)
#%%
from sklearn.metrics import f1_score

def get_feature_importance(j, n):
  s = accuracy_score(Yvalid_Alltime, valid_y_pred) # baseline score
  total = 0.0
  for i in range(n):
    perm = np.random.permutation(range( Xvalid_past_Alltime.shape[0]))
    X_pasty = Xvalid_past_Alltime.copy()
    X_futuro = Xvalid_future_Alltime.copy()
    
    X_pasty[:, :, j] = Xvalid_past_Alltime[perm, :, j]
    X_futuro[:, :, j] = Xvalid_future_Alltime[perm, :, j]
    
    y_pred_ = (I21Time_model.predict([X_pasty, X_futuro]) > 0.5 ).astype(int)
    s_ij = accuracy_score(Yvalid_Alltime, y_pred_)
    total += s_ij
  return s - total / n
#%%
f = []
for j in range(Xvalid_past_Alltime.shape[2]):
  f_j = get_feature_importance(j, 25)
  f.append(f_j)
# Plot

#%%
%matplotlib inline 
#%%
plt.figure(figsize=(9.5, 5))
plt.bar(range(len(selected_columns)), f, color="r", alpha=0.7)  # Use len(selected_features) for the range
plt.xticks(ticks=range(len(selected_columns)), labels=selected_columns, rotation=45)  # Set feature names as xticklabels
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature importances (Move data set)")
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.show()

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
