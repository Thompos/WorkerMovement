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
#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

exec(open(f'{path_stem}UWB functions.py').read())
#%%
#%%
Train_set = pd.read_pickle('Train_set.pkl')

import pickle

with open('UWB_datasets_21lstmAtt_time_pred.pkl', 'rb') as f:
    datasets = pickle.load(f)

#%%
with open('train_valid_dict.pkl', 'rb') as f:
    train_valid_dict = pickle.load(f)
    
with open('y_TV_dict.pkl', 'rb') as f:
    y_TV_dict = pickle.load(f)

with open('XpastTV_dict.pkl', 'rb') as f:
    XpastTV_dict = pickle.load(f)

with open('XfutureTV_dict.pkl', 'rb') as f:
    XfutureTV_dict = pickle.load(f)
    
with open('train_time_dfs_alt.pkl', 'rb') as f:
    train_time_dfs_alt = pickle.load(f)

#%%
Tri_dat = pd.concat( datasets.values() )
#%%
#%%
selected_columns = ['difX', 'difY', 'dif_time', 'ed_dif', 'difSp', 'spread_past', 
                    'centroid_dist_past', 'dif_sin', 'dif_cos'] 
#%%
window_size = 40
padding = pd.DataFrame(0, index=range(window_size-1), columns=selected_columns)

padding = padding.reset_index(drop=True)
#%%
def calculate_spread_dists(data):
    
    centroid_x = data['X'].mean()
    centroid_y = data['Y'].mean()
    distances = np.sqrt((data['X'] - centroid_x)**2 + (data['Y'] - centroid_y)**2)
    spread = ma.sqrt(np.mean(distances) )
    
    dist_past = np.sqrt((data['X'].iloc[-1] - centroid_x)**2 + (data['Y'].iloc[-1] - centroid_y)**2)
    
    return spread, dist_past
#%%
#%%
#%%
for key in train_time_dfs_alt.keys():
    
    rig_data = train_time_dfs_alt[key].reset_index(drop = True).copy()
    rig_data['spread_past'] = 0
    rig_data['centroid_dist_past'] = 0
    rig_data['spread_past'] = rig_data['spread_past'].astype(float)
    rig_data['centroid_dist_past'] = rig_data['centroid_dist_past'].astype(float)
    
    half = int(0.5*window_size)
        
    for i in range(1, len(rig_data)):
        
        if i >= half:
            start = i - half
            end = min(i+half, len(rig_data) )
        else:
            start = 0
            end = i+half
             
        window_dat = rig_data[start:end]
        
        spread_past, dif_past = calculate_spread_dists(window_dat)
            
        rig_data.loc[i, 'spread_past'] = spread_past
        rig_data.loc[i, 'centroid_dist_past'] = dif_past
    
    train_time_dfs_alt[key] = rig_data[['X','Y','time_lapsed_all','move_cat'] + selected_columns].copy()
         
#%%
plt.scatter(rig_data.loc[rig_data['move_cat'] == 0, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 0, 'spread_past'] ,
            s = 3 )
plt.scatter(rig_data.loc[rig_data['move_cat'] == 1, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 1, 'spread_past'] ,
            s = 3 )
plt.show()
#%%
#%%
plt.scatter(rig_data.loc[rig_data['move_cat'] == 0, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 0, 'centroid_dist_past'] ,
            s = 3 )
plt.scatter(rig_data.loc[rig_data['move_cat'] == 1, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 1, 'centroid_dist_past'] ,
            s = 3 )
plt.show()
#%%
#%%
scaler = StandardScaler()
#%%
X_dfs_alt = {}
y_dfs_alt = {}

for key, df in train_time_dfs_alt.items():
    
   X = df[selected_columns].copy().reset_index(drop = True)
   temp = np.where(pd.isnull(X))
   nanX_positions = list(set(temp[0]) )
   
   X_drop = X.drop(nanX_positions).reset_index(drop=True).copy()
   
   y = df['move_cat'].copy().reset_index(drop = True)
   y_dfs_alt[key] = y.drop(nanX_positions).reset_index(drop=True).copy()
     
   X_padded = pd.concat([padding, X_drop], ignore_index=True)
   X_padded = pd.concat([X_padded, padding], ignore_index=True)
     
   X_sc = scaler.fit_transform(X_padded)
   
   X_dfs_alt[key] = X_sc
   print(len(X_sc))
     
#%%
joblib.dump(scaler, 'I22_time_spread_scaler.pkl')
#%%
XpastTV_dict = {}
XfutureTV_dict = {}
y_TV_dict = {}

for key in train_time_dfs_alt.keys():
   
    Xpast = []
    Xfuture = []
    
    df = X_dfs_alt[key].copy()

    n_features = df.shape[1]
    
    y = y_dfs_alt[key].copy()
    temp = np.where(pd.isnull(y))
    nanY_positions = list(set(temp[0]) )

    for i in range( len(y) ):
        Xpast.append(df[i:(i+window_size) ])
        Xfuture.append(df[(i+2*window_size-2):(i+window_size-2):-1 ])
       
    Xpast, Xfuture = np.array(Xpast), np.array(Xfuture)
       
    TV_y = y.drop(nanY_positions).reset_index(drop=True).copy()
    TV_past = np.delete(Xpast, nanY_positions, axis=0)
    TV_past = TV_past[np.arange(TV_past.shape[0])]

    TV_future = np.delete(Xfuture, nanY_positions, axis=0)
    TV_future = TV_future[np.arange(TV_future.shape[0])]

    XpastTV_dict[key] = TV_past
    print(f'X length: {len(TV_past)}')
    XfutureTV_dict[key] = TV_future
    
    y_TV_dict[key] = TV_y
    print(f'y length: {len(TV_y)}')

#%%
#%%
for key in train_time_dfs_alt.keys():
    
    if len(y_TV_dict[key]) == len(XpastTV_dict[key]) == len(XfutureTV_dict[key]) and np.all(y_TV_dict[key].index == np.arange(len(XpastTV_dict[key]))) and np.all(y_TV_dict[key].index == np.arange(len(XfutureTV_dict[key]))):
        print("Indices match after removing instances.")
    else:
        print("Indices do not match after removing instances.")
    
#%%
#%%
trainX_pasts_dict = {}
trainX_futures_dict = {}
validX_pasts_dict = {}
validX_futures_dict = {}

trainY_dict = {}
validY_dict = {}

for key in train_time_dfs_alt.keys():
    
    TV_y = y_TV_dict[key].copy()
    trainsize = int(0.8*len(TV_y))
    validsize = len(TV_y) - trainsize
    print(f'{key} {trainsize} {validsize}')
    
    TV_past = XpastTV_dict[key].copy()
    TV_future = XfutureTV_dict[key].copy()
    trainX_pasts_dict[key] = TV_past[:trainsize]
    trainX_futures_dict[key] = TV_future[:trainsize]
    validX_pasts_dict[key] = TV_past[trainsize:]
    validX_futures_dict[key] = TV_future[trainsize:]
    
    trainY_dict[key] = np.array(TV_y[:trainsize] )
    validY_dict[key] = np.array(TV_y[trainsize:] )
    
    y_TV_dict[key] = np.array(TV_y)
        
#%%
#%%
for key in train_time_dfs_alt.keys():
    
    print(f'training sets {key}')
    print(len(trainX_pasts_dict[key]))
    print(len(trainX_futures_dict[key]))
    print(len(trainY_dict[key]))
    print(f'valid sets {key}')
    print(len(validX_pasts_dict[key]))
    print(len(validX_futures_dict[key]))
    print(len(validY_dict[key]))
    print(f'TV sets {key}')
    print(len(XpastTV_dict[key]))
    print(len(XfutureTV_dict[key]))
    print(len(y_TV_dict[key]))
    
#%%
for key in train_time_dfs_alt.keys():
       
    print(f'TV sets {key}')
    print(len(XpastTV_dict[key]))
    print(len(XfutureTV_dict[key]))
    print(len(y_TV_dict[key]))
    
#%%
Xtrain_past_Alltime = np.concatenate( list(trainX_pasts_dict.values()), axis=0  )
Xtrain_future_Alltime = np.concatenate( list(trainX_futures_dict.values()), axis=0  )
Ytrain_Alltime = np.concatenate( list(trainY_dict.values()), axis=0  )

Xvalid_past_Alltime = np.concatenate( list(validX_pasts_dict.values()), axis=0  )
Xvalid_future_Alltime = np.concatenate( list(validX_futures_dict.values()), axis=0  )
Yvalid_Alltime = np.concatenate( list(validY_dict.values()), axis=0  )

XTV_past_Alltime = np.concatenate( list(XpastTV_dict.values()), axis=0  )
XTV_future_Alltime = np.concatenate( list(XfutureTV_dict.values()), axis=0  )
yTV_Alltime = np.concatenate( list(y_TV_dict.values()), axis=0  )

#%%
#%%
#%%
def create_model(features, timesteps,
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
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
#%%
#%%
keras_classifier = KerasClassifier(build_fn=create_model, verbose=0)
 
param_grid = [{
    'hidden_layers': [2], # 
    'dense_units': [25], # lower than 30
    'lstm_units': [40], # higher than 30
    'dropout_rate': [0, 0.05], # 0.05 best for this script
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
        
        model = create_model(Xtrain_past_Alltime.shape[2], Xtrain_past_Alltime.shape[1],
                             hidden_layers = perm['hidden_layers'],
                             dense_units = perm['dense_units'],
                             lstm_units = perm['lstm_units'],
                             dropout_rate = perm['dropout_rate'], 
                             dense_activation = perm['dense_activation'],
                             dense_initializer = perm['dense_initializer'] )
        
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
#  0.9844
#%%
checkpoint_filepath = 'lstm_modI22TimeNV_callback.h5'
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
I22Time_model = load_model('lstm_modI22TimeNV_callback.h5')
#%%
y_pred = (I22Time_model.predict([XTV_past_Alltime, XTV_future_Alltime]) > 0.5 ).astype(int)
accuracy = accuracy_score(yTV_Alltime,  y_pred)
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

for ind, key in enumerate(train_time_dfs_alt.keys() ):
     
    size = len( y_TV_dict[key]  )
    temp_pred = y_pred[count:(count+size)]
    count += size
    ypred_TV_dict[key] = temp_pred

    print(len(temp_pred))

#%%
#%%
###############################################################################
#################Making predictions for all workers #############################
###############################################################################
#%%
#%%
for r in range(1,7):
    
    dat = datasets[f'rig_data{r}'].reset_index(drop=True).copy()
    
    dat['spread_past'] = 0
    dat['centroid_dist_past'] = 0
    dat['spread_past'] = dat['spread_past'].astype(float)
    dat['centroid_dist_past'] = dat['centroid_dist_past'].astype(float)
    
    dat = dat[['X', 'Y'] + selected_columns].copy()
    
    for i in range(1, len(dat)):
        if i >= half:
            start = i - half
            end = min(i+half, len(dat) )
        else:
            start = 0
            end = i+half
            
        window_dat = dat[start:end]
        spread_past, dif_past = calculate_spread_dists(window_dat)
        
        dat.loc[i, 'spread_past'] = spread_past
        dat.loc[i, 'centroid_dist_past'] = dif_past
        
    dat_select = dat[selected_columns].copy()   
    temp = np.where(pd.isnull(dat_select))
    nanX_positions = list(set(temp[0]) )
    dat_rm = dat_select.drop(nanX_positions).reset_index(drop=True)
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
    
    pred = (I22Time_model.predict([datpast, datfuture]) > 0.5 ).astype(int)
    dat['pred_lstm_NV_alt'] = np.nan
    dat.loc[~dat.index.isin(nanX_positions), 'pred_lstm_NV_alt'] = pred
    
    lstmpreds =  dat['pred_lstm_NV_alt'].values
    
    rig_data = datasets[f'rig_data{r}'].copy()
    rig_data['pred_lstm_NV_alt'] = lstmpreds
    
    datasets[f'rig_data{r}'] = rig_data
##              
#%%
#%%
#%%
with open('UWB_datasets_22lstm_timeNV_pred.pkl', 'wb') as f:
    pickle.dump(datasets, f)
#%%
#%%
#%%
#%%
%matplotlib qt 
animation_fun_pred(datasets['rig_data1'], col = 'pred_lstm_NV_alt', pace = 0.2, testing = False)
#%%
animation_fun_pred(datasets['rig_data2'], col = 'pred_lstm_NV_alt', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data3'], col = 'pred_lstm_NV_alt', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data4'], col = 'pred_lstm_NV_alt', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data5'], col = 'pred_lstm_NV_alt', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data6'], col = 'pred_lstm_NV_alt', pace = 0.01, testing = False, recent = 0.2, atatime = 0.1)
#%%
#%%

Xvalid_past = validX_pasts_dict['Train_set_actual']
Xvalid_future = validX_futures_dict['Train_set_actual']

Yvalid = validY_dict['Train_set_actual']
#%%
valid_y_pred = (I22Time_model.predict([Xvalid_past, Xvalid_future]) > 0.5 ).astype(int)
#%%
from sklearn.metrics import f1_score

def get_feature_importance(j, n):
  s = accuracy_score(Yvalid, valid_y_pred) # baseline score
  total = 0.0
  for i in range(n):
    perm = np.random.permutation(range( Xvalid_past.shape[0]))
    X_pasty = Xvalid_past.copy()
    X_futuro = Xvalid_future.copy()
    
    X_pasty[:, :, j] = Xvalid_past[perm, :, j]
    X_futuro[:, :, j] = Xvalid_future[perm, :, j]
    
    y_pred_ = (I22Time_model.predict([X_pasty, X_futuro]) > 0.5 ).astype(int)
    s_ij = accuracy_score(Yvalid, y_pred_)
    total += s_ij
  return s - total / n
#%%
f = []
for j in range(Xvalid_past.shape[2]):
  f_j = get_feature_importance(j, 20)
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