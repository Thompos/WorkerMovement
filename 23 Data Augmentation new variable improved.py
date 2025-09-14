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
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 35)
#%%
%matplotlib inline
#%%
#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

exec(open(f'{path_stem}UWB functions.py').read())
#%%
#%%
Train_set = pd.read_pickle('Train_set.pkl')
# 'XY_spreadp2',  'nt_XYma2', 'cos_1', sin_1',  
import pickle

with open('UWB_datasets_NO_INT.pkl', 'rb') as f:
    datasets = pickle.load(f)

#%% 
#%%
#%%
selected_columns = [ 'difX', 'difY', 'dif_time', 'ed_dif', 'difSp1', 'XY_spread2', 'ed_dif_1', # 'Xdiff_2', 'Ydiff_2',
                     'dif_angle1_abs', 'centroid_dist2', 'dif_angle_abs' ]  # 'dif_angle',
other_columns = [ 'Xdiff_1', 'Ydiff_1', 'Xdiff_2', 'Ydiff_2']
#%%
window_size = 40
padding = pd.DataFrame(0, index=range(window_size-1), columns=selected_columns)

padding = padding.reset_index(drop=True)
#%%
#%%
# def calculate_spread(data):
    
#     centroid_x = data['X'].mean()
#     centroid_y = data['Y'].mean()
#     distances = np.sqrt((data['X'] - centroid_x)**2 + (data['Y'] - centroid_y)**2)
#     spread = ma.sqrt(np.mean(distances) )
    
#     return spread

def calculate_spread_dists(data):
    
    centroid_x = data['X'].mean()
    centroid_y = data['Y'].mean()
    distances = np.sqrt((data['X'] - centroid_x)**2 + (data['Y'] - centroid_y)**2)
    spread = np.mean(distances)
    
    dist_past = np.sqrt((data['X'].iloc[-1] - centroid_x)**2 + (data['Y'].iloc[-1] - centroid_y)**2)
    
    return spread, dist_past

#%%
#%%
ma1 = '1S'
ma2 = '2S'
td2 = pd.Timedelta(seconds=1)
#%%
for rig in range(1,7):
    
    rig_data = datasets[f'rig_data{rig}'].copy()
    
    rig_data['Xdiff_2'] = rig_data['difX'].rolling(ma2, min_periods=1, center = True).sum().values
    rig_data['Ydiff_2'] = rig_data['difY'].rolling(ma2, min_periods=1, center = True).sum().values
    
    rig_data['Xdiff_1'] = rig_data['difX'].rolling(ma1, min_periods=1, center = True).sum().values
    rig_data['Ydiff_1'] = rig_data['difY'].rolling(ma1, min_periods=1, center = True).sum().values
    rig_data['dif_time1'] = rig_data['dif_time'].rolling(ma1, min_periods=1, center = True).sum().values
    
    rig_data['ed_dif_2'] = ( rig_data['Xdiff_2']**2 + rig_data['Ydiff_2']**2 )**0.5
    rig_data['ed_dif_1'] = ( rig_data['Xdiff_1']**2 + rig_data['Ydiff_1']**2 )**0.5
    
    rig_data['angle'] = calculate_angle(rig_data['difX'].values, rig_data['difY'].values, rig_data['ed_dif'].values )
    angles_nans_dealt = rig_data['angle'].fillna(method='ffill')
    angles_nans_dealt.iloc[0] = angles_nans_dealt.iloc[1]
    dif_angles = np.diff(angles_nans_dealt)
    dif_angles = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in dif_angles]

    dif_angles = np.insert(dif_angles, 0, 0)
    rig_data['dif_angle_abs'] = [abs(angle) for angle in dif_angles]
    

    rig_data['angle1'] = calculate_angle(rig_data['Xdiff_1'].values, rig_data['Ydiff_1'].values, rig_data['ed_dif_1'].values )
    angles_nans_dealt = rig_data['angle1'].fillna(method='ffill')
    angles_nans_dealt.iloc[0] = angles_nans_dealt.iloc[1]
    dif_angles = np.diff(angles_nans_dealt)
    dif_angles = np.insert(dif_angles, 0, 0)
    dif_angles = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in dif_angles]
    rig_data['dif_angle1_abs'] = [abs(angle) for angle in dif_angles]
    
    rig_data['difSp1'] = ((rig_data['Xdiff_1']**2 + rig_data['Ydiff_1']**2)**0.5)/rig_data['dif_time1']
    
    spreads2 = []
    centroid_dists2 = []
    for j in range(len(rig_data)):
        timestamp = rig_data.index[j]
        data_in_td2 = rig_data.loc[timestamp - td2 : timestamp + td2 ]
        spreads, centroid_dist = calculate_spread_dists(data_in_td2)
        spreads2.append(spreads)
        centroid_dists2.append(centroid_dist)
        
    rig_data['XY_spread2'] = spreads2
    rig_data['centroid_dist2'] = centroid_dists2
    
    datasets[f'rig_data{rig}'] = rig_data.copy()
    
#%%
with open('UWB_datasets_22lstm_timeNV_pred.pkl', 'rb') as f:
    Interpol_datasets = pickle.load(f)

Temp_dat = Interpol_datasets['rig_data6'].copy()

Temp_dat['move_cat'] = Train_set['move_cat'].values

#%%
time_to_binary_map = dict(zip(Temp_dat.index, Temp_dat['move_cat']))

rig_data = datasets['rig_data6'].copy()

rig_data['move_cat'] = rig_data.index.map(lambda x: time_to_binary_map.get(x, np.nan))

datasets[f'rig_data6'] = rig_data.copy()
#with open('datasets_23_NV.pkl', 'wb') as f:
 #   pickle.dump(datasets, f)
#%%
Tri_dat = pd.concat( datasets.values() )    
#%%
def calc_variables(df, augmented = True):
       
    if augmented == True:
        
        difs_time = np.diff(df['time_lapsed_all'])
        difs_X = np.diff(df['X'])
        difs_Y = np.diff(df['Y'])
     
        df['dif_time'] = np.insert(difs_time, 0, np.nan)
        df['difX'] = np.insert(difs_X, 0, np.nan)
        df['difY'] = np.insert(difs_Y, 0, np.nan)
        
        df['ed_dif'] = ( df['difX']**2 + df['difY']**2 )**0.5
 
    df['Xdiff_2'] = df['difX'].rolling(ma2, min_periods=1, center = True).sum().values
    df['Ydiff_2'] = df['difY'].rolling(ma2, min_periods=1, center = True).sum().values
    
    df['Xdiff_1'] = df['difX'].rolling(ma1, min_periods=1, center = True).sum().values
    df['Ydiff_1'] = df['difY'].rolling(ma1, min_periods=1, center = True).sum().values
    
    df['ed_dif_2'] = ( df['Xdiff_2']**2 + df['Ydiff_2']**2 )**0.5
    df['ed_dif_1'] = ( df['Xdiff_1']**2 + df['Ydiff_1']**2 )**0.5
         
    df['angle'] = calculate_angle(df['difX'].values, df['difY'].values, df['ed_dif'].values )
    angles_nans_dealt = df['angle'].fillna(method='ffill')
    angles_nans_dealt.iloc[0] = angles_nans_dealt.iloc[1]
    dif_angles = np.diff(angles_nans_dealt)
    dif_angles = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in dif_angles]
    dif_angles = np.insert(dif_angles, 0, 0)
    df['dif_angle_abs'] = [abs(angle) for angle in dif_angles]
    
    df['angle1'] = calculate_angle(df['Xdiff_1'].values, df['Ydiff_1'].values, df['ed_dif_1'].values )
    angles_nans_dealt = df['angle1'].fillna(method='ffill')
    angles_nans_dealt.iloc[0] = angles_nans_dealt.iloc[1]
    dif_angles = np.diff(angles_nans_dealt)
    dif_angles = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in dif_angles]
    dif_angles = np.insert(dif_angles, 0, 0)
    df['dif_angle1_abs'] = [abs(angle) for angle in dif_angles]
     
    df['difSp1'] = (df['Xdiff_1']**2 + df['Ydiff_1']**2)**0.5
     
    spreads2 = []
    centroid_dists2 = []
    for j in range(len(df)):
        timestamp = df.index[j]
        data_in_td2 = df.loc[timestamp - td2 : timestamp + td2 ]
        spreads, centroid_dist = calculate_spread_dists(data_in_td2)
        spreads2.append(spreads)
        centroid_dists2.append(centroid_dist)
        
    df['XY_spread2'] = spreads2
    df['centroid_dist2'] = centroid_dists2
    
    return df.copy()

#%%
#
time_zero = Tri_dat.index.min()
temp_dat = Tri_dat.copy()
#%%
#%%
def adjust_time_speed(df, factor, time_zero = time_zero):
    df_aug = df.copy()
    start_time = time_zero
    time_elapsed = df_aug['time_lapsed_all'] * factor
    df_aug.index = start_time + pd.to_timedelta(time_elapsed, unit='s')
    return df_aug
#%%
#%%
Tri_dat_fast = adjust_time_speed(temp_dat, 1/2)
Tri_dat_slow = adjust_time_speed(temp_dat, 2)
#%%
#%%
Tri_dat_fast['time_lapsed_all'] = np.nan

temp = (Tri_dat_fast.index - time_zero ).total_seconds().values
Tri_dat_fast[  'time_lapsed_all' ] = temp
#%%
Tri_dat_slow['time_lapsed_all'] = np.nan

temp = (Tri_dat_slow.index - time_zero ).total_seconds().values
Tri_dat_slow[ 'time_lapsed_all' ] = temp
#%%
#%%
datasets_fast = {}

for rig in range(1,7):    
    rig_data = Tri_dat_fast[Tri_dat_fast['rig'] == rig].copy()
    if rig == 6:
        rig_move = datasets['rig_data6']['move_cat'].copy()
        rig_data['move_cat'] =  rig_move.values
#
    rig_data_alt = rig_data.iloc[range(1, len(rig_data), 2)].copy()
#
    datasets_fast[f'rig_data{rig}'] = calc_variables(rig_data_alt)
#%%
noise_std = 0.005
datasets_slow = {}

for rig in range(1,7):    
    rig_data = Tri_dat_slow[Tri_dat_slow['rig'] == rig].copy()
    
    new_time = rig_data['time_lapsed_all'].iloc[:-1].values + 0.5*(rig_data['time_lapsed_all'].diff().iloc[1:].values)
    new_X = 0.5*(rig_data['X'].iloc[:-1].values + rig_data['X'].shift(-1).iloc[:-1].values) + np.random.normal(0, noise_std)
    new_Y = 0.5*(rig_data['Y'].iloc[:-1].values + rig_data['Y'].shift(-1).iloc[:-1].values) + np.random.normal(0, noise_std)
    
    if rig == 6:
        rig_move = datasets['rig_data6']['move_cat'].copy()
        rig_data['move_cat'] =  rig_move.values
# 
        new_move = []
        for i in range(len(rig_data)-1): 
            move1 = rig_data['move_cat'].iloc[i]
            move2 = rig_data['move_cat'].iloc[i+1]
            if (np.isnan(move1)|np.isnan(move2)):
                new_move.append(np.nan)
            elif move1 == move2:
                new_move.append(move1)
            else:
                new_move.append(round(0.5*(move1 + move2) ) )
    else:    
       rig_data['move_cat'] = np.nan
       new_move = [np.nan]*(len(rig_data) - 1)

   
    new_dat_temp = {
        'X': pd.concat([rig_data['X'], pd.Series(new_X)], ignore_index=True),
        'Y': pd.concat([rig_data['Y'], pd.Series(new_Y)], ignore_index=True),
        'time_lapsed_all': pd.concat([rig_data['time_lapsed_all'], pd.Series(new_time)], ignore_index=True),
        'move_cat': pd.concat([rig_data['move_cat'], pd.Series(new_move)], ignore_index=True)
    }     
        
    rig_dat_alt = pd.DataFrame(new_dat_temp)   
        
    rig_dat_alt = rig_dat_alt.sort_values(by='time_lapsed_all')
    
    time_diff = rig_data.index.to_series().diff()
    half_time_diff = time_diff.iloc[1:] / 2
    new_timestamps = rig_data.index.to_series()[1:] - half_time_diff
    new_index = rig_data.index.union(new_timestamps)
    
    rig_dat_alt.index = new_index

    datasets_slow[f'rig_data{rig}'] = calc_variables(rig_dat_alt)
#%%
#%% 
with open('datasets_fast_23_NV.pkl', 'wb') as f:
    pickle.dump(datasets_fast, f)
 
with open('datasets_slow_23_NV.pkl', 'wb') as f:
    pickle.dump(datasets_slow, f)
    
#with open('datasets_fast_23_NV.pkl', 'rb') as f:
 #    datasets_fast = pickle.load(f)
    
# with open('datasets_slow_23_NV.pkl', 'rb') as f:
#     datasets_slow = pickle.load(f)
    
#%%              
#%%
Train_set_actual = datasets['rig_data6'][['X','Y', 'time_lapsed_all', 'move_cat'] + selected_columns+other_columns].reset_index(drop = True).copy()
Train_set_fast = datasets_fast['rig_data6'][['X','Y', 'time_lapsed_all', 'move_cat'] + selected_columns+other_columns].reset_index(drop = True).copy()
Train_set_slow = datasets_slow['rig_data6'][['X','Y', 'time_lapsed_all', 'move_cat'] + selected_columns+other_columns].reset_index(drop = True).copy()

#%%
#%%
#%%
train_time_dfs_alt = {
    'Train_set_actual': Train_set_actual.reset_index(drop = True).copy(),
    'Train_set_fast': Train_set_fast.reset_index(drop = True).copy(),
    'Train_set_slow': Train_set_slow.reset_index(drop = True).copy()
}

#%%
plt.scatter(rig_data.loc[rig_data['move_cat'] == 0, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 0, 'dif_angle_abs'] ,
            s = 3 )
plt.scatter(rig_data.loc[rig_data['move_cat'] == 1, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 1, 'dif_angle_abs'] ,
            s = 3 )
plt.show()
#%%
plt.scatter(rig_data.loc[rig_data['move_cat'] == 0, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 0, 'dif_angle1_abs'] ,
            s = 3 )
plt.scatter(rig_data.loc[rig_data['move_cat'] == 1, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 1, 'dif_angle1_abs'] ,
            s = 3 )
plt.show()
#%%
plt.scatter(rig_data.loc[rig_data['move_cat'] == 0, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 0, 'difSp1'] ,
            s = 3 )
plt.scatter(rig_data.loc[rig_data['move_cat'] == 1, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 1, 'difSp1'] ,
            s = 3 )
plt.show()
#%%
plt.scatter(rig_data.loc[rig_data['move_cat'] == 0, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 0, 'ed_dif_2'] ,
            s = 3 )
plt.scatter(rig_data.loc[rig_data['move_cat'] == 1, 'time_lapsed_all'],
            rig_data.loc[rig_data['move_cat'] == 1, 'ed_dif_2'] ,
            s = 3 )
plt.show()
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
joblib.dump(scaler, 'I23_time_NVs_scaler.pkl')
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
    'dense_units': [15, 25], # lower than 25.
    'lstm_units': [50], # higher than 40
    'dropout_rate': [0, 0.05], # 0.05 best for this script
    'dense_activation': ['relu'],
    'dense_initializer': ['he_uniform']
}]

manual_param_grid = list(ParameterGrid(param_grid) )

scorer = make_scorer(accuracy_score) 
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
#  0.9818
#%%
checkpoint_filepath = 'lstm_modI23TimeNV_callback.h5'
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
                      epochs=65, batch_size=32, verbose=1, callbacks=[early_stopping, checkpoint])
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
I23Time_model = load_model('lstm_modI23TimeNV_callback.h5')
#%%
y_pred = (I23Time_model.predict([XTV_past_Alltime, XTV_future_Alltime]) > 0.5 ).astype(int)
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
def model_predictions(dataframes):
    
    dict_data = dataframes.copy()
    
    for r in range(1,7):
            
        dat = dataframes[f'rig_data{r}'].reset_index(drop=True).copy()
    
        X = dat[selected_columns].reset_index(drop=True).copy()
        temp = np.where(pd.isnull(X))
        nanX_positions = list(set(temp[0]) )
       
        X_drop = X.drop(nanX_positions).reset_index(drop=True).copy()
        dat_len = len(X_drop)
       
        X_padded = pd.concat([padding, X_drop], ignore_index=True)
        X_padded = pd.concat([X_padded, padding], ignore_index=True)
         
        X_sc = scaler.transform(X_padded)
          
        Xpast = []
        Xfuture = []
        
        for i in range( dat_len ):
            Xpast.append(X_sc[i:(i+window_size) ])
            Xfuture.append(X_sc[(i+2*window_size-2):(i+window_size-2):-1 ])
           
        Xpast, Xfuture = np.array(Xpast), np.array(Xfuture)
            
        pred = (I23Time_model.predict([Xpast, Xfuture]) > 0.5 ).astype(int)
        dat['pred_lstm23_NV'] = np.nan
        dat.loc[~dat.index.isin(nanX_positions), 'pred_lstm23_NV'] = pred
        
        lstmpreds = dat['pred_lstm23_NV'].values
        
        rig_data = dict_data[f'rig_data{r}'].copy()
        rig_data['pred_lstm23_NV'] = lstmpreds
        
        dict_data[f'rig_data{r}'] = rig_data.copy()
        
    return dict_data

#%%
datasets_pred = model_predictions(datasets)
#%%
datasets_fast_pred = model_predictions(datasets_fast)
#%%
datasets_slow_pred = model_predictions(datasets_slow)
#%%
with open('datasets_pred_23_NV.pkl', 'wb') as f:
    pickle.dump(datasets_pred, f)
    
with open('datasets_fast_pred_23_NV.pkl', 'wb') as f:
    pickle.dump(datasets_fast_pred, f)
    
with open('datasets_slow_pred_23_NV.pkl', 'wb') as f:
    pickle.dump(datasets_slow_pred, f)

#%%
import pickle

with open('datasets_pred_23_NV.pkl', 'rb') as f:
    datasets = pickle.load(f)

with open('datasets_fast_pred_23_NV.pkl', 'rb') as f:
    datasets_fast = pickle.load(f)

with open('datasets_slow_pred_23_NV.pkl', 'rb') as f:
    datasets_slow = pickle.load(f)

Tri_dat = pd.concat( datasets.values() )  
#%%
#%%
#%%
%matplotlib qt 
animation_fun_pred(datasets['rig_data1'], col = 'pred_lstm23_NV', pace = 0.25, testing = False)
#%%
animation_fun_pred(datasets_pred['rig_data2'], col = 'pred_lstm23_NV', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets_pred['rig_data3'], col = 'pred_lstm23_NV', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets_pred['rig_data4'], col = 'pred_lstm23_NV', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets_pred['rig_data5'], col = 'pred_lstm23_NV', pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets_pred['rig_data6'], col = 'pred_lstm23_NV', pace = 0.01, testing = False, recent = 0.2, atatime = 0.1)
#%%
#%%

#%%
Xvalid_past = validX_pasts_dict['Train_set_actual']
Xvalid_future = validX_futures_dict['Train_set_actual']
Yvalid = validY_dict['Train_set_actual']
valid_y_pred = (I23Time_model.predict([Xvalid_past, Xvalid_future]) > 0.5 ).astype(int)
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
    
    y_pred_ = (I23Time_model.predict([X_pasty, X_futuro]) > 0.5 ).astype(int)
    s_ij = accuracy_score(Yvalid, y_pred_)
    total += s_ij
  return s - total / n
#%%
f = []
for j in range(Xvalid_past.shape[2]):
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