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
from sklearn.inspection import permutation_importance
import math as ma
from tensorflow.keras import layers, Sequential
import time
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers, callbacks 
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model
import joblib
from keras.layers import Dense, Dropout
from sklearn.metrics import make_scorer, accuracy_score, f1_score, classification_report
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
random_seed = 69
tf.random.set_seed(69)
np.random.seed(69)
#%%
%matplotlib inline
#%%
Train_set = pd.read_pickle('UWB_SSI2_Train_dat.pkl')

#Tri_dat = pd.read_pickle('UWB_dat_post5.pkl')

I2_model = load_model('best_move_modI2_callback.h5')

scaler = joblib.load('I2_move_mod_scaler.pkl')

#%%
I2_model.summary()
#%%
import pickle

with open('UWB_datasets_5.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
selected_columns = ['nt_XYma1', 'nt_XYma3',
                    'nt_XYmaf1', 'nt_XYmaf3',
                    'XY_spreadp1', 'XY_spreadp3',
                    'XY_spreadf1', 'XY_spreadf3']
#%%
for i in range(1,7):
    rig_data = datasets[f'rig_data{i}']
    
    data = rig_data[selected_columns]
    data = data.dropna()
    
    X_scaled = scaler.transform(data)
    
    rig_pred = (I2_model.predict(X_scaled) > 0.5).astype(int)
    
    rig_data['pred'] = np.nan
    
    for j, p in zip(data.index, rig_pred):    
        rig_data.loc[j, 'pred'] = p
    
    datasets[f'rig_data{i}'] = rig_data
#%%
#%%
#%%
def animation_fun_pred(data, recent = 1, atatime = 0.5, psize = 6, pace = 0.025, testing = False):
       fig, ax = plt.subplots(figsize=(8, 6))

       recentTime = pd.Timedelta(minutes=recent)  
       period = pd.Timedelta(minutes=atatime)
       
       na_color = 'gray'
       zero_color = 'red'
       nonzero_color = 'green'

       data['cols'] = np.where(data['pred'].isna(), na_color, np.where(data['pred'] == 0, zero_color, nonzero_color))

       time_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq=period)
       
       initial_timestamp = data.index[0]
       initial_data = data.loc[initial_timestamp:(initial_timestamp+period)]
       plt.scatter(initial_data['X'], initial_data['Y'], marker='o', s=psize,
                   c= initial_data['cols'] )
       
       plt.scatter([], [], marker='o', s=psize)
       minX = min(data['X'])
       maxX = max(data['X'])
       minY = min(data['Y'])
       maxY = max(data['Y'])+1
       
       plt.xlim(minX, maxX)
       plt.ylim(minY, maxY)

       plt.xlabel('X')
       plt.ylabel('Y')
       plt.title('0 mins lapsed')
       #legend_elements = [Line2D([0], [0], marker='o',  c= initial_data['predmod1'], cmap='viridis',
        #                         markersize=10,  label=f"Previous {int(recent)} mins"),
         #                 Line2D([0], [0], marker='o',  c= initial_data['predmod1'], cmap='viridis',
          #                       markersize=10, label=f"Previous to the last {int(recent)} mins") ,
           #               Line2D([0], [0], marker='o',  c= initial_data['predmod1'], cmap='viridis',
            #                     markersize=10, label=f"Previous to the last {int(recent)} mins") ]     

     #  plt.legend(handles=legend_elements, loc='upper left')

       for ind, timestamp in enumerate(time_range):
           plt.pause(pace)
           plt.cla()
           plt.xlim(minX, maxX)
           plt.ylim(minY, maxY )
           plt.xlabel('X')
           plt.ylabel('Y')
        #   plt.legend(handles=legend_handles, loc='upper left')
           
           x_new = data.loc[timestamp:(timestamp+period), 'X']
           y_new = data.loc[timestamp:(timestamp+period), 'Y']
           new_cols = data.loc[timestamp:(timestamp+period), 'cols']
           
           plt.scatter(x_new, y_new, c= new_cols, s = psize)

           dat = data.loc[timestamp:(timestamp+period)]
           locations = [data.index.get_loc(timestamp) for timestamp in dat.index.values]
           
           try:
               plt.title(f"{timestamp.strftime('%H:%M:%S')} to {(timestamp+period).strftime('%H:%M:%S')}, {locations[0]} ")
           except:
               pass
            
           if ind >= (len(time_range)-1):
               print('The End!')
#%%
#%%
#%%
#%%
%matplotlib qt 
animation_fun_pred(datasets['rig_data1'], pace = 0.1, testing = False)
#%%
animation_fun_pred(datasets['rig_data2'], pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data3'], pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data4'], pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data5'], pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data6'], pace = 0.01, testing = False, recent = 0.2, atatime = 0.1)
#%%
#%%
#%%
#%%
with open('UWB_datasets_9_NNpred.pkl', 'wb') as f:
    pickle.dump(datasets, f)

#%%
#%%
Train_set['move_cat'] = Train_set['pred'].copy()
#%%
move_col = Train_set.columns.get_loc('move_cat')
#%%
#%%
##
starts_nan = [17798,18220,18340,18690,19040,22096,23790,27122,28270,28360,28490,
32790,33370,33950,34243,35830,38880,39230,42800,43940,46300,46678,46990,49520,51100,57300,51790,57705,57880,
58200,58700,58752,61800,69430,70200,70910,71120,72500,74350,31400,34760,38250,46678,
46990,59028,60550,72750,74400,19090,51140,43490]
ends_nan =   [17998,18270,18370,18750,19140,22126,23840,27187,28430,28510,28522,
32900,33480,34030,34330,35910,38950,39300,42970,44140,46635,46740,47020,49670,51690,52700,57675,57860,58300,
58332,58737,58810,62500,69550,70350,71060,72000,72600,74700,31600,34860,38880,46740,
47120,59450,60750,72850,74900,19145,51450,43540]
##
starts0 = [115,18370,18500,21023,21240,21418,21709,22126,22272,22783,23060,
23700,23840,24250,25300,25680,26910,27187,28522,30240,31600,33532,34213,34860,
36280,37250,37600,41050,42000,42970,44250,46648,46740,47320,49670,
51690,52080,52380,52650,53700,58332,59450,60750,62500,62950,72600,72350,
33750,40760,69320,72850,73000,74900]
ends0 =   [315,18540,18620,21073,21300,21475,21810,22456,22600,22850,23612,
23790,24040,24390,25600,25840,27110,27350,28582,31400,32790,33732,34243,35160,
36430,37290,38250,41350,42800,43170,46300,46678,46990,49520,51100,
51790,52150,52600,53400,56500,58550,60550,61800,62850,68950,72750,72500,
33790,40790,69430,74350,74300,75255]
##
starts1 = [33155,33480,71080,18845,33800,34030,40730,57675,57860,58737,69200,46635]
ends1=    [33255,33520,71120,18880,33950,34100,40760,57705,57880,58752,69320,46648]
##
#%%
%matplotlib inline
for start,end in zip(starts1, ends1):    
    plt.figure(figsize=(10, 6))
    indices = np.arange(start, end)
    norm_indices = (indices - start) / (end - start)
    na_color = 'gray'
    zero_color = 'red'
    nonzero_color = 'green'

    colors = np.where(Train_set['move_cat'].iloc[start:end].isna(), na_color,
                            np.where(Train_set['move_cat'].iloc[start:end] == 0, zero_color, nonzero_color))
    plt.xlim(min(Train_set['X']), max(Train_set['X']))
    plt.ylim(min(Train_set['Y']), max(Train_set['Y'])+1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{start} to {end}')
    plt.scatter(Train_set['X'].iloc[start:end], Train_set['Y'].iloc[start:end],
                marker = 'o', s = 7.5, c=colors)    
#%%
# 43490, 44590
start = 62500
end = 62850

plt.figure(figsize=(10, 6))
indices = np.arange(start, end)
norm_indices = (indices - start) / (end - start)

na_color = 'gray'
zero_color = 'red'
nonzero_color = 'green'

colors = np.where(Train_set['move_cat'][start:end].isna(), na_color,
                        np.where(Train_set['move_cat'][start:end] == 0, zero_color, nonzero_color))
plt.xlim(min(Train_set['X']), max(Train_set['X']))
plt.ylim(min(Train_set['Y']), max(Train_set['Y'])+1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'{start} to {end}')
plt.scatter(Train_set['X'][start:end], Train_set['Y'][start:end],
            marker = 'o', s = 7.5, c=colors) 
#%%
#%%
for start,end in zip(starts_nan, ends_nan):      
    Train_set['move_cat'].iloc[start:end] = np.nan

for start,end in zip(starts0, ends0):      
    Train_set['move_cat'].iloc[start:end] = 0
    
for start,end in zip(starts1, ends1):      
    Train_set['move_cat'].iloc[start:end] = 1
#%%
#%%
#%%
#%%
selected_columns.append('move_cat')
#%%
data = Train_set[selected_columns]
data = data.dropna().reset_index(drop = True)

X = data[selected_columns[:-1]]

y = data['move_cat']
#%%
#%%
X_scaled = scaler.transform(X)

train_ind = int(0.8*len(X) )

X_train = X_scaled[:train_ind]
X_valid = X_scaled[train_ind:]

y_train = y[:train_ind]
y_valid = y[train_ind:]

#%%
#%%
def create_model(hidden_layers=4, units=20, activation='relu', dropout_rate=0):
    model = Sequential()
    model.add(Dense(units, activation=activation, input_dim=X.shape[1]))
    model.add(Dropout(dropout_rate)) 
    for _ in range(hidden_layers - 1):
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate)) 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
#%%
keras_classifier = KerasClassifier(build_fn=create_model, verbose=0)
 
param_grid = {
    'hidden_layers': [4],
    'units': [20],
    'dropout_rate': [0, 0.1], 
    'activation': ['relu']
}

scorer = make_scorer(accuracy_score) 

#%%
#%%
#%%
early_stopping = EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)

classes = np.unique(y)
class_weights = compute_class_weight(None, classes = classes, y = y)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

accuracies = []
f1_scores = []
validation_losses = []
params = []

for hidden_layers in param_grid['hidden_layers']:
    for units in param_grid['units']:
        for dropout_rate in param_grid['dropout_rate']:

            model = create_model(hidden_layers=hidden_layers, units=units, activation='relu', dropout_rate = dropout_rate )
            
            model.set_weights(I2_model.get_weights()) 
            history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                                class_weight=class_weights_dict,
                                epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])
               
            y_pred = (model.predict(X_valid) > 0.5).astype(int) 
            accuracy = accuracy_score(y_valid, y_pred)           
            f1 = f1_score(y_valid, y_pred, average='weighted')
            val_loss = history.history['val_loss'][-1]
            
            accuracies.append(accuracy)
            f1_scores.append(f1)
            validation_losses.append(val_loss)
            params.append([hidden_layers, units, dropout_rate])

#%%
best_index = np.argmax(f1_scores)
best_params = params[best_index]

#%%
checkpoint_filepath = 'best_move_modI3b_callback.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=11, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)
#%%
#%%
new_model = create_model(hidden_layers=best_params[0], units=best_params[1], activation='relu', dropout_rate=best_params[2])
new_model.set_weights(I2_model.get_weights())
new_model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
               epochs=120, batch_size=32, callbacks=[early_stopping, checkpoint])
#%%
#%%
I3_model = load_model('best_move_modI3b_callback.h5')
#%%
rig_data = Train_set[selected_columns[:-1]].copy()
rig_data = rig_data.dropna()
rd_scaled = scaler.transform(rig_data)
#%%
rig_pred = (I3_model.predict(rd_scaled) > 0.5).astype(int)
#%%
#%%
Train_set['pred'] = np.nan
#%%
for i, p in zip(rig_data.index, rig_pred):    
    Train_set.loc[i, 'pred'] = p
      
#%%
testing_dat = datasets['rig_data6'].copy() 
#%%
testing_dat['pred'] = Train_set['pred'].values
#%%
#%%
#%%
%matplotlib qt 
animation_fun_pred(testing_dat , pace = 0.4, testing = True)
#%% ######################################################################################################
#%% ######################################################################################################
#%% # predicting again with revised model
%matplotlib qt
#%%
I3_model.summary()
#%%
#%%
#%%
for i in range(1,7):
    rig_data = datasets[f'rig_data{i}']
    
    data = rig_data[selected_columns[:-1]]
    data = data.dropna()
    
    X_scaled = scaler.transform(data)
    
    rig_pred = np.argmax(I3_model.predict(X_scaled), axis = 1)
    
    rig_data['pred'] = np.nan
    
    for j, p in zip(data.index, rig_pred):    
        rig_data.loc[j, 'pred'] = p
    
    datasets[f'rig_data{i}'] =  rig_data
#%%
#%%
#%%
%matplotlib qt
animation_fun_pred(datasets['rig_data1'], pace = 0.1, testing = False)
#%%
animation_fun_pred(datasets['rig_data2'], pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data3'], pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data4'], pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data5'], pace = 0.15, testing = False)
#%%
animation_fun_pred(datasets['rig_data6'], pace = 0.01, testing = False, recent = 0.2, atatime = 0.1)
#%%
#%%
# Tri_dat.to_pickle('UWB_dat_post_I3pred.pkl')
Train_set.to_pickle('Train_set.pkl')
Train_set.to_csv('Train_set.csv', index=False)

#%%
with open('UWB_datasets_9pred.pkl', 'wb') as f:
    pickle.dump(datasets, f)

# Tri_dat.to_csv('Move_pred4Jack.csv', index=False)
#%%
y_pred = (I3_model.predict(X_valid) > 0.5).astype(int)
#%%
from sklearn.metrics import f1_score

def get_feature_importance(j, n):
  s = accuracy_score(y_valid, y_pred) # baseline score
  total = 0.0
  for i in range(n):
    perm = np.random.permutation(range(X_valid.shape[0]))
    X_valid_ = X_valid.copy()
    X_valid_[:, j] = X_valid[perm, j]
    y_pred_ = np.argmax(I3_model.predict(X_valid_), axis = 1)
    s_ij = accuracy_score(y_valid, y_pred_)
    total += s_ij
  return s - total / n
#%%
f = []
for j in range(X_valid.shape[1]):
  f_j = get_feature_importance(j, 40)
  f.append(f_j)
# Plot

#%%
%matplotlib inline 
#%%
plt.figure(figsize=(9.5, 5))
plt.bar(range(len(selected_columns)-1), f, color="r", alpha=0.7)  # Use len(selected_features) for the range
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