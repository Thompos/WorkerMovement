# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:19:36 2024
@author: 
"""
#%%
import tensorflow as tf
from tensorflow.keras import layers, models
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math as ma
from tensorflow.keras import layers, Sequential
import time
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
# from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers, callbacks 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import joblib
from keras.layers import Dense, Dropout
random_seed = 69
tf.random.set_seed(69)
np.random.seed(69)
#%%
import pickle
from scikeras.wrappers import KerasRegressor, KerasClassifier
import tensorflow as tf
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
#%%
Train_set = pd.read_pickle('UWB_SS_Train_dat.pkl')

# Tri_dat = pd.read_pickle('UWB_dat_post5.pkl')
#%%
import pickle

with open('UWB_datasets_5.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
#%%
start = 100
end = 52000 
#%%,
indices = np.arange(start, end)
norm_indices = (indices - start) / (end - start)
colors = ['red' if value == 0 else 'green' for value in Train_set['move_cat'].iloc[start:end]]
plt.xlim(min(Train_set['X']), max(Train_set['X']))
plt.ylim(min(Train_set['Y']), max(Train_set['Y'])+1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'{start} to {end}')
plt.scatter(Train_set['X'][start:end], Train_set['Y'][start:end],
            marker = 'o', s = 7, c=colors)
plt.show()
#%%
move_col = Train_set.columns.get_loc('move_cat')
#%%
Train_set.iloc[21750:21830, move_col] = 0
Train_set.iloc[23500:23700, move_col] = 0
Train_set.iloc[26900:27140, move_col] = 0
Train_set.iloc[31610:31760, move_col] = 0
Train_set.iloc[49700:50000, move_col] = 0
#%%
#%%
#%%
selected_columns = ['nt_XYma1', 'nt_XYma3',
                    'nt_XYmaf1', 'nt_XYmaf3',
                    'XY_spreadp1', 'XY_spreadp3',
                    'XY_spreadf1', 'XY_spreadf3', 'move_cat']
data = Train_set[selected_columns]
data = data.dropna().reset_index(drop = True)

X = data[selected_columns[:-1]]

y = data['move_cat']
#%%
best_model = load_model('best_mod_noweights_weights.h5')
# scaler = joblib.load('move_model_scaler.pkl')

#%%
best_model.summary()
#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_ind = int(0.8*len(X) )

X_train = X_scaled[:train_ind]
X_valid = X_scaled[train_ind:]

y_train = y[:train_ind]
y_valid = y[train_ind:]

#%%
joblib.dump(scaler, 'I2_move_mod_scaler.pkl')
#%%
def create_model(hidden_layers=2, units=15, activation='relu', dropout_rate=0):
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
    'dropout_rate': [0], 
    'activation': ['relu']
}

scorer = make_scorer(accuracy_score) 

#%%
#%%
#%%
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes = classes, y = y)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

accuracies = []
f1_scores = []
validation_losses = []
params = []

for hidden_layers in param_grid['hidden_layers']:
    for units in param_grid['units']:
        for dropout_rate in param_grid['dropout_rate']:

            model = create_model(hidden_layers=hidden_layers, units=units, activation='relu', dropout_rate = dropout_rate )
            model.set_weights(best_model.get_weights()) 
            history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                                class_weight=class_weights_dict,
                                epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])
            #
            y_pred = (model.predict(X_valid) > 0.5).astype(int) 
            accuracy = accuracy_score(y_valid, y_pred)            
            #
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
checkpoint_filepath = 'best_move_modI2_callback.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)
#%%
#%%
new_model = create_model(hidden_layers=best_params[0], units=best_params[1], activation='relu', dropout_rate=0)
new_model.set_weights(best_model.get_weights())
new_model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
               epochs=120, batch_size=32, callbacks=[early_stopping, checkpoint])
#%%
#%%
I2_model = load_model('best_move_modI2_callback.h5')
#%%
rig6_pred = (I2_model.predict(X_scaled) > 0.5).astype(int) 
#%%
data_sort = Train_set[selected_columns[:-1]]
data_sort = data_sort.dropna()

#%%
Train_set['pred'] = np.nan
#%%
for i, p in zip(data_sort.index, rig6_pred):    
    Train_set.loc[i, 'pred'] = p
      
#%%
testing_dat = datasets['rig_data6'].copy()
#%%
testing_dat['pred'] = Train_set['pred'].values
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
       plt.scatter(initial_data['X'], initial_data['Y'], marker='o', s=psize, c= initial_data['cols'] )
       
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
%matplotlib qt 
animation_fun_pred(testing_dat, pace = 0.1, testing = False)
#%%
#%%
Train_set['move_cat'] = Train_set['pred']
#%%#%%
#%%
#%%
#%%
data = Train_set[selected_columns]
data = data.dropna().reset_index(drop = True)

X = data[selected_columns[:-1]]

y = data['move_cat']
#%%
#%%
I2_model.summary()
#%%
#%%
#%%
Train_set.to_pickle('UWB_SSI2_Train_dat.pkl')
#%%
#%%
#%%
#%%