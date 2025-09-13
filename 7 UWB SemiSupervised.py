# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:19:36 2024
@author: 
"""
#%%
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import load_model
import joblib
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math as ma
from tensorflow.keras import layers, Sequential
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers, callbacks 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from keras.callbacks import ModelCheckpoint
import pickle
from scikeras.wrappers import KerasRegressor, KerasClassifier
import tensorflow as tf
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
pd.set_option('display.max_columns', 30)
random_seed = 69
tf.random.set_seed(69)
np.random.seed(69)
#%%
%matplotlib inline
#%%
import pickle

with open('UWB_datasets_5.pkl', 'rb') as f:
    datasets = pickle.load(f)

#%%
rig_data = datasets['rig_data6'] 
rig_data['move_cat'] = np.nan

#%%
#%%
start = 34147
end = 34169
#%%,
indices = np.arange(start, end)
norm_indices = (indices - start) / (end - start)
cmap = plt.cm.coolwarm
plt.xlim(min(rig_data['X']), max(rig_data['X']))
plt.ylim(min(rig_data['Y']), max(rig_data['Y'])+1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'{start} to {end}')
plt.scatter(rig_data['X'][start:end], rig_data['Y'][start:end],
            marker = 'o', s = 7, c=norm_indices, cmap=cmap)
#%%
move_col = rig_data.columns.get_loc('move_cat')
#%%
rig_data.iloc[288:15000, move_col] = 0
rig_data.iloc[23000:23500, move_col] = 0
rig_data.iloc[24000:27100, move_col] = 0
#%%
#%%
rig_data.iloc[21915:21955, move_col] = 1
rig_data.iloc[21910:21950, move_col] = 1
rig_data.iloc[28400:28490, move_col] = 1
rig_data.iloc[32830:32842, move_col] = 1
rig_data.iloc[33800:33915, move_col] = 1
rig_data.iloc[34037:34055, move_col] = 1
rig_data.iloc[34147:34169, move_col] = 1
#%%
#%%
datasets['rig_data6'] = rig_data
#%%
#%%
#%%
#%%
Train_set = rig_data.reset_index(drop = True)
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_ind = int(0.8*len(X) )

X_train = X_scaled[:train_ind]
X_valid = X_scaled[train_ind:]

y_train = y[:train_ind]
y_valid = y[train_ind:]

#%%
joblib.dump(scaler, 'move_model_scaler.pkl')

#%%
from keras.layers import Dense, Dropout
def create_model(hidden_layers=1, units=12, activation='relu', dropout_rate=0.2):
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
    'hidden_layers': [2, 3],
    'units': [15, 20],
    'dropout_rate': [0, 0.1], 
    'activation': ['relu']
}

scorer = make_scorer(accuracy_score) 

#%%
#%%
#%%
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=69)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

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
            
            fold_accuracies = []
            fold_f1_scores = []
            fold_validation_losses = []

            for train_idx, test_idx in kfold.split(X_train, y_train):
                
                Xf_train, Xf_val = X_train[train_idx], X_train[test_idx]
                yf_train, yf_val = y_train[train_idx], y_train[test_idx]

                model = create_model(hidden_layers=hidden_layers, units=units, activation='relu', dropout_rate = dropout_rate )
                
                history = model.fit(Xf_train, yf_train, validation_data=(Xf_val, yf_val), class_weight=class_weights_dict,
                                    epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])
               
                y_pred = (model.predict(Xf_val) > 0.5).astype(int) 
                accuracy = accuracy_score(yf_val, y_pred)
                f1 = f1_score(yf_val, y_pred)
                val_loss = history.history['val_loss'][-1]
                
                fold_accuracies.append(accuracy)
                fold_f1_scores.append(f1)
                fold_validation_losses.append(val_loss)

            avg_accuracy = np.mean(fold_accuracies)
            avg_f1_score = np.mean(fold_f1_scores)
            avg_val_loss = np.mean(fold_validation_losses)

            accuracies.append(avg_accuracy)
            f1_scores.append(avg_f1_score)
            validation_losses.append(avg_val_loss)
            params.append([hidden_layers, units, dropout_rate])

            print(f'Hidden Layers: {hidden_layers}, Units: {units}, Activation: {dropout_rate}')
            print(f'Average Accuracy: {avg_accuracy}, Average F1 Score: {avg_f1_score}, Average Validation Loss: {avg_val_loss}')
#%%
best_index = np.argmax(f1_scores)
best_params = params[best_index]
#%%
#%%
param_grid2 = {
    'hidden_layers': [3,4],
    'units': [18, 20],
    'dropout_rate': [0], 
    'activation': ['relu']
}
#%%
early_stopping = EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)

accuracies = []
f1_scores = []
validation_losses = []
params = []

for hidden_layers in param_grid2['hidden_layers']:
    for units in param_grid2['units']:
        for dropout_rate in param_grid2['dropout_rate']:

            model = create_model(hidden_layers=hidden_layers, units=units, activation='relu', dropout_rate = dropout_rate )
                
            history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), class_weight=class_weights_dict,
                                epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])
               
            y_pred = (model.predict(X_valid) > 0.5).astype(int)
            accuracy = accuracy_score(y_valid, y_pred )
           
            f1 = f1_score(y_valid, y_pred)
            val_loss = history.history['val_loss'][-1]

            accuracies.append(accuracy)
            f1_scores.append(f1)
            validation_losses.append(val_loss)
            params.append([hidden_layers, units, dropout_rate])

#%%
best_index = np.argmax(f1_scores)
best_params = params[best_index]
# 4, 20, 0
#%%
checkpoint_filepath = 'best_mod_noweights_weights.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

#%%
best_model = create_model(hidden_layers=best_params[0], units=best_params[1], activation='relu', dropout_rate=best_params[2])
best_model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
               epochs=100, batch_size=32, callbacks=[early_stopping, checkpoint])
#%%
best_model = load_model('best_mod_noweights_weights.h5')
scaler = joblib.load('move_model_scaler.pkl')
#%%
data = Train_set[selected_columns[:-1]]
data = data.dropna()
X_new_scaled = scaler.transform(data)

#%%
rig6_pred = (best_model.predict(X_new_scaled) > 0.5).astype(int)

#%%
data['pred'] = rig6_pred
#%%
Train_set['pred'] = np.nan
#%%
for i, p in zip(data.index, rig6_pred):
    
    Train_set.loc[i, 'pred'] = p
      
#%%
testing_dat = rig_data.copy() 
#%%
testing_dat['pred'] = Train_set['pred'].values
#%%
#%%
#%%
#%%
#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

exec(open(f'{path_stem}UWB functions.py').read())
#%%
#%%
%matplotlib qt 
animation_fun_pred(testing_dat , pace = 0.1, testing = True)
#%%
#%%
#%%
Train_set['move_cat'] = [x if not pd.isna(x) else y for x,y in zip(Train_set['move_cat'], Train_set['pred']) ]
#%%
#%%
#%%
Train_set.to_pickle('UWB_SS_Train_dat.pkl')
#%%
#%%
#%%
%matplotlib inline
#%%
fig, axes = plt.subplots(1, 4, figsize=(8.5, 4.5))
# Boxplot for 'ed_XYma5'
Train_set.boxplot(column='ed_XYma1', by='move_cat', ax=axes[0])
axes[0].set_title('Boxplot of ed_XYma1 by move_cat')
axes[0].set_xlabel('move_cat')
axes[0].set_ylabel('ed_XYma1')
# Boxplot for 'ed_XYma10'
Train_set.boxplot(column='ed_XYmaf1', by='move_cat', ax=axes[1])
axes[1].set_title('Boxplot of ed_XYmaf1 by move_cat')
axes[1].set_xlabel('move_cat')
axes[1].set_ylabel('ed_XYmaf1')
# Boxplot for 'ed_XYma20'
Train_set.boxplot(column='ed_XYma3', by='move_cat', ax=axes[2])
axes[2].set_title('Boxplot of ed_XYma3 by move_cat')
axes[2].set_xlabel('move_cat')
axes[2].set_ylabel('ed_XYma3')

Train_set.boxplot(column='ed_XYmaf3', by='move_cat', ax=axes[3])
axes[3].set_title('Boxplot of ed_XYmaf3 by move_cat')
axes[3].set_xlabel('move_cat')
axes[3].set_ylabel('ed_XYmaf3')
# Adjust layout
plt.tight_layout()
plt.show()

#%%
#%%
fig, axes = plt.subplots(1, 4, figsize=(8.5, 4.5))
# Boxplot for 'ed_XYma5'
Train_set.boxplot(column='XY_spreadp1', by='move_cat', ax=axes[0])
axes[0].set_title('Boxplot of XY_spreadp1 by move_cat')
axes[0].set_xlabel('move_cat')
axes[0].set_ylabel('XY_spreadp1')
# Boxplot for 'ed_XYma10'
Train_set.boxplot(column='XY_spreadp1', by='move_cat', ax=axes[1])
axes[1].set_title('Boxplot of XY_spreadp1 by move_cat')
axes[1].set_xlabel('move_cat')
axes[1].set_ylabel('XY_spreadp1')
# Boxplot for 'ed_XYma20'
Train_set.boxplot(column='XY_spreadp3', by='move_cat', ax=axes[2])
axes[2].set_title('Boxplot of XY_spreadp3 by move_cat')
axes[2].set_xlabel('move_cat')
axes[2].set_ylabel('XY_spreadp3')

Train_set.boxplot(column='XY_spreadf3', by='move_cat', ax=axes[3])
axes[3].set_title('Boxplot of XY_spreadf3 by move_cat')
axes[3].set_xlabel('move_cat')
axes[3].set_ylabel('XY_spreadf3')
# Adjust layout
plt.tight_layout()
plt.show()
#%%