# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:53:57 2024

@author:
"""

#%%
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import itertools
from keras.models import Model, Sequential, load_model
# TF_ENABLE_ONEDNN_OPTS=0
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import joblib
import pickle
from sklearn.metrics import mean_squared_error
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.patches as mpatches
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Concatenate,Dense,Flatten,Reshape,Dropout,AveragePooling2D,GlobalAveragePooling2D
from keras import backend as K
from tensorflow.keras.regularizers import l1, l2

seed = 43
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)

#%%
#%%
dwell_clusterids_df = pd.read_pickle('dwell_df_clean.pkl')
dwell_cluster_df = pd.read_pickle('dwell_cluster_df.pkl')

path_clusters_df = pd.read_pickle('path_clusters_df.pkl')

#%%
import pickle
with open('datasets_28.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
with open('transit_indi_grids_padded.pkl', 'rb') as f:
     transit_indi_grids_padded = pickle.load(f)
#%%
#%%
Tri_dat = pd.concat( datasets.values() )

#%%
#%%
#%%
min_x = ma.floor(min(Tri_dat['X']) )
max_x = ma.ceil(max(Tri_dat['X']) )
min_y = ma.floor(min(Tri_dat['Y']) )
max_y = ma.ceil(max(Tri_dat['Y']) )

grid_spacing = 0.25  # Grid spacing
#%%
#%%
ncols = int((max_x-min_x)/grid_spacing)
nrows = int((max_y-min_y)/grid_spacing)

#%%
#%%
#%%
def train_test_split(data_dict, train_size=0.85, shuffle=True, seed = seed):
    
    np.random.seed(seed)

    if shuffle:
        keys = list(data_dict.keys())
        np.random.shuffle(keys)
    else:
        keys = list(data_dict.keys())

    num_train_samples = int(train_size*len(keys))

    # Split keys into training and validation sets
    train_keys = keys[:num_train_samples]
    test_keys = keys[num_train_samples:]

    # Create dictionaries for training and validation data
    train_data_dict = {key: data_dict[key] for key in train_keys}
    test_data_dict = {key: data_dict[key] for key in test_keys}

    return train_data_dict, test_data_dict

#%%
TV_data_dict, test_data_dict = train_test_split(transit_indi_grids_padded, train_size=0.87,
                                                  shuffle=True)

#%%
# train_data_dict, val_data_dict = train_test_split(TV_data_dict, train_size=0.92,
  #                                                shuffle=True)
#%%
#%%
X_train = np.array(list(TV_data_dict.values()))
# X_val = np.array(list(val_data_dict.values()))
X_test = np.array(list(test_data_dict.values()))
#%%
X_train_extra = []

for transit in X_train:
    
    rotated_90 = np.rot90(transit)
    rotated_180 = np.rot90(transit, 2)
    reflected_x = np.flipud(transit)
    reflected_y = np.fliplr(transit)
    
    X_train_extra.append(rotated_90)
    X_train_extra.append(rotated_180)
    X_train_extra.append(reflected_x)
    X_train_extra.append(reflected_y)

X_train_extra = np.array(X_train_extra)

X_train_augmented = np.concatenate((X_train, X_train_extra), axis=0)

#%%
plt.imshow(X_train[0], interpolation='nearest')
plt.show()
for i in range(4):
    plt.imshow(X_train_extra[i], interpolation='nearest')
    plt.colorbar()  # Optional: to show a color bar
    plt.show()

#%%
with open('AE_X_train_aug.pkl', 'wb') as f:
    pickle.dump(X_train_augmented, f)
    
#%%
#%%
#%%
k_size = 3
#%%
def conv_encoder(X_tr, X_val, filters_vec,
                 dropout_rate_d = 0, dropout_rate_c = 0,
                 hidden_d_layers = 0, hidden_d = 400, l1_rate = 1e-5,
                 latent_vars = 32, convolutional_dropout = False, d_activation = 'relu',
                 pooling = False, pooling_type = 'max', c_activation = 'relu'):
    
    if pooling == False:
        stride = (2,2)
    else:
        stride = (1,1)
    
    input_shape = (X_tr.shape[1], X_tr.shape[2], 1)   
    inputs = Input(shape=input_shape)
        
    x = Conv2D(filters_vec[0], (k_size, k_size), activation=c_activation, padding='same', strides = stride )(inputs)
    if pooling == True:
        if pooling_type == 'avg':
            x = AveragePooling2D ((2, 2), padding='same')(x)
        else:
            x = MaxPooling2D ((2, 2), padding='same')(x)
    x = Dropout(dropout_rate_c)(x)
    
    x = Conv2D(filters_vec[1], (k_size, k_size), activation=c_activation, padding='same', strides = stride)(x)
    if pooling == True:
        if pooling_type == 'avg':
            x = AveragePooling2D ((2, 2), padding='same')(x)
        else:
            x = MaxPooling2D ((2, 2), padding='same')(x)
    x = Dropout(dropout_rate_c)(x)
    
    x = Conv2D(filters_vec[2], (k_size, k_size), activation=c_activation, padding='same', strides = stride)(x)
    if pooling == True:
        if pooling_type == 'avg':
            x = AveragePooling2D ((2, 2), padding='same')(x)
        else:
            x = MaxPooling2D ((2, 2), padding='same')(x)
    x = Dropout(dropout_rate_c)(x)
    
    x = Conv2D(filters_vec[3], (k_size, k_size), activation=c_activation, padding='same', strides = stride)(x)
    if pooling == True:
        if pooling_type == 'avg':
            x = AveragePooling2D ((2, 2), padding='same')(x)
        else:
            x = MaxPooling2D ((2, 2), padding='same')(x)
    x = Dropout(dropout_rate_c)(x)
    
    x = Flatten()(x)
    for _ in range(hidden_d_layers):
        x = Dense(hidden_d, activation=d_activation)(x)
        x = Dropout(dropout_rate_d)(x)
    
    x = Dense(latent_vars, activation=d_activation, activity_regularizer=l1(l1_rate))(x)
   
    latent_space = x
    
    encoder_model = Model(inputs, latent_space)
    
    return encoder_model
#%%
#%%
def conv_decoder(latent_dim, latent_vars, dropout_rate_d, filters_vec, c_activation='relu',
                 d_activation = 'relu', upsampling_interpolation = 'bilinear'):
    
   # latent_inputs = Input(shape=(latent_dim,))
    latent_inputs = Input(shape=latent_dim)
    x = latent_inputs
    
    x = Dense(4*4*latent_vars, activation=d_activation)(latent_inputs)
    x = Dropout(dropout_rate_d)(x)
    x = Reshape((4,4,latent_vars))(x)
    
    x = Conv2D(filters_vec[3], (k_size, k_size), activation=c_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(filters_vec[2], (k_size, k_size), activation=c_activation, padding='same')(x)
    x = UpSampling2D((2,2))(x)
    
    x = Conv2D(filters_vec[1], (k_size, k_size), activation=c_activation, padding='same')(x)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(filters_vec[0], (k_size, k_size), activation=c_activation, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    reconstructed_output = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)
    
    decoder_model = Model(latent_inputs, reconstructed_output)
    
    return decoder_model
#%%
def c_auto_encoder(train, val, mod_pars_encoder, mod_pars_decoder, start_size,
                   filter_factor = 2, learning_rate = 0.0001):
    
    train_ready = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
    val_ready = val.reshape(val.shape[0], val.shape[1], val.shape[2], 1)
    
    start = start_size
    filters_vec = [int(start*filter_factor**x) for x in range(5)]
    
    mod_encoder = conv_encoder(train_ready, val_ready, **mod_pars_encoder, filters_vec=filters_vec)
    mod_decoder = conv_decoder(latent_dim = mod_encoder.output_shape[1:], filters_vec=filters_vec,
                               **mod_pars_decoder)
    #print(mod_encoder.output_shape[1:])
    autoencoder_input = Input(shape=(train.shape[1], train.shape[2], 1))
    latent_rep = mod_encoder(autoencoder_input)
    reconstructed_output = mod_decoder(latent_rep)
    
    autoencoder = Model(autoencoder_input, reconstructed_output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
     
    return autoencoder

#%% 
param_grid_conv_encoder = {
    'hidden_d_layers': [0], # could try
    'dropout_rate_c': [0],  # dropout doesn't appear to be useful here
    'pooling': [True], # better to use stride
    'pooling_type': ['avg'],
    'l1_rate': [1e-5]
}

param_grid_common = {
    'dropout_rate_d': [0],
    'latent_vars': [32], # yes
    'd_activation': ['elu'],
    'c_activation': ['relu'],
}

param_grid_conv_decoder = {
    'upsampling_interpolation': ['nearest'] # nearest is best
}

param_grid_other = {  
    'filter_factor': [2], # 2 preferred to 1.5
    'start_size': [16] # more than 4
}

# param_combinations_ModA = list(itertools.product(*param_grid_modA.values()))
comb_dict = {**param_grid_conv_encoder, **param_grid_common,
                                **param_grid_conv_decoder, **param_grid_other}
encoder_dict ={**param_grid_conv_encoder, **param_grid_common}
decoder_dict ={**param_grid_common, **param_grid_conv_decoder}

param_combinations_ModC = list(itertools.product(*comb_dict.values()))
#%%
#%%
def auto_encoder_CV(param_combinations, param_grid_encoder, param_grid_decoder, param_grid_oth,
                    function_name,
                    thresh = 0.5, cv_splits = 3, seed = seed):    
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    bce_losses = []

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    for ind, params in enumerate(param_combinations):
        
        model_params_encoder = dict(zip(param_grid_encoder.keys(), params[:len(param_grid_encoder.keys() ) ]  ) )
        model_params_decoder = dict(zip(param_grid_decoder.keys(), params[-len(param_grid_decoder.keys() )-len(param_grid_other) :-len(param_grid_other) ]  ) )
        model_other_pars = dict(zip(param_grid_oth.keys(), params[-len(param_grid_oth): ]  ) )

        accs = []
        precs = []
        recs = []
        f1s = []
        losses = []
        
        for train_idx, val_idx in kf.split(X_train_augmented):
            early_stopping = EarlyStopping(monitor='val_loss', patience=8,
                                           restore_best_weights=True)
        
            X_train_fold, X_val_fold = X_train_augmented[train_idx], X_train_augmented[val_idx]
            model = function_name(X_train_fold, X_val_fold, model_params_encoder, model_params_decoder, **model_other_pars )
            model.fit(X_train_fold, X_train_fold,
                             epochs=50,
                             batch_size=32,
                             shuffle=True,
                             validation_data=(X_val_fold, X_val_fold),
                             verbose=1, callbacks=[early_stopping]
                             )
           
            predictions = model.predict(X_val_fold)
            r_predictions = (predictions > thresh)
            r_predictions = r_predictions.astype(np.int32)
        
            loss = model.evaluate(X_val_fold, X_val_fold, verbose=1)
            accuracy = accuracy_score(X_val_fold.flatten(), r_predictions.flatten())
            precision = precision_score(X_val_fold.flatten(), r_predictions.flatten())
            recall = recall_score(X_val_fold.flatten(), r_predictions.flatten())
            f1 = f1_score(X_val_fold.flatten(), r_predictions.flatten())
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1-score:", f1)
            print("val_loss:", loss)
        
            losses.append(loss)
            accs.append(accuracy)
            precs.append(precision)
            recs.append(recall)
            f1s.append(f1)

        accuracies.append(np.mean(accs) )
        precisions.append(np.mean(precs))
        recalls.append(np.mean(recs))
        f1scores.append(np.mean(f1s))
        bce_losses.append(np.mean(losses))
    
        print(f'Run {ind+1} of {len(param_combinations)} complete')

    return f1scores, recalls, precisions, accuracies, bce_losses

#%%
f1scores, recalls, precisions, accuracies, BCEs = auto_encoder_CV(param_combinations = param_combinations_ModC,
                                                                  param_grid_encoder = encoder_dict,
                                                                  param_grid_decoder = decoder_dict,
                                                                  param_grid_oth = param_grid_other,
                                                                  function_name = c_auto_encoder,
                                                                  thresh = 0.5, cv_splits = 3, seed = seed  )
    
#%%
#%%
best_index = np.argmax(f1scores)
best_index_alt = np.argmin(BCEs)
print(f'Best f1 score: {f1scores[best_index]}' ) # ~85.1
print(f'Accuracy of permatation selected: {accuracies[best_index]} ') # ~ 99.5
print(f'Precision of permatation selected: {precisions[best_index]} ') # ~83.4
print(f'Recall of permatation selected: {recalls[best_index]} ') # ~ 86.9

print("Best parameters:", param_combinations_ModC[best_index])
best_params = {x:y for x,y in zip(comb_dict.keys(), param_combinations_ModC[best_index]) }
best_encoder_params = {x:y for x,y in zip(encoder_dict.keys(), param_combinations_ModC[best_index][:len(encoder_dict)]) }
best_decoder_params = {x:y for x,y in zip(decoder_dict.keys(), param_combinations_ModC[best_index][len(param_grid_conv_encoder):len(param_grid_conv_encoder)+len(decoder_dict)]) }
# best losses around 0.025, that's when also trying to keep the latent layer size something reasonable.
#%%
plt.scatter(range(len(f1scores)), f1scores)
#%%
for i in range(len(param_combinations_ModC[0]) ):
    
    plt.scatter([x[i] for x in param_combinations_ModC], f1scores)
    plt.title(f'{list(comb_dict.keys())[i]}')
    plt.show()
    
#%%
#%%
final_mod_filename = 'autoCONV_AUG_29_callback.keras'
#%%
checkpoint_filepath = final_mod_filename
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

#%%
model = c_auto_encoder(X_train_augmented, X_test, 
                       best_encoder_params, best_decoder_params,
                       best_params['start_size'],
                       best_params['filter_factor'],
                                            )
#%%
history = model.fit(x=X_train_augmented, y=X_train_augmented,
                        validation_data=(X_test, X_test), shuffle = True,
                    #class_weight=class_weights_dict,
                      epochs=80, batch_size=32,
                      verbose=1, callbacks=[early_stopping, checkpoint])
#%%
AE_CONV_AUG_29_mod = load_model(final_mod_filename)
#%%
AE_CONV_AUG_29_mod.summary()
#%%
model = AE_CONV_AUG_29_mod
thresh = 0.5
decoded_test_data = tf.squeeze(model.predict(X_test), axis = -1)
decoded_train_data = tf.squeeze(model.predict(X_train), axis = -1)
#decoded_valid_data = tf.squeeze(model.predict(X_val), axis = -1)

test_loss = np.mean(binary_crossentropy(X_test, decoded_test_data))
train_loss = np.mean(binary_crossentropy(X_train, decoded_train_data))
#valid_loss = np.mean(binary_crossentropy(X_val, decoded_valid_data))

print("Reconstruction test Loss (bce-loss):", test_loss )
print("Reconstruction train Loss (bce-loss):", train_loss )
#print("Reconstruction valid Loss (bce-loss):", valid_loss )
#
binary_decoded_test_data = tf.cast(decoded_test_data > thresh, tf.int32)
binary_decoded_train_data = tf.cast(decoded_train_data > thresh, tf.int32)

#%%
# test
# Convert TensorFlow tensor to NumPy array
accuracy = accuracy_score(X_test.flatten(), tf.reshape(binary_decoded_test_data, [-1]) )
precision = precision_score(X_test.flatten(), tf.reshape(binary_decoded_test_data, [-1]) )
recall = recall_score(X_test.flatten(), tf.reshape(binary_decoded_test_data, [-1]) )
f1 = f1_score(X_test.flatten(), tf.reshape(binary_decoded_test_data, [-1]) )
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1-score:", f1) # 87.8
#
accuracy = accuracy_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
precision = precision_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
recall = recall_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
f1 = f1_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
print("Train Accuracy:", accuracy)
print("Train Precision:", precision)
print("Train Recall:", recall)
print("Train F1-score:", f1) # 92.0
#
#%%
def threshold_opt(preds, truth):
    
    thresholds = np.linspace(0, 1, 21)
    best_threshold = []
    best_f1_score = 0
    
    for threshold in thresholds:

        binary_preds = tf.cast(preds > threshold, tf.int32)   
        f1 = f1_score(tf.reshape(truth, [-1]), tf.reshape(binary_preds, [-1])  )
    
        if f1 > best_f1_score:
            best_threshold.append(threshold)
            best_threshold[:] = best_threshold[-1:]
            best_f1_score = f1
        elif f1 == best_f1_score:
            best_threshold.append(threshold)
            
    print("Best Threshold:", best_threshold)
    print("Best F1-score:", best_f1_score)
    return best_threshold
    
#%%
threshold = threshold_opt(decoded_train_data, X_train) # 92
#%%
threshold = threshold_opt(decoded_test_data, X_test) # 87.9
#%%
#%%
thresh = 0.5
binary_decoded_test_data = tf.cast(decoded_test_data > thresh, tf.int32)   
#%%
def grid_ref_fun(x, y):
    xg = ma.floor((x - min_x)/grid_spacing )
    yg = ma.ceil((max_y - y)/grid_spacing ) - 1
    return xg, yg
#%%
for i in range(len(decoded_test_data)):

    plt.figure(figsize=(11, 7))

    actual_patch = mpatches.Patch(color='blue', alpha=0.4, label='Actual')
    predicted_patch = mpatches.Patch(color='red', alpha=0.4, label='Predicted')

    plt.imshow(X_test[i], cmap='Blues', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y], alpha = 1, 
           vmin=0.0, vmax=1)

    plt.imshow(binary_decoded_test_data[i], cmap='Reds', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y], alpha = 0.6, 
           vmin=0.0, vmax=1)

    x_vals = np.around(np.arange(min_x, max_x, grid_spacing), 2)
    y_vals = np.around(np.flip(np.arange(min_y, max_y, grid_spacing)), 2)

    x_labels = [str(int(x)) if x%1 == 0 else '' for x in x_vals]
    y_labels = [str(int(y)) if y%1 == 0 else '' for y in y_vals]

    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

    plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
    plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
    plt.xlabel(' (X)')
    plt.ylabel(' (Y)')
    plt.title('')
    plt.legend(handles=[actual_patch, predicted_patch])

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