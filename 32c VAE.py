# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:53:57 2024

@author:
"""

#%%
import sys
import site

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import itertools
import tensorflow as tf
import keras

from keras.models import Model, Sequential, load_model
#TF_ENABLE_ONEDNN_OPTS=0
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
import pickle
with open('AE_X_train_aug.pkl', 'rb') as f:
    X_train_augmented = pickle.load(f)
    
with open('AE_X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    
with open('AE_X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

#%%    
#%%
#%%
k_size = 3
batch_size = 32
epochs_CV = 30
epochs_final = 80
spacial = True
#%%
def V_encoder(X_tr, X_val, filters_vec,
                 dropout_rate_d = 0, dropout_rate_c = 0,
                 hidden_d_layers = 0, c_layers = 4,
                 hidden_d = 400, l1_rate = 1e-5,
                 latent_vars = 40, d_activation = 'relu',
                 pooling = True, pooling_type = 'avg', c_activation = 'relu',
                 spacial = False):
    
    width = 64
    
    if pooling == False:
        stride = (2,2)
    else:
        stride = (1,1)
    
    def conv_downsampling(layer, filter_vec, kernal = k_size, activation = c_activation):
        layer = Conv2D(filter_vec, (kernal, kernal), activation=activation,
                                       padding='same', strides = stride )(layer)
        if pooling:
            if pooling_type == 'avg':
                layer = AveragePooling2D ((2, 2), padding='same')(layer)
            else:
                layer = MaxPooling2D ((2, 2), padding='same')(layer)
        layer = Dropout(dropout_rate_c)(layer)
        return layer
    
    input_shape = (X_tr.shape[1], X_tr.shape[2], 1)   
    inputs = Input(shape=input_shape)
    x = inputs    
    
    for c_layer in range(c_layers):
        
        x = conv_downsampling(x, filters_vec[c_layer])
        width = width/2
    
    if spacial == True:
        
        y = Flatten()(x)
        y = Dense(latent_vars)(y)
        while width > 2:
            x = conv_downsampling(x, latent_vars)
            width = width/2
            print(width)
        x = conv_downsampling(x, latent_vars, kernal=2, activation = 'linear')
        x = Flatten()(x)
                            
    else:
        x = Flatten()(x)
        for _ in range(hidden_d_layers):
            x = Dense(hidden_d, activation=d_activation)(x)
            x = Dropout(dropout_rate_d)(x)
        # x = Dense(latent_vars, activation=d_activation, activity_regularizer=l1(l1_rate))(x)
        # y = Dense(latent_vars, activation=d_activation, activity_regularizer=l1(l1_rate))(x)
        x = Dense(latent_vars)(x)
        y = Dense(latent_vars)(x)
    
    z_mean = x
    z_sds = y
    
    def reparam_trick(means, sds):
        batch = K.shape(means)[0]
        dim = K.int_shape(means)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return means + K.exp(0.5*sds)*epsilon
       
    z_samples = reparam_trick(z_mean, z_sds)
    
    encoder_model = Model(inputs, [z_mean, z_sds, z_samples], name = 'encoder')
    
    return encoder_model
#%%
#%%
def V_decoder(latent_dim, latent_vars, dropout_rate_d, filters_vec, c_activation='relu',
                 d_activation = 'relu', upsampling_interpolation = 'bilinear'):
    
   # latent_inputs = Input(shape=(latent_dim,))
    latent_inputs = Input(shape=latent_dim)
    x = latent_inputs
    
    x = Dense(4*4*latent_vars, activation=d_activation)(x)
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
    
    x = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)
    
    reconstructed_output = x
    
    decoder_model = Model(latent_inputs, reconstructed_output, name = 'decoder')
    
    return decoder_model
#%%
def VAE(train, val, mod_pars_encoder, mod_pars_decoder, start_size,
                   filter_factor = 2, learning_rate = 0.0001, beta = 0):
    
    train_ready = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
    val_ready = val.reshape(val.shape[0], val.shape[1], val.shape[2], 1)
    
    start = start_size
    filters_vec = [int(start*filter_factor**x) for x in range(5)]
    
    mod_encoder = V_encoder(train_ready, val_ready, **mod_pars_encoder, filters_vec=filters_vec)
    #print(mod_encoder.output_shape)
    print(mod_encoder.output_shape[-1][1:])
    decoder_dim = mod_encoder.output_shape[-1][1:]
    mod_decoder = V_decoder(latent_dim = decoder_dim, filters_vec=filters_vec,
                               **mod_pars_decoder)
    autoencoder_input = Input(shape=(train.shape[1], train.shape[2], 1))
    z_mean, z_log_var, z_samples = mod_encoder(autoencoder_input)
    reconstructed_output = mod_decoder(z_samples)
    
    vae = Model(autoencoder_input, reconstructed_output)
    
    loss = vae_loss(autoencoder_input, reconstructed_output, z_mean, z_log_var, beta)
    
    vae.add_loss(loss)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    vae.compile(optimizer=optimizer)
     
    return vae

#%%
import tensorflow.keras.backend as K

def vae_loss(inputs, outputs, z_mean, z_log_var, beta):
    # Reconstruction loss
    reconstruction_loss = K.mean(K.binary_crossentropy(K.flatten(inputs), K.flatten(outputs)))
    # KL Divergence loss
    kl_loss = -0.5 * K.mean(K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
    # Total VAE loss
    total_loss = reconstruction_loss + beta*kl_loss
    return total_loss

#%%
def VAE_predictor(encoder, decoder, X):
    # Define input for the new prediction model
    #inputs = Input(shape=input_shape)
    
    z_mean, _, _ = encoder.predict(X)
    
    reconstructed_output = decoder.predict(z_mean)

    return reconstructed_output
#%%
param_grid_V_encoder = {
    'hidden_d_layers': [0], # could try
    'dropout_rate_c': [0],  # dropout doesn't appear to be useful here
    'pooling': [True], # better to use stride
    'pooling_type': ['avg'],
    'l1_rate': [1e-5],
   # param_grid_conv_encoder
    'spacial': [spacial]
}

param_grid_common = {
    'dropout_rate_d': [0],
    'latent_vars': [40], # yes
    'd_activation': ['elu'],
    'c_activation': ['relu']
}

param_grid_V_decoder = {
    'upsampling_interpolation': ['nearest'] # nearest is best
}

param_grid_other = {  
    'filter_factor': [2], # 2 preferred to 1.5
    'start_size': [16], # more than 4
    'beta': [5e-6]
}

comb_dict = {**param_grid_V_encoder, **param_grid_common,
                                **param_grid_V_decoder, **param_grid_other}
encoder_dict ={**param_grid_V_encoder, **param_grid_common}
decoder_dict ={**param_grid_common, **param_grid_V_decoder}

param_combinations_ModC = list(itertools.product(*comb_dict.values()))

#%%  0.0000005, 0.0000001, 0.000005, 0.000001
#%%
def auto_encoder_CV(param_combinations, param_grid_encoder,
                    param_grid_decoder, param_grid_oth,
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
            model = function_name(X_train_fold, X_val_fold, model_params_encoder,
                                  model_params_decoder, **model_other_pars )
            model.fit(X_train_fold, X_train_fold,
                             epochs=epochs_CV,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_val_fold, X_val_fold),
                             verbose=1, callbacks=[early_stopping]
                             )
            encoder = model.get_layer('encoder')
            decoder = model.get_layer('decoder')
            predictions = VAE_predictor(encoder, decoder, X_val_fold)
          #  predictions = model.predict(X_val_fold)
            r_predictions = (predictions > thresh)
           # print(r_predictions.sum() )
            r_predictions = r_predictions.astype(np.int32)
            
            def calculate_reconstruction_loss(true, pred):
                return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.keras.layers.Flatten()(true), 
                                                              tf.keras.layers.Flatten()(pred)))
        
            loss = calculate_reconstruction_loss(X_val_fold, predictions).numpy()
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
                                                                  function_name = VAE,
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
best_decoder_params = {x:y for x,y in zip(decoder_dict.keys(), param_combinations_ModC[best_index][len(param_grid_V_encoder):len(param_grid_V_encoder)+len(decoder_dict)]) }
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
if spacial == True:
    final_mod_filename = 'VAE_AUG_spacial_32b_callback.keras'
else:
    final_mod_filename = 'VAE_AUG_32b_callback.keras'

#%%
checkpoint_filepath = final_mod_filename
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

#%%
model = VAE(X_train_augmented, X_test, 
                       best_encoder_params, best_decoder_params,
                       best_params['start_size'],
                       best_params['filter_factor'],
                                            )
#%%
history = model.fit(x=X_train_augmented, y=X_train_augmented,
                        validation_data=(X_test, X_test), shuffle = True,
                    #class_weight=class_weights_dict,
                      epochs=epochs_final, batch_size=32,
                      verbose=1, callbacks=[early_stopping, checkpoint])
#%%
VAE_AUG_32_mod = load_model(final_mod_filename)
#%%
VAE_AUG_32_mod.summary()
#%%
model = VAE_AUG_32_mod
thresh = 0.5

encoder = model.get_layer('encoder')
decoder = model.get_layer('decoder')
decoded_test_data = tf.squeeze(VAE_predictor(encoder, decoder, X_test), axis = -1  )
decoded_train_data = tf.squeeze(VAE_predictor(encoder, decoder, X_train), axis = -1  )

# decoded_test_data = tf.squeeze(model.predict(X_test), axis = -1)
# decoded_train_data = tf.squeeze(model.predict(X_train), axis = -1)
# #decoded_valid_data = tf.squeeze(model.predict(X_val), axis = -1)

test_loss = np.mean(binary_crossentropy(X_test, decoded_test_data))
train_loss = np.mean(binary_crossentropy(X_train, decoded_train_data))

print("Reconstruction test Loss (bce-loss):", test_loss )
print("Reconstruction train Loss (bce-loss):", train_loss )
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
print("Test F1-score:", f1) # 86.5, 80.5 spacial
#
accuracy = accuracy_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
precision = precision_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
recall = recall_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
f1 = f1_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
print("Train Accuracy:", accuracy)
print("Train Precision:", precision)
print("Train Recall:", recall)
print("Train F1-score:", f1) # 85.8, 79.9 spacial
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
threshold = threshold_opt(decoded_train_data, X_train) # 86.1, 80.6 spacial
#%%
threshold = threshold_opt(decoded_test_data, X_test) # 86.5, 81.9 spacial
#%%
#%%
thresh = 0.4
binary_decoded_test_data = tf.cast(decoded_test_data > thresh, tf.int32) 
#%%
#%%
import pickle
with open('datasets_28.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
Tri_dat = pd.concat( datasets.values() )

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