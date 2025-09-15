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
TF_ENABLE_ONEDNN_OPTS=0
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
with open('AE_X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('AE_X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

#%%
unique_classes, class_counts = np.unique(X_train, return_counts=True)

# class_weights = [ sum(X_train)/np.prod(X_train.shape), (np.prod(X_train.shape)-sum(X_train) )/np.prod(X_train.shape) ]

#class_weights = np.array([ 0.15, 0.85 ], dtype=np.float32)
class_weights = tf.constant([ 0.15, 0.85 ], dtype=tf.float32) 
#%%
def custom_loss(y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.float32)
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    
    weights = tf.gather(class_weights, tf.cast(y_true, dtype=tf.int32))
    
    weights = tf.reshape(weights, [-1, 1, 1, 1])
    
    weighted_loss = bce_loss * weights
    
    return tf.reduce_mean(weighted_loss)
#%%
#custom_objects = {'custom_loss': custom_loss}
#%%
def create_modelA(X_tr, X_v, encoding_dim, latent_dim, hidden_encoder_layers=0, hidden_decoder_layers=0,
              encoder_activation = 'relu', decoder_activation='relu',
              dropout_rate = 0, l1_par = 0.00001, hidden_dim = 50, decoding_dim = 500
              ):
        
    input_shape = (X_tr.shape[1],X_tr.shape[2])
    # Encode
    inputs = Input(shape=input_shape)
    
    flat_input = Flatten()(inputs)
    
    encoded = Dense(encoding_dim, activation=encoder_activation)(flat_input)
    encoded_dropout = Dropout(dropout_rate)(encoded)
    
    for _ in range(hidden_encoder_layers):
       encoded = Dense(hidden_dim, activation=encoder_activation)(encoded_dropout)
       encoded_dropout = Dropout(dropout_rate)(encoded)
    
    latent = Dense(latent_dim, activation=encoder_activation, activity_regularizer=l1(l1_par))(encoded_dropout)
    latent_dropout = Dropout(dropout_rate)(latent)
        
    decoded = latent_dropout
    decoded = Dense(encoding_dim, activation=decoder_activation)(decoded)
    decoded = Dropout(dropout_rate)(decoded)
    for _ in range(hidden_decoder_layers):
       decoded = Dense(decoding_dim, activation=decoder_activation)(decoded)
       decoded = Dropout(dropout_rate)(decoded)

    decoded = Dense(X_tr.shape[1]*X_tr.shape[2], activation='sigmoid')(decoded)
    decoded = Reshape((X_tr.shape[1], X_tr.shape[2]))(decoded)
    
    # Autoencoder model
    autoencoder = Model(inputs, decoded)
    # Compile model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder 
#%%
#%%
param_grid_modA = {
    'encoding_dim': [600], # it picked over 500, less than 800
    'latent_dim': [32], # selcted greater than 100 but we need to try and reduce this
    'hidden_encoder_layers': [0], # 0 
    'hidden_decoder_layers': [0], # 0
    'encoder_activation': ['relu'], 
    'decoder_activation': ['relu'], #
    'dropout_rate': [0.1], # >0.05, <0.2
    'hidden_dim': [100],
    'decoding_dim': [700],
    'l1_par' : [5e-7,1e-6] 
}

param_grid_other = {
    'epochs': [70],
    'batch_size': [32]
}

# param_combinations_ModA = list(itertools.product(*param_grid_modA.values()))
comb_dict = merged_dict = {**param_grid_modA, **param_grid_other}
param_combinations_ModA = list(itertools.product(*comb_dict.values()))

#%%
#%%
def auto_encoder_CV(param_combinations, param_grid_mod, param_grid_oth, function_name, 
                    thresh = 0.5, cv_splits = 2, seed = seed):
    
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    bce_losses = []

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    for ind, params in enumerate(param_combinations):
        
        model_params = dict(zip(param_grid_mod.keys(), params[:len(param_grid_mod.keys() ) ]  ) )
        other_params = dict(zip(param_grid_oth.keys(), params[len(param_grid_mod.keys()): ]  ) )
    
        accs = []
        precs = []
        recs = []
        f1s = []
        losses = []
        
        for train_idx, val_idx in kf.split(X_train):
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
 #          
            model = function_name(X_train_fold, X_val_fold, **model_params)
          
            model.fit(X_train_fold, X_train_fold,
                             epochs=other_params['epochs'],
                             batch_size=other_params['batch_size'],
                             shuffle=True,
                             validation_data=(X_val_fold, X_val_fold),
                             verbose=1, callbacks=[early_stopping]
                             )
            predictions = model.predict(X_val_fold)
            r_predictions = (predictions > thresh)
            r_predictions = r_predictions.astype(int)
        
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
f1scores, recalls, precisions, accuracies, BCEs = auto_encoder_CV(param_combinations = param_combinations_ModA,
                                                            param_grid_mod = param_grid_modA,
                                                            param_grid_oth = param_grid_other,
                                                            function_name = create_modelA)


#%%
best_index = np.argmax(f1scores)
best_index_alt = np.argmin(BCEs)
print(f'Best f1 score: {f1scores[best_index]}' ) #~74.5
print(f'Accuracy of permatation selected: {accuracies[best_index]} ') # ~ 99.2
print(f'Precision of permatation selected: {precisions[best_index]} ') # ~76
print(f'Recall of permatation selected: {recalls[best_index]} ') # ~ 73

print("Best parameters:", param_combinations_ModA[best_index])
best_params = {x:y for x,y in zip(comb_dict.keys(), param_combinations_ModA[best_index]) }
# best losses around 0.025, that's when also trying to keep the latent layer size something reasonable.
#%%
plt.scatter(range(len(f1scores)), f1scores)
#%%
for i in range(len(param_combinations_ModA[0]) ):
    
    plt.scatter([x[i] for x in param_combinations_ModA], f1scores)
    plt.title(f'{list(comb_dict.keys())[i]}')
    plt.show()
#%%
# reconstructed_matrix = autoencoder.predict(boolean_matrix)
#%%
#%%
#%%
final_mod_filename = 'autoNN_29_callback.keras'
#%%
checkpoint_filepath = final_mod_filename
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

#%%
model = create_modelA(X_train, X_test, # X_val,
                    encoding_dim = best_params['encoding_dim'],
                    latent_dim = best_params['latent_dim'],
                    hidden_encoder_layers = best_params['hidden_encoder_layers'],
                    hidden_decoder_layers = best_params['hidden_decoder_layers'],
                    encoder_activation = best_params['encoder_activation'],
                    decoder_activation = best_params['decoder_activation'],
                    dropout_rate = best_params['dropout_rate'],
                    hidden_dim = best_params['hidden_dim'],
                    decoding_dim = best_params['decoding_dim'],
                    l1_par = best_params['l1_par']
                                            )
#%%
history = model.fit(x=X_train, y=X_train,
                        validation_data=(X_test, X_test), shuffle = True,
                      epochs=best_params['epochs'], batch_size=best_params['batch_size'],
                      verbose=1, callbacks=[early_stopping, checkpoint])
#%%
AE_NN_29_mod = load_model(final_mod_filename, custom_objects=custom_objects)
#%%
AE_NN_29_mod.summary()
#%%
thresh = 0.5
model = AE_NN_29_mod
decoded_test_data = model.predict(X_test)
decoded_train_data = model.predict(X_train)
#decoded_valid_data = model.predict(X_val)

test_loss = np.mean(binary_crossentropy(X_test, decoded_test_data))
train_loss = np.mean(binary_crossentropy(X_train, decoded_train_data))
#valid_loss = np.mean(binary_crossentropy(X_val, decoded_valid_data))

print("Reconstruction test Loss (bce-loss):", test_loss )
print("Reconstruction train Loss (bce-loss):", train_loss )
#print("Reconstruction valid Loss (bce-loss):", valid_loss )
#
binary_decoded_test_data = (decoded_test_data > thresh).astype(int)
binary_decoded_train_data = (decoded_train_data > thresh).astype(int)
#binary_decoded_valid_data = (decoded_valid_data > thresh).astype(int)

#%%
# test

accuracy = accuracy_score(X_test.flatten(), binary_decoded_test_data.flatten())
precision = precision_score(X_test.flatten(), binary_decoded_test_data.flatten())
recall = recall_score(X_test.flatten(), binary_decoded_test_data.flatten())
f1 = f1_score(X_test.flatten(), binary_decoded_test_data.flatten())
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1-score:", f1) # 79.8
#
accuracy = accuracy_score(X_train.flatten(), binary_decoded_train_data.flatten())
precision = precision_score(X_train.flatten(), binary_decoded_train_data.flatten())
recall = recall_score(X_train.flatten(), binary_decoded_train_data.flatten())
f1 = f1_score(X_train.flatten(), binary_decoded_train_data.flatten())
print("Train Accuracy:", accuracy)
print("Train Precision:", precision)
print("Train Recall:", recall)
print("Train F1-score:", f1) # 92
#
# accuracy = accuracy_score(X_val.flatten(), binary_decoded_valid_data.flatten())
# precision = precision_score(X_val.flatten(), binary_decoded_valid_data.flatten())
# recall = recall_score(X_val.flatten(), binary_decoded_valid_data.flatten())
# f1 = f1_score(X_val.flatten(), binary_decoded_valid_data.flatten())
# print("Val Accuracy:", accuracy)
# print("Val Precision:", precision)
# print("Val Recall:", recall)
# print("Val F1-score:", f1)
#%%
def threshold_opt(preds, truth):
    
    thresholds = np.linspace(0, 1, 21)
    best_threshold = []
    best_f1_score = 0
    
    for threshold in thresholds:

        binary_preds = (preds > threshold).astype(int)    
        f1 = f1_score(truth.flatten(), binary_preds.flatten())
    
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
threshold = threshold_opt(decoded_train_data, X_train)
#%%
#%%
threshold = threshold_opt(decoded_test_data, X_test)
#%%
#%%
thresh = 0.4
binary_decoded_test_data = (decoded_test_data > thresh).astype(int)
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
k_size = 3
#%%
def conv_encoder(X_tr, X_val, filters_vec,
                 dropout_rate_d = 0, dropout_rate_c = 0,
                 hidden_d_layers = 0, hidden_d = 400, l1_rate = 1e-4,
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
    
   # x = GlobalAveragePooling2D()(x)
    
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
                   filter_factor = 2):
    
    train_ready = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
    val_ready = val.reshape(val.shape[0], val.shape[1], val.shape[2], 1)
    
    start = start_size
    filters_vec = [int(start*filter_factor**x) for x in range(5)]
    
    mod_encoder = conv_encoder(train_ready, val_ready, **mod_pars_encoder, filters_vec=filters_vec)
    mod_decoder = conv_decoder(latent_dim = mod_encoder.output_shape[1:], filters_vec=filters_vec,
                               **mod_pars_decoder)

    autoencoder_input = Input(shape=(train.shape[1], train.shape[2], 1))
    latent_rep = mod_encoder(autoencoder_input)
    reconstructed_output = mod_decoder(latent_rep)
    
    autoencoder = Model(autoencoder_input, reconstructed_output)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
     
    return autoencoder

#%% 
param_grid_conv_encoder = {
    'hidden_d_layers': [0], # could try
    'dropout_rate_c': [0],  # dropout doesn't appear to be useful here
    'pooling': [True], # better to use stride
    'pooling_type': ['avg'],
    'l1_rate': [1e-4,1e-5]
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
    'start_size': [8] # more than 4
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
        
        for train_idx, val_idx in kf.split(X_train):
            early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
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
print(f'Best f1 score: {f1scores[best_index]}' ) # ~79.3
print(f'Accuracy of permatation selected: {accuracies[best_index]} ') # ~ 99.3
print(f'Precision of permatation selected: {precisions[best_index]} ') # ~77.6
print(f'Recall of permatation selected: {recalls[best_index]} ') # ~ 81.1

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
final_mod_filename = 'autoCONV_29_callback.keras'
#%%
checkpoint_filepath = final_mod_filename
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

#%%
model = c_auto_encoder(X_train, X_test, 
                       best_encoder_params, best_decoder_params,
                       best_params['start_size'],
                       best_params['filter_factor'],
                                            )
#%%
history = model.fit(x=X_train, y=X_train,
                        validation_data=(X_test, X_test), shuffle = True,
                    #class_weight=class_weights_dict,
                      epochs=80, batch_size=32,
                      verbose=1, callbacks=[early_stopping, checkpoint])
#%%
AE_CONV_29_mod = load_model(final_mod_filename) # , custom_objects=custom_objects)
#%%
AE_CONV_29_mod.summary()
#%%
model = AE_CONV_29_mod
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
#binary_decoded_valid_data = tf.cast(decoded_valid_data > thresh, tf.int32)

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
print("Test F1-score:", f1) # 82.1
#
accuracy = accuracy_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
precision = precision_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
recall = recall_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
f1 = f1_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
print("Train Accuracy:", accuracy)
print("Train Precision:", precision)
print("Train Recall:", recall)
print("Train F1-score:", f1) # 85.6
#
# accuracy = accuracy_score(X_val.flatten(), tf.reshape(binary_decoded_valid_data, [-1]) )
# precision = precision_score(X_val.flatten(), tf.reshape(binary_decoded_valid_data, [-1]) )
# recall = recall_score(X_val.flatten(), tf.reshape(binary_decoded_valid_data, [-1]) )
# f1 = f1_score(X_val.flatten(), tf.reshape(binary_decoded_valid_data, [-1]) )
# print("Val Accuracy:", accuracy)
# print("Val Precision:", precision)
# print("Val Recall:", recall)
# print("Val F1-score:", f1)
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
threshold = threshold_opt(decoded_train_data, X_train) # 85.8
#%%
threshold = threshold_opt(decoded_test_data, X_test) # 82.6
#%%
#%%
thresh = 0.4
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
# let's try a hybrid of the two models
#%%
def conv_2_encoder(X_tr, X_val, filters_vec,
                   dropout_rate_d = 0, dropout_rate_c = 0,
                   hidden_d_layers = 0, hidden_d = 400,  l1_rate = 1e5,
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
    
   # x = GlobalAveragePooling2D()(x)
    
    x = Flatten()(x)
    for _ in range(hidden_d_layers):
        x = Dense(hidden_d, activation=d_activation)(x)
        x = Dropout(dropout_rate_d)(x)
    
    x = Dense(latent_vars, activation=d_activation, activity_regularizer = l1(l1_rate) )(x)
   
    latent_space = x
    
    encoder_model = Model(inputs, latent_space)
    
    return encoder_model
#%%
#%%
def conv_2_decoder(latent_dim, latent_vars, dropout_rate_d, filters_vec, c_activation='relu',
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

def c2_auto_encoder(train, val, mod_pars_encoder, mod_pars_decoder, start_size,
                   filter_factor = 2):
    
    train_ready = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
    val_ready = val.reshape(val.shape[0], val.shape[1], val.shape[2], 1)
    
    start = start_size
    filters_vec = [int(start*filter_factor**x) for x in range(5)]
    
    mod_encoder = conv_2_encoder(train_ready, val_ready, **mod_pars_encoder, filters_vec=filters_vec)
    mod_decoder = conv_2_decoder(latent_dim = mod_encoder.output_shape[1:], filters_vec=filters_vec,
                               **mod_pars_decoder)

    autoencoder_input = Input(shape=(train.shape[1], train.shape[2], 1))
    latent_rep = mod_encoder(autoencoder_input)
    reconstructed_output = mod_decoder(latent_rep)
    
    autoencoder = Model(autoencoder_input, reconstructed_output)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
     
    return autoencoder

#%% 
param_grid_conv_encoder = {
    'hidden_d_layers': [0], # could try
    'dropout_rate_c': [0],  # dropout doesn't appear to be useful here
    'pooling': [True], # better to use stride
    'pooling_type': ['max','avg'],
    'hidden_d' : [400],
    'l1_rate' : [1e-5,1e-6]
}

param_grid_common = {
    'dropout_rate_d': [0,0.05],
    'latent_vars': [32], # yes
    'd_activation': ['elu'],
    'c_activation': ['elu'],
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
k_size = 3
#%%
f1scores, recalls, precisions, accuracies, BCEs = auto_encoder_CV(param_combinations = param_combinations_ModC,
                                                                  param_grid_encoder = encoder_dict,
                                                                  param_grid_decoder = decoder_dict,
                                                                  param_grid_oth = param_grid_other,
                                                                  function_name = c2_auto_encoder,
                                                                  thresh = 0.5, cv_splits = 3, seed = seed  )

#%%
#%%
best_index = np.argmax(f1scores)
best_index_alt = np.argmin(BCEs)
print(f'Best f1 score: {f1scores[best_index]}' ) # ~79.7
print(f'Accuracy of permatation selected: {accuracies[best_index]} ') # ~ 99.4
print(f'Precision of permatation selected: {precisions[best_index]} ') # ~79.4
print(f'Recall of permatation selected: {recalls[best_index]} ') # ~ 80

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
final_mod_filename = 'autoCONVDense_29_callback.keras'
#%%
checkpoint_filepath = final_mod_filename
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

#%%
model = c2_auto_encoder(X_train, X_test, 
                       best_encoder_params, best_decoder_params,
                       best_params['start_size'],
                       best_params['filter_factor'],
                                            )
#%%
history = model.fit(x=X_train, y=X_train,
                        validation_data=(X_test, X_test), shuffle = True,
                    #class_weight=class_weights_dict,
                      epochs=80, batch_size=32,
                      verbose=1, callbacks=[early_stopping, checkpoint])
#%%
AE_CONVDENSE_29_mod = load_model(final_mod_filename) #, custom_objects=custom_objects)
#%%
AE_CONVDENSE_29_mod.summary()
#%%
model = AE_CONVDENSE_29_mod
thresh = 0.5
decoded_test_data = tf.squeeze(model.predict(X_test), axis = -1)
decoded_train_data = tf.squeeze(model.predict(X_train), axis = -1)

test_loss = np.mean(binary_crossentropy(X_test, decoded_test_data))
train_loss = np.mean(binary_crossentropy(X_train, decoded_train_data))
print("Reconstruction test Loss (bce-loss):", test_loss )
print("Reconstruction train Loss (bce-loss):", train_loss )

binary_decoded_test_data = tf.cast(decoded_test_data > thresh, tf.int32)
binary_decoded_train_data = tf.cast(decoded_train_data > thresh, tf.int32)

#%%
# test
accuracy = accuracy_score(X_test.flatten(), tf.reshape(binary_decoded_test_data, [-1]) )
precision = precision_score(X_test.flatten(), tf.reshape(binary_decoded_test_data, [-1]) )
recall = recall_score(X_test.flatten(), tf.reshape(binary_decoded_test_data, [-1]) )
f1 = f1_score(X_test.flatten(), tf.reshape(binary_decoded_test_data, [-1]) )
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1-score:", f1) # 82.1
#
accuracy = accuracy_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
precision = precision_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
recall = recall_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
f1 = f1_score(X_train.flatten(), tf.reshape(binary_decoded_train_data, [-1]) )
print("Train Accuracy:", accuracy)
print("Train Precision:", precision)
print("Train Recall:", recall)
print("Train F1-score:", f1) # 86.8
#
#%%
#%%
threshold = threshold_opt(decoded_train_data, X_train) # 86.8
#%%
threshold = threshold_opt(decoded_test_data, X_test) #82.1
#%% # 82.4
#%%
thresh = 0.45
binary_decoded_test_data = tf.cast(decoded_test_data > thresh, tf.int32)   
#%%
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