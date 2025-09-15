# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:13:37 2024

@author: 
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import itertools
from keras.models import Model, Sequential, load_model
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import joblib
import pickle
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_samples
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Concatenate,Dense,Flatten,Reshape,Dropout,AveragePooling2D,GlobalAveragePooling2D

seed = 69
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)

#%%
import pickle
with open('AE_X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
#%%
spacial = True
if spacial == True:
    final_mod_filename = 'VAE_AUG_spacial_32b_callback.keras'
else:
    final_mod_filename = 'VAE_AUG_32b_callback.keras'
#%%
VAE_32_mod = load_model(final_mod_filename) #, custom_objects=custom_objects)
#%%
VAE_32_mod.summary()
#%%
encoder_vae = VAE_32_mod.get_layer('encoder')
decoder_vae = VAE_32_mod.get_layer('decoder')

encoder_vae.summary()
decoder_vae.summary()

#%% 
def VAE_inference(input_data, encoder_model, decoder_model, predict = True): 
    
    z_mean, z_log_var, z_samples = encoder_model.predict(input_data)
    
    latent_rep = z_mean if predict else z_samples  
    reconstructed_output = decoder_vae(latent_rep) 
         
    return reconstructed_output  

#%%
z_mean, z_log_var, z_samples = encoder_vae.predict(X_train)
#%%
#%%
#%%
NNlatent_df = pd.DataFrame(z_mean)
#%%
#%%
import pickle
with open('datasets_28.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
#%%
Tri_dat = pd.concat( datasets.values() )
#%%
min_x = Tri_dat['X'].min()
max_x = Tri_dat['X'].max()
min_y = Tri_dat['Y'].min()
max_y = Tri_dat['Y'].max()

#%%        
#%%
#%%
#%% ##########################################################################
############################################################################## 
min_x = ma.floor(min(Tri_dat['X']) )
max_x = ma.ceil(max(Tri_dat['X']) )
min_y = ma.floor(min(Tri_dat['Y']) )
max_y = ma.ceil(max(Tri_dat['Y']) )

grid_spacing = 0.25  # Grid spacing
#%%
#%%
ncols = int((max_x-min_x)/grid_spacing)
nrows = int((max_y-min_y)/grid_spacing)

x_vals = np.around(np.arange(min_x, max_x, grid_spacing), 2)
y_vals = np.around(np.flip(np.arange(min_y, max_y, grid_spacing)), 2)

x_labels = [str(int(x)) if x%1 == 0 else '' for x in x_vals]
y_labels = [str(int(y)) if y%1 == 0 else '' for y in y_vals]
#%%
def grid_ref_fun(x, y):
  xg = ma.floor((x - min_x)/grid_spacing )
  yg = ma.ceil((max_y - y)/grid_spacing ) - 1
  return xg, yg

#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(NNlatent_df)
#%%
link_method = 'average'
#%%
similarity_matrix = pairwise_distances(X_scaled, metric='euclidean')  # 'sqeuclidean', 'seuclidean',

hierarchical_link_mat = linkage(similarity_matrix, method=link_method)
plt.figure(figsize=(11.5, 6))
plt.title('Dendrogram for Hierarchical Clustering')
dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=50)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

#%%
HC_thresh = 19
hier_cluster_assignments = fcluster(hierarchical_link_mat, HC_thresh, criterion='distance')
clusters = np.unique(hier_cluster_assignments)
print( len(clusters ) )
#%%
#%%
#%%
def plt_clusters(data_array, assignments, clusters):
    
    for clust in clusters:
        
        mask_cluster = (assignments == clust)
        X_cluster = data_array[mask_cluster]
        
        rows = max(1 + len(X_cluster)//4,2)
        cols = 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 7))
        
        row = 0
        col = 0
        
        for path in X_cluster:
         
            axes[row,col].imshow(path, cmap='Blues', aspect='auto', interpolation='nearest',
                    extent=[min_x, max_x, min_y, max_y], alpha = 1, 
                    vmin=0.0, vmax=1)
            
          #  axes[row,col].grid(which='both', color='black', linestyle='-', linewidth=0.2)       
        #    axes[row,col].set_xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
        #    axes[row,col].set_yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
           
            if col == 0:
                axes[row,col].set_ylabel(' (Y)')
            if rows == 0:
                axes[row,col].xlabel(' (X)')
           
            row = row if ((col+1) < cols) else row+1
            col = col+1 if ((col+1) < cols) else 0
            
        plt.show()
#%%
plt_clusters(X_train, hier_cluster_assignments, clusters)
#%%
#%%
def silloutette_score_fun(link_mat, scaled_dat, weighted = True, singlet_val = 0,
                          max_score = 30, start = 0, plots = True):
    
    silhouette_scores_check = []
    n_clusters = []
    best_threshold = start
    best_clusters = len(scaled_dat)
    max_silhouette_score = float('-inf')

    thresholds = np.linspace(start,max_score,20)
    for threshold in thresholds:
        clusters = fcluster(link_mat, threshold, criterion='distance')
        no_clusters = len(np.unique(clusters))
        if (no_clusters > 1) & (no_clusters < len(scaled_dat)) :  # Check if multiple clusters are formed
       
            silhouette_samples_values = silhouette_samples(scaled_dat, clusters)
            unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
            
            if weighted == True:
                clustered_indices = np.where(np.isin(clusters, unique_clusters[cluster_counts > 1]))[0]
                clustered_silhouette_scores = silhouette_samples_values[clustered_indices]
                clustered_score_value = clustered_silhouette_scores.mean()
                
                clustered_wt = len(clustered_silhouette_scores)/(len(clustered_silhouette_scores)+len(silhouette_samples_values))
                all_wt = 1-clustered_wt
                silhouette_score_value = clustered_wt*clustered_score_value + all_wt*silhouette_samples_values.mean() 
            else:
                if singlet_val == 1:
                    clustered_indices = np.where(np.isin(clusters, unique_clusters[cluster_counts > 1]))[0]
                    clustered_silhouette_scores = silhouette_samples_values[clustered_indices]
                    silhouette_score_value = clustered_silhouette_scores.mean()    
                else:
                    silhouette_samples_values[silhouette_samples_values == 0] = singlet_val
                    silhouette_score_value = silhouette_samples_values.mean() 
            
            silhouette_scores_check.append(silhouette_score_value)
            n_clusters.append(no_clusters)
            if round(silhouette_score_value,4) >= round(max_silhouette_score,4):
                max_silhouette_score = silhouette_score_value
                best_threshold = threshold
                best_clusters = len(np.unique(clusters))
        else:
             silhouette_scores_check.append(0)
    
    if plots == True:       
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        ax1.plot(thresholds, silhouette_scores_check, 'bo-', label='Silhouette Score')
        ax1.set_xlabel('Distance Threshold')
        ax1.set_ylabel('Silhouette Score', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(thresholds, n_clusters, 'ro-', label='Number of Clusters')
        ax2.set_ylabel('Number of Clusters', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('Silhouette Scores for Different Thresholds')
        
        plt.show()

    print("Best threshold:", best_threshold)
    print("Number of clusters for the best threshold:", best_clusters)
    print("Highest silhouette score:", max_silhouette_score)
    
    return best_threshold, best_clusters, silhouette_scores_check
    
#%% ####################################################################
HC_thresh, clusters, sil_scores = silloutette_score_fun(hierarchical_link_mat, X_scaled,
                                                        weighted=True, singlet_val=0,
                                                        max_score=40)
#%%
#%%
hier_cluster_assignments = fcluster(hierarchical_link_mat, HC_thresh, criterion='distance')
clusters = np.unique(hier_cluster_assignments)
print( len(clusters ) )
#%%
plt_clusters(X_train, hier_cluster_assignments, clusters)
#%%
#%%
X_scaled_df = pd.DataFrame(X_scaled)
corr_matrix = X_scaled_df.corr()
#%%
# Set up the matplotlib figure
#plt.figure(figsize=(9, 6))

#sns.heatmap(corr_matrix, annot=False, cmap='plasma', vmin=-1, vmax=1)

#plt.title('Correlation map of learned features')

#plt.show()
#%%
#plt.hist(node_samples)
#plt.show()
features = 40
#%%
case_no = 10

for node in range(features): # 40
    
    mean = z_mean[case_no][node]
#    sd = z_log_var[case_no][node] #
    sd = ( np.exp(0.5*z_log_var[case_no][node] ) )  # **0.5
    sd_lim = 100
    seq_len = 16
    sequence = np.linspace(mean - sd*sd_lim, mean + sd*sd_lim, num=seq_len,
                           dtype=np.float32)
        
    explore_df = pd.DataFrame([z_mean[case_no]]*seq_len)    
    explore_df.iloc[:, node] = sequence
    
    explore_array = explore_df.to_numpy()
    
    reconstructions = decoder_vae.predict(explore_array)
    
    thresh = 0.45
    binary_decoded_reconstructions = tf.cast(reconstructions > thresh, tf.int32)
  
    fig, axes = plt.subplots(4, 4, figsize=(10, 6.5))

    axes = axes.flatten()
    
    x_vals = np.around(np.arange(min_x, max_x, grid_spacing), 2)
    y_vals = np.around(np.flip(np.arange(min_y, max_y, grid_spacing)), 2)
    
    x_labels = [str(int(x)) if x % 1 == 0 else '' for x in x_vals]
    y_labels = [str(int(y)) if y % 1 == 0 else '' for y in y_vals]
     
    for i in range(len(reconstructions)):
        ax = axes[i]
      
        actual_patch = mpatches.Patch(color='blue', alpha=0.4, label='Actual')
        predicted_patch = mpatches.Patch(color='red', alpha=0.4, label='Predicted')
    
        ax.imshow(X_train[case_no], cmap='Blues', aspect='auto', interpolation='nearest',
                   extent=[min_x, max_x, min_y, max_y], alpha=1, vmin=0.0, vmax=1)
          
        ax.imshow(binary_decoded_reconstructions[i], cmap='Reds', aspect='auto', interpolation='nearest',
                  extent=[min_x, max_x, min_y, max_y], alpha=0.6, vmin=0.0, vmax=1)
        
        ax.grid(which='both', color='black', linestyle='-', linewidth=0.4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        if i == 0 or i == seq_len-1:
            ax.set_title(f"{sd_lim} sd's from mean", fontsize = 9)
        else:
            ax.set_title(f"Node {node}", fontsize = 9)

    plt.show()
 
#%%
#%%
case_no = 11
compare_means = []

for node in range(features):
    
    mean = z_mean[case_no][node]
#    sd = z_log_var[case_no][node] #
    sd = ( np.exp(0.5*z_log_var[case_no][node] ) )
    compare_means.append(mean)
     
print(compare_means)
#%%

for node in range(features):
    compare_means = []
    compare_sds = []
    for case_no in range(len(X_train)):
    
        mean = z_mean[case_no][node]
        #    sd = z_log_var[case_no][node] #
        sd = ( np.exp(0.5*z_log_var[case_no][node] ) )
        compare_means.append(mean)
        compare_sds.append(sd)
    compare_means = np.array(compare_means)
    compare_sds = np.array(compare_sds)
    
    minx = np.array(compare_means.min(), compare_sds.min() ).min()
    maxx = np.array(compare_means.max(), compare_sds.max() ).max()
    plt.hist(compare_means, range=[minx, maxx])
    plt.hist(compare_sds, color = 'orange')
    plt.vlines(compare_means.mean(), 0, 200, color= 'r' )
    plt.vlines(sd, 0, 400, color  = 'Orange')
    plt.title(f'Node {node}')
    plt.show()
#print(compare_means)

# NOTE THAT THE SD'S REFER TO THE PREDICTION SD, RATHER THAN THE SD ASSOCIATED 
# WITH THE MEANS GOVEN TO ALL THE MEANS IN THE DATASET

#%%
node_mins = []
node_maxs = []

for node in range(40):
    compare_means = []

    for case_no in range(len(X_train)):
    
        mean = z_mean[case_no][node]
        compare_means.append(mean)
    
    node_mins.append(min(compare_means) )
    node_maxs.append(max(compare_means) )
    
    
#%%
case_no = 10

for node in range(40):
    
    mean = z_mean[case_no][node]
    node_min = node_mins[node]
    node_max = node_maxs[node]
    seq_len = 16
    sequence = np.linspace(node_min, node_max, num=seq_len,
                           dtype=np.float32)
        
    explore_df = pd.DataFrame([z_mean[case_no]]*seq_len)    
    explore_df.iloc[:, node] = sequence
    
    explore_array = explore_df.to_numpy()
    
    reconstructions = decoder_vae.predict(explore_array)
    
    thresh = 0.45
    binary_decoded_reconstructions = tf.cast(reconstructions > thresh, tf.int32)
  
    fig, axes = plt.subplots(4, 4, figsize=(10, 6.5))

    axes = axes.flatten()
        
    for i in range(len(reconstructions)):
        ax = axes[i]
      
        actual_patch = mpatches.Patch(color='blue', alpha=0.4, label='Actual')
        predicted_patch = mpatches.Patch(color='red', alpha=0.4, label='Predicted')
    
        ax.imshow(X_train[case_no], cmap='Blues', aspect='auto', interpolation='nearest',
                   extent=[min_x, max_x, min_y, max_y], alpha=1, vmin=0.0, vmax=1)
          
        ax.imshow(binary_decoded_reconstructions[i], cmap='Reds', aspect='auto', interpolation='nearest',
                  extent=[min_x, max_x, min_y, max_y], alpha=0.6, vmin=0.0, vmax=1)
        
        ax.grid(which='both', color='black', linestyle='-', linewidth=0.4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        if i == 0 or i == seq_len-1:
            ax.set_title(f"{sd_lim} sd's from mean", fontsize = 9)
        else:
            ax.set_title(f"Node {node}", fontsize = 9)

    plt.show()
 
#%% 

#%%
#%%
#%%
#%%
