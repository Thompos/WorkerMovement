# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:13:37 2024

@author: 
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math as ma
import itertools
from keras.models import Model, Sequential, load_model
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import joblib
import pickle
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
final_mod_filename = 'autoCONV_AUG_29_callback.keras'
#%%
AE_29_mod = load_model(final_mod_filename) #, custom_objects=custom_objects)
#%%
AE_29_mod.summary()
#%%
print(len(AE_29_mod.layers) )
encoder_sub_mod = AE_29_mod.get_layer('functional_313')
decoder_sub_mod = AE_29_mod.get_layer('functional_315')

encoder_sub_mod.summary()
decoder_sub_mod.summary()

#%%
#%%
latent_representations = encoder_sub_mod.predict(X_train)

#%%
NNlatent_df = pd.DataFrame(latent_representations)
#%%
thresh_high_corr = 0.5
corrs = []
high_corrs = []
rows = 5
cols = 7
for i in range(NNlatent_df.shape[1]):

    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    
    for j in range(NNlatent_df.shape[1]):
        
        row = (j+1)//cols
        col = (j+1) - (row)*cols
        
        axes[row,col].scatter(NNlatent_df.iloc[:, i], NNlatent_df.iloc[:, j] )
        axes[row,col].set_title(f'{i} Vs {j}')
        
        correlation_AB = NNlatent_df.iloc[:, i].corr(NNlatent_df.iloc[:, j])
      #  print(f"Correlation between {i} and {j}: {correlation_AB}")
        corrs.append(correlation_AB)
        if abs(correlation_AB) > thresh_high_corr:
            high_corrs.append([i,j])
        
    plt.show()

#%%
total = (31*32)/2
print(f'proportion of features with correlation > 0.9 {(((abs(np.array(corrs)) > 0.9).sum() - 32)/2)/total}')
print(f'proportion of features with correlation > 0.8 {(((abs(np.array(corrs)) > 0.8).sum() - 32)/2)/total}')
print(f'proportion of features with correlation > 0.7 {(((abs(np.array(corrs)) > 0.7).sum() - 32)/2)/total}')
print(f'proportion of features with correlation > 0.6 {(((abs(np.array(corrs)) > 0.6).sum() - 32)/2)/total}')

print(f'proportion of features with correlation > 0.5 {(((abs(np.array(corrs)) > 0.5).sum() - 32)/2)/total}')
# 31/25, 31/9, 31/21, 
# could the magnitude be indicative of the importance?
print(f'proportion of features with correlation < 0.2 {(((abs(np.array(corrs)) < 0.2).sum() - 32)/2)/total}' )
print(f'proportion of features with correlation < 0.3 {(((abs(np.array(corrs)) < 0.3).sum() - 32)/2)/total}' )
print(f'proportion of features with correlation < 0.4 {(((abs(np.array(corrs)) < 0.4).sum() - 32)/2)/total}' )
print(f'proportion of features with correlation < 0.5 {(((abs(np.array(corrs)) < 0.5).sum() - 32)/2)/total}' )

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
        
        fig, axes = plt.subplots(rows, cols, figsize=(11, 8))
        
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
HC_thresh, clusters, sil_scores = silloutette_score_fun(hierarchical_link_mat, X_scaled, weighted=True, singlet_val=0,
                                                        max_score=50)
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
plt.figure(figsize=(10, 7))

sns.heatmap(corr_matrix, annot=False, cmap='plasma', vmin=-1, vmax=1)

plt.title('Correlation map of learned features')

plt.show()
#%%
#%%