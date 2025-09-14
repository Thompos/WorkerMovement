# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:44:07 2024

@author: Mr T
"""

#%%
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
#%%
#%%
dwell_clusterids_df = pd.read_pickle('dwell_df_clean.pkl')
dwell_cluster_df = pd.read_pickle('dwell_cluster_df.pkl')

path_clusters_df = pd.read_pickle('path_clusters_df.pkl')

import pickle
with open('UWB_datasets_14_clust_checking.pkl', 'rb') as f:
    datasets = pickle.load(f)

#%%
#%%
ma1 = '1S'
#%%
for i in range(1,7):
    
    rig_data = datasets[f'rig_data{i}']

    rig_data['angle_diff_mab1'] = rig_data['angle_dif'].rolling(ma1, min_periods=1).mean().values
    rig_data['angle_diff_ma1'] = rig_data['angle_dif'].rolling(ma1, min_periods=1, center = True ).mean().values   
    rig_data['angle_diff_maf1'] = rig_data['angle_dif'][::-1].rolling(ma1, min_periods=1).mean()[::-1].values
    
#%%
#%%
Tri_dat = pd.concat( datasets.values() )              
#%%
xmin = Tri_dat['X'].min()
xmax = Tri_dat['X'].max()
ymin = Tri_dat['Y'].min()
ymax = Tri_dat['Y'].max()
#%%
#%%
Tri_move_dat = Tri_dat[Tri_dat['pred_lstmAtt_clean'] == 1].copy()
#%%
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 5))

for i, ax in enumerate(axes.flatten(), start=1):
    condition1 = (Tri_dat.rig == i) & (Tri_dat['pred_lstmAtt_clean'] > 0.5)
    hb = ax.scatter(Tri_dat.loc[condition1, 'X'], 
                   Tri_dat.loc[condition1, 'Y'], alpha = 0.02)
    ax.set_title(f'worker {i} dwelling locoations')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)

plt.tight_layout()
plt.show()
#%%
#%%
selected_columns = ['X', 'Y'] #, 'angle_diff_ma1', 'angle_diff_maf1']

select_dat = Tri_move_dat[selected_columns].reset_index(drop = True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(select_dat)
#%%

inertia = []
silhouette_scores = []

k_range = range(5, 49)

for k in k_range:

    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    
    inertia.append(-gmm.score(X_scaled))
    silhouette_scores.append(silhouette_score(X_scaled, gmm.predict(X_scaled)))

# Plotting elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Plotting silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.show()
#%%

n_clusters = 19
gmm = GaussianMixture(n_components=n_clusters, random_state=69)
gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)

# Plotting GMM clusters
plt.figure(figsize=(8, 6))
plt.scatter(select_dat['X'], select_dat['Y'], c=labels, s=50, cmap='tab20', edgecolors='k')
plt.title('Scatter Plot with GMM Clusters')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

#%%
#%%
selected_columns = ['X', 'Y', 'path_time_ow', 'path_count']

select_dat = Tri_move_dat[selected_columns].reset_index(drop = True)
select_dat.dropna(inplace = True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(select_dat)

#%%
# Apply DBSCAN
eps_vals = [0.1, 0.4, 0.7, 0.9] 
min_samples = 30

for eps in eps_vals:
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    plt.figure(figsize=(8.5, 4.5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, edgecolors='k')
    plt.title('DBSCAN Clustering')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()
    
    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')


#%%
#%%
#%%
#%%
################################################################################
################## Trying with path cluster DF #################################
################################################################################
# functions 
def print_all_clusters_fun(agg_dat, master_dat, cluster_assignments):
    
    fig, axes = plt.subplots(figsize=(10, 6))
    
    xmin = master_dat['X'].min() - 0.5
    xmax = master_dat['X'].max() + 0.5
    ymin = master_dat['Y'].min() - 0.5
    ymax = master_dat['Y'].max() + 0.5
    
    agg_dat['path_cluster'] = cluster_assignments
    master_dat['path_cluser_temp'] = master_dat['path_count'].map(agg_dat.set_index('path_id')['path_cluster'])
    unique_clusters = np.unique(agg_dat['path_cluster'])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters))) 
    color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}
    cols = [color_map[value] for value in master_dat['path_cluser_temp'] ]

    plt.scatter(master_dat['X'], master_dat['Y'], alpha=0.7, c=cols, cmap=plt.cm.tab20)

    for cluster, color in color_map.items():
        plt.scatter([], [], label=f'Cluster {cluster}', color=color)  

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.legend()
    plt.show()

#%%
#%%
def print_indi_clusters_fun(agg_dat, master_dat, columns, cluster_assignments):
    #
    path_counts = plot_dat['path_count'].unique()
    
    xmin = master_dat['X'].min() - 0.5
    xmax = master_dat['X'].max() + 0.5
    ymin = master_dat['Y'].min() - 0.5
    ymax = master_dat['Y'].max() + 0.5

    # Create a figure and a grid of subplots
    num_plots = len(path_counts)
    colms = columns  # Number of columns in the grid
    agg_dat['path_cluster'] = cluster_assignments
    master_dat['path_cluser_temp'] = master_dat['path_count'].map(agg_dat.set_index('path_id')['path_cluster'])
    unique_clusters = np.unique(agg_dat['path_cluster'])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}
    rows = (num_plots + colms - 1) // colms  # Calculate number of rows needed
    fig, axs = plt.subplots(rows, colms, figsize=(13, 2 * rows))

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    for i, path_count in enumerate(path_counts):
        row = i // colms
        colm = i % colms
        if len(path_counts) > colms:
            ax = axs[row, colm]
        else:
            ax = axs[colm]
        # Filter data for the current path count
        subset = master_dat[master_dat['path_count'] == path_count]
        
        cols = [color_map[value] for value in subset['path_cluser_temp'] ]
        # Plot the subset
        ax.scatter(subset['X'], subset['Y'], alpha=1, c=cols, cmap=plt.cm.tab20)
        ax.set_title(f'Path Count: {path_count}')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


    # Hide unused subplots
    for i in range(num_plots, rows * colms):
        axs.flatten()[i].axis('off')

    # Show the plot
    plt.show()

#%%
def silloutette_score_fun(link_mat, scaled_dat):
    
    silhouette_scores_check = []
    best_threshold = 1
    best_clusters = 1*len(scaled_dat)
    max_silhouette_score = float('-inf')

    thresholds = np.linspace(0, 11, 20)  # Using linkage distances as thresholds
    for threshold in thresholds:
        clusters = fcluster(link_mat, threshold, criterion='distance')
        no_clusters = len(np.unique(clusters))
        if (no_clusters > 1) & (no_clusters < len(scaled_dat)) :  # Check if multiple clusters are formed
       
            silhouette_samples_values = silhouette_samples(scaled_dat, clusters)
            unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        
            valid_indices = np.where(np.isin(clusters, unique_clusters[cluster_counts > 1]))[0]
            valid_silhouette_scores = silhouette_samples_values[valid_indices]
            silhouette_score_value = valid_silhouette_scores.mean()
            silhouette_scores_check.append(silhouette_score_value)
            if round(silhouette_score_value,4) > round(max_silhouette_score,4):
                max_silhouette_score = silhouette_score_value
                best_threshold = threshold
                best_clusters = len(np.unique(clusters))
       # elif silhouette_score_value == max_silhouette_score and len(np.unique(clusters)) < best_clusters:
        #    best_threshold = threshold
         #   best_clusters = len(np.unique(clusters))
        else:
             silhouette_scores_check.append(0)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, silhouette_scores_check, marker='o')
    plt.title('Silhouette Scores for Different Thresholds')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Silhouette Score')
    plt.show()

    print("Best threshold:", best_threshold)
    print("Number of clusters for the best threshold:", best_clusters)
    print("Highest silhouette score:", max_silhouette_score)
    
    return best_threshold, best_clusters, silhouette_scores_check
    
#%% # here ###################################################################
#%% ##########################################################################
##############################################################################
##############################################################################
path_clust_ID = 9

test_dat = path_clusters_df[ path_clusters_df['perm_clust_id'] == path_clust_ID ].copy()

selected_columns = [
        'min_X', 'max_X', 'min_Y', 'max_Y', 'pathway_len'
         ]

for i in percentiles:
  #  selected_columns.append(f'Xdifpc_{i}')
   # selected_columns.append(f'Ydifpc_{i}')
    
    selected_columns.append(f'Xpc_{i}')
    selected_columns.append(f'Ypc_{i}')
    
  #  selected_columns.append(f'dist2start_pc_{i}')
    
  #  selected_columns.append(f'path_dist_pc_{i}')
    selected_columns.append(f'angle_pc_{i}')
    selected_columns.append(f'sin_pc_{i}')
    selected_columns.append(f'cos_pc_{i}')

test_dat_select = test_dat[selected_columns]
#%%
fig, axes = plt.subplots(figsize=(10, 6))

plot_dat = Tri_dat[Tri_dat['perm_clust_ow_id']  == path_clust_ID].copy()

condition1 = ( (path_clusters_df['perm_clust_id'] == path_clust_ID)  ) 

start = path_clusters_df.loc[path_clusters_df['perm_clust_id'] == path_clust_ID, 'start_clust'].iloc[0]
end = path_clusters_df.loc[path_clusters_df['perm_clust_id'] == path_clust_ID, 'end_clust'].iloc[0]

plt.scatter(plot_dat['X'], plot_dat['Y'], label=f'path cluster', alpha=0.4)
plt.scatter(dwell_cluster_df.loc[dwell_cluster_df['cluster_id'] ==  start, 'Xu_wt'],
            dwell_cluster_df.loc[dwell_cluster_df['cluster_id'] == start, 'Yu_wt'],
            label=f'path cluster', alpha=1, c = 'red')
plt.scatter(dwell_cluster_df.loc[dwell_cluster_df['cluster_id'] == end, 'Xu_wt'],
            dwell_cluster_df.loc[dwell_cluster_df['cluster_id'] == end, 'Yu_wt'],
            label=f'path cluster', alpha=1, c = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(ymin,ymax)
plt.xlim(xmin,xmax)
plt.grid(True)
plt.show()

#%%
path_cluster_prep_dat = path_clusters_df[selected_columns].copy()
scaler = StandardScaler()
All_scaled = scaler.fit_transform(path_cluster_prep_dat)

#%%
X_scaled = All_scaled[path_clusters_df['perm_clust_id'] == path_clust_ID ]
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(test_dat_select)

#%%
k_values = range(2, len(X_scaled))

wcss = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=69)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_values, wcss, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()
#%%
#%%
if len(X_scaled) > 1:
    similarity_matrix = pairwise_distances(X_scaled, metric='euclidean')  # 'sqeuclidean', 'seuclidean',

    link_method = 'average'
    hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

    plt.figure(figsize=(10, 5))
    plt.title('Dendrogram for Hierarchical Clustering')
    dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
#%%

hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat, X_scaled)

#%%
if len(X_scaled) > 1:
    hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')
else:
    hier_cluster_assignments = 1
#%%
print_all_clusters_fun(test_dat, plot_dat, hier_cluster_assignments) 
#%%
print_indi_clusters_fun(test_dat, plot_dat, 5, hier_cluster_assignments)
   
#%%
#%%
#%%
#%%
# %%
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_scaled)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Number of Components')
plt.grid(True)
plt.show()
#%%
n_components = np.where( pca.explained_variance_ratio_ < 0.025)[0][0] + 1
pca_final = PCA(n_components=n_components)
pca_final.fit(X_scaled)

# Transform the original data into the principal components
df_pca = pca_final.transform(X_scaled)


pca_column_names = [f'PC{i}' for i in range(1, n_components + 1)]

col_names = pca_column_names

# Convert the PCA-transformed data into a DataFrame for further analysis or visualization
df_pca = pd.DataFrame(df_pca, columns=col_names)


principal_components  = pca_final.components_
components_df = pd.DataFrame(principal_components)
# %%

# %%
pca_similarity_matrix = pairwise_distances(df_pca, metric='euclidean')  # 'sqeuclidean', 'seuclidean',

pca_linkage_matrix = linkage(pca_similarity_matrix, method='average')

plt.figure(figsize=(10, 5))
plt.title('Dendrogram for Hierarchical Clustering')
dendrogram(linkage(pca_similarity_matrix, method='average'), truncate_mode='level', p=0)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
#%% # here
pca_hier_best_thresh, pca_hier_best_clusters, pca_hier_silhouette_scores = silloutette_score_fun(pca_linkage_matrix, df_pca)

#%%
pca_cluster_assignments = fcluster(pca_linkage_matrix, pca_hier_best_thresh, criterion='distance')
#%%
print_all_clusters_fun(test_dat, plot_dat, pca_cluster_assignments) 
#%%
print_indi_clusters_fun(test_dat, plot_dat, 5, pca_cluster_assignments)
#%%

#%%
#%%############################################################################
#%%
#%%
for i in range(1,7):    
    datasets[f'rig_data{i}'] = Tri_dat.loc[Tri_dat['rig'] == i]
    
#%%
with open('UWB_datasets_13_path_clustered.pkl', 'wb') as f:
    pickle.dump(datasets, f)

#%%
path_clusters_df.to_pickle('path_clusters_df.pkl')
#%%
#%%
#%%