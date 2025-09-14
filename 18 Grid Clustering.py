# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:14:56 2024

@author: 
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_samples

#%%
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
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
#%%
Tri_dat = pd.concat( datasets.values() )
#%%
import pickle
with open('dwell_indi_grids.pkl', 'rb') as f:
    dwell_indi_grids = pickle.load(f)
    
with open('transit_indi_grids.pkl', 'rb') as f:
    transit_indi_grids = pickle.load(f)
    
with open('transit_clust_grids_padded.pkl', 'rb') as f:
    transit_clust_grids_padded = pickle.load(f)
    
with open('dwell_indi_grids.pkl', 'rb') as f:
    dwell_indi_grids = pickle.load(f)
    
with open('path_prop_similarity_mat.pkl', 'rb') as f:
    path_prop_similarity_mat = pickle.load(f)
    
with open('transit_indi_grids_padded.pkl', 'rb') as f:
    transit_indi_grids_padded = pickle.load(f)
    
with open('path_propunion_similarity_mat.pkl', 'rb') as f:
    path_propunion_similarity_mat = pickle.load(f)
    
with open('intersection_mat.pkl', 'rb') as f:
    intersection_mat = pickle.load(f)

#%%
#%%
dwell_dat = Tri_dat[Tri_dat['cleaned_pred'] < 0.5].copy()
transit_dat = Tri_dat[Tri_dat['cleaned_pred'] >= 0.5].copy()
#%%
#%%
#%%
#%%
#%%
pathIDs_trim = np.sort([int(x) for x in path_clusters_df['path_id'] ]  )
pathIDs = np.sort([int(x) for x in path_propunion_similarity_mat.columns ]  )

wanted_bools = [True if x in pathIDs_trim else False for x in pathIDs ]
           
#%%
sorted_path_clusters_df = path_clusters_df.sort_values(by='path_id')
#%%
#%%
trim_path_propunion_similarity_mat = path_propunion_similarity_mat.loc[wanted_bools, wanted_bools]
#%%
#%%
#%%
link_method = 'average'
#%%
hierarchical_link_mat = linkage(trim_path_propunion_similarity_mat, method=link_method)
plt.figure(figsize=(11.5, 6))
plt.title('Dendrogram for Hierarchical Clustering')
dendrogram(linkage(trim_path_propunion_similarity_mat, method=link_method), truncate_mode='level', p=0)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

#%%
hier_cluster_assignments = fcluster(hierarchical_link_mat, 1.15, criterion='distance')
sorted_path_clusters_df['grid_cluster'] = hier_cluster_assignments
transit_dat['grid_cluster_temp'] = transit_dat['path_count'].map(sorted_path_clusters_df.set_index('path_id')['grid_cluster'])
#%%
print(f'new clustering n of cluster: {len(np.unique(hier_cluster_assignments))}' )
print(f"new clustering n of cluster: {len(sorted_path_clusters_df['path_cluster'].unique())}" )
#%%
path_cluster_IDs = np.unique(hier_cluster_assignments) 
#%%

for path_clust in path_cluster_IDs:
    
    path_cluster_dat = sorted_path_clusters_df[sorted_path_clusters_df['grid_cluster'] ==  path_clust].copy()
    master_dat = transit_dat[transit_dat['grid_cluster_temp'] == path_clust].copy()
    
    xmin = master_dat['X'].min() - 1
    xmax = master_dat['X'].max() + 1
    ymin = master_dat['Y'].min() - 1
    ymax = master_dat['Y'].max() + 1    
    
    paths_included = np.sort(path_cluster_dat['path_id'].unique())
    
    num_plots = len(paths_included)
    num_cols = 5
    num_rows = (num_plots + num_cols - 1) // num_cols  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 2* num_rows))
    axes = axes.flatten()
    
    for i, path_id in enumerate(paths_included):
        ax = axes[i]
        temp_dat = master_dat[master_dat['path_count'] == path_id].copy()
        
        ax.scatter(temp_dat['X'], temp_dat['Y'], alpha=0.7, c='blue' )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if i == 0:
            ax.set_title(f'Path ID {path_id}, cluster {path_clust}')
        else:
            ax.set_title(f'Path ID {path_id}')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

    for j in range(len(paths_included), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()    

#%%
#%%
path_dwell_perms = np.sort(transit_dat['perm_clust_ow_id'].unique() )
path_dwell_perms = [x for x in path_dwell_perms if not np.isnan(x) ]

for perm in path_dwell_perms:
    
    master_dat = transit_dat[transit_dat['perm_clust_ow_id'] == perm].copy()
    
    unique_clusters = np.sort(master_dat['grid_cluster_temp'].unique() )
    
    xmin = master_dat['X'].min() - 1
    xmax = master_dat['X'].max() + 1
    ymin = master_dat['Y'].min() - 1
    ymax = master_dat['Y'].max() + 1    
    
    paths_included = np.sort(master_dat['path_count'].unique())
    
    num_plots = len(paths_included)
    num_cols = 5
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 2* num_rows))
    axes = axes.flatten()
    
    for i, path_id in enumerate(paths_included):
        ax = axes[i]
        
        temp_dat = master_dat[master_dat['path_count'] == path_id].copy()
        
        cols = [color_map[value] for value in temp_dat['grid_cluster_temp'] ]
                
        ax.scatter(temp_dat['X'], temp_dat['Y'], alpha=0.7, c=cols, cmap=plt.cm.tab20 )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if i == 0:
            ax.set_title(f'Path ID {path_id}, perm {perm}')
        else:
            ax.set_title(f'Path ID {path_id}')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

    for j in range(len(paths_included), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()    
   
#%%
#%%
Tri_dat['grid_path_cluster'] = Tri_dat['path_count'].map(sorted_path_clusters_df.set_index('path_id')['grid_cluster'])

for i in range(1,7):    
    datasets[f'rig_data{i}'] = Tri_dat.loc[Tri_dat['rig'] == i].copy()

#%%
with open('trim_path_propunion_similarity_mat.pkl', 'wb') as f:
    pickle.dump(trim_path_propunion_similarity_mat, f)
    
with open('path_clusters_df_18.pkl', 'wb') as f:
    pickle.dump(sorted_path_clusters_df, f)

with open('path_clusters_df_18.pkl', 'wb') as f:
    pickle.dump(sorted_path_clusters_df, f)

with open('UWB_datasets_18_gridpath_clustered.pkl', 'wb') as f:
    pickle.dump(datasets, f) 
#%%
#%%


