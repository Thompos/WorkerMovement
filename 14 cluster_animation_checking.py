# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:20:16 2024

"""
#%%
#%%
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
from matplotlib.patches import Ellipse

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
with open('UWB_datasets_13_path_clustered.pkl', 'rb') as f:
    datasets = pickle.load(f)

#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

exec(open(f'{path_stem}UWB functions.py').read())
#%%
#%%
#%%
Tri_dat = pd.concat( datasets.values() )

#%%
#%%
Tri_dat['X_clust_centroid'] = Tri_dat['dwell_cluster'].map(dwell_cluster_df.set_index('cluster_id')['Xu_wt'])
Tri_dat['Y_clust_centroid'] = Tri_dat['dwell_cluster'].map(dwell_cluster_df.set_index('cluster_id')['Yu_wt'])

Tri_dat['X_clust_q05'] = Tri_dat['dwell_cluster'].map(dwell_cluster_df.set_index('cluster_id')['Xq05_wt'])
Tri_dat['X_clust_q95'] = Tri_dat['dwell_cluster'].map(dwell_cluster_df.set_index('cluster_id')['Xq95_wt'])
Tri_dat['Y_clust_q05'] = Tri_dat['dwell_cluster'].map(dwell_cluster_df.set_index('cluster_id')['Yq05_wt'])
Tri_dat['Y_clust_q95'] = Tri_dat['dwell_cluster'].map(dwell_cluster_df.set_index('cluster_id')['Yq95_wt'])
#%%
#%%
Tri_dat['X_path'] = [x if not ma.isnan(p) else np.nan for x,p in zip(Tri_dat['Xma_1'], Tri_dat['path_count'] ) ]
Tri_dat['Y_path'] = [y if not ma.isnan(p) else np.nan for y,p in zip(Tri_dat['Yma_1'], Tri_dat['path_count'] ) ]
#%%
#%%
for i in range(1,7):    
    datasets[f'rig_data{i}'] = Tri_dat.loc[Tri_dat['rig'] == i]
    
#%%
with open('UWB_datasets_14_clust_checking.pkl', 'wb') as f:
    pickle.dump(datasets, f)
#%%
#%%
#%
#%%
plt.figure(figsize=(6, 6))
for i in range(len(dwell_cluster_df)):
    
    x_quantile_lower = dwell_cluster_df['Xq05_wt'][i]
    x_quantile_upper = dwell_cluster_df['Xq95_wt'][i]
    y_quantile_lower = dwell_cluster_df['Yq05_wt'][i]
    y_quantile_upper = dwell_cluster_df['Yq95_wt'][i]
    
    x_center = (x_quantile_upper + x_quantile_lower) / 2
    y_center = (y_quantile_upper + y_quantile_lower) / 2
    x_radius = (x_quantile_upper - x_quantile_lower) / 2
    y_radius = (y_quantile_upper - y_quantile_lower) / 2
    
    ellipse = Ellipse((x_center, y_center), 2 * x_radius, 2 * y_radius, edgecolor='red', fc='None', lw=2)
    plt.gca().add_patch(ellipse)
    
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Ellipse Representation')
plt.xlim(0, 10)  # Adjust limits as needed
plt.ylim(0, 10)  # Adjust limits as needed
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%
#%
#%%
##############################################################################
#%%
%matplotlib qt
animation_fun_pred_cluster(datasets['rig_data1'], col = 'pred_26_NV', pace = 0.5, testing = False, view_actual = True)
#%%
animation_fun_pred_cluster(datasets['rig_data2'], col = 'pred_26_NV', pace = 0.15, testing = False)
#%%
animation_fun_pred_cluster(datasets['rig_data3'], col = 'pred_26_NV', pace = 0.15, testing = False)
#%%
animation_fun_pred_cluster(datasets['rig_data4'], col = 'pred_26_NV', pace = 0.15, testing = False)
#%%
animation_fun_pred_cluster(datasets['rig_data5'], col = 'pred_26_NV', pace = 0.15, testing = False)
#%%
animation_fun_pred_cluster(datasets['rig_data6'], col = 'pred_26_NV', pace = 0.01, testing = False, recent = 0.2, atatime = 0.1)
#%%

#%%
