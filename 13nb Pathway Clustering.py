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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_samples

pd.set_option('display.max_columns', 78)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
#%%
#%%
dwell_clusterids_df = pd.read_pickle('dwell_df_clean.pkl')
dwell_cluster_df = pd.read_pickle('dwell_cluster_df.pkl')
#Tri_dat = pd.read_pickle('Tri_dat_clustered_12.pkl')
#%%
import pickle
with open('UWB_datasets_12_dwell_clustered.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
#%% 
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
def calculate_dif_percentile_interpol(data, cumulative_distances, total_length, percentile,
                                      centX, centY):
    
    percentage_distance = percentile*total_length/100
    index = np.searchsorted(cumulative_distances, percentage_distance)
    if index == 0:
        xdif_percentile = data['difX'].iloc[0]
        ydif_percentile = data['difY'].iloc[0]
        x_percentile = data['X'].iloc[0]
        y_percentile = data['Y'].iloc[0]
        x_pc_ma1 = data['Xma_1'].iloc[0]
        y_pc_ma1 = data['Yma_1'].iloc[0]
            
    elif index == len(cumulative_distances):
        xdif_percentile = data['difX'].iloc[-1]
        ydif_percentile = data['difY'].iloc[-1]
        x_percentile = data['X'].iloc[-1]
        y_percentile = data['Y'].iloc[-1]
        x_pc_ma1 = data['Xma_1'].iloc[-1]
        y_pc_ma1 = data['Yma_1'].iloc[-1]
          
    else:
        prop_between = (percentage_distance - cumulative_distances[index-1])/(cumulative_distances[index] - cumulative_distances[index-1])
        xdif_percentile = data['difX'].iloc[:(index-1)].sum() + prop_between*data['difX'].iloc[index]
        ydif_percentile = data['difY'].iloc[:(index-1)].sum() + prop_between*data['difY'].iloc[index]
        
        x_percentile = data['X'].iloc[index-1] + prop_between*(data['X'].iloc[index] - data['X'].iloc[index-1] )
        y_percentile = data['Y'].iloc[index-1] + prop_between*(data['Y'].iloc[index] - data['Y'].iloc[index-1] )
        
        x_pc_ma1 =  data['Xma_1'].iloc[index-1] + (data['Xma_1'].iloc[index] - data['Xma_1'].iloc[index-1] )
        y_pc_ma1 =  data['Yma_1'].iloc[index-1] + (data['Yma_1'].iloc[index] - data['Yma_1'].iloc[index-1] )
                   
    dist_percentile = np.sqrt( (x_percentile-centX)**2 + (y_percentile-centY)**2 )
    path_dist_pc = percentile*total_length/100
    sin_percentile = (y_percentile-centY)/dist_percentile
    cos_percentile = (x_percentile-centX)/dist_percentile    
    tan_percentile = sin_percentile/cos_percentile
        
    return xdif_percentile, ydif_percentile, x_percentile, y_percentile, x_pc_ma1, y_pc_ma1, dist_percentile, path_dist_pc, sin_percentile, cos_percentile, tan_percentile
                                                                                                                         
#%%
#%%
temp_dat = Tri_dat.copy().reset_index(drop = True)
step = 10
percentiles = range(step,100+step,step)

cols = ['path_id', 'rig', 'start_clust', 'end_clust',
        'start_X', 'end_X', 'start_Y', 'end_Y',
        'time_start', 'time_end', 'time_path',
        'pathway_len', 'min_X', 'max_X', 'min_Y', 'max_Y'
         ]

for i in percentiles:
    cols.append(f'Xdifpc_{i}')
    cols.append(f'Ydifpc_{i}')
    
    cols.append(f'Xpc_{i}')
    cols.append(f'Ypc_{i}')
    
    cols.append(f'Xma1pc_{i}')
    cols.append(f'Yma1pc_{i}')
    
    cols.append(f'dist2start_pc_{i}')
    
    cols.append(f'path_dist_pc_{i}')
    
    cols.append(f'sin_pc_{i}')
    cols.append(f'cos_pc_{i}')
    cols.append(f'tan_pc_{i}')

path_clusters_df = pd.DataFrame(columns=cols)

for path_count, group_data in temp_dat.groupby('path_count'):
    
    first_index = group_data.index[0]
    last_index = group_data.index[-1]
    preceding_row_number = first_index - 1 if first_index > 0 else None
    next_row_number = last_index + 1 if last_index < len(temp_dat) else None

    current_worker = group_data['rig'].iloc[0]
    prev_worker = temp_dat['rig'].iloc[preceding_row_number] if preceding_row_number is not None else group_data['rig'].iloc[0]
    next_worker = temp_dat['rig'].iloc[next_row_number] if next_row_number is not None else group_data['rig'].iloc[0]
    
    if (prev_worker == current_worker):    
        start_clust = temp_dat['dwell_cluster'].iloc[preceding_row_number]
    else:
        start_clust = None
    
    if (next_worker == current_worker):    
        end_clust = temp_dat['dwell_cluster'].iloc[next_row_number]
    else:
        end_clust = None    
    
    clust_df = dwell_cluster_df[dwell_cluster_df['cluster_id'] == start_clust]
    clust_dist_95 = 1
   
    start_clust_centroid_X = clust_df['Xu_wt'].iloc[0]
    start_clust_centroid_Y = clust_df['Yu_wt'].iloc[0]
    start_dists = np.sqrt( (group_data['X']-start_clust_centroid_X )**2 + (group_data['Y'] - start_clust_centroid_Y)**2)
    
    condition1 = (len(group_data) < 2)
    condition2 = (group_data['ed_dif'].sum() == 0 )
    condition3 = (clust_dist_95 >= start_dists).all()
    condition4 = (start_dists > 1).sum() < 2
     
    if ( condition1 | condition2 | condition3):
        pass
    else:
        start_X = group_data['Xma_05'].iloc[0]
        end_X = group_data['Xma_05'].iloc[-1]
        start_Y = group_data['Yma_05'].iloc[0]
        end_Y = group_data['Yma_05'].iloc[-1]
    
        time_start = group_data['time_lapsed_all'].iloc[0]
        time_end = group_data['time_lapsed_all'].iloc[-1]
        time_path = time_end - time_start
    
        pathway_len = group_data['ed_dif'].sum()
    
        min_X = group_data['X'].min()
        max_X = group_data['X'].max()
        min_Y = group_data['Y'].min()
        max_Y = group_data['Y'].max()
    
        pathway_cumsum = cumsum(group_data['ed_dif']).values
        #
        pecentile_vars = []
      
        for i in percentiles:
            quant_vars = calculate_dif_percentile_interpol(group_data, pathway_cumsum,
                                                                       pathway_len,
                                                                       i,
                                                                       start_clust_centroid_X,
                                                                       start_clust_centroid_Y )
           
            for j in quant_vars:
                pecentile_vars.append(j)            
                
        new_row_vars = [path_count, current_worker, start_clust, end_clust,
                        start_X, end_X, start_Y, end_Y,
                        time_start, time_end, time_path,
                        pathway_len, min_X, max_X, min_Y, max_Y
                         ]

        new_row_vars += pecentile_vars
        
        path_clusters_df.loc[len(path_clusters_df)] = new_row_vars
    
path_clusters_df.dropna(inplace=True)
#%%  

path_clusters_df['X_len'] = path_clusters_df['max_X'] - path_clusters_df['min_X']
path_clusters_df['Y_len'] = path_clusters_df['max_Y'] - path_clusters_df['min_Y']

path_clusters_df['X_len'][path_clusters_df['X_len'] == 0] = 0.005
path_clusters_df['Y_len'][path_clusters_df['Y_len'] == 0] = 0.005

path_clusters_df['height2width_ratio'] = path_clusters_df['Y_len'] / path_clusters_df['X_len']

#%%
#Tri_dat['pinteraction'] =         
#%%
#%%
#%%
no_dwell_clusters = Tri_dat['dwell_cluster'].max() + 1
#%%
perm_clusters = []
perm_clusters_reversable = []
    
for ind, row in path_clusters_df.iterrows():
    
    temp_perm = [ row['start_clust'], row['end_clust'] ]
   
    if sorted(temp_perm) not in perm_clusters_reversable:
        perm_clusters_reversable.append( sorted(temp_perm) )
    
    if temp_perm not in perm_clusters:
        perm_clusters.append( temp_perm )

perm_clusters = sorted(perm_clusters)
perm_clusters_reversable = sorted(perm_clusters_reversable)

#%%
#%%
path_clusters_df['perm_clust'] = path_clusters_df['perm_clust_rev'] = [[] for _ in range(len(path_clusters_df))]

for i in range(len(path_clusters_df)):
    
    row_perm = [path_clusters_df.loc[i, 'start_clust'], path_clusters_df.loc[i, 'end_clust'] ]
    path_clusters_df.at[i, 'perm_clust'] = row_perm
    
    match_index = None
    for j, x in enumerate(perm_clusters):
        if x == row_perm:
            match_index = j
            break
    
    path_clusters_df.at[i, 'perm_clust_id'] = match_index
    
    row_perm_rev = sorted([path_clusters_df.loc[i, 'start_clust'], path_clusters_df.loc[i, 'end_clust'] ])
    path_clusters_df.at[i, 'perm_clust_rev'] = row_perm_rev
    
    match_index = None
    for j, x in enumerate(perm_clusters_reversable):
        if x == row_perm_rev:
            match_index = j
            break
        
    path_clusters_df.at[i, 'perm_clust_rev_id'] = match_index        
    
#%%
#%%
for i in range(1,7):
    
    rig_data = datasets[f'rig_data{i}']
    
    clust_dat = path_clusters_df.loc[path_clusters_df['rig']==i]
    
    rig_data['perm_clust_rev_id'] = rig_data['path_count'].map(clust_dat.set_index('path_id')['perm_clust_rev_id'])
    rig_data['perm_clust_rev'] = rig_data['path_count'].map(clust_dat.set_index('path_id')['perm_clust_rev'])
    
    rig_data['perm_clust_ow_id'] = rig_data['path_count'].map(clust_dat.set_index('path_id')['perm_clust_id'])
    rig_data['perm_clust_ow'] = rig_data['path_count'].map(clust_dat.set_index('path_id')['perm_clust'])

    rig_data['path_time_start'] = rig_data['path_count'].map(clust_dat.set_index('path_id')['time_start'])
    
    rig_data['path_length'] = rig_data['path_count'].map(clust_dat.set_index('path_id')['pathway_len'])
    
    rig_data['path_time_ow'] =  rig_data['time_lapsed_all'] -  rig_data['path_time_start']

    datasets[f'rig_data{i}'] = rig_data
#%%
Tri_dat = pd.concat( datasets.values() )   
#%%
Tri_dat['path_distance_bw'] = np.nan     
#%%
for path_count, group_data in Tri_dat.groupby('path_count'):
    
    if group_data['perm_clust_ow'].iloc[0] == group_data['perm_clust_rev'].iloc[0]:
        distances = cumsum(group_data['ed_dif'])
        times = cumsum(group_data['dif_time'])
    else:
        distances = cumsum(group_data['ed_dif'][::-1])
        times = cumsum(group_data['dif_time'][::-1])
    
    Tri_dat.loc[ Tri_dat['path_count'] == path_count, 'path_distance_bw'] = distances.values
    Tri_dat.loc[ Tri_dat['path_count'] == path_count, 'path_time_bw'] = times.values
        
#%%
#%%
#%%
len(perm_clusters_reversable)

test_dat = Tri_dat[ Tri_dat['perm_clust_ow'] == 0]

#%%
#%%
################################################################################
################## Trying with (path) cluster DF #################################
################################################################################
# functions 
def print_all_clusters_fun(agg_dat, master_dat, cluster_assignments):
    
    fig, axes = plt.subplots(figsize=(10, 6))
    
 #   xmin = master_dat['X'].min() - 0.5
  #  xmax = master_dat['X'].max() + 0.5
   # ymin = master_dat['Y'].min() - 0.5
   # ymax = master_dat['Y'].max() + 0.5
    
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
def print_indi_clusters_fun(agg_dat, master_dat, columns, cluster_assignments, path_perm, final = 'NF'):
    #
    path_counts = plot_dat['path_count'].unique()
       
   # xmin = master_dat['X'].min() - 0.5
   # xmax = master_dat['X'].max() + 0.5
   # ymin = master_dat['Y'].min() - 0.5
   # ymax = master_dat['Y'].max() + 0.5

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
        ax.set_title(f'Path ID {final}: {path_count}, {path_perm}')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


    # Hide unused subplots
    for i in range(num_plots, rows * colms):
        axs.flatten()[i].axis('off')

    # Show the plot
    plt.show()

#%%
def silloutette_score_fun(link_mat, scaled_dat, max_score = 11, start = 0, plots = True):
    
    silhouette_scores_check = []
    best_threshold = start
    best_clusters = 1*len(scaled_dat)
    max_silhouette_score = float('-inf')

    thresholds = np.linspace(start,max_score,20)  # Using linkage distances as thresholds
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
            if round(silhouette_score_value,4) >= round(max_silhouette_score,4):
                max_silhouette_score = silhouette_score_value
                best_threshold = threshold
                best_clusters = len(np.unique(clusters))
        else:
             silhouette_scores_check.append(0)
    
    if plots == True:       
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
    
#%% ####################################################################
#%% ##########################################################################
############################################################################## 
##############################################################################

selected_columns = np.array([ 'pathway_len', 'X_len', 'Y_len', 'min_X', 'max_X', 'min_Y', 'max_Y', 'height2width_ratio'] )


for i in percentiles:
   # selected_columns.append(f'Xdifpc_{i}')
   # selected_columns.append(f'Ydifpc_{i}')    
    selected_columns = np.append(selected_columns, f'Xpc_{i}')
    selected_columns = np.append(selected_columns, f'Ypc_{i}')    
    # selected_columns.append(f'Xma1pc_{i}')
    # selected_columns.append(f'Yma1pc_{i}')    
    selected_columns = np.append(selected_columns, f'dist2start_pc_{i}')    
  #  selected_columns.append(f'path_dist_pc_{i}')
    selected_columns = np.append(selected_columns, f'sin_pc_{i}')
    selected_columns = np.append(selected_columns, f'cos_pc_{i}')
    selected_columns = np.append(selected_columns, f'tan_pc_{i}')

columns1 = np.isin(selected_columns, np.array(['pathway_len', 'X_len', 'Y_len']) )

columns2 = np.isin(selected_columns, np.array(['X_len', 'min_X', 'max_X']) )
columns3 = np.isin(selected_columns, np.array(['Y_len', 'min_Y', 'max_Y']) )
columns4 = [ 'Xpc' in col for col in selected_columns]

columns5 = [ 'Ypc' in col for col in selected_columns]
columns6 = [ 'dist2' in col for col in selected_columns]

columns7 = [ 'cos' in col for col in selected_columns]
columns8 = [ 'sin' in col for col in selected_columns]
columns9 = [ 'tan' in col for col in selected_columns]

columns10 = np.isin(selected_columns, np.array(['height2width_ratio']) )

colums11 = [x or y for x,y in zip(columns4,columns5) ]


#test_dat_select = test_dat[columns1 + columns2]
#%%
path_cluster_prep_dat_select = path_clusters_df[selected_columns].copy()
All_scaled_df = path_cluster_prep_dat_select.copy()
# start_dist_factor = 0.35
minXY = min(Tri_dat['Y'])
maxXY = max(Tri_dat['Y'])

for col in All_scaled_df.columns:    
        minV = min(path_cluster_prep_dat_select[col])
        maxV = max(path_cluster_prep_dat_select[col])
        All_scaled_df[col] = (path_cluster_prep_dat_select[col] - minV) / (maxXY - minV)
#%%
All_scaled = np.array(All_scaled_df)
#%%
#%% #######################
##############################################################################
########################### 78

path_clust_ID = 78

test_dat = path_clusters_df[ path_clusters_df['perm_clust_id'] == path_clust_ID ].copy()
#%%
fig, axes = plt.subplots(figsize=(9.5, 6))

plot_dat = Tri_dat[Tri_dat['perm_clust_ow_id']  == path_clust_ID].copy()

xmin = plot_dat['X'].min() - 0.5
xmax = plot_dat['X'].max() + 0.5
ymin = plot_dat['Y'].min() - 0.5
ymax = plot_dat['Y'].max() + 0.5

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
last = 10
#%%
select_rows = path_clusters_df['perm_clust_id'] == path_clust_ID

X_scaled = {}

for i in range(1,last+1):
    
    X_scaled[i] = All_scaled[select_rows][:, eval(f'columns{i}' ) ]

#%%
check_array = np.array(path_cluster_prep_dat_select)

check_dict = {}

for i in range(1,last+1):
    
    check_dict[i] = check_array[select_rows][:, eval(f'columns{i}' ) ]
    
#%%
start1 = 0.82
start2 = 0.285  # 0.235
start3 = 0.285  # 0.235
start4 = 0.335
start5 = 0.335
start6 = 0.8
start7 = 0.39
start8 = 0.39
start9 = 2
start10 = 0.2
# start11 = 1.5

max_score1 = start1 + 2
max_score2 = start2 + 2
max_score3 = start3 + 2
max_score4 = start4 + 2
max_score5 = start5 + 2
max_score6 = start6 + 2
max_score7 = start7 + 2
max_score8 = start8 + 2
max_score9 = start9 + 2
max_score9 = start9 + 2
max_score10 = start10 + 2
# max_score11 = start11 + 2

link_method = 'average'
#%%
test_dat['cluster_1'] = ''

if len(X_scaled[1]) > 1:
    similarity_matrix = pairwise_distances(X_scaled[1], metric='euclidean')  # 'sqeuclidean', 'seuclidean',
   
    hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

    plt.figure(figsize=(10, 5))
    plt.title('Dendrogram for Hierarchical Clustering')
    dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    
    hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                         X_scaled[1],
                                                                                         max_score=max_score1,
                                                                                         start = start1 )
    hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')
    
else:
    hier_cluster_assignments = 1
        
#print_all_clusters_fun(test_dat, plot_dat, hier_cluster_assignments_first) 
print_indi_clusters_fun(test_dat, plot_dat, 5, hier_cluster_assignments, path_perm=path_clust_ID)
   
test_dat['cluster_1'] = hier_cluster_assignments
#%%
clust = 2
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}')  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster                                                                   
                                                                                                                            
#%%
clust = 3
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}')  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster                    

#%%
clust = 4
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}')  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster                           
#%%
clust = 5
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}')  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster   
         
#%%
clust = 6
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

   
        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}')  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster                    
#%%
clust = 7
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

    
        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}'  )  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster 

#%%
clust = 8
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

 
        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}'  )  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster
#%%
clust = 9
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

 
        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}'  )  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster

#%%
clust = 10
test_dat[f'cluster_{clust}'] = ''

for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
    subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
    cases = test_dat[f'cluster_{clust-1}'] == i
    
    if len(subset_X) > 1:
        similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

 
        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        plt.figure(figsize=(10, 5))
        plt.title('Dendrogram for Hierarchical Clustering')
        dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}'  )  )
        
        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
    else:
        hier_cluster_assignments = [1]
        
    subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
    subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
    print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
                            final = 'Final')
    
    cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
                                                              hier_cluster_assignments ) ]
                                                              
    test_dat.loc[cases, f'cluster_{clust}'] = cluster  
#%%
# clust = 11
# test_dat[f'cluster_{clust}'] = ''

# for i in sorted( np.unique( test_dat[f'cluster_{clust-1}'] ) ):
    
#     subset_X = X_scaled[clust][ test_dat[f'cluster_{clust-1}'] == i ]
#     cases = test_dat[f'cluster_{clust-1}'] == i
    
#     if len(subset_X) > 1:
#         similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

 
#         hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

#         plt.figure(figsize=(10, 5))
#         plt.title('Dendrogram for Hierarchical Clustering')
#         dendrogram(linkage(similarity_matrix, method=link_method), truncate_mode='level', p=0)
#         plt.xlabel('Index')
#         plt.ylabel('Distance')
#         plt.show()
        
#         hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
#                                                                                     subset_X,
#                                                                                     max_score= eval(f'max_score{clust}'),
#                                                                             start = eval(f'start{clust}'  )  )
        
#         hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')       
#     else:
#         hier_cluster_assignments = [1]
        
#     subtest_dat = test_dat[test_dat[f'cluster_{clust-1}'] == i]
#     subplot_dat = plot_dat[plot_dat['path_count'].isin(subtest_dat['path_id'])].copy()
    
    
#     print_indi_clusters_fun(subtest_dat, subplot_dat, 5, hier_cluster_assignments, path_perm = path_clust_ID,
#                             final = 'Final')
    
#     cluster = [(str(x) + '_' + str(y) ) for x,y in zip( test_dat[f'cluster_{clust-1}'][cases],
#                                                               hier_cluster_assignments ) ]
                                                              
#     test_dat.loc[cases, f'cluster_{clust}'] = cluster                                                     
#%%
#%%
#%%
path_perms = sorted(path_clusters_df['perm_clust_id'].unique() )
    
#%%                              
#%%
#%%
for path_id in path_perms:
      
    test_dat = path_clusters_df[ path_clusters_df['perm_clust_id'] == path_id ].copy()
    test_dat_select = test_dat[selected_columns].copy()
    
    plot_dat = Tri_dat[Tri_dat['perm_clust_ow_id']  == path_id].copy()
    
    xmin = plot_dat['X'].min() - 0.5
    xmax = plot_dat['X'].max() + 0.5
    ymin = plot_dat['Y'].min() - 0.5
    ymax = plot_dat['Y'].max() + 0.5
    
    X_scaled = {}
    
    select_rows = path_clusters_df['perm_clust_id'] == path_id
    
    for i in range(1, last + 1): 
        X_scaled[i] = All_scaled[select_rows][:, eval(f'columns{i}') ]
    
    test_dat_select['cluster_1'] = ''
      
    if len(X_scaled[1]) > 1:
        similarity_matrix = pairwise_distances(X_scaled[1], metric='euclidean')

        hierarchical_link_mat = linkage(similarity_matrix, method=link_method)

        hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                              X_scaled[1],
                                                                                              max_score=eval(f'max_score{1}'),
                                                                                              start = start1, plots = False )

        hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')
        
        test_dat_select['cluster_1'] =  [str(int(path_id) ) + '_' + str(x) for x in hier_cluster_assignments]
                    
    else:
        hier_cluster_assignments = 1
        test_dat_select['cluster_1'] =  str(int(path_id) ) + '_' + str(hier_cluster_assignments)

    for clust in range(2, last + 1):
        
        test_dat_select[f'cluster_{clust}'] = ''

        for i in sorted( np.unique( test_dat_select[f'cluster_{clust-1}'] ) ):
    
            subset_X = X_scaled[clust][ test_dat_select[f'cluster_{clust-1}'] == i ]
            cases = test_dat_select[f'cluster_{clust-1}'] == i
    
            if len(subset_X) > 1:
                similarity_matrix = pairwise_distances(subset_X, metric='euclidean')

                hierarchical_link_mat = linkage(similarity_matrix, method=link_method)
        
                hier_best_thresh, hier_best_clusters, hier_silhouette_scores = silloutette_score_fun(hierarchical_link_mat,
                                                                                    subset_X,
                                                                                    max_score= eval(f'max_score{clust}'),
                                                                            start = eval(f'start{clust}'), plots = False )
        
                hier_cluster_assignments = fcluster(hierarchical_link_mat, hier_best_thresh, criterion='distance')
                cluster = [(str(x) + '_' + str(y) ) for x,y in zip(test_dat_select[f'cluster_{clust-1}'][cases],
                                                                         hier_cluster_assignments) ]
            else:
                hier_cluster_assignments = 1
                cluster =  str(i) + '_' + str(hier_cluster_assignments)
        
            test_dat_select.loc[cases, f'cluster_{clust}'] = cluster
            
    
    path_clusters_df.loc[ path_clusters_df['perm_clust_id'] == path_id, f'cluster_{last}' ] = test_dat_select[f'cluster_{last}']    
                     
    print_indi_clusters_fun(test_dat, plot_dat, 5,  test_dat_select[f'cluster_{last}'], path_perm = path_id, final = 'Final')
#%%
#%%
#%%
path_cluster_labels = sorted(path_clusters_df[f'cluster_{last}'].unique() )

path_cluster_dict = {label: idx for idx, label in enumerate(path_cluster_labels)}

path_clusters_df['path_cluster'] = [path_cluster_dict[x] for x in path_clusters_df[f'cluster_{last}'] ]

#%%

Tri_dat['path_cluster'] = Tri_dat['path_count'].map(path_clusters_df.set_index('path_id')['path_cluster'])

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
with open('UWB_datasets_28_path_clustered.pkl', 'wb') as f:
    pickle.dump(datasets, f)
#%%

