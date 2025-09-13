# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:52:38 2024

@author: 
"""
#%%
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, ParameterGrid


pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
#%%
import pickle
with open('datasets_pred_26_NV.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%   
#with open('datasets_pred_23_NV.pkl', 'rb') as f:
 #   datasets = pickle.load(f)
#%%
# Tri_dat = pd.read_pickle('UWB_dat_post_I5lstmAtt_pred.pkl')
pred_col = 'pred_26_NV'
#%%
Tri_dat = pd.concat( datasets.values() )
#%%
#%%
Tri_dat['w1_d'] = np.nan
Tri_dat['w2_d'] = np.nan
Tri_dat['w3_d'] = np.nan
Tri_dat['w4_d'] = np.nan
Tri_dat['w5_d'] = np.nan
Tri_dat['w6_d'] = np.nan
#%%
#%%
for worker in range(1,7):
    
    col_no = Tri_dat.columns.get_loc(f'w{worker}_d')
    
    worker_dat = Tri_dat.loc[Tri_dat['rig'] == worker].reset_index(drop = True)
    worker_lapsed = worker_dat['time_lapsed_all']
    
    for i in range(len(Tri_dat)):
        row = Tri_dat.iloc[i]
        if row['rig'] == worker:
            Tri_dat.iloc[i, col_no] = 0
        else:
            closest_i = np.argmin(abs(row['time_lapsed_all']-worker_lapsed))
            Tri_dat.iloc[i, col_no] = ma.sqrt((worker_dat.loc[closest_i, 'X'] - row['X'])**2 + (worker_dat.loc[closest_i, 'Y'] - row['Y'])**2)
            
#%% #
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
condition1 = (Tri_dat.rig == 1)

for rig, ax in zip(range(1,7), axes.flatten()):
        
    ax.scatter(Tri_dat.loc[condition1, 'time_lapsed_all'], Tri_dat.loc[condition1, f'w{rig}_d'], s = 3, alpha = 0.1 )
    ax.set_title(f'Worker {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axhline(y=1, color='r', linestyle='--')

plt.tight_layout()
plt.show()
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
condition1 = (Tri_dat.rig == 2)

for rig, ax in zip(range(1,7), axes.flatten()):
        
    ax.scatter(Tri_dat.loc[condition1, 'time_lapsed_all'], Tri_dat.loc[condition1, f'w{rig}_d'], s = 3, c = 'red', alpha = 0.1 )
    ax.set_title(f'Worker {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axhline(y=1, color='b', linestyle='--')

plt.tight_layout()
plt.show()
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
condition1 = (Tri_dat.rig == 3)

for rig, ax in zip(range(1,7), axes.flatten()):
        
    ax.scatter(Tri_dat.loc[condition1, 'time_lapsed_all'], Tri_dat.loc[condition1, f'w{rig}_d'], s = 3, c = 'green', alpha = 0.1 )
    ax.set_title(f'Worker {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axhline(y=1, color='r', linestyle='--')

plt.tight_layout()
plt.show()
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
condition1 = (Tri_dat.rig == 4)

for rig, ax in zip(range(1,7), axes.flatten()):
        
    ax.scatter(Tri_dat.loc[condition1, 'time_lapsed_all'], Tri_dat.loc[condition1, f'w{rig}_d'], s = 3, c = 'orange', alpha = 0.1 )
    ax.set_title(f'Worker {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axhline(y=1, color='r', linestyle='--')

plt.tight_layout()
plt.show()
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
condition1 = (Tri_dat.rig == 5)

for rig, ax in zip(range(1,7), axes.flatten()):
        
    ax.scatter(Tri_dat.loc[condition1, 'time_lapsed_all'], Tri_dat.loc[condition1, f'w{rig}_d'], s = 3, c = 'grey', alpha = 0.1 )
    ax.set_title(f'Worker {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axhline(y=1, color='r', linestyle='--')

plt.tight_layout()
plt.show()
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
condition1 = (Tri_dat.rig == 4)

for rig, ax in zip(range(1,7), axes.flatten()):
        
    ax.scatter(Tri_dat.loc[condition1, 'time_lapsed_all'], Tri_dat.loc[condition1, f'w{rig}_d'], s = 3, c = 'purple', alpha = 0.1 )
    ax.set_title(f'Worker {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axhline(y=1, color='r', linestyle='--')

plt.tight_layout()
plt.show()
#%% here
#%%
thresh = 1

for i in range(1,7):
    Tri_dat[f'pinteraction_w{i}'] = [1 if (x < thresh and i != y) else 0 for x,y in zip(Tri_dat[f'w{i}_d'], Tri_dat['rig']) ]
     
#%%
Tri_dat['pinteraction'] = [x+y+z+a+b+c for x,y,z,a,b,c in zip(Tri_dat['pinteraction_w1'],
                                                                   Tri_dat['pinteraction_w2'],
                                                                   Tri_dat['pinteraction_w3'],
                                                                   Tri_dat['pinteraction_w4'],
                                                                   Tri_dat['pinteraction_w5'],
                                                                   Tri_dat['pinteraction_w6'])]
#%%
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

for rig, ax in zip(range(1,7), axes.flatten()):
    condition1 = (Tri_dat.rig == rig) & (Tri_dat.pinteraction > 0)
    ax.scatter(Tri_dat.loc[condition1, 'X'], Tri_dat.loc[condition1, 'Y'], s = 3, c = 'purple', alpha = 0.1 )
    ax.set_title(f'Worker {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.tight_layout()
plt.show()

#%%
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 5.5))
colours = ['red', 'black', 'blue', 'purple', 'darkorange', 'grey']

handles = [Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color) for color in colours]
legend_labels = [f'Worker {i}' for i in range(1, 7)]

for rig, ax in zip(range(1, 7), axes.flatten()):
    for i in range(1, 7):
        if i == rig:
            pass
        else:
            condition1 = (Tri_dat.rig == rig) & (Tri_dat[f'pinteraction_w{i}'] > 0)
            ax.scatter(Tri_dat.loc[condition1, 'X'], Tri_dat.loc[condition1, 'Y'], s=i, c=colours[i - 1], alpha=0.1)
    ax.set_title(f'Worker {rig} potential interactions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

fig.legend(handles, legend_labels, loc='upper left', title='Workers', bbox_to_anchor=(1, 0.9))

plt.subplots_adjust(right=0.85)
plt.tight_layout()
plt.show()

#%%
Tri_dat.to_pickle('UWB_dat_post_pred&interactionsexplore.pkl')

#Tri_dat = pd.read_pickle('UWB_dat_post_pred&interactionsexplore.pkl')
#%%
#%%
plt.hexbin(Tri_dat['X'], Tri_dat['Y'], gridsize=50, cmap='Blues')
plt.colorbar(label='Density')
plt.title('Density Plot with Matplotlib')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
#%%
#for i in range(1,7):
    
 #   datasets[f'rig_data{i}'] = Tri_dat.loc[Tri_dat['rig'] == i]
#%%
#with open('UWB_datasets_23with worker_interactions.pkl', 'wb') as f:
    
#    pickle.dump(datasets, f)

#%%
#%%
xmin = Tri_dat['X'].min()
xmax = Tri_dat['X'].max()
ymin = Tri_dat['Y'].min()
ymax = Tri_dat['Y'].max()

plt.hexbin(Tri_dat.loc[Tri_dat[pred_col]==0, 'X' ], Tri_dat.loc[Tri_dat[pred_col]==0, 'Y'],
           gridsize=50, cmap='Reds', extent=[xmin, xmax, ymin, ymax], label='Binary Variable 0')
plt.colorbar(label='Density')
plt.title('Density Plot of worker no-movement or movement on the spot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#%%
plt.hexbin(Tri_dat.loc[Tri_dat[pred_col]==1, 'X' ], Tri_dat.loc[Tri_dat[pred_col]==1, 'Y'],
           gridsize=50, cmap='Blues', extent=[xmin, xmax, ymin, ymax], label='Binary Variable 1')
plt.colorbar(label='Density')
plt.title('Density Plot of worker movement away from a location')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
#%%
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

for i, ax in enumerate(axes.flatten(), start=1):
    condition1 = (Tri_dat.rig == i) & (Tri_dat[pred_col] == 0)
    hb = ax.hexbin(Tri_dat.loc[condition1, 'X'], 
                   Tri_dat.loc[condition1, 'Y'],
                   gridsize=50, cmap='Blues', extent=[xmin, xmax, ymin, ymax], label='Binary Variable 1')
    ax.set_title(f'Density worker no-movement or movement on the spot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.tight_layout()
plt.show()
#%%
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

for i, ax in enumerate(axes.flatten(), start=1):
    condition1 = (Tri_dat.rig == i) & (Tri_dat[pred_col] == 0)
    hb = ax.scatter(Tri_dat.loc[condition1, 'X'], 
                   Tri_dat.loc[condition1, 'Y'], alpha = 0.02)
    ax.set_title(f'worker {i} worker dwell')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)

plt.tight_layout()
plt.show()
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

for i, ax in enumerate(axes.flatten(), start=1):
    condition1 = (Tri_dat.rig == i) & (Tri_dat[pred_col] == 1)
    hb = ax.hexbin(Tri_dat.loc[condition1, 'X'], 
                   Tri_dat.loc[condition1, 'Y'],
                   gridsize=50, cmap='Greens', extent=[xmin, xmax, ymin, ymax], label='Binary Variable 1')
    ax.set_title(f'Density Plot of worker {i} movement away from a location')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)

plt.tight_layout()
plt.show()
#%%
# let's smooth out  predictions
#%%
window_size_steps = 4

for i in range(1,7):
    temp = Tri_dat.loc[ Tri_dat.rig == i , ]
    
   # ma3tempPL = temp['pred_lstm'].rolling(window=window_size_steps, min_periods=1, center = True).mean() 
    ma3tempPLA = temp[pred_col].rolling(window=window_size_steps, min_periods=1, center = True).mean() 
    ma3tempP = temp[pred_col].rolling(window=window_size_steps, min_periods=1, center = True).mean() 
   
   # Tri_dat.loc[ Tri_dat.rig == i, 'pred_lstm_clean'] = ma3tempPL.values
    Tri_dat.loc[ Tri_dat.rig == i, pred_col] = ma3tempPLA.values
    Tri_dat.loc[ Tri_dat.rig == i, 'pred_clean'] = ma3tempP.values
    
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

for i, ax in enumerate(axes.flatten(), start=1):
    condition1 = (Tri_dat.rig == i) & (Tri_dat[pred_col] < 0.5)
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
# let's count the dwelling zones
#%%
Tri_dat['cleaned_pred'] = round(Tri_dat[pred_col]).values
dwell_count = 0
path_count = 0

for i in range(1,7):
    temp = Tri_dat.loc[ Tri_dat.rig == i, ].copy().reset_index(drop = True)
    
    dwell_count_temp = [np.nan]*len(temp)
    path_count_temp = [np.nan]*len(temp)
    for j in range(len(temp)):
        if j == 0:
            if temp['cleaned_pred'].iloc[j] < 0.5:
                dwell_count += 1
                dwell_count_temp[j] = dwell_count
                
            elif temp['cleaned_pred'].iloc[j] > 0.5:
                path_count += 1
                path_count_temp[j] = path_count         
        elif j > 0:
           if temp['cleaned_pred'].iloc[j] == temp['cleaned_pred'].iloc[j-1]:
               if temp['cleaned_pred'].iloc[j] < 0.5:
                   dwell_count_temp[j] = dwell_count
               elif temp['cleaned_pred'].iloc[j] > 0.5:
                   path_count_temp[j] = path_count  
           
           elif temp['cleaned_pred'].iloc[j] != temp['cleaned_pred'].iloc[j-1]:
               if temp['cleaned_pred'].iloc[j] < 0.5:
                   dwell_count += 1
                   dwell_count_temp[j] = dwell_count
               elif temp['cleaned_pred'].iloc[j] > 0.5:
                   path_count += 1
                   path_count_temp[j] = path_count
                           
    Tri_dat.loc[ Tri_dat.rig == i, 'dwell_count'] = dwell_count_temp
    Tri_dat.loc[ Tri_dat.rig == i, 'path_count'] = path_count_temp
#%%
#%%
print(f" Total dwells = {Tri_dat['dwell_count'].max()}, Total paths = {Tri_dat['path_count'].max()}  ")
#%%
#%%
def calculate_spread(data):
    centroid_x = data['X'].mean()
    centroid_y = data['Y'].mean()
    distances = np.sqrt((data['X'] - centroid_x)**2 + (data['Y'] - centroid_y)**2)
    spread = ma.sqrt(np.mean(distances) )
    return spread

#%%    
dwell_df = pd.DataFrame(columns=['dwell_id','rig', 'Xu', 'Yu', 'Xmed', 'Ymed',
                                 'max_dist', 'q95_dist', 
                                 'Xlq', 'Xuq', 'Ylq', 'Yuq', 
                                 'Xmin', 'Xmax', 'Ymin', 'Ymax',
                                 'Xq05', 'Xq95', 'Yq05', 'Yq95',                                 
                                 'XY_spread',
                                 'time_start', 'time_end', 'time_dwell'])

for dwell_count, group_data in Tri_dat.groupby('dwell_count'):
    
    Xu = group_data['X'].mean()
    Yu = group_data['Y'].mean()
    
    max_dist = np.sqrt( (group_data['X']-Xu)**2 + (group_data['Y']-Yu)**2 ).max()  
    q95_dist = np.percentile( np.sqrt( (group_data['X']-Xu)**2 + (group_data['Y']-Yu)**2 ), 95)
    
    Xmed = group_data['X'].median()
    Ymed = group_data['Y'].median()
    
    Xlq = np.percentile(group_data['X'], 25) 
    Xuq = np.percentile(group_data['X'], 75) 
    Ylq = np.percentile(group_data['Y'], 25)
    Yuq = np.percentile(group_data['Y'], 75)
    
    Xmin = group_data['X'].min()
    Xmax = group_data['X'].max()
    Ymin = group_data['Y'].min()
    Ymax = group_data['Y'].max()
    
    Xq05 = np.percentile(group_data['X'], 5) 
    Xq95 = np.percentile(group_data['X'], 95) 
    Yq05 = np.percentile(group_data['Y'], 5)
    Yq95 = np.percentile(group_data['Y'], 95)
    
    rig = group_data['rig'].iloc[0]
    
    time_start = group_data['time_lapsed_all'].iloc[0]
    time_end = group_data['time_lapsed_all'].iloc[-1]
    
    time_dwell = time_end - time_start
        
    XY_spread = calculate_spread(group_data)
    
    dwell_df.loc[len(dwell_df)] = [dwell_count, rig, Xu, Yu, Xmed, Ymed,
                                   max_dist, q95_dist, 
                                   Xlq, Xuq, Ylq, Yuq,
                                   Xmin, Xmax, Ymin, Ymax,
                                   Xq05, Xq95, Yq05, Yq95,  
                                   XY_spread, time_start, time_end, time_dwell]
#%%
print(dwell_df)
#%%
dwell_df.to_pickle('dwell_df.pkl')
#%%
#%%
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
dwell_df_clean = dwell_df[dwell_df['time_dwell'] > 2].copy().reset_index(drop = True)
KM_data = dwell_df_clean.drop(['rig', 'dwell_id', 'time_start', 'time_end', 'time_dwell',
                               'max_dist', 
                               'q95_dist',
                               'Xmin', 'Xmax', 'Ymin', 'Ymax',
                               'Xq05', 'Xq95', 'Yq05', 'Yq95',  
                               'XY_spread'], axis = 1)

X_scaled = scaler.fit_transform(KM_data)
#%%
#%%
k_values = range(4, 30)

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
n_clusters = 14

kmeans = KMeans(n_clusters=n_clusters, random_state=69)
kmeans.fit(X_scaled)

cluster_assignments = kmeans.labels_
#%%
dwell_df_clean['dwell_cluster'] = cluster_assignments
#%%
#%%
plt.figure(figsize=(8, 6))

colors = plt.cm.tab20(np.linspace(0, 1, 13)) 

for cluster_id, color in enumerate(colors):
    cluster_points = dwell_df_clean[cluster_assignments == cluster_id]
    
    plt.scatter(cluster_points['Xu'], cluster_points['Yu'], label=f'Cluster {cluster_id}', alpha=0.3, color = color)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering')
plt.legend()
plt.grid(True)
plt.show()

#%%
#%%
xmin = Tri_dat['X'].min()
xmax = Tri_dat['X'].max()
ymin = Tri_dat['Y'].min()
ymax = Tri_dat['Y'].max()

# Define a colormap with distinct colors for each cluster
colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))  # Using 'tab20' colormap

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9.5, 5))

for rig, ax in zip(range(1, 7), axes.flatten()):
    dat = dwell_df_clean[dwell_df_clean.rig == rig]

    for cluster_id, color in enumerate(colors):
        cluster_points = dat[dat['dwell_cluster'] == cluster_id]

        ax.scatter(cluster_points['Xu'], cluster_points['Yu'], label=f'Cluster {cluster_id}', alpha=0.7, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Worker {rig} dwell zones')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

handles, labels = ax.get_legend_handles_labels()

# Display a single legend outside the loop
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.show()
#%%
dwel_df_clust = dwell_df.copy()
dwel_df_clust = dwel_df_clust.drop(['rig', 'dwell_id', 'time_start', 'time_end', 'time_dwell',
                                    'max_dist', 
                                    'q95_dist',
                                    'Xmin', 'Xmax', 'Ymin', 'Ymax', 
                                    'Xq05', 'Xq95', 'Yq05', 'Yq95', 
                                    'XY_spread'], axis = 1)
new_data_scaled = scaler.transform(dwel_df_clust)  
predicted_clusters = kmeans.predict(new_data_scaled)
#%%
dwell_df['dwell_cluster'] = predicted_clusters
#%%
#%%
#%%
cluster_dict = {}

grouped_clusters = dwell_df.groupby('dwell_cluster')['dwell_id'].apply(list)

cluster_dict = grouped_clusters.to_dict()

#%%
Tri_dat['dwell_cluster'] = Tri_dat['dwell_count'].map(lambda x: next((cluster for cluster, dwell_ids in cluster_dict.items() if x in dwell_ids), None))

#%%
#%%
dwell_cluster_df = pd.DataFrame(columns=['cluster_id', 'rigs',
                                         'Xu_wt', 'Yu_wt', 'Xmed_wt', 'Ymed_wt',
                                         'max_dist_wt', 
                                         'q95_dist_wt',
                                 'Xlq_wt', 'Xuq_wt', 'Ylq_wt', 'Yuq_wt',
                                 'Xmin_wt', 'Xmax_wt', 'Ymin_wt', 'Ymax_wt',
                                 'Xq05_wt', 'Xq95_wt', 'Yq05_wt', 'Yq95_wt', 
                                 'time_u', 'time_lq', 'time_uq', 'time_min', 'time_max',
                                 'rig_cluster_freq', 'rig_cluster_totaltime', 'rig_cluster_utime'
                                 ])

for cluster_count, group_data in dwell_df_clean.groupby('dwell_cluster'):
    
    total_group_dwell = group_data['time_dwell'].sum()
    weights = group_data['time_dwell']/total_group_dwell
    
    Xu_wt = (group_data['Xu']*weights).sum()
    Yu_wt = (group_data['Yu']*weights).sum()
    
    Xmed_wt = (group_data['Xmed']*weights).sum()
    Ymed_wt = (group_data['Ymed']*weights).sum()
    
    max_dist_wt = (group_data['max_dist']*weights).sum()
    q95_dist_wt = (group_data['q95_dist']*weights).sum()
    
    Xlq_wt = (group_data['Xlq']*weights).sum()
    Xuq_wt = (group_data['Xuq']*weights).sum()
    Ylq_wt = (group_data['Ylq']*weights).sum()
    Yuq_wt = (group_data['Yuq']*weights).sum()
    
    Xmin_wt = (group_data['Xmin']*weights).sum()
    Xmax_wt = (group_data['Xmax']*weights).sum()
    Ymin_wt = (group_data['Ymin']*weights).sum()
    Ymax_wt = (group_data['Ymax']*weights).sum()
    
    Xq05_wt = (group_data['Xq05']*weights).sum()
    Xq95_wt = (group_data['Xq95']*weights).sum()
    Yq05_wt = (group_data['Yq05']*weights).sum()
    Yq95_wt = (group_data['Yq95']*weights).sum()
    
    rigs = list(set(group_data['rig']) )
    
    time_u = group_data['time_dwell'].mean()
    time_lq = np.percentile(group_data['time_dwell'], 25)
    time_uq = np.percentile(group_data['time_dwell'], 75)
    time_min = group_data['time_dwell'].min()
    time_max = group_data['time_dwell'].max()
    
    rig_cluster_freq = []
    rig_cluster_totaltime = []
    rig_cluster_utime = []
    
    for i in range(1,7):
        rig_cluster_freq.append((group_data['rig'] == i).sum() )
        rig_clust_dat = group_data.loc[group_data['rig'] == i]
        if len(rig_clust_dat) > 0:
            rig_cluster_totaltime.append(rig_clust_dat['time_dwell'].sum() ) 
            rig_cluster_utime.append(rig_clust_dat['time_dwell'].mean() ) 
        else:
            rig_cluster_totaltime.append(0 ) 
            rig_cluster_utime.append(0) 
              
    dwell_cluster_df.loc[len(dwell_cluster_df)] = [cluster_count, rigs, Xu_wt, Yu_wt, Xmed_wt, Ymed_wt,
                                     max_dist_wt, q95_dist_wt, 
                                     Xlq_wt, Xuq_wt, Ylq_wt, Yuq_wt,
                                     Xmin_wt, Xmax_wt, Ymin_wt, Ymax_wt,
                                     Xq05_wt, Xq95_wt, Yq05_wt, Yq95_wt,  
                                     time_u, time_lq, time_uq, time_min, time_max,
                                     rig_cluster_freq, rig_cluster_totaltime, rig_cluster_utime]
#%%
#%%
for i in range(1,7):
    
    datasets[f'rig_data{i}'] = Tri_dat.loc[Tri_dat['rig'] == i]
    
#%%
with open('UWB_datasets_12_dwell_clustered.pkl', 'wb') as f:
    
    pickle.dump(datasets, f)
#%%
#%%
dwell_df_clean.to_pickle('dwell_df_clean.pkl')

dwell_cluster_df.to_pickle('dwell_cluster_df.pkl')

Tri_dat.to_pickle('Tri_dat_clustered_12.pkl')
#%%
#%%
#%%

