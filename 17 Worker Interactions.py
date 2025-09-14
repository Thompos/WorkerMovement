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
pred_col = 'pred_26_NV'
#%%
dwell_dat = Tri_dat[Tri_dat[pred_col] < 0.5].copy()
transit_dat = Tri_dat[Tri_dat[pred_col] >= 0.5].copy()

#%%
dwell_dat_TO = dwell_dat.sort_values(by='time_lapsed_all').reset_index(drop= True).copy()
#%%

plt.figure(figsize=(15, 6))
ax = plt.gca()

custom_colors = ['lightcoral', 'chocolate', 'lawngreen', 'forestgreen',
                 'turquoise', 'dodgerblue', 'slategrey', 'blueviolet',
                 'magenta', 'lightpink', 'silver', 'yellow', 'gray', 'black']
# Loop through each set of y-values and plot them
for rig in range(1,7):
    rig_dwell_data = dwell_dat[dwell_dat['rig'] == rig ].copy()
    
    for dwell in rig_dwell_data['dwell_count'].unique().astype(int):
        
        rig_dwell_ID_data = rig_dwell_data[rig_dwell_data['dwell_count'] == dwell ].copy()
        clust = int(rig_dwell_ID_data['dwell_cluster'].iloc[0])
        
        start = rig_dwell_ID_data['time_lapsed_all'].iloc[0]
        end = rig_dwell_ID_data['time_lapsed_all'].iloc[-1]
        
        plt.hlines(rig, start/60, end/60,
                   linestyle='-', color = custom_colors[clust], linewidth = 45 )
    
for second in range(0, ma.ceil(dwell_dat_TO['time_lapsed_all'].max()) ):
               
    time_condition = (dwell_dat_TO['time_lapsed_all'] >= second) & \
                     (dwell_dat_TO['time_lapsed_all'] <= (second+1)) #& \
                     # (dwell_dat_TO['dwell_cluster'] == float(clust) )
    
    same_dwell_times = dwell_dat_TO.loc[time_condition, : ]
    
    count = same_dwell_times.groupby('dwell_cluster')['rig'].nunique()
            
    if len(count[count > 1]) > 0:
        
        plt.plot((second+0.5)/60, 0.2, 'o', color = 'red', markersize = 0.1 )
    else:
        pass

legend_patches = [mpatches.Patch(color=color, label=f'Cluster {i}') for i, color in enumerate(custom_colors)]

# Add the legend to the current axes
plt.gca().legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))


plt.xlabel('Time lapsed (mins)')
plt.ylim([0,7])
plt.ylabel('Worker')
plt.title('')

plt.show()

#%%
#%%
transit_dat_TO = transit_dat.sort_values(by='time_lapsed_all').reset_index(drop= True).copy()
#%%
#%%
#%%
#test = [0,0,0,0,0,0]
plt.figure(figsize=(15, 7.5))
ax = plt.gca()

n_workers = len(Tri_dat['rig'].unique() )

simultanious_transit = pd.DataFrame(np.zeros((n_workers, n_workers )),
                                    index = list(range(1,7)), columns = list(range(1,7)) )

simultanious_transit_path_intersection = pd.DataFrame(np.zeros((n_workers, n_workers )),
                                    index = list(range(1,7)), columns = list(range(1,7)) )

transit_pot_interactions = pd.DataFrame(np.zeros((n_workers, n_workers )),
                                    index = list(range(1,7)), columns = list(range(1,7)) )

simulataneous_transit_pot_interactions = pd.DataFrame(np.zeros((n_workers, n_workers )),
                                    index = list(range(1,7)), columns = list(range(1,7)) )

pinteraction_cols =  [ind for ind, col in enumerate(Tri_dat.columns) if 'pinteraction_w' in col ]
# Loop through each set of y-values and plot them
for rig in range(1,7):
    rig_transit_data = transit_dat[transit_dat['rig'] == rig ].copy()
    
    rig_counts = rig_transit_data['path_count'].unique()
    rig_counts = [int(x) for x in rig_counts if not pd.isna(x) ]
    
    for transit in rig_counts:
        
        rig_transit_ID_data = rig_transit_data[rig_transit_data['path_count'] == transit ].copy()
        
        if pd.isna(rig_transit_ID_data['path_cluster'].iloc[0]):
            pass
        else:
            worker_cluster = int(rig_transit_ID_data['path_cluster'].iloc[0] )
            
            
            start = rig_transit_ID_data['time_lapsed_all'].iloc[0]
            end = rig_transit_ID_data['time_lapsed_all'].iloc[-1]
            #if end - start < 12:
            plt.plot( 0.5*(start+end)/60, rig, 'o', color = 'green', markersize = 0.7 )
            #else:
            plt.hlines(rig, start/60, end/60, linestyle='-', color = 'green', linewidth = 45 )
            
            time_condition = (transit_dat_TO['time_lapsed_all'] >= start) & \
                              (transit_dat_TO['time_lapsed_all'] <= end) 
            same_transit_times = transit_dat_TO.loc[time_condition, : ].copy()
            
            pot_interactions_temp = np.any(rig_transit_ID_data.iloc[ : , pinteraction_cols] > 0, axis=0)
            pot_interactions_temp = [ind for ind, x in enumerate(pot_interactions_temp) if x]
            if len(pot_interactions_temp) > 0:
                transit_pot_interactions.iloc[rig-1, pot_interactions_temp + [rig-1] ] += 1
                plt.plot( 0.5*(start+end)/60, rig + 0.15, 'o', color = 'orange', markersize = 1 )
            
            count = same_transit_times['rig'].value_counts()
            
            if len(count) > 1:
                plt.plot( 0.5*(start+end)/60, rig - 0.15, 'o', color = 'red', markersize = 0.6 )
                transit_workers = list(sorted((same_transit_times['rig']).unique().astype(int) - 1) )
                
                simultanious_transit.iloc[rig-1, transit_workers] += 1
              
                transits = np.unique(same_transit_times['path_cluster'])
                transit_cols = [ind for ind,x in enumerate(intersection_mat.columns) if int(x) in transits ]
               
                worker_col = [ind for ind,x in enumerate(intersection_mat.columns) if int(x) == worker_cluster ]            
              
                temp_mat = intersection_mat.iloc[worker_col, transit_cols]
        
                if temp_mat.sum().sum() > 1:
                    plt.plot(0.5*(start+end)/60, rig - 0.25, 'o', color = 'blue', markersize = 0.7 )
                    
                    cases_bool = np.array(temp_mat) > 0
                    cases_bool = cases_bool.flatten()
                    paths_intersecting = temp_mat.columns[cases_bool]
                    paths_intersecting = [int(x) for x in paths_intersecting]  
                    workers = np.unique([x-1 for x,y in zip(same_transit_times['rig'], same_transit_times['path_cluster']) if y in paths_intersecting])
                    workers = list(np.sort(workers ) )
                    if len(workers) > 1:
                        simultanious_transit_path_intersection.iloc[rig-1, workers] += 1
                
                sim_transit_pot_interactions = [x for x in pot_interactions_temp if x in transit_workers ]
                if len(sim_transit_pot_interactions) > 0:
                    # double transit colli
                   simulataneous_transit_pot_interactions.iloc[rig-1, sim_transit_pot_interactions+[rig-1] ] += 1
                   plt.plot( 0.5*(start+end)/60, rig + 0.3, 'o', color = 'black', markersize = 1.5 )                  
                  # test[rig-1] += 1       
            else:
                pass
        
plt.xlabel('Time lapsed (mins)')
plt.xlim([-1,1+ma.ceil(Tri_dat['time_lapsed_all'].max())/60])
plt.ylim([0,7])
plt.ylabel('Worker')
plt.title('')
#plt.grid(True)
#plt.legend(['Line 1', 'Line 2', 'Line 3'])

# Show the plot
plt.show()
#%%
#%%
#%%
plt.figure(figsize=(12, 7))
sns.heatmap(simultanious_transit, annot=False, cmap="Greens", square=True)

# Manually iterate over columns and rows to add annotations
for i, row in enumerate(simultanious_transit.values):
    for j, val in enumerate(row):
        plt.text(j + 0.5, i + 0.5, "{:d}".format(int(val)),
                 ha="center", va="center", color="red", fontsize=11, fontweight = 'bold')
plt.yticks(rotation = 0)
plt.title('Simulataneous transits')
plt.show()


#%%
#%%
plt.figure(figsize=(12, 7))
sns.heatmap(simultanious_transit_path_intersection, annot=False, cmap="Greens", square=True)

# Manually iterate over columns and rows to add annotations
for i, row in enumerate(simultanious_transit_path_intersection.values):
    for j, val in enumerate(row):
        plt.text(j + 0.5, i + 0.5, "{:d}".format(int(val)),
                 ha="center", va="center", color="red", fontsize=11, fontweight = 'bold')
plt.yticks(rotation = 0)
plt.title('Potential transit path intersections')
plt.show()

#%%
#%%
plt.figure(figsize=(12, 7))
sns.heatmap(transit_pot_interactions, annot=False, cmap="Greens", square=True)

# Manually iterate over columns and rows to add annotations
for i, row in enumerate(transit_pot_interactions.values):
    for j, val in enumerate(row):
        plt.text(j + 0.5, i + 0.5, "{:d}".format(int(val)),
                 ha="center", va="center", color="red", fontsize=11, fontweight = 'bold')
plt.yticks(rotation = 0)
plt.title('Potential interactions of transits')
plt.show()

#%%
plt.figure(figsize=(12, 7))
sns.heatmap(simulataneous_transit_pot_interactions, annot=False, cmap="Greens", square=True)

# Manually iterate over columns and rows to add annotations
for i, row in enumerate(simulataneous_transit_pot_interactions.values):
    for j, val in enumerate(row):
        plt.text(j + 0.5, i + 0.5, "{:d}".format(int(val)),
                 ha="center", va="center", color="red", fontsize=11, fontweight = 'bold')
plt.yticks(rotation = 0)
plt.title('Potential interactions of co-transits')
plt.show()

#%%
#%%
#%%
#%%
cols_to_add = ['X', 'Y', 'rig' ]
sim_transit_location_data = pd.DataFrame(columns = ['X','Y','rig'])
pot_transit_interaction_location_data = pd.DataFrame(columns = ['X','Y','rig'])

# Loop through each second
for second in range(0, ma.ceil(transit_dat_TO['time_lapsed_all'].max()) ):
               
    time_condition = (transit_dat_TO['time_lapsed_all'] >= second) & \
                     (transit_dat_TO['time_lapsed_all'] <= (second+1))
    
    same_transit_time_dat = transit_dat_TO.loc[time_condition, : ].copy()
    
    presence_of_ones_pot = same_transit_time_dat.iloc[:, pinteraction_cols].sum(axis=1) > 0
    pot_data_to_add = same_transit_time_dat.loc[presence_of_ones_pot, cols_to_add]
    if len(pot_data_to_add) > 0:
        pot_transit_interaction_location_data = pd.concat([pot_transit_interaction_location_data, pot_data_to_add ], ignore_index=True)
    
    
    transit_workers = list(sorted((same_transit_time_dat['rig']).unique().astype(int) - 1) )
    int_cols = [pinteraction_cols[ind] for ind, x in enumerate(pinteraction_cols) if ind in transit_workers ]
    
    
    columns = cols_to_add + list(same_transit_times.columns[int_cols])
    temp_dat = same_transit_time_dat[columns].copy()
    temp_pinteraction_cols = [ind for ind, col in enumerate(temp_dat.columns) if 'pinteraction_w' in col ]
    
    presence_of_ones = temp_dat.iloc[:, temp_pinteraction_cols].sum(axis=1) > 0
    data_to_add = temp_dat.loc[presence_of_ones, cols_to_add]
    
    if len(data_to_add) > 0:
        sim_transit_location_data = pd.concat([sim_transit_location_data, data_to_add ], ignore_index=True)
    
            
#%%
#%%
plt.figure(figsize=(9.5, 6))
xmin = Tri_dat['X'].min()
xmax = Tri_dat['X'].max()
ymin = Tri_dat['Y'].min()
ymax = Tri_dat['Y'].max()

plt.scatter(transit_dat_TO['X'], transit_dat_TO['Y'], label=f'Transit', alpha=0.5, color='green')
plt.scatter(pot_transit_interaction_location_data['X'], pot_transit_interaction_location_data['Y'], 
            label=f'Transit', alpha=0.5, marker= '*', color='orange')
plt.scatter(sim_transit_location_data['X'], sim_transit_location_data['Y'], 
            label=f'Transit', alpha=0.5, marker= 'x', color='red')

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Worker transit areas')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

custom_colors = ['green', 'orange', 'red']
states = ['Transit', 'Potential interaction whilst a worker in transit', 'Potential interaction from multiple in transit']
legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(custom_colors, states)]
plt.gca().legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, -0.25))

plt.tight_layout()
plt.show()
#%%
#%%
#%%
#%%
#%%
#%%
#%%