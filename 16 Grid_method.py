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
import seaborn as sns
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_samples
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_dilation

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

grid_ref_dict = {}

id_counter = 0

for x in np.arange(min_x, max_x, grid_spacing):
    for y in np.arange(min_y, max_y, grid_spacing):
        grid_ref_dict[(round(x, 2), round(y, 2))] = id_counter
        id_counter += 1

# Print the grid reference dictionary
for grid_ref, value in grid_ref_dict.items():
    print(grid_ref, value)

#%%
def grid_ref_fun(x, y):
    xg = ma.floor((x - min_x)/grid_spacing )
    yg = ma.ceil((max_y - y)/grid_spacing ) - 1
    return xg, yg
    
#%%
ncols = int((max_x-min_x)/grid_spacing)
nrows = int((max_y-min_y)/grid_spacing)

#%%
blank_grid = np.zeros((nrows, ncols ))
test_grid = blank_grid.copy()

testx = 3
testy = 1

xg, yg = grid_ref_fun(testx, testy)
test_grid[yg , xg] = 1

# Plot the blank grid
plt.figure(figsize=(11, 7))

plt.imshow(test_grid, cmap='Blues', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=1)

x_vals = np.around(np.arange(min_x, max_x, grid_spacing), 2)
y_vals = np.around(np.flip(np.arange(min_y, max_y, grid_spacing)), 2)  # Reverse y-coordinates

x_labels = [str(int(x)) if x%1 == 0 else '' for x in x_vals]
y_labels = [str(int(y)) if y%1 == 0 else '' for y in y_vals]

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel(' (X)')
plt.ylabel(' (Y)')
plt.title('')

plt.colorbar(label='Presence')
plt.show()

#%%
#%%
dwell_grid = blank_grid.copy()

dwell_dat = Tri_dat[Tri_dat['cleaned_pred'] < 0.5].copy()

# check which grid positions are used for dweeling
for x,y in zip(dwell_dat['X'], dwell_dat['Y'] ):
    
    xg, yg = grid_ref_fun(x, y)
    dwell_grid[yg, xg] = 1

plt.figure(figsize=(11, 7))

plt.imshow(dwell_grid, cmap='Reds', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=1)

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel(' (X)')
plt.ylabel(' (Y)')
plt.title('Overall dwelling areas')

plt.colorbar(label='Presence')
plt.show()


#%%
transit_grid = blank_grid.copy()

transit_dat = Tri_dat[Tri_dat['cleaned_pred'] >= 0.5].copy()

#
for x,y in zip(transit_dat['X'], transit_dat['Y'] ):
    
    xg, yg = grid_ref_fun(x, y)
    transit_grid[yg, xg] = 1

plt.figure(figsize=(11, 7))

plt.imshow(transit_grid, cmap='Greens', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=1)

x_vals = np.around(np.arange(min_x, max_x, grid_spacing), 2)
y_vals = np.around(np.flip(np.arange(min_y, max_y, grid_spacing)), 2)  # Reverse y-coordinates

x_labels = [str(int(x)) if x%1 == 0 else '' for x in x_vals]
y_labels = [str(int(y)) if y%1 == 0 else '' for y in y_vals]

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel(' (X)')
plt.ylabel(' (Y)')
plt.title('Overall Transit areas')

plt.colorbar(label='Presence')
plt.show()

#%%
#%%

comb_grid = transit_grid + 2*dwell_grid

custom_colors = ['white', 'green', 'red', 'blue']

cmap = ListedColormap(custom_colors)

plt.figure(figsize=(11, 7))

plt.imshow(comb_grid, cmap=cmap, aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=3)

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel(' (X)')
plt.ylabel(' (Y)')
plt.title('Overalll Dwelling, transit or both areas')

legend_patches = [
    mpatches.Patch(color='white', label='No activity'),
    mpatches.Patch(color='green', label='Transit only'),
    mpatches.Patch(color='red', label='Dwelling only'),
    mpatches.Patch(color='blue', label='Dwelling & Transit')
]

#plt.legend(handles=legend_patches, title='Key', loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(handles=legend_patches, title='Values', loc='center left', bbox_to_anchor=(1, 0.5),
           frameon=True, edgecolor='black', facecolor='lightgrey')


#plt.colorbar(label='Presence')
plt.show()


#%%
pred_col = 'pred_26_NV'
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6))

rig_dwell_grids = {}
rig_transit_grids = {}

for rig, ax in zip(range(1,7), axes.flatten()):

    rig_data = datasets[f'rig_data{rig}']
    
    rig_dwell_grid = blank_grid.copy()
    rig_transit_grid = blank_grid.copy()
    
    condition_dwell = (rig_data[pred_col] < 0.5)
    condition_transit = (rig_data[pred_col] >= 0.5)

    rig_dwell_dat = rig_data[condition_dwell].copy()
    rig_transit_dat = rig_data[condition_transit].copy()
    
    for x,y in zip(rig_dwell_dat['X'], rig_dwell_dat['Y'] ): 
         xg, yg = grid_ref_fun(x, y)
         rig_dwell_grid[yg, xg] = 1
        
    for x,y in zip(rig_transit_dat['X'], rig_transit_dat['Y'] ):
         xg, yg = grid_ref_fun(x, y)
         rig_transit_grid[yg, xg] = 1

    rig_comb_grid = rig_transit_grid + 2*rig_dwell_grid

    ax.imshow(rig_comb_grid, cmap=cmap, aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=3)

    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)

    ax.set_xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
    ax.set_yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
    ax.set_xlabel(' (X)')
    ax.set_ylabel(' (Y)')
    ax.set_title('')
    
    rig_dwell_grids[rig] = rig_dwell_grid
    rig_transit_grids[rig] = rig_transit_grid

legend_patches = [
        mpatches.Patch(color='white', label='No activity'),
        mpatches.Patch(color='green', label='Transit only'),
        mpatches.Patch(color='red', label='Dwelling only'),
        mpatches.Patch(color='blue', label='Dwelling & Transit')
        ]

plt.legend(handles=legend_patches, title='Values', loc='center left', bbox_to_anchor=(1, 1.1),
           frameon=True, edgecolor='black', facecolor='lightgrey')

plt.show()

#%%
#%%
comb_dwell_grid = blank_grid.copy()
for rig in range(1, 7):
    comb_dwell_grid += rig_dwell_grids[rig]

plt.figure(figsize=(11, 7))

plt.imshow(comb_dwell_grid, cmap='Reds', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=6)

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dwelling areas - Number of workers using area for dwelling')

plt.colorbar(label='No. of different workers')

plt.show()

#%%
comb_transit_grid = blank_grid.copy()

for rig in range(1, 7):
    comb_transit_grid += rig_transit_grids[rig]

plt.figure(figsize=(11, 7))

plt.imshow(comb_transit_grid, cmap='Greens', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=6)

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Transit areas - no. of workers using as transit area')

plt.colorbar(label='No. of different workers')

plt.show()
#%%
#%%
comb_comb_grid = blank_grid.copy()

for rig in range(1, 7):
    comb_comb_grid += ((rig_transit_grids[rig] + rig_dwell_grids[rig]) > 0).astype(int)

plt.figure(figsize=(11, 7))

plt.imshow(comb_comb_grid, cmap='Blues', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=6)

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Overall Workers areas - no. of workers using the area')

plt.colorbar(label='No. of different workers')

plt.show()
#%%
#%%
# number of dwell zones
#%% # come back here
dwell_indi_grids = {}
dwell_indi_grids2 = {}
dwell_indi_times = {}

custom_colors = ['lightcoral', 'chocolate', 'lawngreen', 'forestgreen',
                 'turquoise', 'dodgerblue', 'slategrey', 'blueviolet',
                 'magenta', 'lightpink', 'silver', 'yellow', 'firebrick', 'grey']

for dwell in dwell_dat['dwell_count'].unique().astype(int):
    
    condition = dwell_dat['dwell_count'] == dwell
    temp_dwell = dwell_dat[condition].copy()
    
    clust = int(temp_dwell['dwell_cluster'].iloc[0])
    
    temp_dwell_grid = blank_grid.copy()
    temp_dwell_grid2 = blank_grid.copy()
      
    for x,y in zip(temp_dwell['X'], temp_dwell['Y'] ): 
         xg, yg = grid_ref_fun(x, y)
         temp_dwell_grid[yg, xg] += 1
         temp_dwell_grid2[yg, xg] = 1
         
    plt.figure(figsize=(11, 7))
    
    # cols = ['white', custom_colors[clust]]
    cmap = LinearSegmentedColormap.from_list('custom_colormap', [(1, 1, 1), custom_colors[clust]])
    # cmap = ListedColormap(cols)

    plt.imshow(temp_dwell_grid, cmap=cmap, aspect='auto', interpolation='nearest',
              extent=[min_x, max_x, min_y, max_y],
              vmin=0.0, vmax=temp_dwell_grid.max())

    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
 
    plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
    plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
    plt.xlabel(' (X)')
    plt.ylabel(' (Y)')
    plt.title(f'{dwell}')
    
    plt.colorbar(label='No. of dwelling readings')
 
    plt.show()
    
    dwell_indi_grids[dwell] = temp_dwell_grid
    dwell_indi_grids2[dwell] = temp_dwell_grid2
    dwell_indi_times[dwell] = temp_dwell['time_lapsed_all'].iloc[-1] - temp_dwell['time_lapsed_all'].iloc[0]
    
#%%
with open('dwell_indi_grids.pkl', 'wb') as f:
    pickle.dump(dwell_indi_grids, f)
    
with open('dwell_indi_times.pkl', 'wb') as f:
    pickle.dump(dwell_indi_times, f)
#%%
dwell_count_grid = blank_grid.copy()
dwell_time_grid = blank_grid.copy()

for dwell in dwell_dat['dwell_count'].unique().astype(int):
    
    dwell_count_grid += dwell_indi_grids[dwell]
    dwell_time_grid += dwell_indi_grids2[dwell]*dwell_indi_times[dwell]
    
#%%
plt.figure(figsize=(11, 7))

plt.imshow(dwell_count_grid, cmap='Reds', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=dwell_count_grid.max() )

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('No of Dwell readings')

plt.colorbar(label='No. of different dwelling periods')

plt.show()
#%%
plt.figure(figsize=(11, 7))

plt.imshow(dwell_time_grid/60, cmap='Reds', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=dwell_time_grid.max()/60 )

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dwelling periods - Total time (mins)')

plt.colorbar(label='Total time (mins)')

plt.show()
#%%
#%%
transit_readingscount_grid = blank_grid.copy()

for x,y in zip(transit_dat['X'], transit_dat['Y'] ):
    
    xg, yg = grid_ref_fun(x, y)
    transit_readingscount_grid[yg, xg] += 1

plt.figure(figsize=(11, 7))

plt.imshow(transit_readingscount_grid, cmap='Greens', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=transit_readingscount_grid.max() )

plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel(' (X)')
plt.ylabel(' (Y)')
plt.title('Overall Transit - number of readings')

plt.colorbar(label='Number of readings')
plt.show()
dwell_clust_grids2 = {}

#%%
#%% # individual path transits
transit_indi_grids = {}

for path in sorted(np.unique(transit_dat['path_count']) ):
    
    condition = transit_dat['path_count'] == path
    temp_path = transit_dat[condition].copy()
    
    temp_transit_grid = blank_grid.copy()
      
    for x,y in zip(temp_path['X'], temp_path['Y'] ): 
         xg, yg = grid_ref_fun(x, y)
         temp_transit_grid[yg, xg] = 1
         
    plt.figure(figsize=(11, 7))

    plt.imshow(temp_transit_grid, cmap='Greens', aspect='auto', interpolation='nearest',
              extent=[min_x, max_x, min_y, max_y],
              vmin=0.0, vmax=temp_transit_grid.max())

    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
 
    plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
    plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
    plt.xlabel(' (X)')
    plt.ylabel(' (Y)')
    plt.title(f'{path}')
    
    plt.colorbar(label='No. of transit readings')
 
    plt.show()
    
    transit_indi_grids[path] = temp_transit_grid
    
#%%
with open('transit_indi_grids.pkl', 'wb') as f:
    pickle.dump(transit_indi_grids, f)
#%%
transit_clust_grids = {}
transit_clust_n_occur = []

path_clusters = np.sort(np.unique(transit_dat['path_cluster']) )
path_clusters = [str(int(x)) for x in path_clusters if not np.isnan(x)]

for path_clust in path_clusters:
    
    condition = transit_dat['path_cluster'] == int(path_clust)
    temp_path_clust = transit_dat[condition].copy()
    
    transit_clust_n_occur.append(len(temp_path_clust['path_count'].unique() ) )
        
    temp_path_grid = blank_grid.copy()
      
    for x,y in zip(temp_path_clust['X'], temp_path_clust['Y'] ): 
         xg, yg = grid_ref_fun(x, y)
         temp_path_grid[yg, xg] = 1
         
    plt.figure(figsize=(11, 7))
    
    plt.imshow(temp_path_grid, cmap='Greens', aspect='auto', interpolation='nearest',
              extent=[min_x, max_x, min_y, max_y],
              vmin=0.0, vmax=temp_path_grid.max() )

    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
 
    plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
    plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Path cluster representations - {path_clust}')
 
    plt.colorbar(label='Presence')
    plt.show()
    
    transit_clust_grids[path_clust] = temp_path_grid
    
#%%
#%%
#%%
transit_clust_grids_padded = {}

structure = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])

for path_clust in path_clusters:
        
       temp_grid = (transit_clust_grids[path_clust] > 0)*1
        
       padded_path_grid = binary_dilation(temp_grid, structure)
       
       plt.figure(figsize=(12, 7))
       plt.imshow(padded_path_grid, cmap='Greens', aspect='auto', interpolation='nearest',
                 extent=[min_x, max_x, min_y, max_y],
                 vmin=0.0, vmax=padded_path_grid.max() )

       plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

       plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
       plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
       plt.xlabel('X')
       plt.ylabel('Y')
       plt.title(f'Path cluster representations - {path_clust}')

       plt.colorbar(label='Presence')
       plt.show()

       transit_clust_grids_padded[path_clust] = padded_path_grid
#%%
with open('transit_clust_grids_padded.pkl', 'wb') as f:
    pickle.dump(transit_clust_grids_padded, f)

#%%
#%% doing the same for path count (individual transits)
transit_indi_grids_padded = {}

path_ids = np.sort(np.unique(transit_dat['path_count']) )
path_ids = [str(int(x)) for x in path_ids if not np.isnan(x)]

for transit in path_ids:
        
       temp_grid = (transit_indi_grids[int(transit)] > 0)*1
        
       padded_path_grid = binary_dilation(temp_grid, structure)
       
       plt.figure(figsize=(12, 7))
       plt.imshow(padded_path_grid, cmap='Greens', aspect='auto', interpolation='nearest',
                 extent=[min_x, max_x, min_y, max_y],
                 vmin=0.0, vmax=padded_path_grid.max() )

       plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)

       plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
       plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
       plt.xlabel('X')
       plt.ylabel('Y')
       plt.title(f'Transit representations - {transit}')

       plt.colorbar(label='Presence')
       plt.show()

       transit_indi_grids_padded[transit] = padded_path_grid    
#%%
with open('transit_indi_grids_padded.pkl', 'wb') as f:
     pickle.dump(transit_indi_grids_padded, f)
#%%
#%%
# forming some useful matrices for interactions
intersection_mat = pd.DataFrame(np.zeros((len(transit_clust_grids_padded), len(transit_clust_grids_padded) )),
                                index = path_clusters, columns = path_clusters )

path_prop_similarity_mat = pd.DataFrame(np.zeros((len(transit_clust_grids_padded),
                                                       len(transit_clust_grids_padded) )),
                                               index = path_clusters, columns = path_clusters )

for path_clust in path_clusters:
    
    row_clust = transit_clust_grids_padded[path_clust]
    
    for comparison_clust in path_clusters:
        
        col_clust = transit_clust_grids_padded[comparison_clust]
        n_common = ((row_clust != 0) & (col_clust != 0)).sum()
        
        if n_common > 0:    
            intersection_mat.loc[path_clust, comparison_clust] = 1
            path_total = row_clust.sum()          
            path_prop_similarity_mat.loc[path_clust, comparison_clust] = n_common/path_total
        else:
            pass
#%%
path_indis = np.sort(np.unique(transit_dat['path_count']) )
path_indis = [str(int(x)) for x in path_indis if not np.isnan(x)]
# forming useful matrices for proportion of common grid squares
path_propunion_similarity_mat = pd.DataFrame(np.zeros((len(transit_indi_grids_padded),
                                                       len(transit_indi_grids_padded) )),
                                               index = path_indis, columns = path_indis )

for transit in path_indis:
    
    row_clust = transit_indi_grids_padded[transit]
    
    for comparison_transit in path_indis:
        
        col_clust = transit_indi_grids_padded[comparison_transit]
        n_common = ((row_clust != 0) & (col_clust != 0)).sum()
        
        if n_common > 0:
            n_total = ((row_clust != 0) | (col_clust != 0)).sum()
            path_propunion_similarity_mat.loc[transit, comparison_transit] = n_common/n_total
        else:
            pass

#%%
# Plot the heatmap
sns.set(style="white")  
plt.figure(figsize=(12, 8))  # Set figure size
sns.heatmap(intersection_mat, annot=False, cmap="binary", square=True
          )
plt.xlabel("Pathway cluster ID")  # Set x-axis label
plt.ylabel("Pathway cluster ID")  # Set y-axis label
plt.title("Pathway Intersections")
plt.show()
#%%
sns.set(style="white")  
plt.figure(figsize=(12, 8))  # Set figure size
sns.heatmap(path_propunion_similarity_mat, annot=False, cmap="inferno", square=True
          )
plt.xlabel("Pathway cluster ID")  # Set x-axis label
plt.ylabel("Pathway cluster ID")  # Set y-axis label
plt.title("Pathway ")
plt.show()
#%%
sns.set(style="white")  
plt.figure(figsize=(12, 8))  # Set figure size
sns.heatmap(path_propunion_similarity_mat, annot=False, cmap="binary", square=True
          )
plt.xlabel("Pathway cluster ID")  # Set x-axis label
plt.ylabel("Pathway cluster ID")  # Set y-axis label
plt.title("Pathway ")
plt.show()

#%%
with open('path_prop_similarity_mat.pkl', 'wb') as f:
    pickle.dump(path_prop_similarity_mat, f)
    
with open('path_propunion_similarity_mat.pkl', 'wb') as f:
    pickle.dump(path_propunion_similarity_mat, f)  
    
with open('intersection_mat.pkl', 'wb') as f:
     pickle.dump(intersection_mat, f)  
     
#%%
very_similar_thresh = 0.75
very_similar_clusts = []

for ind, path_clust in enumerate(path_clusters):
    
    condition = ((path_prop_similarity_mat.iloc[:, ind] > very_similar_thresh) & (path_prop_similarity_mat.iloc[ind, :] > very_similar_thresh) )
    similar = [[path_clust, path_clusters[i] ] for i, select in enumerate(condition) if (select and (path_clust != path_clusters[i]) )]
    if len(similar) > 0:
        very_similar_clusts.append(similar)
    
#%%

for pair in very_similar_clusts:
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    
    grid1 = transit_clust_grids_padded[pair[0][0]]
    grid2 = transit_clust_grids_padded[pair[0][1]]
    
    ax = axs[0]
    im1 = ax.imshow(grid1, cmap='Blues', aspect='auto', interpolation='nearest',
                  extent=[min_x, max_x, min_y, max_y],
                  vmin=0.0, vmax=grid1.max())
    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(min_x, max_x, grid_spacing))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(min_y, max_y, grid_spacing))
    ax.set_yticklabels(y_labels[::-1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Transit areas - no. of workers using as transit area')

    ax = axs[1]
    im2 = ax.imshow(grid2, cmap='Oranges', aspect='auto', interpolation='nearest',
                  extent=[min_x, max_x, min_y, max_y],
                  vmin=0.0, vmax=grid2.max())

    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(min_x, max_x, grid_spacing))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(min_y, max_y, grid_spacing))
    ax.set_yticklabels(y_labels[::-1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Transit areas - no. of workers using as transit area')

    plt.show()
#%%

very_similar_thresh = 0.7
very_similar_transits = []

for ind, transit in enumerate(path_indis):
    
    condition = ((path_propunion_similarity_mat.iloc[:, ind] > very_similar_thresh) & (path_propunion_similarity_mat.iloc[ind, :] > very_similar_thresh) )
    similar = [[transit, path_indis[i] ] for i, select in enumerate(condition) if (select and (transit != path_indis[i]) )]
    if len(similar) > 0:
        very_similar_transits.append(similar)
    
#%%

for pair in very_similar_transits:
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    
    grid1 = transit_indi_grids_padded[pair[0][0]]
    grid2 = transit_indi_grids_padded[pair[0][1]]
    
    ax = axs[0]
    im1 = ax.imshow(grid1, cmap='Blues', aspect='auto', interpolation='nearest',
                  extent=[min_x, max_x, min_y, max_y],
                  vmin=0.0, vmax=grid1.max())
    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(min_x, max_x, grid_spacing))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(min_y, max_y, grid_spacing))
    ax.set_yticklabels(y_labels[::-1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Transit areas - no. of workers using as transit area')

    ax = axs[1]
    im2 = ax.imshow(grid2, cmap='Oranges', aspect='auto', interpolation='nearest',
                  extent=[min_x, max_x, min_y, max_y],
                  vmin=0.0, vmax=grid2.max())

    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(min_x, max_x, grid_spacing))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(min_y, max_y, grid_spacing))
    ax.set_yticklabels(y_labels[::-1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Exploring similar transits')

    plt.show()
#%%
transit_clust_n_occur = np.array(transit_clust_n_occur)
condition = transit_clust_n_occur > 5
common_paths = [path_clusters[i] for i, select in enumerate(condition) if select ]

for path_clust in common_paths:
    
    plt.figure(figsize=(12, 7))
    
    grid = transit_clust_grids_padded[path_clust]
      
    plt.imshow(grid, cmap='Blues', aspect='auto', interpolation='nearest',
                  extent=[min_x, max_x, min_y, max_y],
                  vmin=0.0, vmax=grid.max())
    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
    plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Path cluster representations - {path_clust}')

    plt.colorbar(label='Presence')
    plt.show()

    plt.show() 
#%%
#%%
#%%
grid_spacing = 0.25  # Grid spacing
min_x = ma.floor(min(Tri_dat['X']) ) - grid_spacing
max_x = ma.ceil(max(Tri_dat['X']) ) + grid_spacing
min_y = ma.floor(min(Tri_dat['Y']) ) - grid_spacing
max_y = ma.ceil(max(Tri_dat['Y']) ) + grid_spacing

grid_ref_dict = {}

id_counter = 0

for x in np.arange(min_x, max_x, grid_spacing):
    for y in np.arange(min_y, max_y, grid_spacing):
        grid_ref_dict[(round(x, 2), round(y, 2))] = id_counter
        id_counter += 1

# Print the grid reference dictionary
for grid_ref, value in grid_ref_dict.items():
    print(grid_ref, value)
    
    
ncols = int((max_x-min_x)/grid_spacing)
nrows = int((max_y-min_y)/grid_spacing)
#%%
blank_grid = np.zeros((nrows, ncols ))
test_grid = blank_grid.copy()

testx = -6
testy = 2

xg, yg = grid_ref_fun(testx, testy)
test_grid[yg , xg] = 1

# up
xg, yg = grid_ref_fun(testx, testy+grid_spacing)
test_grid[yg, xg] += 1
# down
xg, yg = grid_ref_fun(testx, testy-grid_spacing)
test_grid[yg, xg] += 1
# right
xg, yg = grid_ref_fun(testx+grid_spacing, testy)
test_grid[yg, xg] += 1
# left
xg, yg = grid_ref_fun(testx-grid_spacing, testy)
test_grid[yg, xg] += 1
# upright
xg, yg = grid_ref_fun(testx+grid_spacing, testy+grid_spacing)
test_grid[yg, xg] += 1
# upleft
xg, yg = grid_ref_fun(testx-grid_spacing, testy+grid_spacing)
test_grid[yg, xg] += 1
# downright
xg, yg = grid_ref_fun(testx+grid_spacing, testy-grid_spacing)
test_grid[yg, xg] += 1
# downleft
xg, yg = grid_ref_fun(testx-grid_spacing, testy-grid_spacing)
test_grid[yg, xg] += 1


# Plot the blank grid
plt.figure(figsize=(11, 7))

plt.imshow(test_grid, cmap='Blues', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax=1)

x_vals = np.around(np.arange(min_x, max_x, grid_spacing), 2)
y_vals = np.around(np.flip(np.arange(min_y, max_y, grid_spacing)), 2)  # Reverse y-coordinates

x_labels = [str(int(x)) if x%1 == 0 else '' for x in x_vals]
y_labels = [str(int(y)) if y%1 == 0 else '' for y in y_vals]

plt.grid(which='both', color='black', linestyle='-', linewidth=0.35)

plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
plt.xlabel(' (X)')
plt.ylabel(' (Y)')
plt.title('')

plt.colorbar(label='Presence')
plt.show()

#%%

dwell_indi_padded_grids = {}

for dwell in dwell_dat['dwell_count'].unique().astype(int):
    
    condition = dwell_dat['dwell_count'] == dwell
    temp_dwell = dwell_dat[condition].copy()
    
    temp_dwell_grid = blank_grid.copy()
      
    for x,y in zip(temp_dwell['X'], temp_dwell['Y'] ): 
         xg, yg = grid_ref_fun(x, y)
         temp_dwell_grid[yg, xg] += 1
         # up
         xg, yg = grid_ref_fun(x, y+grid_spacing)
         temp_dwell_grid[yg, xg] += 1
         # down
         xg, yg = grid_ref_fun(x, y-grid_spacing)
         temp_dwell_grid[yg, xg] += 1
         # right
         xg, yg = grid_ref_fun(x+grid_spacing, y)
         temp_dwell_grid[yg, xg] += 1
         # left
         xg, yg = grid_ref_fun(x-grid_spacing, y)
         temp_dwell_grid[yg, xg] += 1
         # upright
         xg, yg = grid_ref_fun(x+grid_spacing, y+grid_spacing)
         temp_dwell_grid[yg, xg] += 1
         # upleft
         xg, yg = grid_ref_fun(x-grid_spacing, y+grid_spacing)
         temp_dwell_grid[yg, xg] += 1
         # downright
         xg, yg = grid_ref_fun(x+grid_spacing, y-grid_spacing)
         temp_dwell_grid[yg, xg] += 1
         # downleft
         xg, yg = grid_ref_fun(x-grid_spacing, y-grid_spacing)
         temp_dwell_grid[yg, xg] += 1
   
         
    plt.figure(figsize=(11, 7))

    plt.imshow(temp_dwell_grid, cmap='Reds', aspect='auto', interpolation='nearest',
              extent=[min_x, max_x, min_y, max_y],
              vmin=0.0, vmax=temp_dwell_grid.max())

    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
 
    plt.xticks(np.arange(min_x, max_x, grid_spacing), x_labels)
    plt.yticks(np.arange(min_y, max_y, grid_spacing), y_labels[::-1])
    plt.xlabel(' (X)')
    plt.ylabel(' (Y)')
    plt.title(f'{dwell}')
    
    plt.colorbar(label='No. of dwelling readings')
 
    plt.show()
    
    dwell_indi_padded_grids[dwell] = temp_dwell_grid
  
#%%
with open('dwell_indi_padded_grids.pkl', 'wb') as f:
    pickle.dump(dwell_indi_padded_grids, f)

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
  # if loading the grids !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
with open('dwell_indi_grids.pkl', 'rb') as f:
    dwell_indi_grids = pickle.load(f)
    
with open('dwell_indi_times.pkl', 'rb') as f:
    dwell_indi_times = pickle.load(f)
    
with open('transit_indi_grids.pkl', 'rb') as f:
    transit_indi_grids = pickle.load(f)

with open('transit_clust_grids_padded.pkl', 'rb') as f:
    transit_clust_grids_padded = pickle.load(f)
    
with open('transit_indi_grids_padded.pkl', 'rb') as f:
     transit_indi_grids_padded = pickle.load(f)
    
#%%