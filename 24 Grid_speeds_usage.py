# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:20:16 2024

"""
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
from matplotlib.lines import Line2D

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

exec(open(f'{path_stem}UWB functions.py').read())
#%%
dwell_clusterids_df = pd.read_pickle('dwell_df_clean.pkl')
dwell_cluster_df = pd.read_pickle('dwell_cluster_df.pkl')

path_clusters_df = pd.read_pickle('path_clusters_df.pkl')

#%%
import pickle
with open('UWB_datasets_23with worker_interactions.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
#%%
Tri_dat = pd.concat( datasets.values() )
#%%
transit_dat = Tri_dat[Tri_dat['pred_lstm23_NV'] >= 0.5].copy()
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
        grid_ref_dict[id_counter] = (round(x, 2), round(y, 2))
        id_counter += 1

# Print the grid reference dictionary
for grid_ref, value in grid_ref_dict.items():
    print(grid_ref, value)

#%%
def grid_ref(x, y):
    xg = ma.floor((x - min_x)/grid_spacing )
    yg = ma.ceil((max_y - y)/grid_spacing ) - 1
    return xg, yg
    
#%%
ncols = int((max_x-min_x)/grid_spacing)
nrows = int((max_y-min_y)/grid_spacing)

#%%
#%%
blank_grid = np.zeros((nrows, ncols ))
#%%
# with open('grid_top_speeds_All_dict.pkl', 'rb') as f:
#       grid_top_speeds_All_dict = pickle.load(f)
# with open('grid_mean_speeds_All_dict.pkl', 'rb') as f:
#       grid_mean_speeds_All_dict = pickle.load(f)
# with open('grid_proptime_transit_dict.pkl', 'rb') as f:
#       grid_proptime_transit_dict = pickle.load(f)
#%%
grid_top_speeds_All_dict = {}
grid_mean_speeds_All_dict = {}
grid_proptime_transit_dict = {}
pred_col = 'pred_26_NV'

for grid in grid_ref_dict.keys():
    
    xlower = grid_ref_dict[grid][0]
    xupper = xlower + grid_spacing
    ylower = grid_ref_dict[grid][1]
    yupper = ylower + grid_spacing
    
    condition = (transit_dat['X'] >= xlower) & (transit_dat['X'] < xupper) & (transit_dat['Y'] >= ylower) & (transit_dat['Y'] < yupper)    
    
    if condition.sum() == 0:
        top_speed = 0
        mean_speed = 0
        n_second_transit = 0    
    else:
        grid_dat = transit_dat.loc[condition].copy()
    
        top_speed = max(grid_dat['difSp1'] ) 
        mean_speed = (grid_dat['difSp1'] ).mean() 
    
        n_grid_points = len(grid_dat)
        n_second_transit = 0
        n_accounted4 = 0
        second = 0
        
        while n_accounted4 < n_grid_points:
            
            condition_time = (grid_dat['time_lapsed_all'] >= second) & (grid_dat['time_lapsed_all'] < (second + 1) )
            temp_n = condition_time.sum()
            second += 1
            if temp_n > 0:
                n_second_transit += 1
                n_accounted4 += temp_n
                print(f'{n_accounted4/n_grid_points} - {grid}')
            else:
                pass
        
    grid_top_speeds_All_dict[grid] = top_speed
    grid_mean_speeds_All_dict[grid] = mean_speed
    grid_proptime_transit_dict[grid] = n_second_transit

#%%
blank_grid = np.zeros((nrows, ncols ))
#%%

top_speed_grid = blank_grid.copy()

for grid in grid_ref_dict.keys():
    
     testx = grid_ref_dict[grid][0] + grid_spacing/2
     testy = grid_ref_dict[grid][1] + grid_spacing/2
     
     xg, yg = grid_ref(testx, testy)
     top_speed_grid[yg , xg] = grid_top_speeds_All_dict[grid]
     
# Plot the blank grid
plt.figure(figsize=(11, 7))

max_value = np.max(top_speed_grid)
plt.imshow(top_speed_grid, cmap='Blues', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax = max_value )

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

plt.colorbar(label='Top transit speed (m/s)')
plt.show()

#%%

mean_speed_grid = blank_grid.copy()

for grid in grid_ref_dict.keys():
    
     testx = grid_ref_dict[grid][0] + grid_spacing/2
     testy = grid_ref_dict[grid][1] + grid_spacing/2
     
     xg, yg = grid_ref(testx, testy)
     mean_speed_grid[yg , xg] = grid_mean_speeds_All_dict[grid]
     
# Plot the blank grid
plt.figure(figsize=(11, 7))

max_value = np.max(mean_speed_grid)
plt.imshow(mean_speed_grid, cmap='Blues', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax = max_value )

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

plt.colorbar(label='Mean transit speed (m/s)')
plt.show()
#%%
proptime_transit_grid = blank_grid.copy()

for grid in grid_ref_dict.keys():
    
     testx = grid_ref_dict[grid][0] + grid_spacing/2
     testy = grid_ref_dict[grid][1] + grid_spacing/2
     
     xg, yg = grid_ref(testx, testy)
     proptime_transit_grid[yg , xg] = grid_proptime_transit_dict[grid]
     
# Plot the blank grid
plt.figure(figsize=(11, 7))

max_value = np.max(proptime_transit_grid)
plt.imshow(proptime_transit_grid, cmap='Greens', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax = max_value )

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

plt.colorbar(label='Number of second intervals including a measurement')
plt.show()
#%%
#%%
#%%
transit_time_total = 0
transit_time_workers = [0]*6

total_time_allworkers = 0
total_time_worker = [0]*6

transits = transit_dat['path_count'].unique()
transits = [x for x in transits if not np.isnan(x) ]

for path in transits:
    
    path_dat = transit_dat.loc[transit_dat['path_count'] == path].reset_index(drop= True).copy()
    
    transit_time = max(path_dat['time_lapsed_all']) - min(path_dat['time_lapsed_all'])
    transit_time_total += transit_time
    
    rig = path_dat['rig'].iloc[0]
    
    transit_time_workers[rig-1] += transit_time  
    
for rig in range(1,7):

    total = max(datasets[f'rig_data{rig}']['time_lapsed_all']) - min(datasets[f'rig_data{rig}']['time_lapsed_all'])
    total_time_worker[rig-1] = total
    total_time_allworkers += total
    
    
#%%
print(f'Percentage of working time in transit All workers: {round(transit_time_total/total_time_allworkers, 3)*100} ')
print(f'Percentage of working time in transit for each worker: {[round(100*x/y,2) for x,y in zip(transit_time_workers,total_time_worker)]}')

#%%
with open('grid_top_speeds_All_dict.pkl', 'wb') as f:
     pickle.dump(grid_top_speeds_All_dict, f)
with open('grid_mean_speeds_All_dict.pkl', 'wb') as f:
     pickle.dump(grid_mean_speeds_All_dict, f)
with open('grid_proptime_transit_dict.pkl', 'wb') as f:
     pickle.dump(grid_proptime_transit_dict, f)
#%%     
import pickle
with open('grid_top_speeds_All_dict.pkl', 'rb') as f:
      grid_top_speeds_All_dict = pickle.load(f)
with open('grid_mean_speeds_All_dict.pkl', 'rb') as f:
      grid_mean_speeds_All_dict = pickle.load(f)
with open('grid_proptime_transit_dict.pkl', 'rb') as f:
      grid_proptime_transit_dict = pickle.load(f)
#%%
#%%    
#%%
top_speed_grid = blank_grid.copy()

for grid in grid_ref_dict.keys():
    
     testx = grid_ref_dict[grid][0] + grid_spacing/2
     testy = grid_ref_dict[grid][1] + grid_spacing/2
     
     xg, yg = grid_ref(testx, testy)
     top_speed_grid[yg , xg] = grid_top_speeds_All_dict[grid]
     
# Plot the blank grid
plt.figure(figsize=(11, 7))

max_value = np.max(top_speed_grid)
plt.imshow(top_speed_grid, cmap='Blues', aspect='auto', interpolation='nearest',
           extent=[min_x, max_x, min_y, max_y],
           vmin=0.0, vmax = max_value )

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

plt.colorbar(label='Top transit speed (m/s)')
plt.show()

#%%
#%%    
#%%
# Exploring angles
# overall
angles_diffs = np.diff(Tri_dat['angle'])
angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]
plt.hist(angles_diffs)
plt.title('Angle Differences - full dataset')
plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Frequency')
plt.show()
#%%
angles1 = calculate_angle(Tri_dat['Xdiff_1'].values, Tri_dat['Ydiff_1'].values, Tri_dat['ed_dif_1'].values )
angles_diffs = np.diff(angles1)
angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]
plt.hist(angles_diffs)
plt.title('Angle Differences (smoothed) - full dataset')
plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Frequency')
plt.show()
#%%
#%%
condition = Tri_dat['pred_lstm23_NV'] >= 0.5
# transit
angles_diffs = np.diff(Tri_dat.loc[condition, 'angle'])
angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]
plt.hist(angles_diffs)
plt.title('Angle Differences - All workers in transit')
plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Frequency')
plt.show()
#%%
angles1 = calculate_angle(Tri_dat.loc[condition, 'Xdiff_1'].values, Tri_dat.loc[condition, 'Ydiff_1'].values, Tri_dat.loc[condition, 'ed_dif_1'].values )
angles_diffs = np.diff(angles1)
angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]
plt.hist(angles_diffs)
plt.title('Angle Differences (smoothed) - All workers in transit')
plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Frequency')
plt.show()

#%%
condition = Tri_dat['pred_lstm23_NV'] < 0.5
# transit
angles_diffs = np.diff(Tri_dat.loc[condition, 'angle'])
angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]
plt.hist(angles_diffs)
plt.title('Angle Differences - All workers dwelling')
plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Frequency')
plt.show()
#%%
angles1 = calculate_angle(Tri_dat.loc[condition, 'Xdiff_1'].values, Tri_dat.loc[condition, 'Ydiff_1'].values, Tri_dat.loc[condition, 'ed_dif_1'].values )
angles_diffs = np.diff(angles1)
angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]
plt.hist(angles_diffs)
plt.title('Angle Differences (smoothed) - All workers dwelling')
plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Frequency')
plt.show()
#%%
#%%
fig, axs = plt.subplots(2, 3, figsize=(11, 6))

for rig in range(1,7):
# transit
    rig_data = datasets[f'rig_data{rig}'].copy()
    condition = rig_data['pred_lstm23_NV'] >= 0.5
    angles_diffs = np.diff(rig_data.loc[condition, 'angle'])
    angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]
    i = (rig - 1) // 3  
    j = (rig - 1) % 3   
    axs[i, j].hist(angles_diffs)
    axs[i, j].set_title(f'Angle Differences - worker {rig} in transit')
    axs[i, j].set_xlabel('Angle Difference (degrees)')
    axs[i, j].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
#%%
fig, axs = plt.subplots(2, 3, figsize=(11, 6))

for rig in range(1,7):
# transit
    rig_data = datasets[f'rig_data{rig}'].copy()
    condition = rig_data['pred_lstm23_NV'] >= 0.5
    angles1 = calculate_angle(rig_data.loc[condition, 'Xdiff_1'].values, rig_data.loc[condition, 'Ydiff_1'].values, rig_data.loc[condition, 'ed_dif_1'].values )
    angles_diffs = np.diff(angles1)
    angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]

    i = (rig - 1) // 3  
    j = (rig - 1) % 3   
    axs[i, j].hist(angles_diffs)
    axs[i, j].set_title(f'delta Angle (smoothed) - worker {rig} transit')
    axs[i, j].set_xlabel('Angle Difference (degrees)')
    axs[i, j].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
#%%
fig, axs = plt.subplots(2, 3, figsize=(11, 6))

for rig in range(1,7):
# transit
    rig_data = datasets[f'rig_data{rig}'].copy()
    condition = rig_data['pred_lstm23_NV'] < 0.5
    angles_diffs = np.diff(rig_data.loc[condition, 'angle'])
    angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]
    i = (rig - 1) // 3  
    j = (rig - 1) % 3   
    axs[i, j].hist(angles_diffs)
    axs[i, j].set_title(f'delta Angle - worker {rig} dwelling')
    axs[i, j].set_xlabel('Angle Difference (degrees)')
    axs[i, j].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
#%% 
#%%
fig, axs = plt.subplots(2, 3, figsize=(11, 6))

for rig in range(1,7):
# transit
    rig_data = datasets[f'rig_data{rig}'].copy()
    condition = rig_data['pred_lstm23_NV'] < 0.5
    angles1 = calculate_angle(rig_data.loc[condition, 'Xdiff_1'].values, rig_data.loc[condition, 'Ydiff_1'].values, rig_data.loc[condition, 'ed_dif_1'].values )
    angles_diffs = np.diff(angles1)
    angles_diffs = [x if (x <= 180 and x > -180) else x+360 if x <= -180 else x-360 for x in angles_diffs]

    i = (rig - 1) // 3  
    j = (rig - 1) % 3   
    axs[i, j].hist(angles_diffs)
    axs[i, j].set_title(f'delta Angle (smoothed) - worker {rig} dwelling')
    axs[i, j].set_xlabel('Angle Difference (degrees)')
    axs[i, j].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
#%% 
#%%
%matplotlib qt

xmin = min(Tri_dat['X'])
xmax = max(Tri_dat['X'])
ymin = min(Tri_dat['Y'])
ymax = max(Tri_dat['Y'])
#%%
def speed_animation1(worker_data, speed_col = 'difSp1'):
    
    fig, ax = plt.subplots(figsize=(10, 7))

    step = 0.1
    speed = 0

    worker_color =  np.array(['blue', 'red', 'black', 'darkgreen', 'indigo', 'orangered'])

    columns_needed = ['X','Y','time_lapsed_all', 'difSp1', 'rig', 'difSp']
    data = worker_data.loc[worker_data[speed_col] >= speed, columns_needed].copy()
    colors = data['rig'].values - 1
    ax.scatter(data['X'], data['Y'], marker='o', s=10, c=worker_color[colors], alpha = 0.7)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{round(speed,2)} m/s or more')
    
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=worker,
                             markerfacecolor=color, markersize=10) for worker, 
                          color in zip(range(1,7), worker_color)]
    ax.legend(handles=legend_handles, loc='upper left')
    
    while len(data) > 0:
         
        speed += step
        data = data.loc[data[speed_col] >= speed, columns_needed].copy()
        colors = data['rig'].values - 1
        plt.pause(0.8)
        plt.cla()
        ax.scatter(data['X'], data['Y'], marker='o', s=10, c=worker_color[colors], alpha = 0.7)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        ax.set_title(f'{round(speed,2)} m/s or more')
        ax.legend(handles=legend_handles, loc='upper left')
    
    # plt.show()   
#%%
speed_animation1(Tri_dat, 'difSp1')
#%%
speed_animation1(transit_dat, 'difSp1')
#%%
# ([(x**2+y**2)**0.5 for x, y in zip(Tri_dat['ed_dif'],Tri_dat['dif_time']) ]/Tri_dat['dif_time']).max()

#%%
def speed_animation2(worker_data, speed_col = 'difSp1'):
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    step = 0.5
    low_speed = 0
    upper_speed = low_speed + step
    
    worker_color =  np.array(['blue', 'red', 'black', 'darkgreen', 'indigo', 'orangered'])
    
    
    columns_needed = ['X','Y','time_lapsed_all', 'difSp1', 'rig', 'difSp']
    condition = (worker_data[speed_col] >= low_speed) & (worker_data[speed_col] < upper_speed)
    data = worker_data.loc[condition, columns_needed].copy()
    colors = data['rig'].values - 1
    ax.scatter(data['X'], data['Y'], marker='o', s=12, c=worker_color[colors], alpha = 0.7)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{round(low_speed,1)} to {round(upper_speed,1)} m/s or more')
    
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=worker,
                             markerfacecolor=color, markersize=10) for worker, 
                          color in zip(range(1,7), worker_color)]
    ax.legend(handles=legend_handles, loc='upper left')
    
    while len(data) > 0:
         
        low_speed += step
        upper_speed += step
        condition = (worker_data[speed_col] >= low_speed) & (worker_data['difSp1'] < upper_speed)
        data = worker_data.loc[condition, columns_needed].copy()
        colors = data['rig'].values - 1
        plt.pause(4)
        plt.cla()
        ax.scatter(data['X'], data['Y'], marker='o', s=12, c=worker_color[colors], alpha = 0.7)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        ax.set_title(f'{round(low_speed,1)} to {round(upper_speed,1)} m/s or more')
        ax.legend(handles=legend_handles, loc='upper left')
        
#%%
speed_animation2(Tri_dat)
#%%
speed_animation2(transit_dat)
#%%
#%%
def speed_animation_indi(worker_data, speed_col = 'difSp1'):
    
    step = 0.1
    speed = 0

    worker_color =  np.array(['blue', 'red', 'black', 'darkgreen', 'indigo', 'orangered'])
    
    num_rows = 2
    num_cols = 3
    
    columns_needed = ['X','Y','time_lapsed_all', 'difSp1', 'rig', 'difSp']
    data = worker_data.loc[worker_data[speed_col] >= speed, columns_needed].copy()
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(11, 8))
    for i in range(num_rows):
        for j in range(num_cols):
            rig = i*(num_cols)+(j+1)#1,2, 3
            mask = data['rig']==rig
            colors = data.loc[mask, 'rig'].values - 1
            axes[i, j].scatter(data.loc[mask, 'X'], data.loc[mask, 'Y'], marker='o', s=10, c=worker_color[colors], alpha = 0.7)
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_ylim(ymin, ymax)
            axes[i, j].set_xlabel('X')
            axes[i, j].set_ylabel('Y')
            axes[i, j].set_title(f'{round(speed,2)} m/s or more - (w{rig})')
    
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=worker,
                             markerfacecolor=color, markersize=10) for worker, 
                          color in zip(range(1,7), worker_color)]
   
    while len(data) > 0:
         
        speed += step
        data = data.loc[data[speed_col] >= speed, columns_needed].copy()
        colors = data['rig'].values - 1
        plt.pause(0.8)

        for i in range(num_rows):
            for j in range(num_cols):
                axes[i, j].cla()
                rig = i*(num_cols)+(j+1)
                mask = data['rig']==rig
                colors = data.loc[mask, 'rig'].values - 1
                axes[i, j].scatter(data.loc[mask,'X'], data.loc[mask,'Y'], marker='o', s=10, c=worker_color[colors], alpha = 0.7)
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].set_ylim(ymin, ymax)
        
                axes[i, j].set_title(f'{round(speed,2)} m/s or more - (w{rig})')
         
#%%
speed_animation_indi(Tri_dat)
#%%  
speed_animation_indi(transit_dat)
#%%
#%%  
animation_fun_speed(datasets['rig_data1'], speed_col = 'difSp1')
#%%
#%%
#%%
Tri_dat['t1'] = np.nan
Tri_dat['t2'] = np.nan
Tri_dat['t3'] = np.nan
Tri_dat['t4'] = np.nan
Tri_dat['t5'] = np.nan
Tri_dat['t6'] = np.nan
#%%
#%%
pred_col = Tri_dat.columns.get_loc('pred_lstm23_NV')

for worker in range(1,7):
    
    col_no = Tri_dat.columns.get_loc(f't{worker}')
    
    worker_dat = Tri_dat.loc[Tri_dat['rig'] == worker].reset_index(drop = True).copy()
    worker_lapsed = worker_dat['time_lapsed_all']
    
    for i in range(len(Tri_dat)):
        row = Tri_dat.iloc[i]
        if row['rig'] == worker:
            Tri_dat.iloc[i, col_no] = Tri_dat.iloc[i, pred_col]
        else:
            closest_i = np.argmin(abs(row['time_lapsed_all']-worker_lapsed))
            Tri_dat.iloc[i, col_no] = worker_dat.iloc[closest_i, pred_col]
  
#%%
Tri_dat_TO = Tri_dat.sort_values(by='time_lapsed_all')

#%%
#%%
animation_fun_speed_indi(Tri_dat_TO, speed_col = 'difSp1', pace = 0.5)
#%%
#%%
animation_fun_speed_transit_indi(Tri_dat_TO, speed_col = 'difSp1', pace = 2.5)
#%%
#%%
#%%
for r in range(1,7):
    
    rig_data = Tri_dat.loc[Tri_dat['rig'] == r].copy()
    
    datasets[f'rig_data{r}'] = rig_data
#%%
#%%
with open('UWB_datasets_24.pkl', 'wb') as f:
    pickle.dump(datasets, f)
#%%
#%%
#%%
#%%
#%%
#%%