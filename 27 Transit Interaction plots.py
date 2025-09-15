# -*- coding: utf-8 -*-
"""
Created on Fri May  3 08:12:16 2024

@author: 
"""

#%%
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import time
import joblib
import seaborn as sns

pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 35)
#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

exec(open(f'{path_stem}UWB functions.py').read())

#%%
#%%
import pickle
with open('datasets_pred_26_NV.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
%matplotlib inline
#%%
pred_col = 'pred_26_NV'
#%%
# find starts and ends for each worker with each worker, we need to know if there's more than two also
cols_wanted = ['X','Y','time_lapsed_all','rig', pred_col]
# also whether they are dwelling and who is in transit
# time of interactive period
int_dict = {}

for rig in range(1,7):
    
    rig_data = datasets[f'rig_data{rig}'].reset_index(drop = True).copy()
    
    pinteraction_cols = [col for col in rig_data.columns if 'pinteraction_w' in col ]
    columns2select = cols_wanted + pinteraction_cols
    
    for interaction_col in pinteraction_cols:
        
        worker_int = int(interaction_col[-1] )
        
        if rig == worker_int:
            pass
        else:
            worker_int_df = rig_data.loc[rig_data[interaction_col] == 1, columns2select].copy()
            diffs = np.concatenate(([worker_int_df.index[0]], np.diff(worker_int_df.index) )) 
            
            start_ends = np.where(diffs > 1)[0] 
            
            entries = []
            for i in range(len(start_ends)-1):
                 
                entry = {
                    "start": rig_data.loc[worker_int_df.index[start_ends[i]], 'time_lapsed_all' ],
                    "end": rig_data.loc[worker_int_df.index[start_ends[i+1]-1] , 'time_lapsed_all' ],
                    }
                entries.append(entry)
                
            int_dict[f'worker{rig}_worker{worker_int}'] = entries
            
#%%
Tri_dat = pd.concat( datasets.values() )
Tri_dat_TO = Tri_dat.sort_values(by='time_lapsed_all').reset_index(drop= True).copy()

#%%
xmin = min(Tri_dat_TO['X'])
xmax = max(Tri_dat_TO['X'])
ymin = min(Tri_dat_TO['Y'])
ymax = max(Tri_dat_TO['Y'])

transit_colors = ['red', 'lime']

worker_colors = ['royalblue', 'sienna', 'gray', 'lightsalmon', 'indigo', 'm']

transit_markers = ['s','o']

#%%
start = int_dict['worker1_worker2'][0]['start']
end = int_dict['worker1_worker2'][0]['end']
interval_mask = (Tri_dat_TO['time_lapsed_all'] >= start) & (Tri_dat_TO['time_lapsed_all'] <= end)
subset_data = Tri_dat_TO.loc[interval_mask, columns2select].copy()

interact_mask = subset_data[pinteraction_cols].eq(1).any(axis=1)
subset_interact_dat = subset_data.loc[interact_mask]

colors = [worker_colors[int(x)-1] for x in subset_interact_dat['rig']]
edgey_colors = [transit_colors[int(x)] if not np.isnan(x) else 'white' for x in subset_interact_dat[pred_col]]
marker_types = [transit_markers[int(x)] if not np.isnan(x) else 'o' for x in subset_interact_dat[pred_col]]  # Default marker i


plt.figure(figsize=(10, 6))
sns.set_palette(worker_colors)
sns.scatterplot(x='X', y='Y', data=subset_interact_dat,
                edgecolor=edgey_colors,
                c = colors,
#                hue=colors,
                marker='s', s=400, alpha=1, linewidth=0)
sns.set_palette(transit_colors)
sns.scatterplot(x='X', y='Y', data=subset_interact_dat, 
                edgecolor=edgey_colors,
                c = edgey_colors,
               # hue=edgey_colors,
                marker='o', s=18, alpha=1, linewidth=0)

# Set limits and labels
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Time lapsed (mins) {round(start/60,2)} to {round(end/60,2)}')

legend_elements = [Line2D([0], [0], marker='o', color='black', label='Dwelling', markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='o', color='black', label='Transit', markerfacecolor='lime', markersize=10)]

for i, worker_color in enumerate(worker_colors[:6], start=1):
    legend_elements.append(Line2D([0], [0], marker='s', color='w', label=f'Worker {i}',
                                  markerfacecolor=worker_color, markersize=15))

plt.legend(handles=legend_elements)

plt.show()

#%%
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#%%
for rig in range(1,7):
    for worker in range(1,7):   
        
        if rig == worker:
            pass
        else:
            temp_dict = int_dict[f'worker{rig}_worker{worker}']
            
            for i in temp_dict:
                
                start = i['start']
                end = i['end']
                
                interval_mask = (Tri_dat_TO['time_lapsed_all'] >= start) & (Tri_dat_TO['time_lapsed_all'] <= end)
                subset_data = Tri_dat_TO.loc[interval_mask, columns2select].copy()
            
              #  interact_mask = subset_data[pinteraction_cols].eq(1).any(axis=1)
                interact_mask = (subset_data['rig'] == rig) | (subset_data[f'pinteraction_w{rig}'] == 1)
                subset_interact_dat = subset_data.loc[interact_mask]
            
                colors = [worker_colors[int(x)-1] for x in subset_interact_dat['rig']]
                edgey_colors = [transit_colors[int(x)] if not np.isnan(x) else 'white' for x in subset_interact_dat[pred_col]]
                marker_types = [transit_markers[int(x)] if not np.isnan(x) else '' for x in subset_interact_dat[pred_col]] 
             
                plt.figure(figsize=(10, 6))
                
                sns.set_palette(worker_colors)
                sns.scatterplot(x='X', y='Y', data=subset_interact_dat, edgecolor= edgey_colors,
                                c=colors, marker='s', s=400, alpha=1, linewidth=0)
                sns.set_palette(transit_colors)
                sns.scatterplot(x='X', y='Y', data=subset_interact_dat, edgecolor=edgey_colors,
                                c=edgey_colors, marker='o', s=18, alpha=1, linewidth=0)
                
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(f'Time lapsed (mins) {round(start/60,2)} to {round(end/60,2)}')
            
                legend_elements = [Line2D([0], [0], marker='o', color='black',
                                          label='Dwelling', markerfacecolor='red', markersize=10),
                                   Line2D([0], [0], marker='o', color='black',
                                          label='Transit', markerfacecolor='lime', markersize=10)]
            
                for j, worker_color in enumerate(worker_colors[:6], start=1):
                    legend_elements.append(Line2D([0], [0], marker='s', color='w', label=f'Worker {j}',
                                                  markerfacecolor=worker_color, markersize=15))
                    
                plt.legend(handles=legend_elements, loc='lower left')
            
                plt.show()

#%%    
#%%
#%%
#%%
with open('interaction_plots_dict.pkl', 'wb') as f:
    pickle.dump(int_dict, f)
    
#%%
#%%
datasets['rig_data1'].columns
#%%
plt.scatter(datasets['rig_data1']['time_lapsed_all'], datasets['rig_data1']['dif_angle1_abs'],
            s = 1)
plt.show()
#%%
plt.scatter(datasets['rig_data1']['time_lapsed_all'], datasets['rig_data1']['difSp1'],
            s = 1)
plt.show()
#%%
#%%
#%%
#%%
#%%
#%%