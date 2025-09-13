# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:37:49 2024

@author: xqb22125
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math as ma

#%%

one_sec_df = pd.read_pickle('UWB_1sec_df.pkl')
two_sec_df = pd.read_pickle('UWB_2sec_df.pkl')

#%%
sns.scatterplot(x='Xu', y='Yu', hue='rig', data=one_sec_df)

#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6))

for rig, ax in zip(range(1,7), axes.flatten()):
    ax.scatter(one_sec_df.loc[one_sec_df.rig == rig  , 'Xu'], 
               one_sec_df.loc[one_sec_df.rig == rig, 'Yu'], s = 5,
               )
    ax.set_title(f'Category {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = one_sec_df[one_sec_df['rig'] == rig]
    ax.plot(rig_data['Xu'], rig_data['Yu'], 
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = one_sec_df[one_sec_df['rig'] == rig]
    ax.plot(rig_data['Xmed'], rig_data['Ymed'], 
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%%
fig, axs = plt.subplots(6, 2, figsize=(11, 6))

for idx, rig_value in enumerate(one_sec_df['rig'].unique()):
    row, col = divmod(idx, 2)  # Calculate the row and column for each subplot
    # Filter the DataFrame for the specific rig
    rig_df = one_sec_df[one_sec_df['rig'] == rig_value]

    ax = axs[int(rig_value-1), 0]
    ax.scatter(rig_df['time_lapsed'], rig_df['ed_u'], label='ed_u', c = 'red' )
    ax = axs[int(rig_value-1), 1]
    ax.scatter(rig_df['time_lapsed'], rig_df['ed_med'], label='ed_med', c = 'green')

    ax.set_title(f'Rig {rig_value}')
    ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

#%%
thresh_med = 1.4

one_sec_df['move_med'] = [float(1) if x > thresh_med else float(0) for x in one_sec_df['ed_med'] ]

plt.scatter(one_sec_df['time_lapsed'], one_sec_df['ed_med'] , s= 8)
plt.axhline(y=thresh_med, color='red', linestyle='--', label='Threshold')

#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = one_sec_df[one_sec_df['rig'] == rig]
    ax.scatter(rig_data['Xu'], rig_data['Yu'],
               c=rig_data['move_med'], cmap='coolwarm', s = 5,
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

#%%
plt.scatter(one_sec_df['Xu'], one_sec_df['Yu'], c=one_sec_df['move_med'],
            cmap='coolwarm', s = 3,
         label=f'Rig {rig}')
plt.show()
#%%

#%%
thresh_u = 0.9

one_sec_df['move_u'] = [float(1) if x > thresh_u else float(0) for x in one_sec_df['ed_u'] ]

plt.scatter(one_sec_df['time_lapsed'], one_sec_df['ed_u'] , s= 8)
plt.axhline(y=thresh_u, color='red', linestyle='--', label='Threshold')

#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = one_sec_df[one_sec_df['rig'] == rig]
    ax.scatter(rig_data['Xu'], rig_data['Yu'],
               c=rig_data['move_u'], cmap='coolwarm', s = 5,
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

#%%
plt.scatter(one_sec_df['Xu'], one_sec_df['Yu'], c=one_sec_df['move_u'],
            cmap='coolwarm', s = 3,
         label=f'Rig {rig}')
plt.show()
#%%
#%%
one_sec_df['sd_xy'] = [x+y for x, y in zip(one_sec_df['sd_X'], one_sec_df['sd_Y']) ] 
#%%
thresh_sdxy = 0.36

one_sec_df['move_sdxy'] = [1.0 if x > thresh_sdxy else 0.0 for x in one_sec_df['sd_xy'] ]

plt.scatter(one_sec_df['time_lapsed'], one_sec_df['sd_xy'] , s= 3)
plt.axhline(y=thresh_sdxy, color='red', linestyle='--', label='Threshold')

#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = one_sec_df[one_sec_df['rig'] == rig]
    ax.scatter(rig_data['Xu'], rig_data['Yu'],
               c=rig_data['move_sdxy'], cmap='coolwarm', s = 5,
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

#%%
plt.scatter(one_sec_df['Xu'], one_sec_df['Yu'], c=one_sec_df['move_sdxy'],
            cmap='coolwarm', s = 3,
         label=f'Rig {rig}')
plt.show()
#%%
#%%
thresh_pval = 0.01
one_sec_df['move_pval_thresh'] = [0.0 if x > thresh_pval else 1.0 for x in one_sec_df['stat_move']]
plt.scatter(one_sec_df['time_lapsed'], one_sec_df['ed_u'] , s = 3,
            c=one_sec_df['move_pval_thresh'] , cmap='coolwarm',)
#%%
plt.scatter(one_sec_df['time_lapsed'], one_sec_df['ed_med'] , s = 3,
            c=one_sec_df['move_pval_thresh'], cmap='coolwarm',)
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = one_sec_df[one_sec_df['rig'] == rig]
    ax.scatter(rig_data['Xu'], rig_data['Yu'],
            c=rig_data['move_pval_thresh'], cmap='coolwarm', s = 5,
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

#%%
plt.scatter(one_sec_df['Xu'], one_sec_df['Yu'], c=one_sec_df['move_pval_thresh'],
            cmap='coolwarm', s = 3,
         label=f'Rig {rig}')
plt.show()
#%%
# one_sec_df['move_pval_thresh'] = [0.0 if x > thresh_pval else 1.0 for x in one_sec_df['stat_move']]
plt.scatter(one_sec_df['time_lapsed'], one_sec_df['ed_u'] , s = 3,
            c=(one_sec_df['slope_difs'])**(0.3) , cmap='coolwarm',)
#%%
plt.scatter(one_sec_df['time_lapsed'], one_sec_df['ed_med'] , s = 3,
            c=(one_sec_df['slope_difs'])**(0.3), cmap='coolwarm',)
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = one_sec_df[one_sec_df['rig'] == rig]
    ax.scatter(rig_data['Xu'], rig_data['Yu'],
            c=rig_data['slope_difs']**(0.15), cmap='coolwarm', s = 5,
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

#%%
plt.scatter(one_sec_df['Xu'], one_sec_df['Yu'], c=(one_sec_df['slope_difs']**(0.15)),
         cmap='coolwarm', s = 3,
         label=f'Rig {rig}')
plt.show()
#%%
#%%
#%%
#%%
one_sec_df.columns
#%%
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = one_sec_df[one_sec_df['rig'] == rig]
    ax.plot(rig_data['time_lapsed'], rig_data['slope_difs'],
 #           c=rig_data['slope_difs']**(0.15), cmap='coolwarm', s = 5,
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

#%%
#%%
plt.scatter(one_sec_df['time_lapsed'], one_sec_df['slope_difs'],
            cmap='coolwarm', s = 3,
         label=f'Rig {rig}')
#%%
plt.hist((one_sec_df['slope_difs'])**(0.3)  ,
   #      bins=range(0, round(max(one_sec_df['slope_difs']) ) + 1),
         edgecolor='black')  # You can adjust the number of bins as needed
#%%
#%%
#%%
#%%
