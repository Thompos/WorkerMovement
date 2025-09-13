# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:22:22 2024

@author: xqb22125
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math as ma
from scipy.stats import wilcoxon
from scipy.stats import linregress
pd.set_option('display.max_columns', 20) 
pd.set_option('display.max_rows', 30)
#%%
Tri_dat = pd.read_pickle('UWB_read.pkl')
#%%
rigs = Tri_dat['rig'].unique()
#%%
Tri_dat['time2'] = pd.to_datetime(Tri_dat['time'])
#%%
#%%
Tri_dat['time_lapsed_all'] = np.nan

for i in rigs:
    temp = (Tri_dat.loc[ Tri_dat.rig == i , 'time2' ] - min(Tri_dat[ 'time2' ])).dt.total_seconds().values
    Tri_dat.loc[ Tri_dat.rig == i , 'time_lapsed_all' ] = temp

#%%
# Plot 'time_lapsed_all'
plt.plot(Tri_dat['time_lapsed_all'], label='time_lapsed_all')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Time')
plt.title('Comparison of time_lapsed_rig and time_lapsed_all')
plt.legend()
#%%
#%%
Tri_dat.set_index('time2', inplace=True)
#%%
datasets = {}

for i in rigs:
    rig_data = Tri_dat.loc[Tri_dat['rig'] == i].copy()
    
    
            
    datasets[f'rig_data{i}'] = rig_data
#%% 
#%%
#%%
for i in rigs:
    
    rig_data = datasets[f'rig_data{i}']
    difsX = np.diff(rig_data['X'])
    difsY = np.diff(rig_data['Y'])
    difs_time = np.diff(rig_data['time_lapsed_all'])
    
    rig_data[ 'difX' ] = np.insert(difsX, 0, np.nan)
    rig_data[ 'difY' ] = np.insert(difsY, 0, np.nan)
        
    rig_data[ 'difSpX' ] = np.insert(difsX, 0, np.nan)/np.insert(difs_time, 0, np.nan)
    rig_data[ 'difSpY' ] = np.insert(difsY, 0, np.nan)/np.insert(difs_time, 0, np.nan)
    rig_data[ 'dif_time' ] = np.insert(difs_time, 0, np.nan)
    
    datasets[f'rig_data{i}'] = rig_data
#%%
for i in rigs:
    
    rig_data = datasets[f'rig_data{i}']

    ed_dif = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['difX'], rig_data['difY'] )  ]
    rig_data['ed_dif'] = ed_dif

    rig_data['difSp'] = ed_dif/rig_data['dif_time' ]
    
    datasets[f'rig_data{i}'] = rig_data

#%%
ma05 = '0.5S'
ma1 = '1S'
ma2 = '2S'
ma3 = '3S'

#%%
def calculate_spread(data):
    centroid_x = data['Xma_1']  # Adjust column names as per your DataFrame
    centroid_y = data['Yma_1']
    distances = np.sqrt((data['X'] - centroid_x)**2 + (data['Y'] - centroid_y)**2)
    spread = np.mean(distances)
    return spread

#%%
for i in rigs:
    rig_data = datasets[f'rig_data{i}']
    
    ma05tempx = rig_data['X'].rolling(ma05, min_periods=1, center = True).mean()
    ma1tempx = rig_data['X'].rolling(ma1, min_periods=1, center = True).mean()
    ma2tempx = rig_data['X'].rolling(ma2, min_periods=1, center = True).mean()
    ma3tempx = rig_data['X'].rolling(ma3, min_periods=1, center = True).mean()

    ma05tempy = rig_data['Y'].rolling(ma05, min_periods=1, center = True).mean()
    ma1tempy = rig_data['Y'].rolling(ma1, min_periods=1, center = True).mean()
    ma2tempy = rig_data['Y'].rolling(ma2, min_periods=1, center = True).mean()
    ma3tempy = rig_data['Y'].rolling(ma3, min_periods=1, center = True).mean()
    
    rig_data['Xma_05'] = ma05tempx.values
    rig_data[ 'Xma_1'] = ma1tempx.values
    rig_data[ 'Xma_2'] = ma2tempx.values
    rig_data[ 'Xma_3'] = ma3tempx.values

    rig_data[ 'Yma_05'] = ma05tempy.values
    rig_data[ 'Yma_1'] = ma1tempy.values
    rig_data[ 'Yma_2'] = ma2tempy.values
    rig_data[ 'Yma_3'] = ma3tempy.values
    
    distances1 = np.sqrt((rig_data['X'] - ma1tempx)**2 + (rig_data['Y'] - ma1tempy)**2)
    distances2 = np.sqrt((rig_data['X'] - ma2tempx)**2 + (rig_data['Y'] - ma2tempy)**2)

    rolling_spread1 = distances1.rolling(ma1, min_periods=1, center = True).mean()
    rolling_spread2 = distances2.rolling(ma2, min_periods=1, center = True).mean()
    
    rig_data[ 'ma_spread1'] = rolling_spread1.values
    rig_data[ 'ma_spread2'] = rolling_spread2.values

    datasets[f'rig_data{i}'] = rig_data
    
#%%
#%%
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = datasets[f'rig_data{i}']
    ax.plot(rig_data['Xma_2'], rig_data['Yma_2'], 
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('Xma_2')
    ax.set_ylabel('Yma_2')

plt.tight_layout()

plt.show()
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = datasets[f'rig_data{i}']
    ax.plot(rig_data['difX'], rig_data['difY'], 
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('diffX')
    ax.set_ylabel('diffY')

plt.tight_layout()

plt.show()
#%%

#%%
import pickle          
#%%
#%%
#%%
with open('UWB_datasets_NO_INT.pkl', 'wb') as f:
    pickle.dump(datasets, f)
#%%
#%%
#%%
#%%
#%%