# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:11:43 2024

@author: 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math as ma
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 25) 
#%%

# Tri_dat = pd.read_pickle('UWB_datasets_NO_INT.pkl')
#%%
import pickle

with open('UWB_datasets_NO_INT.pkl', 'rb') as f:
    datasets = pickle.load(f)

#%%
rig_data = datasets['rig_data1']

plt.plot(rig_data.time_lapsed_all, abs(rig_data.difSpX)/5, linewidth  = 0.7 )
plt.plot(rig_data.time_lapsed_all, abs(rig_data.difSpY)/5, linewidth  = 0.7 )
plt.plot(rig_data.time_lapsed_all, abs(rig_data.ma_spread1), linewidth  = 0.7 )

plt.show()
#%%
def calculate_spread(data):
    try:
        centroid_x = data['X'].mean()
        centroid_y = data['Y'].mean()
        distances = np.sqrt((data['X'] - centroid_x)**2 + (data['Y'] - centroid_y)**2)
        spread = np.mean(distances)
    except:
        spread = np.nan
    return spread
#%%

ma05 = '0.5S'
ma1 = '1S'
ma2 = '2S'
ma3 = '3S'

#%%
for i in range(1,7):
    
    rig_data = datasets[f'rig_data{i}']

    rig_data[ 'Xdiff_ma05'] = rig_data['difX'].rolling(ma05, min_periods=1).sum().values
    rig_data[ 'Ydiff_ma05'] = rig_data['difY'].rolling(ma05, min_periods=1).sum().values
    
    rig_data['Xdiff_ma1'] = rig_data['difX'].rolling(ma1, min_periods=1).sum().values
    rig_data['Ydiff_ma1'] = rig_data['difY'].rolling(ma1, min_periods=1).sum().values

    rig_data[ 'Xdiff_ma2'] = rig_data['difX'].rolling(ma2, min_periods=2).sum().values
    rig_data[ 'Ydiff_ma2'] = rig_data['difY'].rolling(ma2, min_periods=2).sum().values
    
    rig_data[ 'Xdiff_ma3'] = rig_data['difX'].rolling(ma3, min_periods=2).sum().values
    rig_data[ 'Ydiff_ma3'] = rig_data['difY'].rolling(ma3, min_periods=2).sum().values
    
    rig_data[ 'ed_XYma05'] = rig_data['ed_dif'].rolling(ma05, min_periods=1).sum().values
    rig_data[ 'ed_XYma1'] = rig_data['ed_dif'].rolling(ma1, min_periods=1).sum().values
    rig_data[ 'ed_XYma2'] = rig_data['ed_dif'].rolling(ma2, min_periods=2).sum().values     
    rig_data[ 'ed_XYma3'] = rig_data['ed_dif'].rolling(ma3, min_periods=2).sum().values
    
    #           
    start_time =  rig_data.index.min()
    end_time = rig_data.index.max()
    p1 = pd.Timedelta(seconds=1)
    p3 = pd.Timedelta(seconds=3)
    spreadsp1 = []
    spreadsp3 = []
    spreadsf1 = []
    spreadsf3 = []
    
    for j in range(len(rig_data)):
        timestamp = rig_data.index[j]
        data_in_p1 = rig_data.loc[timestamp - p1 : timestamp ]
        spreadsp1.append(calculate_spread(data_in_p1) )
        
        data_in_p3 = rig_data.loc[timestamp - p3 : timestamp  ]
        spreadsp3.append(calculate_spread(data_in_p3) )
        
        data_in_f1 = rig_data.loc[timestamp : timestamp + p1 ]
        spreadsf1.append(calculate_spread(data_in_f1) )
        
        data_in_f3 = rig_data.loc[timestamp : timestamp + p3 ]
        spreadsf3.append(calculate_spread(data_in_f3) )
    
    rig_data[ 'XY_spreadp1'] = spreadsp1
    rig_data[ 'XY_spreadp3'] = spreadsp3 
    rig_data[ 'XY_spreadf1'] = spreadsf1
    rig_data[ 'XY_spreadf3'] = spreadsf3 
    #
    # forwards
    rig_data[ 'Xdiff_maf05'] = rig_data['difX'][::-1].rolling(ma05, min_periods=1).sum()[::-1].values
    rig_data[ 'Ydiff_maf05'] = rig_data['difY'][::-1].rolling(ma05, min_periods=1).sum()[::-1].values
    
    rig_data[ 'Xdiff_maf1'] = rig_data['difX'][::-1].rolling(ma1, min_periods=1).sum()[::-1].values
    rig_data[ 'Ydiff_maf1'] = rig_data['difY'][::-1].rolling(ma1, min_periods=1).sum()[::-1].values

    rig_data[ 'Xdiff_maf2'] = rig_data['difX'][::-1].rolling(ma2, min_periods=2).sum()[::-1].values
    rig_data[ 'Ydiff_maf2'] = rig_data['difY'][::-1].rolling(ma2, min_periods=2).sum()[::-1].values
    
    rig_data[ 'Xdiff_maf3'] = rig_data['difX'][::-1].rolling(ma3, min_periods=2).sum()[::-1].values
    rig_data[ 'Ydiff_maf3'] = rig_data['difY'][::-1].rolling(ma3, min_periods=2).sum()[::-1].values
    
    rig_data[ 'ed_XYmaf05'] = rig_data['ed_dif'][::-1].rolling(ma05, min_periods=1).sum()[::-1].values
    rig_data[ 'ed_XYmaf1'] = rig_data['ed_dif'][::-1].rolling(ma1, min_periods=1).sum()[::-1].values
    rig_data[ 'ed_XYmaf2'] = rig_data['ed_dif'][::-1].rolling(ma2, min_periods=2).sum()[::-1].values     
    rig_data[ 'ed_XYmaf3'] = rig_data['ed_dif'][::-1].rolling(ma3, min_periods=2).sum()[::-1].values
    
    datasets[f'rig_data{i}'] = rig_data
#%%    
#%%
rig_data = datasets['rig_data1']

plt.plot(rig_data.time_lapsed_all, abs(rig_data.difX), linewidth = 0.7 )
plt.plot(rig_data.time_lapsed_all, abs(rig_data.difY), linewidth = 0.7 )
plt.plot(rig_data.time_lapsed_all, abs(rig_data.ed_XYmaf1)/9, linewidth = 0.7 )
plt.plot(rig_data.time_lapsed_all, abs(rig_data.ed_XYma1)/9, linewidth = 0.7 )
plt.show()

#%%
for i in range(1,7):
    
    rig_data = datasets[f'rig_data{i}']

    rig_data[ 'nt_XYma05'] = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['Xdiff_ma05'], rig_data['Ydiff_ma05'] )  ]
    rig_data[ 'nt_XYma1'] = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['Xdiff_ma1'], rig_data['Ydiff_ma1'] )  ]
    rig_data[ 'nt_XYma2'] = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['Xdiff_ma2'], rig_data['Ydiff_ma2'] )  ]   
    rig_data[ 'nt_XYma3'] = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['Xdiff_ma3'], rig_data['Ydiff_ma3'] )  ]
    
    rig_data[ 'nt_XYmaf05'] = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['Xdiff_maf05'], rig_data['Ydiff_maf05'] )  ]
    rig_data[ 'nt_XYmaf1'] = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['Xdiff_maf1'], rig_data['Ydiff_maf1'] )  ]
    rig_data[ 'nt_XYmaf2'] = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['Xdiff_maf2'], rig_data['Ydiff_maf2'] )  ]     
    rig_data[ 'nt_XYmaf3'] = [ma.sqrt(x**2 + y**2) for x, y in zip(rig_data['Xdiff_maf3'], rig_data['Ydiff_maf3'] )  ]

    datasets[f'rig_data{i}'] = rig_data
#%%
#%%
Tri_dat = pd.concat(datasets.values() )
#%%
#%%
#%%
power = 0.4

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

min_value = 0
max_value = (Tri_dat['ed_XYma05']**power).max()

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = Tri_dat[Tri_dat['rig'] == rig]
    ax.scatter(rig_data['X'], rig_data['Y'],
            c=(rig_data['ed_XYma05']**power), cmap='coolwarm', s = 5,
            norm=plt.Normalize(min_value, max_value),
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

#%%
plt.scatter(Tri_dat['X'], Tri_dat['Y'], c=Tri_dat['ed_XYma05']**(0.7),
         cmap='coolwarm', s = 3, norm=plt.Normalize(min_value, max_value),
         label=f'Rig {rig}')
plt.show()

#%%
plt.hist((Tri_dat['ed_XYma05'])**(0.3), edgecolor='black')  
#%%
#%%
# unsupervised learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
#%%
ML_data = Tri_dat.copy()
ML_data = ML_data.dropna()
#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(ML_data[['ed_XYma05']])
#%%
k = 3

kmeans = KMeans(n_clusters=k, random_state=21)
kmeans.fit(X_scaled)
#%%
ML_data['cluster'] = kmeans.labels_
#%%
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

custom_colors = ['blue', 'lightblue', 'orange', 'red', 'green'] 

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = ML_data[ML_data['rig'] == rig]
    ax.scatter(rig_data['X'], rig_data['Y'],
            c=rig_data['cluster'].map(dict(zip(range(k), custom_colors))), 
 #           cmap='tab10',
            s = 4,
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

#%%
plt.scatter(ML_data['X'], ML_data['Y'],
            c=ML_data['cluster'].map(dict(zip(range(k), custom_colors))),
         # cmap='tab10',
         s = 3,
         label=f'Rig {rig}')
plt.show()

#%%
#%%
#%%
#%%
#%%
start = 0
stop = 2000

rig_data = ML_data[ML_data['rig'] == 1]

plt.scatter(rig_data.time_lapsed_all[start:stop], rig_data.ed_XYma05[start:stop],
            s = 1,
            c=rig_data['cluster'][start:stop].map(dict(zip(range(k), custom_colors)))  )
plt.show()
#%%
#%%
#%%
Tri_dat.to_pickle('UWB_dat_post5impute.pkl')
ML_data.to_pickle('UWB_ML_data.pkl')

with open('UWB_datasets_5.pkl', 'wb') as f:
    pickle.dump(datasets, f)


#%%
# startX, endX, startY, endY, time_start, time_end, Xmas, Ymas, rig, 

#%%
#%%
# pathway traffic
# pathways, number of times on pathway
# consideration of prediction vs mapping layout?

#%%
Tri_dat.columns
#%%