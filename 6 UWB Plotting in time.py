# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:42:45 2024

@author: Mr T
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
import pickle
pd.set_option('display.max_columns', 12)

#%%
Tri_dat = pd.read_pickle('UWB_dat_post5impute.pkl')

with open('UWB_datasets_5.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
#%%
%matplotlib inline
%matplotlib qt 
#%%
from matplotlib.animation import FuncAnimation
plt.ion()  # Enable interactive mode
#%%
#%%
#%%            
def animation_fun(data, dark, light, recent = 2, atatime = 0.5, psize = 6, pace = 0.025, testing = False):
       fig, ax = plt.subplots(figsize=(8, 6))

       recentTime = pd.Timedelta(minutes=recent)  
       period = pd.Timedelta(minutes=atatime)

       time_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq=period)
       
       initial_timestamp = data.index[0]
       initial_data = data.loc[initial_timestamp:(initial_timestamp+period)]
       plt.scatter(initial_data['X'], initial_data['Y'], marker='o', s=psize, c= dark)
       
       plt.scatter([], [], c = light, marker='o', s=psize)
       plt.xlim(min(Tri_dat['X']), max(Tri_dat['X']))
       plt.ylim(min(Tri_dat['Y']), max(Tri_dat['Y'])+1)

       plt.xlabel('X')
       plt.ylabel('Y')
       plt.title('0 mins lapsed')
       legend_elements = [Line2D([0], [0], marker='o', color=dark, markersize=10,
                          label=f"Previous {int(recent)} mins"),
                          Line2D([0], [0], marker='o', color=light, markersize=10,
                          label=f"Previous to the last {int(recent)} mins") ]     

       plt.legend(handles=legend_elements, loc='upper left')

       for ind, timestamp in enumerate(time_range):
           x_new = data.loc[timestamp:(timestamp+period), 'X']
           y_new = data.loc[timestamp:(timestamp+period), 'Y']
           plt.pause(pace)
           
           if timestamp >= (data.index[0] + recentTime):
               x_old = data.loc[(timestamp-recentTime):(timestamp-recentTime+period), 'X']
               y_old = data.loc[(timestamp-recentTime):(timestamp-recentTime+period), 'Y']
               plt.scatter(x_old, y_old, c= light, s=psize)
           
           plt.scatter(x_new, y_new, c = dark, s = psize)
           if testing == True:
               dat = data.loc[timestamp:(timestamp+period)]
               locations = [data.index.get_loc(timestamp) for timestamp in dat.index.values]
               plt.title(f"{timestamp.strftime('%H:%M:%S')} to {(timestamp+period).strftime('%H:%M:%S')}, {locations[0]} ")
           if (ind % 2 == 0) and (testing == False):
               plt.title(f"{0.5*ind} mins lapsed")
           
           if ind >= (len(time_range)-1):
               print('The End!')
       
#%%
#%%
animation_fun(datasets['rig_data1'], 'blue', 'lightblue' )
#%%
animation_fun(datasets['rig_data2'], 'red', 'lightpink' )
#%%
animation_fun(datasets['rig_data3'], 'black', 'lightgrey' )
#%%
animation_fun(datasets['rig_data4'], 'darkgreen', 'lightgreen' )
#%%
animation_fun(datasets['rig_data5'], 'indigo', 'lavender' )
#%%
animation_fun(datasets['rig_data6'], 'orangered', 'lightsalmon', testing = True, pace = 0.2 )
#%%
#%%  
Time_order_dat = Tri_dat.sort_index()
#%%*
fig, ax = plt.subplots(figsize=(8, 6))

recently = pd.Timedelta(minutes=1)  
step = pd.Timedelta(minutes=0.5)
p_temp = 6

darkcols = np.array(['blue', 'red', 'black', 'darkgreen', 'indigo', 'orangered'])
lightcols = np.array(['lightblue', 'lightpink', 'lightgrey', 'lightgreen', 'lavender', 'lightsalmon'])
 
time_range = pd.date_range(start=Time_order_dat.index.min(), end=Time_order_dat.index.max(), freq=step)

initial_timestamp = Time_order_dat.index[0]
initial_data = Time_order_dat.loc[initial_timestamp:(initial_timestamp+step)]
ax.scatter(initial_data['X'], initial_data['Y'], marker='o', s=p_temp, c=darkcols[np.array(initial_data['rig']-1)])

plt.xlim(min(Time_order_dat['X']), max(Time_order_dat['X']))
plt.ylim(min(Time_order_dat['Y']), max(Time_order_dat['Y'])+1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('0 mins lapsed')

for ind, timestamp in enumerate(time_range):
    x_new = Time_order_dat.loc[timestamp:(timestamp+step), 'X']
    y_new = Time_order_dat.loc[timestamp:(timestamp+step), 'Y']
    cols_newi = np.array(Time_order_dat.loc[timestamp:(timestamp+step), 'rig']-1)
    plt.pause(0.02)
    
    x_old = Time_order_dat.loc[(timestamp-recently):(timestamp-recently+step), 'X']
    y_old = Time_order_dat.loc[(timestamp-recently):(timestamp-recently+step), 'Y']
    cols_oldi = np.array(Time_order_dat.loc[(timestamp-recently):(timestamp-recently+step), 'rig']-1)
    ax.scatter(x_old, y_old, c= lightcols[cols_oldi], s=p_temp)
    
    ax.scatter(x_new, y_new, c=darkcols[cols_newi], s=p_temp)
    
    if ind % 2 == 0: 
        ax.set_title(f"{0.5*ind} mins lapsed")
    
    if ind >= (len(time_range)-1):
        print('The End!')

plt.show()   
#%%
#%%
fig, ax = plt.subplots(figsize=(8, 6))

rec_par = 1
step_par = 0.5

recently = pd.Timedelta(minutes=rec_par)  
step = pd.Timedelta(minutes=step_par)
p_temp = 6

darkcols = np.array(['blue', 'red', 'black', 'darkgreen', 'indigo', 'orangered'])
lightcols = np.array(['lightblue', 'lightpink', 'lightgrey', 'lightgreen', 'lavender', 'lightsalmon'])
 
time_range = pd.date_range(start=Time_order_dat.index.min(), end=Time_order_dat.index.max(), freq=step)

minX = min(Time_order_dat['X'])
minY = min(Time_order_dat['Y'])
maxX = max(Time_order_dat['X'])
maxY = max(Time_order_dat['Y'])+1
plt.xlim(minX, maxX)
plt.ylim(minY, maxY )
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('0 mins lapsed')

legend_handles = [Line2D([0], [0], marker='o', color='w', label=worker,
                         markerfacecolor=color, markersize=10) for worker, 
                      color in zip(range(1,7), darkcols)]

plt.legend(handles=legend_handles, loc='upper left')
initial_timestamp = Time_order_dat.index[0]
initial_data = Time_order_dat.loc[initial_timestamp:(initial_timestamp+step)]
ax.scatter(initial_data['X'], initial_data['Y'], marker='o', s=p_temp, c=darkcols[np.array(initial_data['rig']-1)])

for ind, timestamp in enumerate(time_range):
    plt.pause(0.5)
    plt.cla()
    plt.xlim(minX, maxX)
    plt.ylim(minY, maxY )
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(handles=legend_handles, loc='upper left')
     
    x_new = Time_order_dat.loc[timestamp:(timestamp+step), 'X']
    y_new = Time_order_dat.loc[timestamp:(timestamp+step), 'Y']
    cols_newi = np.array(Time_order_dat.loc[timestamp:(timestamp+step), 'rig']-1)

    ax.scatter(x_new, y_new, c=darkcols[cols_newi], s=p_temp)
    
    ax.set_title(f"{int(step_par*ind)} mins lapsed")
    
    if ind >= (len(time_range)-1):
        print('The End!')

plt.show()
#%%
#%%
Time_order_dat.to_pickle('UWB_time_order.pkl')
#%%
#%%
#%%
%matplotlib inline
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%