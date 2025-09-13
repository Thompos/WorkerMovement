# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:50:54 2024

@author: 
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%

Tri_dat = pd.read_pickle('UWB_read.pkl')
#%%
# Create a scatter plot
plt.scatter(Tri_dat['X'], Tri_dat['Y'], s = 4)

# Set labels and title
plt.xlabel('X direction')
plt.ylabel('Y direction')

plt.show()
#%%
sns.scatterplot(x='X', y='Y', hue='rig', data=Tri_dat)

#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 5))

for rig, ax in zip(range(1,7), axes.flatten()):
    ax.scatter(Tri_dat.loc[Tri_dat.rig == rig, 'X'], 
               Tri_dat.loc[Tri_dat.rig == rig, 'Y'],
               s = 3)
    ax.set_title(f'Worker {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.tight_layout()

plt.show()
#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

for rig, ax in zip(range(1, 7), axes.flatten()):
    rig_data = Tri_dat[Tri_dat['rig'] == rig]
    ax.plot(rig_data['X'], rig_data['Y'], 
            label=f'Rig {rig}')
    ax.set_title(f'Rig {rig}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.tight_layout()

plt.show()

#%%

