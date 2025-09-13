# -*- coding: utf-8 -*-
#%%
import glob
import pandas as pd
#%%
folders = [f'Rig{num}UWB_raw' for num in range(1, 7)]
added_path = 'data' 

path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\Raw_datas_UWB_Mocap'

folder_path = '\\'.join([path_stem, folders[0], added_path ] )
file_extension = '.txt'  # Change this to the desired file extension
#%%
found_files = glob.glob(f"{folder_path}/**/*{file_extension}", recursive=True)
timestamp_path = '\\'.join([path_stem, folders[0], 'timestamps.txt' ] )
print(found_files[:10]) if found_files else print(f"No {file_extension} files found in the specified folder.")
print(timestamp_path) if timestamp_path else print(f"No {file_extension} files found in the specified folder.")

#%%
dfs = []
df_time = pd.read_csv(timestamp_path, header = None)

for ind, file in enumerate(found_files):
    
    df = pd.read_csv(file, sep=' ', header = None)
    df['rig'] = 1
    df['time'] = df_time.loc[ind]
    dfs.append(df)

test_read = pd.concat(dfs, ignore_index=True)
#%%
dfs = []

for ind, folder in enumerate(folders):
    
    folder_path = '\\'.join([path_stem, folder, added_path ] )
    found_files = glob.glob(f"{folder_path}/**/*{file_extension}", recursive=True)
    timestamps = '\\'.join([path_stem, folder, 'timestamps.txt' ] )
    
    df_time = pd.read_csv(timestamps, header = None)
    
    print(ind)
    
    for indy, file in enumerate(found_files):
        
        df = pd.read_csv(file, sep=' ', header = None)
        df['rig'] = ind + 1
        df['time'] = df_time.loc[indy]
        dfs.append(df)

Data_read = pd.concat(dfs, ignore_index=True)
#%%
Data_read.columns.values[0:2] = ['X', 'Y']

Data_read.to_pickle('UWB_read.pkl')

Data_read.to_csv('UWB_read.csv', index=False)

