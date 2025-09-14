# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:00:42 2024

@author: 
"""

#%%
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import joblib
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
#%%
%matplotlib inline
#%%
path_stem = 'C:\\Users\\xqb22125\\OneDrive - University of Strathclyde\\Manufacturing\\DataSets\\Tricycle\\.spyproject\\'

exec(open(f'{path_stem}UWB functions.py').read())

#%%
import pickle
with open('datasets_pred_26_NV.pkl', 'rb') as f:
    datasets = pickle.load(f)
#%%
with open('datasets_fast_pred_26_NV.pkl', 'rb') as f:
    datasets_fast = pickle.load(f)

with open('datasets_slow_pred_26_NV.pkl', 'rb') as f:
    datasets_slow = pickle.load(f)
#%%
pred_col = 'pred_lstm26_NV'
#%%
#%%
#%%
columns = ['Accuracy_(NormalVsFast)', 'specificity_(NormalVsFast)',
           'Sensitivity_(NormalVsFast)', 'Accuracy_(NormalVsSlow)', 'Specificity_(NormalVsSlow)',
                      'Sensitivity_(NormalVsSlow)']
rownames = ['worker1', 'worker2', 'worker3', 'worker4', 'worker5', 'worker6']
summary_predictions_df = pd.DataFrame(index=rownames, columns=columns)

#%%

for rig in range(1,7):
    
    rig_data = datasets[f'rig_data{rig}'].copy()
    rig_data_fast = datasets_fast[f'rig_data{rig}'].copy()
    rig_data_slow = datasets_slow[f'rig_data{rig}'].copy()
    
    mask = rig_data['time'].isin(rig_data_fast['time'])
    common_dataset_act_fast = rig_data.loc[mask]
    
    preds_fast = rig_data_fast[pred_col].values
    preds_actual = common_dataset_act_fast[pred_col].values
    
    confusion_matrix = pd.crosstab(preds_actual[1:], preds_fast[1:], margins=True)
    confusion_matrix.index = ['Actual_0', 'Actual_1', 'Actual_All']
    confusion_matrix.columns = ['Fast_0', 'Fast_1', 'Fast_All']
    accuracy_fast = round(accuracy_score(preds_actual[1:], preds_fast[1:]), 2)
    specificity_fast = round(confusion_matrix.iloc[0,0]/confusion_matrix.iloc[0,2], 2)
    sensitivity_fast = round(confusion_matrix.iloc[1,1]/confusion_matrix.iloc[1,2], 2)
    print(f'Worker {rig}')
    print(confusion_matrix)
    print(f'Accuracy: {accuracy_fast} ')
    print(f'specificity: {specificity_fast} ')
    print(f'Sensitivity:{sensitivity_fast} ')
    
    times4comparison = rig_data_slow['time_lapsed_all']/2
    mask = round(times4comparison,2).isin(round(rig_data['time_lapsed_all'],2))
    
    common_dataset_act_slow = rig_data_slow.loc[mask]
    
    preds_slow = common_dataset_act_slow[pred_col].values
    preds_actual = rig_data[pred_col].values
    
    confusion_matrix = pd.crosstab(preds_actual[1:], preds_slow[1:], margins=True)
    confusion_matrix.index = ['Actual_0', 'Actual_1', 'Actual_All']
    confusion_matrix.columns = ['Slow_0', 'Slow_1', 'Slow_All']
    accuracy_slow = round(accuracy_score(preds_actual[1:], preds_slow[1:]), 2)
    specificity_slow = round(confusion_matrix.iloc[0,0]/confusion_matrix.iloc[0,2], 2)
    sensitivity_slow = round(confusion_matrix.iloc[1,1]/confusion_matrix.iloc[1,2], 2)
    print(f'Worker {rig}')
    print(confusion_matrix)
    print(f'Accuracy: {accuracy_slow} ')
    print(f'specificity: {specificity_slow} ')
    print(f'Sensitivity:{sensitivity_slow} ')
        
    worker_data = [accuracy_fast, specificity_fast, sensitivity_fast,
                   accuracy_slow, specificity_slow, sensitivity_slow ]
    
    summary_predictions_df.loc[f'worker{rig}'] = worker_data

#%%

def compare_anomoly_preds(comparison = 'fast', worker = 6, window = 50 ):
    
    rig_data = datasets[f'rig_data{worker}'].copy()
    
    xmin = min(rig_data['X'])
    xmax = max(rig_data['X'])
    ymin = min(rig_data['Y'])
    ymax = max(rig_data['Y'])
    
    if comparison == 'fast':
        rig_data_fast = datasets_fast[f'rig_data{rig}'].copy()
        mask = rig_data['time'].isin(rig_data_fast['time'])
        common_dataset_act_fast = rig_data.loc[mask]
        preds_fast = rig_data_fast[pred_col].values
        preds_actual = common_dataset_act_fast[pred_col].values
        different_ilocs = np.where(preds_fast != preds_actual)[0]
    else:
        rig_data_slow = datasets_slow[f'rig_data{rig}'].copy()
        times4comparison = rig_data_slow['time_lapsed_all']/2
        mask = round(times4comparison,2).isin(round(rig_data['time_lapsed_all'],2))
        common_dataset_act_slow = rig_data_slow.loc[mask]
        preds_slow = common_dataset_act_slow[pred_col].values
        preds_actual = rig_data[pred_col].values
        different_ilocs = np.where(preds_slow != preds_actual)[0]
    
    different_intervals = []
    i = 0
    while i < len(different_ilocs):
        
        temp = [ different_ilocs[i] ]
        while (i < (len(different_ilocs)-1)) and (different_ilocs[i+1] == (different_ilocs[i] + 1) ):
            
            temp.append(different_ilocs[i+1])
            i += 1
        different_intervals.append(temp)
        i += 1     
        
    x_col = np.where(rig_data.columns == 'X')
    y_col = np.where(rig_data.columns == 'Y')

    for j in different_intervals:
        
        fudge = 1 if comparison == 'slow' else 2
        lower = j[0]*fudge
        upper = j[-1]*fudge + 1
        plt.scatter(rig_data.iloc[max(0, (lower-window) ):min((upper+window), len(rig_data)), x_col[0][0] ],
                    rig_data.iloc[max(0, (lower-window) ):min((upper+window), len(rig_data)), y_col[0][0] ],
                    c = 'green', alpha = 0.4, s= 9 )
        
        colors = [['black']*fudge if preds_actual[x] == 1 else ['red']*fudge for x in j] 
        colors = [item for sublist in colors for item in sublist]
            
        plt.scatter(rig_data.iloc[lower:upper+fudge-1, x_col[0][0] ],
                    rig_data.iloc[lower:upper+fudge-1, y_col[0][0] ],
                    c = colors, s = 75)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax )
        
        plt.scatter([], [], c='green', s=20, label='Surrounding points')
        plt.scatter([], [], c='red', s=75, label='Actual time dwelling prediction')
        plt.scatter([], [], c='black', s=75, label='Actual time transit prediction')
        plt.legend()
       
        plt.show()
    print(len(different_intervals))

#%%
compare_anomoly_preds( comparison = 'slow')
#%%
compare_anomoly_preds( comparison = 'fast')
#%%
#%%
# checking against labels for worker 6
rig_data = datasets['rig_data6'].reset_index(drop=True).copy()

mask_pred = [~np.isnan(x) for x in rig_data['move_cat'].values]

preds = rig_data[pred_col].values
labels = rig_data[ 'move_cat'].values

mask_different_pred = preds != labels

mask_different_pred_narm = mask_different_pred[mask_pred]

comparison = []
for pred, label in zip(preds, labels):
    
    if np.isnan(pred) or np.isnan(label):
        
        comparison.append(np.nan)
    elif pred == label == 0:
        comparison.append('tn')
    elif pred == label == 1:
        comparison.append('tp')
    elif pred > label:
        comparison.append('fn')
    elif label > pred:
        comparison.append('fp')

from collections import Counter

category_counts = Counter(comparison)
#%%
sensitivity = category_counts['tp']/(category_counts['tp']+category_counts['fn'])
print(f'sensitivity: {sensitivity}')

specificity = category_counts['tn']/(category_counts['tn']+category_counts['fp'])
print(f'specificity: {specificity}')

#%%
comparison = np.array(comparison)
incorrect_cases = np.where(np.logical_or(comparison == 'fp', comparison == 'fn'))[0]

#%%
incorrect_intervals = []

i = 0
while i < len(incorrect_cases):
    
    temp = [ incorrect_cases[i] ]
    while (i < (len(incorrect_cases)-1)) and (incorrect_cases[i+1] == (incorrect_cases[i] + 1) ):
        
        temp.append(incorrect_cases[i+1])
        i += 1
    incorrect_intervals.append(temp)
    i += 1     
        
#%%
window = 50
x_col = np.where(rig_data.columns == 'X')[0][0]
y_col = np.where(rig_data.columns == 'Y')[0][0]
move_col = np.where(rig_data.columns == 'move_cat')[0][0]
transit_color = ['red', 'green']

for i in incorrect_intervals:
    
    lower = i[0]
    upper = i[-1] + 1
    
    lower_w = max(0, (lower-window) )
    upper_w = min((upper+window), len(rig_data) )
     
    s_colors = [transit_color[int(x)] if not ma.isnan(x) else 'grey' for x in rig_data.iloc[lower_w:upper_w, move_col]]
    
    plt.scatter(rig_data.iloc[lower_w:upper_w, x_col ],
                rig_data.iloc[lower_w:upper_w, y_col ],
                c = s_colors, alpha = 0.4, s= 9 )
    
    colors = ['green' if comparison[x] == 'fp' else 'red' for x in i] 
        
    plt.scatter(rig_data.iloc[lower:upper, x_col ],
                rig_data.iloc[lower:upper, y_col ],
                c = colors, s = 80,  edgecolors='black')
    plt.xlim(min(rig_data['X']), max(rig_data['X']))
    plt.ylim(min(rig_data['Y']), max(rig_data['Y']) )
    
    plt.scatter([], [], c='green', s=15, label='Transit')
    plt.scatter([], [], c='red', s=15, label='Dwelling')
    plt.scatter([], [], c='green', s=80, label='False Transit prediction')
    plt.scatter([], [], c='red', s=80, label='False Dwelling prediction')
    plt.scatter([], [], c='grey', s=15, label='Not classified')
    plt.title(f'{i[0]} to {i[-1]}')
    plt.legend()
   
    plt.show()
    print(i)

#%%
#%%
condition = (rig_data[pred_col] > 0.5) & (rig_data['difSp1'] < 0.05)
dubious_cases = np.where(condition == True)[0]

#%%
dubious_intervals = []

i = 0
while i < len(dubious_cases):
    
    temp = [ dubious_cases[i] ]
    while (i < (len(dubious_cases)-1)) and (dubious_cases[i+1] == (dubious_cases[i] + 1) ):
        
        temp.append(dubious_cases[i+1])
        i += 1
    dubious_intervals.append(temp)
    i += 1     
        
#%%
for i in dubious_intervals:
    
    lower = i[0]
    upper = i[-1] + 1
    
    lower_w = max(0, (lower-window) )
    upper_w = min((upper+window), len(rig_data) )
     
    s_colors = [transit_color[int(x)] if not ma.isnan(x) else 'grey' for x in rig_data.iloc[lower_w:upper_w, move_col]]
    
    plt.scatter(rig_data.iloc[lower_w:upper_w, x_col ],
                rig_data.iloc[lower_w:upper_w, y_col ],
                c = s_colors, alpha = 0.4, s= 9 )
          
    plt.scatter(rig_data.iloc[lower:upper, x_col ],
                rig_data.iloc[lower:upper, y_col ],
                c = 'green', s = 80,  edgecolors='black')
    plt.xlim(min(rig_data['X']), max(rig_data['X']))
    plt.ylim(min(rig_data['Y']), max(rig_data['Y']) )
    
    plt.scatter([], [], c='green', s=15, label='Transit')
    plt.scatter([], [], c='red', s=15, label='Dwelling')
    plt.scatter([], [], c='green', s=80, label='Dubious Transit prediction')
    plt.scatter([], [], c='grey', s=15, label='Not classified')
    plt.title(f'{i[0]} to {i[-1]}')
    plt.legend()
   
    plt.show()
    print(i)
#%%
#%%
len(incorrect_intervals)
#%%
len(dubious_intervals)
#%%
#%%
intervals2check = incorrect_intervals + dubious_intervals
#%%
#%%
#%%
with open('TimeAug_prediction_comparison_df.pkl', 'wb') as f:
    pickle.dump(summary_predictions_df, f)
#%%
with open('intervals2check.pkl', 'wb') as f:
    pickle.dump(intervals2check, f)

#%%
#%%
# rough validation set 
Train_set = datasets['rig_data6'].copy()

start = int(0.75*len(Train_set) )

rough_test_set = Train_set[start:].reset_index(drop=True).copy()

#%%
#%%
filter_positions = np.where((np.diff(rough_test_set[pred_col] ) > 0) )[0]
#%%
#%%
rough_test_set['pred_filtered'] = rough_test_set[pred_col].values
#%%
#%%
filter_positions_padded = []
for i in filter_positions:
    
    for j in range(i-2,i+4):
        
        filter_positions_padded.append(j)
        rough_test_set.loc[j, 'pred_filtered'] = np.nan
     
#%%
#%%
preds_filtered = rough_test_set['pred_filtered'].values
y = rough_test_set['move_cat'].values
#%%
#%%
mask = ~np.isnan(preds_filtered) & ~np.isnan(y)
#%%
preds_filtered = preds_filtered[mask]
#%%
y_filtered = y[mask]
#%%
#%%
confusion_matrix = pd.crosstab(y_filtered, preds_filtered, margins=True)
confusion_matrix.index = ['Label_Dwell', 'Label_Transit', 'Actual_All']
confusion_matrix.columns = ['Pred_Dwell', 'Pred_Transit', 'Pred_All']
print(confusion_matrix)
accuracy_filter = round((confusion_matrix.iloc[0,0]+confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[0,2]+confusion_matrix.iloc[1,2]), 2)
specificity_filter = round(confusion_matrix.iloc[0,0]/confusion_matrix.iloc[0,2], 2)
sensitivity_filter = round(confusion_matrix.iloc[1,1]/confusion_matrix.iloc[1,2], 2)

#%%
#%%
#%%
#%%
filter_positions_time = np.where((np.diff(preds_actual) > 0) )[0]
#%%
preds_normal_filtered = preds_actual
#%%
filter_positions_time_padded = []

for i in filter_positions_time:  
    for j in range(i-2,i+4):
        
        filter_positions_time_padded.append(j)
        preds_normal_filtered[j] = np.nan

#%%
mask = ~np.isnan(preds_normal_filtered) & ~np.isnan(preds_slow)
#%%
preds_normal_filtered = preds_normal_filtered[mask]
#%%
preds_slow_filtered= preds_slow[mask]
#%%
#%%
confusion_matrix = pd.crosstab(preds_normal_filtered, preds_slow_filtered, margins=True)
confusion_matrix.index = ['Label_Dwell', 'Label_Transit', 'Actual_All']
confusion_matrix.columns = ['Pred_Dwell', 'Pred_Transit', 'Pred_All']
print(confusion_matrix)
accuracy_filter = round((confusion_matrix.iloc[0,0]+confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[0,2]+confusion_matrix.iloc[1,2]), 2)
specificity_filter = round(confusion_matrix.iloc[0,0]/confusion_matrix.iloc[0,2], 2)
sensitivity_filter = round(confusion_matrix.iloc[1,1]/confusion_matrix.iloc[1,2], 2)

#%%
#%%
#%%
y_pred = rough_test_set['pred_prob26_NV'].values
y_label = rough_test_set['move_cat'].values
#%%
mask = ~np.isnan(y_pred) & ~np.isnan(y_label)
#%%
y_pred_narm = y_pred[mask] 
y_label_narm = y_label[mask]

#%%
from sklearn.metrics import roc_curve, auc
#%%
def plot_roc_curve(true_labels, predicted_probabilities):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

#%%
plot_roc_curve(y_label_narm, y_pred_narm)

#%%
def find_optimal_threshold(true_labels, predicted_probabilities):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    distances_to_corner = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_threshold_index = np.argmin(distances_to_corner)
    optimal_threshold = thresholds[optimal_threshold_index]
    return optimal_threshold

optimal_threshold = find_optimal_threshold(y_label_narm, y_pred_narm)
print("Optimal Threshold:", optimal_threshold)
#%%
