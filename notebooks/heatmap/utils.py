# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:27:28 2019

@author: Angelo Antonio Manzatto
"""


##################################################################################
# Libraries
##################################################################################  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################################################################
# Classes and Methods
############################################################################################

def create_idx(df, grid):
    '''
    This method creates two extra columns in our dataset with the mapping between (lon, lat) int (grid_h, grid_w)
    
    Inputs:
        
        df - Pandas dataframe containing 'lat' and 'lon' columns.
        grid - The matrix division we want to group each of tuple (lat, lon).
    
    Outputs:
        
        A copy of the original dataset containing the grid mapping 
        
    '''
    
    # Make sure that the dataset contains the needed columns
    assert('lat' in df and 'lon' in df)
    
    df_copy = df.copy()
    
    map_height = grid[0]
    map_width = grid[1]
    
    # We add an episolon factor to avoid defining the max indexes above matrix boundaries
    eps = 1e-6
    min_lat = df_copy.lat.min() - eps
    max_lat = df_copy.lat.max() + eps
    min_lon = df_copy.lon.min() - eps
    max_lon = df_copy.lon.max() + eps
    
    map_width_length  = abs(max_lat - min_lat)
    map_height_length = abs(max_lon - min_lon)
    
    map_width_ratio =  map_width  / map_width_length 
    map_height_ratio = map_height / map_height_length
    
    # Print minimum and maximum latitude and longitude
    print("Min Lat: {0} , Max Lat: {1}".format(min_lat, max_lat))
    print("Min Lon: {0} , Max Lon: {1}".format(min_lon ,max_lon))
         
    df_copy[['grid_h','grid_w']] = df_copy[['lon','lat']].apply(lambda x:(
                                                                          abs((x[0] - max_lon) * map_height_ratio),
                                                                              (x[1] - min_lat) * map_width_ratio
                                                                              ),
                                                                          axis=1, 
                                                                          result_type="expand")
    
    return df_copy

def find_best_bounding_map(df, grid=(16,16), threshold = 200):
    '''
    This method tries to find the bounding that best encompassed most of the aggregates incidents into a grid map
    
    Inputs:
        
        df - Pandas dataframe containing 'lat' and 'lon' columns.
        grid - The matrix division we want to group each of tuple (lat, lon).
    
    Outputs:
        
        A copy of the original dataset with a selection of the most relevant data in the spatial dimension
        
    '''
        
    # Define map grid division for width (latitude) and height (longitude)
    grid_h = grid[0]
    grid_w = grid[1]
    
    # Make sure that the dataset contains the needed columns
    assert('lat' in df and 'lon' in df)

    df_indexed = create_idx(df, grid)     
    
    df_indexed[['grid_w','grid_h']] = df_indexed[['grid_h','grid_w']].astype(int)
  
    # Create a global incident heatmap by summing values with the same grid x and grid y location
    global_incident_heatmap = df_indexed.groupby(['grid_w','grid_h'],as_index=True).size()
    global_incident_heatmap = global_incident_heatmap.reset_index()
    global_incident_heatmap.rename(columns ={0:'total'},inplace=True)
    
    global_incident_matrix = np.zeros((grid_h, grid_w))
    
    # Fill each square of the heatmap matris with the total number of crimes commited
    for _, incident_row in global_incident_heatmap.iterrows():
        
        global_incident_matrix[incident_row['grid_h']][incident_row['grid_w']] = incident_row['total']
    
    global_incident_matrix = global_incident_matrix.astype(int)
    
    w_sum = global_incident_matrix.sum(axis=0)
    h_sum = global_incident_matrix.sum(axis=1)
    
    h_min = np.argmin(h_sum < threshold) 
    h_max = grid_h - np.argmin(h_sum[::-1] < threshold) - 1      
    w_min = np.argmin(w_sum < threshold)
    w_max = grid_w - np.argmin(w_sum[::-1] < threshold) - 1 
        
    # Select df based on new boundaries with only relevant occurrences
    selected_df = df_indexed[(df_indexed.grid_w >= w_min) & 
                             (df_indexed.grid_w <= w_max) & 
                             (df_indexed.grid_h >= h_min) & 
                             (df_indexed.grid_h <= h_max)]
    
    # New heatmap with just the selected dataframe
    selected_global_incident_heatmap = selected_df.groupby(['grid_w','grid_h'],as_index=True).size()
    selected_global_incident_heatmap = selected_global_incident_heatmap.reset_index()
    selected_global_incident_heatmap.rename(columns ={0:'total'},inplace=True)

    # Zero indexing to origin
    selected_global_incident_heatmap['grid_h'] = selected_global_incident_heatmap['grid_h'] - h_min
    selected_global_incident_heatmap['grid_w'] = selected_global_incident_heatmap['grid_w'] - w_min
    
    selected_global_incident_matrix = np.zeros(( h_max - h_min + 1, w_max - w_min + 1))
    
    # Fill each square of the heatmap matris with the total number of crimes commited
    for _, incident_row in selected_global_incident_heatmap.iterrows():
        
        selected_global_incident_matrix[incident_row['grid_h']][incident_row['grid_w']] = incident_row['total']
    
    selected_global_incident_matrix = selected_global_incident_matrix.astype(int)
    
    # Plot Heatmap matrices
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_figheight(16)
    f.set_figwidth(16)
    
    ax1.set_title('Original Heatmap')
    ax1.matshow(global_incident_matrix, cmap='jet')
    
    ax2.set_title('Selected Heatmap')
    ax2.matshow(selected_global_incident_matrix, cmap='jet')
    
    for i in range(grid_w-1):
        for j in range(grid_h-1):
            c = global_incident_matrix[j,i]
            ax1.text(i, j, str(c), va='center', ha='center')
            
    for i in range(w_max-w_min+1):
        for j in range(h_max-h_min+1 ):
            c = selected_global_incident_matrix[j,i]
            ax2.text(i, j, str(c), va='center', ha='center')
    
    selected_df = selected_df.drop(['grid_h','grid_w'],1)

    return selected_df

def get_labels(y_pred, y_true):
    
    
    # Coordinate of pixels around a position
    surrounding_areas = [[-1,-1],[-1,0],[-1,1],
                         [0,-1] ,[0, 1],
                         [1,-1] ,[1, 0],[1, 1]]    
    
    # Get list of TP, FN, FP and TN    
    corrects   = np.argwhere((y_pred == 1) & (y_true == 1))
    negatives  = np.argwhere((y_pred == 0) & (y_true == 1))
    positives  = np.argwhere((y_pred == 1) & (y_true == 0))
    neutrals   = np.argwhere((y_pred == 0) & (y_true == 0))
    
    Correct = corrects.tolist()
    Neutral = neutrals.tolist()   
    FP = []
    FN = []
    FPWN = []
    FNWN = []
    
    # Get heatmap shapes
    h_map, w_map = y_true.shape
    
    # Separete negatives into False Negatives and False Negatives With Neighbor
    for risk in negatives:
        
        # Get surrounded coordinates around risk
        surrounding_risks = [x + risk for x in surrounding_areas]
        
        # Update value for each surrounding coordinate
        flag = False
        for surrounding_risk in surrounding_risks:
            
            if flag:
                break
            # If coordinates are outside map boundaries ignore them
            h, w = surrounding_risk
            if h >= h_map or w >= w_map or surrounding_risk[0] < 0 or surrounding_risk[1] < 0:
                continue
            
            # if we find a coordinate in surrouding areas that is true despite the fact we predicted false
            # consider this as a False Negative With Neighbor
            if y_pred[h,w] == 1:
                FNWN.append(risk.tolist())
                flag = True
        
        # If there is no match in surrounding areas consider this risk as False Negative
        if flag == False:    
            FN.append(risk.tolist())
        
    # Separete positives into False Positives and False Positives With Neighbor        
    for risk in positives:
        
        # Get surrounded coordinates around risk
        surrounding_risks = [x + risk for x in surrounding_areas]
        
        # Update value for each surrounding coordinate
        flag = False
        for surrounding_risk in surrounding_risks:
            
            if flag:
                break
            
            # If coordinates are outside map boundaries ignore them
            h, w = surrounding_risk
            if h >= h_map or w >= w_map or surrounding_risk[0] < 0 or surrounding_risk[1] < 0:
                continue
            
            # if we find a coordinate in surrouding areas that is predicted true despite the fact we predicted false
            # on the center risk point, consider this risk as a False Positive With Neighbor
            if y_true[h,w] == 1:
                FPWN.append(risk.tolist())
                flag = True
        
        # If there is no match in surrounding areas consider this risk as False Positive
        if flag == False:    
            FP.append(risk.tolist())
            
    return [Correct, Neutral, FPWN, FNWN, FP, FN]

        
def evaluate_model(X, Y, model, threshold = 96):

    # Get predictions for the whole batch
    Y_pred_eval = model.predict(X)
    
    # threshold percentile
    threshold = np.percentile(Y_pred_eval, threshold)
    
    measures = np.zeros((len(X),6))
    
    for i, (x, y) in enumerate(zip(X, Y)):
       
        y_true = np.squeeze(y)
    
        y_pred = model.predict(x[np.newaxis,...])
        y_pred = np.squeeze(y_pred)
    
        # Set values above threshold as risk and under to neutral
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        
        # get all labels for each coordinate predicted and true
        labels = get_labels(y_pred, y_true)
        
        measures[i][0] = len(labels[0])
        measures[i][1] = len(labels[1]) 
        measures[i][2] = len(labels[2]) 
        measures[i][3] = len(labels[3]) 
        measures[i][4] = len(labels[4]) 
        measures[i][5] = len(labels[5]) 
    
    return measures

def billinear_interpolation(image, scale):
    
    orig_h, orig_w = image.shape
    new_h, new_w = scale
    
    h_ratio = orig_h / new_h
    w_ratio = orig_w / new_w 
    
    new_image = np.empty((new_h,new_w), dtype=np.float)
    
    for h in range(new_h):
        for w in range(new_w):
            
            h_scale = h * h_ratio
            w_scale = w * w_ratio
            
            h0 = int(h_scale)
            w0 = int(w_scale)
            h1 = min(h0 + 1, orig_h-1) 
            w1 = min(w0 + 1, orig_w-1)
            
            idx_a = image[h0,w0]
            idx_b = image[h0,w1]
            idx_c = image[h1,w0]
            idx_d = image[h1,w1]    
            
            b = (w_scale - w0) * idx_b + (1. - (w_scale - w0)) * idx_a
            t = (w_scale - w0) * idx_d + (1. - (w_scale - w0)) * idx_c
            pxf = (h_scale - h0) * t + (1. - (h_scale - h0)) * b
            pxf = int(pxf+0.5)
            
            new_image[h,w] = pxf
            
    return new_image

# Prediction
def find_percentiles(X, model, percentils):
    
    y = model.predict(X)
    
    values = []
    
    for p in percentils:
        values.append(np.percentile(y, p))
        
    return values