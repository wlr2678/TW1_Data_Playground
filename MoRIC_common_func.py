from pds4_tools import pds4_read       # Core PDS4 package

# For data manipulation
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
from skimage import img_as_float

import glob, os, shutil # For file operations

def format_path(path):
    return path.replace(os.sep, '/')

def enter_dir(directory):
    '''Change working directory to directory'''
    os.chdir(format_path(directory))

def load_data_list(search_condition):
    '''Load all condition-meeting data paths in the working directory into a list'''
    p = glob.glob(search_condition)
    print('File list loaded.')
    return p

def read_pds(path):
    '''Read PDS4 data: images only for NaTeCam and with coordinates for MoRIC data'''
    data = pds4_read(path, quiet=True)
    img = np.array(data[0].data)
    img = img_as_float(img)
    coor = pd.DataFrame(data[1].data)[['Row', 'Column', 'Longitude', 'Latitude']]
    return img, coor # NumPy array for image, pandas DataFrame for table

def create_plot_layout(layout, gs_kw, figsize=(5, 5)):
    '''Create plot layout from layout definition (2D array), gridspec kw (dict of ratios) and figsize (tuple)'''
    fig, axes = plt.subplot_mosaic(layout, gridspec_kw=gs_kw, figsize=figsize, layout='constrained')
    return fig, axes

def prune_coor(coor):
    '''
    Clean up coordinates for displaying only borders, returning the upper edge 
    and other edges separately for direction finding
    '''
    size_x = coor.iloc[-1]['Column']
    size_y = coor.iloc[-1]['Row']
    upper_edge = coor[coor['Row'] == 1]
    lower_edge = coor[coor['Row'] == size_y]
    left_edge = coor[coor['Column'] == 1]
    right_edge = coor[coor['Column'] == size_x]
    return upper_edge, pd.concat([lower_edge, left_edge, right_edge]).drop_duplicates().reset_index(drop=True)

def plot_base_map(ax_map, path_map):
    '''Create a plot of base Mars map'''
    map_image = plt.imread(path_map)
    ax_map.imshow(map_image, extent = [-180, 180, -90, 90], aspect=1)
    
def plot_coor(ax_map, upper_edge, other_edges, style1='.r', style2='.y'):
    '''Plot coordinates for an image'''
    ax_map.plot(upper_edge['Longitude'], upper_edge['Latitude'], style1, alpha=1, markersize=1.5)
    ax_map.plot(other_edges['Longitude'], other_edges['Latitude'], style2, alpha=1, markersize=1.5)

def find_distance(coor1, coor2):
    '''Find great-circle distance between two martian coordinates (DataFrame format) using the haversine formula'''
    R = 3389.5e3 # Mean radius of Mars
    # Degrees
    lon_d1 = coor1['Longitude']
    lon_d2 = coor2['Longitude']
    lat_d1 = coor1['Latitude']
    lat_d2 = coor2['Latitude']
    
    # Radians
    lat_r1 = lat_d1 * np.pi/180
    lat_r2 = lat_d2 * np.pi/180
    dlat = (lat_d2-lat_d1) * np.pi/180
    dlon = (lon_d2-lon_d1) * np.pi/180
    
    a = np.power(np.sin(dlat/2), 2) + np.cos(lat_r1) * np.cos(lat_r2) * np.power(np.sin(dlon/2), 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
