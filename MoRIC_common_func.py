from pds4_tools import pds4_read       # Core PDS4 package

# For data manipulation
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
from skimage import exposure
from skimage import img_as_float

import glob, os, shutil # For file operations

import operator # For sorting

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
    #img = img_as_float(img)
    img = img.astype(np.float32)
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
    lon_d1 = coor1['Longitude'][0]
    lon_d2 = coor2['Longitude']
    lat_d1 = coor1['Latitude'][0]
    lat_d2 = coor2['Latitude']
    
    # Radians
    lat_r1 = lat_d1 * np.pi/180
    lat_r2 = lat_d2 * np.pi/180
    dlat = (lat_d2-lat_d1) * np.pi/180
    dlon = (lon_d2-lon_d1) * np.pi/180
    
    a = np.power(np.sin(dlat/2), 2) + np.cos(lat_r1) * np.cos(lat_r2) * np.power(np.sin(dlon/2), 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c / 1000

def gen_coor(gmars_format):
    '''Generate coordinate DataFrame from given string type coordinate specification copied from Google Mars'''
    lon = gmars_format.split(',')[1].strip()
    lat = gmars_format.split(',')[0].strip()
    if lon[-1] == 'E':
        lon = float(lon[:-1])
    elif lon[-1] == 'W':
        lon = -float(lon[:-1])
    if lat[-1] == 'N':
        lat = float(lat[:-1])
    elif lat[-1] == 'S':
        lat = -float(lat[:-1])
    print('Longitude: ' + str(lon) + ', Latitude: ' + str(lat))

    return pd.DataFrame({'Longitude': lon, 'Latitude': lat}, index=['POI'])

def find_closest_images(df_POI, df_coor, search_radius=1000):
    '''Iterate through all image coordinates provided in df_coor,
     sort and return closest images by distance (if search radius is provided only
     return those within the radius)'''
    dist = {}
    for i, row in df_coor.iterrows():
        dist[i] = find_distance(df_POI, row)
    
    dist_sorted = dict(sorted(dist.items(), key=operator.itemgetter(1)))
    dist_chosen = {}
    for key in dist_sorted:
        if not (isinstance(dist_sorted[key], float) and dist_sorted[key] > search_radius):
            dist_chosen[key] = dist_sorted[key]
    if not dist_chosen:
        print('No images found within the search radius, returning empty dict')
    return dist_chosen

def stretch_img(img, percent=0.5):
    # cf https://www.harrisgeospatial.com/docs/BackgroundStretchTypes.html
    # Adapted for different percentages
    p2, p98 = np.percentile(img, (0+percent, 100-percent))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img

def export_img(name, img):
    pil_img = Image.fromarray(np.clip(img*255, 0, 255).astype('uint8'))
    pil_img.save(name)

def export_image_list(path_list, list_name):
    folder_out = 'POI Batch - ' + list_name
    print(f'Creating subfolder {folder_out}')
    if folder_out not in os.listdir(os.getcwd()):
        os.mkdir(folder_out)
    else:
        print('Error: folder already exists!')
        return
    for filename in path_list:
        img = read_pds(filename)[0]
        print('Processing and exporting file ({index}) {name}'.format(index=path_list.index(filename)+1, name=filename.removesuffix('.2CL')))

        #tonemap1 = cv.createTonemap(gamma=1)
        tonemap1 = cv.createTonemapMantiuk(gamma=2, scale=1.1, saturation=1.6)
        #tonemap1 = cv.createTonemapReinhard(gamma=0.5, intensity=-2, light_adapt=1, color_adapt=0.5)
        #tonemap1 = cv.createTonemapDrago(gamma=0.9, saturation=0.8, bias=0.8)
        img = tonemap1.process(img)
        img = np.nan_to_num(img, copy=True, posinf=1, neginf=0)
        img = stretch_img(img)

        export_img('{subfolder}\{name}.jpg'.format(subfolder=folder_out, name=filename.removesuffix('.2CL')), img)
        print('DONE')