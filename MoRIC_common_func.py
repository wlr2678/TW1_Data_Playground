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

from itertools import combinations # For pairwise comparison of images

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
    return R * c / 1000

def gen_coor(gmars_format):
    '''Generate coordinate DataFrame from given string type coordinate specification'''
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

def find_bounded_images(bound_coor, df_coor):
    '''Iterate through image coordinates in df_coor, check if each image is 
    within the rectangular region defined by latitudes and longitudes in bound 
    ((lat1, lat2), (lon1, lon2)) and return the images in bounds'''
    lat1 = min(bound_coor[0])
    lat2 = max(bound_coor[0])
    lon1 = min(bound_coor[1])
    lon2 = max(bound_coor[1])
    chosen = []
    for i, row in df_coor.iterrows():
        lat_row = row['Latitude']
        lon_row = row['Longitude']
        if lat1 <= lat_row <= lat2:
            if lon1 <= lon_row <= lon2:
                chosen.append(i) # Image passes check for boundedness
    return chosen

def remove_quasi_duplicates(img_list, df_coor, threshold_centre = 50, threshold_corner = 50):
    paths_to_remove = []
    orbits_modified = []
    for path1, path2 in combinations(img_list, 2):
        centre_diff = find_distance(df_coor.loc[path1], df_coor.loc[path2])
        if centre_diff <= threshold_centre:
            uppercorner1 = pd.DataFrame(pds4_read(path1, quiet=True)[1].data)[['Longitude', 'Latitude']].iloc[0]
            uppercorner2 =  pd.DataFrame(pds4_read(path2, quiet=True)[1].data)[['Longitude', 'Latitude']].iloc[0]
            corner_diff = find_distance(uppercorner1, uppercorner2)
            if corner_diff <= threshold_corner:
                print(f'Image {path1}')
                print(f'Image {path2} \n in close proximity.\nCentre:{centre_diff}, corner:{corner_diff}')
                orbit_num1 = int(path1.split('_')[-2])
                orbit_num2 = int(path2.split('_')[-2])
                
                if path1 in paths_to_remove and path2 in paths_to_remove:
                    pass

                elif path1 not in paths_to_remove and path2 not in paths_to_remove:
                    if (orbit_num1 not in orbits_modified and orbit_num2 not in orbits_modified) or (orbit_num1 in orbits_modified and orbit_num2 in orbits_modified):
                        if orbit_num1 > orbit_num2:
                            path_to_remove = path1
                            orbit_to_remove = orbit_num1
                        else:
                            path_to_remove = path2
                            orbit_to_remove = orbit_num2
                    else:
                        if orbit_num1 in orbits_modified:
                            path_to_remove = path1
                            orbit_to_remove = orbit_num1
                        else:
                            path_to_remove = path2
                            orbit_to_remove = orbit_num2
                elif path1 in paths_to_remove:
                    path_to_remove = path2
                    orbit_to_remove = orbit_num2
                elif path2 in paths_to_remove:
                    path_to_remove = path1
                    orbit_to_remove = orbit_num1
                
                paths_to_remove.append(path_to_remove)
                orbits_modified.append(orbit_to_remove)
    return paths_to_remove

def stretch_img(img, percent=0.5):
    # cf https://www.harrisgeospatial.com/docs/BackgroundStretchTypes.html
    # Adapted for different percentages
    p2, p98 = np.percentile(img, (0+percent, 100-percent))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img

def export_img(name, img):
    pil_img = Image.fromarray(np.clip(img*255, 0, 255).astype('uint8'))
    pil_img.save(name, quality=95, subsampling=0)

def export_image_list(path_list, list_name, create_subfolder=False):
    folder_out = 'POI Batch - ' + list_name
    print(f'Creating folder {folder_out}')
    if folder_out not in os.listdir(os.getcwd()):
        os.mkdir(folder_out)
    else:
        print('Error: folder already exists!')
        return
    # Write path_list to a text file
    with open(f'{folder_out}\datalabels.txt', 'w') as fp:
        for filename in path_list:
            fp.write(f"{filename.removesuffix('.2CL')}\n")

    for filename in path_list:
        img = read_pds(filename)[0]
        print('Processing and exporting file ({index}) {name}'.format(index=path_list.index(filename)+1, name=filename.removesuffix('.2CL')))

        #tonemap1 = cv.createTonemap(gamma=1)
        #tonemap1 = cv.createTonemapMantiuk(gamma=2, scale=1.1, saturation=1.6)
        #tonemap1 = cv.createTonemapReinhard(gamma=0.5, intensity=-2, light_adapt=1, color_adapt=0.5)
        tonemap1 = cv.createTonemapDrago(gamma=0.9, saturation=0.7, bias=0.8)
        img = tonemap1.process(img)
        img = np.nan_to_num(img, copy=True, posinf=1, neginf=0)
        img = stretch_img(img)

        data_number = filename.split('_')[2][-4:]
        orbit_number = filename.split('_')[-2]
        timestamp = filename.split('_')[5]
        if create_subfolder:
            sub_folder = orbit_number
            if sub_folder not in os.listdir(folder_out):
                print(f'Creating subfolder for orbit: {orbit_number}')
                os.mkdir(folder_out + '\\' + sub_folder)
            export_img('{folder}\\{subfolder}\\'.format(folder=folder_out, subfolder=sub_folder) + orbit_number + '_' + data_number + '_' + timestamp + '.jpg', img)
        else:
            export_img('{folder}\\'.format(folder=folder_out) + orbit_number + '_' + data_number + '_' + timestamp + '.jpg', img)
        print('DONE')