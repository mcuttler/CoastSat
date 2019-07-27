# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:41:27 2019

@author: 00084142
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:07:29 2019

@author: 00084142
"""
# load modules
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
from islands import SDS_island_tools, SDS_island_shorelines, SDS_island_transects

import matplotlib.pyplot as plt
import pdb
import pandas as pd

# image processing modules
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
#from skimage.measure import label, regionprops

# machine learning modules
from sklearn.externals import joblib
from shapely.geometry import LineString, LinearRing, Polygon
from shapely import ops
# other modules
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib import gridspec
from pylab import ginput

#%%
# region of interest (longitude, latitude in WGS84), can be loaded from a .kml polygon
polygon = SDS_tools.polygon_from_kml(os.path.join(os.getcwd(), 'KMLs','FLY.kml'))
               
# date range
dates = ['2013-01-01', '2019-05-01']  
# satellite missions
sat_list = ['S2']  
# name of the site
sitename = 'FLY'    
# filepath where data will be stored
filepath_data = os.path.join(os.getcwd(), 'data')    
#island file - info about island slope and center coordinates
island_file = os.path.join(os.getcwd(), 'data',sitename, sitename + '_info.csv')    
# put all the inputs into a dictionnary
inputs = {
        'polygon': polygon,
        'dates': dates,
        'sat_list': sat_list,
        'sitename': sitename,
        'filepath': filepath_data,
        'island_file': island_file
            }    
# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs)   
#for only S2 imagery  
#metadata = {'S2': metadata['S2']}   
        
# settings for the shoreline extraction
settings = { 
        # general parameters:
        'cloud_thresh': 0,        # threshold on maximum cloud cover
        'output_epsg': 28350,       # epsg code of spatial reference system desired for the output - 28350 = GDA94 zone 50
        # quality control:
        'check_detection': False,    # if True, shows each shoreline detection to the user for validation
        'check_detection_sand_poly': True, #if True, uses sand polygon for detection and shows user for validation 
        # add the inputs defined previously
        'inputs': inputs,
        # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
        'min_beach_area': 50,     # minimum area (in metres^2) for an object to be labelled as a beach
        'buffer_size': 100,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
        'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
        'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images
        'dark_sand': False,         # only switch to True if your site has dark sand (e.g. black sand beach)
        'zref': 0   #reference height datum for tidal correction 
    }
    
#read additional settings for island info - adds: 
settings = SDS_island_tools.read_island_info(island_file,settings)
        
## [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections); required if using sand_polygon
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
### set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 100             
# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    output = pickle.load(f)         


#%%
sitename = settings['inputs']['sitename']
filepath_data = settings['inputs']['filepath']
       
# create a subfolder to store the .jpg images showing the detection
filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'jpg_shorelines')
if not os.path.exists(filepath_jpg):      
    os.makedirs(filepath_jpg)

satname = 'L8'

# get images
filepath = SDS_tools.get_filepath(settings['inputs'],satname)
filenames = output['filename']
        
#output_sand_area = []       #area of sandy pixles identified from classification 
#output_sand_perimeter = []  #perimieter of sandy pixels
#output_sand_centroid = []   #coordinates center of mass of sandy pixels
#output_sand_points = []     #coordinates of sandy pixels
#output_sand_eccentricity = []
#output_sand_orientation = []

# load classifiers and convert settings['min_beach_area'] and settings['buffer_size']
# from metres to pixels
if satname in ['L5','L7','L8']:
    if settings['dark_sand']:
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_Landsat_dark.pkl'))
    else:
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_Landsat.pkl'))
        pixel_size = 15
elif satname == 'S2':
    clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_S2.pkl'))
    pixel_size = 10
buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
if 'reference_shoreline' in settings.keys():
    max_dist_ref_pixels = np.ceil(settings['max_dist_ref']/pixel_size)
    
#%%
# loop through the images
for i in range(len(filenames)):
    print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')

# get image filename
    fn = SDS_tools.get_filenames(filenames[i],filepath, satname)       
# preprocess image (cloud mask + pansharpening/downsampling)
    im_ms, georef, cloud_mask, im_extra, imQA = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
    # get image spatial reference system (epsg code) from metadata dict
    image_epsg = metadata[satname]['epsg'][i]
    # calculate cloud cover
    cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
    # skip image if cloud cover is above threshold
#           if cloud_cover > settings['cloud_thresh']:
#                continue

# classify image in 4 classes (sand, whitewater, water, other) with NN classifier
    im_classif, im_labels = SDS_island_shorelines.classify_image_NN(im_ms, im_extra, cloud_mask,
                                    min_beach_area_pixels, clf)

            #######################################################################################
            # SAND POLYGONS (kilian)
            #######################################################################################
            #######################################################################################

            # create binary image with True where the sand pixels
    im_binary_sand = (im_classif == 1)
            # fill the interior of the ring of sand around the island
    im_binary_sand_closed = morphology.remove_small_holes(im_binary_sand, area_threshold=3000, connectivity=1)
            # vectorise the contours
    sand_contours = measure.find_contours(im_binary_sand_closed, 0.5)
            
            # if several contours, it means there is a gap --> merge sand and non-classified pixels
    if len(sand_contours) > 1:
        im_binary_sand = np.logical_or(im_classif == 1, im_classif == 0)
        im_binary_sand_closed = morphology.remove_small_holes(im_binary_sand, area_threshold=3000, connectivity=1)
        sand_contours = measure.find_contours(im_binary_sand_closed, 0.5)
                # if there are still more than one contour, only keep the one with more points
        if len(sand_contours) > 1:
            n_points = []
            for j in range(len(sand_contours)):
                n_points.append(sand_contours[j].shape[0])
            sand_contours = [sand_contours[np.argmax(n_points)]]
                    
                    # convert to world coordinates
            sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
            sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                    # make a shapely polygon
            linear_ring = LinearRing(coordinates=sand_contours_coords)
            sand_polygon = Polygon(shell=linear_ring, holes=None)
        else:    
                    # convert to world coordinates
            sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
            sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
            # make a shapely polygon
            linear_ring = LinearRing(coordinates=sand_contours_coords)
            sand_polygon = Polygon(shell=linear_ring, holes=None)
                                      
    else:    
                # convert to world coordinates
        sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
        sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                # make a shapely polygon
        linear_ring = LinearRing(coordinates=sand_contours_coords)
        sand_polygon = Polygon(shell=linear_ring, holes=None)
            
            # check if perimeter of polygon matches with reference shoreline
            # if much longer (1.5 times) then also merge sand and non-classified pixels
    if linear_ring.length > 1.5*LineString(settings['reference_shoreline']).length:
        im_binary_sand = np.logical_or(im_classif == 1, im_classif == 0)
        im_binary_sand_closed = morphology.remove_small_holes(im_binary_sand, area_threshold=3000, connectivity=1)
        sand_contours = measure.find_contours(im_binary_sand_closed, 0.5)
        # if there are still more than one contour, only keep the one with more points
        if len(sand_contours) > 1:
            n_points = []
            for j in range(len(sand_contours)):
                n_points.append(sand_contours[j].shape[0])
            sand_contours = [sand_contours[np.argmax(n_points)]]   
                # convert to world coordinates
            sand_contours_world = SDS_tools.convert_pix2world(sand_contours[0],georef)
            sand_contours_coords = SDS_tools.convert_epsg(sand_contours_world, image_epsg, settings['output_epsg'])[:,:-1]               
                # make a shapely polygon
            linear_ring = LinearRing(coordinates=sand_contours_coords)
            sand_polygon = Polygon(shell=linear_ring, holes=None)
                
            # calculate the attributes of sand polygon
    sand_area = sand_polygon.area
    sand_perimeter = sand_polygon.exterior.length
    sand_centroid = np.array(sand_polygon.centroid.coords)
    sand_points = np.array(sand_polygon.exterior.coords)
            
            #calculate attributes of binary sand image
    label_img = measure.label(im_binary_sand_closed)
    regions = measure.regionprops(label_img)
            #weird error if length of regions larger than 1; only keep largest region
    if len(regions)>1:
        bbox = []
        for j in range(len(regions)):
            bbox.append(regions[j]['bbox_area'])
        sand_eccentricity = regions[np.argmax(bbox)]['eccentricity']
        sand_orientation = regions[np.argmax(bbox)]['orientation']
    else:                    
        sand_eccentricity = regions[0]['eccentricity']
        sand_orientation = regions[0]['orientation']
            
    output_sand_area.append(sand_area)
    output_sand_perimeter.append(sand_perimeter)
    output_sand_centroid.append(sand_centroid)
    output_sand_points.append(sand_points)
    output_sand_eccentricity.append(sand_eccentricity)
    output_sand_orientation.append(sand_orientation)

#%% save and export output
output_corrected['sand_eccentricity'] = output_sand_eccentricity
output_corrected['sand_orientation'] = output_sand_orientation

outpath = 'G:\CUTTLER_GitHub\CoastSat\data'
csv_path = os.path.join(outpath,sitename,sitename + '_output_corrected_regionprops.csv')
data_out = pd.DataFrame.from_dict(output_corrected)
    
data_out.to_csv(csv_path)


plt.figure()
plt.plot(output_corrected['dates'],output_corrected['sand_eccentricity'])
plt.ylabel('Eccentricity')
plt.title(sitename)

plt.figure()
plt.plot(output['dates'],np.degrees(np.array(output_corrected['sand_orientation'])))
plt.ylabel('Orientation')
plt.title(sitename)


                       
 
        

    