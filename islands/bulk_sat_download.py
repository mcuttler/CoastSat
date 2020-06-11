# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 05:04:01 2019

@author: 00084142
"""

#%% bulk download
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
from islands import SDS_island_tools, SDS_island_shorelines, SDS_island_transects

    
# region of interest (longitude, latitude in WGS84), can be loaded from a .kml polygon
locations = ['EVA','Y','FLY','OBSERVATION']
for i,kml in enumerate(locations):
    polygon = SDS_tools.polygon_from_kml(os.path.join(os.getcwd(), 'KMLs',kml+'.kml'))
                
    # date range
    dates = ['1999-01-01', '2008-01-01']
    
    # satellite missions
    sat_list = ['L7']
    
    # name of the site
    sitename = kml
    
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
    
  # retrieve satellite images from GEE
    print('Downloading for ' + kml)
    metadata = SDS_download.retrieve_images(inputs)
    
    # settings for the shoreline extraction
    settings = { 
        # general parameters:
        'cloud_thresh': 0.5,        # threshold on maximum cloud cover
        'output_epsg': 28350,       # epsg code of spatial reference system desired for the output - 28350 = GDA94 zone 50
        # quality control:
        'check_detection': False,    # if True, shows each shoreline detection to the user for validation
        'check_detection_sand_poly': True, #if True, uses sand polygon for detection and shows user for validation 
        'save_figure': False,        # if True, saves a figure showing the mapped shoreline for each image
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
    
    settings = SDS_island_tools.read_island_info(island_file,settings)
    
    #[OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
    SDS_preprocess.save_jpg(metadata, settings)