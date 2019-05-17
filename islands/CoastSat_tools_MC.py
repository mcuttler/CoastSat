# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:28:09 2019

@author: M Cuttler - UWA
"""
#%% Code for looping through numerous sites, downloading images, and pre-processing to jpgs

def get_sat_data(polygons, im_dates, sat_list):
    """
    this code loops through the polygons defined in 'polygons' to get data
    within defined date range and satellite missions. this is essentially 
    ripped and modified from K Vos' CoastSat toolbox - main.py script
    
    polygons is a dictionary with {site: KML} defining the site and KML for 
    bounding polygon of interest
    
    date range is list with ['start', 'end']
    
    sat_list is list with ['L5', 'L7', 'L8', 'S2']
    """
    import os
    os.chdir('P:\HANSEN_UWA-UNSW_Linkage\Analysis\CoastSat\CoastSat')
    
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt
    import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
    
    #add this as if/else statement to main.py
    for kml in polygons.values():
        # load modules

        
        # region of interest (longitude, latitude), can also be loaded from a .kml polygon
        #open .kml as text file and double check format of coordinates
        kmlfile = os.path.join('P:\HANSEN_UWA-UNSW_Linkage\Analysis\CoastSat','CoastSat_MC', 'KMLs', kml)
        

        polygon = SDS_tools.coords_from_kml(kmlfile)              
       
        # name of the site
        sitename = kml[0:-4]
        
        # put all the inputs into a dictionnary
        inputs = {
                'polygon': polygon,
                'dates': im_dates,
                'sat_list': sat_list,
                'sitename': sitename
                }        

        # retrieve satellite images from GEE
        metadata = SDS_download.retrieve_images(inputs)
        
        # settings for the shoreline extraction
        settings = { 
                # general parameters:
                'cloud_thresh': 0.2,        # threshold on maximum cloud cover
                'output_epsg': 28350,       # epsg code of spatial reference system desired for the output; GDA94, zone 50   
                # quality control:
                'check_detection': True,    # if True, shows each shoreline detection to the user for validation
                
                # add the inputs defined previously
                'inputs': inputs,
                
                # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
                'min_beach_area': 50,     # minimum area (in metres^2) for an object to be labelled as a beach
                'buffer_size': 150,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
                'min_length_sl': 200,       # minimum length (in metres) of shoreline perimeter to be valid 
                }
        
        SDS_preprocess.save_jpg(metadata, settings)
        
#%% Code for extracting transect start and end points from CSV

def get_transect_data(filepath,filename):

        import pandas as pd
        import numpy as np
        import os
        
        fullfile = os.path.join(filepath, filename)
        
        df = pd.read_csv(fullfile)
        
        #extract Start and End coordinates and conver to numpy array
        data = np.array(df.loc[:,'StartX':'EndY'])
        #create transects dictionary for CoastSat code
        transects = dict([])
        
        for i, data2 in enumerate(data):
            tID = 'Transect ' + str(i+1)
            transects[tID] = np.array([data2[0:2], data2[2:4]])
            
            #pickle to save for later use
            import pickle
            import os
            
            filepath = 'P:\HANSEN_UWA-UNSW_Linkage\Analysis\CoastSat\CoastSat\data\MANDURAH_OCEAN'
            sitename = 'MANDURAH_OCEAN'
            with open(os.path.join(filepath, sitename + '_transects_MC.pkl'), 'wb') as f:
                pickle.dump(transects, f)
                
#%%  code for calculating transects radiating from single point (i.e. for islands)  

def calc_island_transects(x,y,trans_length,heading,fig):
    """ 
    This code is for calculating transecs radiating from a single point. It uses the 
    x,y (input) as the origin for the transects and calculates transects of given length
    and heading (clockwise from North)
    
    Edit so that Transects is an output dictionary with key = transect number
    
    """
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    
    #create dictionary for output
    Transects = dict([])
#    Transects = {'StartXY':np.zeros([heading.size,2]),'EndXY':np.zeros([heading.size,2])}                                   
    
    for i,j in enumerate(heading):
        
        #calculate x and y
        xx = (math.sin(math.radians(j))*trans_length)+x
        yy = (math.cos(math.radians(j))*trans_length)+y
        
        Transects[i] = np.array([[x, y], [xx, yy]])
    
    if fig == 1:
        fig = plt.figure()
                  
        plt.plot([Transects['StartXY'][:,0],Transects['EndXY'][:,0]],
                 [Transects['StartXY'][:,1],Transects['EndXY'][:,1]])
        plt.show


        
    return Transects 
        
    
    
        
        