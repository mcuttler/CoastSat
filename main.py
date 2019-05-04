    #==========================================================#
    # Shoreline extraction from satellite images
    #==========================================================#
    
    # Kilian Vos WRL 2018
    
    #%% 1. Initial settings
    
    # load modules
    import os
    import numpy as np    
    import pickle
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt
    import SDS_download, SDS_preprocess, SDS_tools, SDS_transects, SDS_shoreline
    
    # region of interest (longitude, latitude in WGS84), can be loaded from a .kml polygon
    polygon = SDS_tools.coords_from_kml('FLY.kml')
                
    # date range
    dates = ['2013-01-01', '2019-05-01']
    
    # satellite missions
    sat_list = ['L8','S2']
    
    # name of the site
    sitename = 'FLY'
    
    # filepath where data will be stored
    filepath_data = os.path.join(os.getcwd(), 'data')
    
    #island file - info about island slope and center coordinates
    island_file = sitename + '_info.csv'
    
    # put all the inputs into a dictionnary
    inputs = {
        'polygon': polygon,
        'dates': dates,
        'sat_list': sat_list,
        'sitename': sitename,
        'filepath': filepath_data,
        'island_file': island_file
            }
    
    #%% 2. Retrieve images
    
    # retrieve satellite images from GEE
#    metadata = SDS_download.retrieve_images(inputs)
    
    # if you have already downloaded the images, just load the metadata file
    filepath = os.path.join(inputs['filepath'], sitename)
    with open(os.path.join(filepath, sitename + '_metadata' + '.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    ##################################################################################################
    # create a subset of the metadata for testing
    
#    n = 20 # number of images
#    metadata.pop('S2')
#    metadata2 = dict([])
#    metadata2['L8'] = dict([])
#    for key in metadata['L8'].keys():
#        metadata2['L8'][key] =  [metadata['L8'][key][i] for i in range(n)]
#    
    
    #%% 3. Batch shoreline detection
        
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
        'dark_sand': False,
        'zref': 0   #reference height datum for tidal correction 
    }
    
    #read additional settings for island info - adds:
    #settings['island_center'] = center coordinates of island
    #settings['beach_slope'] = slope for tidal correction 
    settings = SDS_tools.read_island_info(island_file,settings)
    
    # [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
    #SDS_preprocess.save_jpg(metadata, settings)
    
    ## [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections); required if using sand_polygon
    settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
    ### set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
    settings['max_dist_ref'] = 100        
    ##
    ### extract shorelines from all images (also saves output.pkl and shorelines.kml)
    output = SDS_shoreline.extract_shorelines(metadata, settings)
#    
    #plot time series of beach area
    #fig = plt.figure()
    #plt.plot(output['dates'],output['sand_area'],'b-x')
    #plt.grid('on')
    #plt.xlabel('Date')
    #plt.ylabel('Sub-aerial sand area (m^2)')
    #fig.set_size_inches([8,  4])
    #%% make figures showing timeseries of beach area and centroid movement
    #plot centroid data
    
    #from matplotlib import gridspec
    #import numpy as np
    #fig = plt.figure()
    #gs = gridspec.GridSpec(2,1)
    ##gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
    #ax1 = fig.add_subplot(gs[0,0])
    #for i, coords in enumerate(output['sand_centroid']): 
    #    plt.plot(coords[0][0],coords[0][1],'b.')
    #plt.grid('on')
    #plt.xlabel('Easting (m)');
    #plt.ylabel('Northing (m)');
    #plt.title('Centroid Movement')
    #
    #ax2 = fig.add_subplot(gs[1,0])
    #EvaCenter = [234731.70, 7573554.25]
    ##FlyCenter = [246858.55, 7586598.73]
    #centroidX = []
    #centroidY = []
    #for i,dum in enumerate(output['sand_centroid']):
    #    centroidX.append([dum[0][0]-EvaCenter[0]])
    #    centroidY.append([dum[0][1]-EvaCenter[1]])
    #centroidX = np.array(centroidX)
    #centroidY = np.array(centroidY)
    ##plot change in East/West coordinate of centroid (compared to island center)
    #plt.plot(output['dates'],centroidX,'b-',label='East-West movement')
    #
    ##plot change in North/South coordinate of centroid (compared to island center)
    #plt.plot(output['dates'],centroidY,'r-',label='North-South movement')
    #plt.grid('on')
    #plt.legend()
    #plt.xlabel('Date')
    #plt.ylabel('Change in centroid coordinate (m)')
    #
    #fig.set_size_inches([8,  6])
    #l,b,w,h = ax1.get_position().bounds
    #ax1.set_position([l,b+0.05,w,h])
    
    #%% make a figure of the time coverage
    #from matplotlib import gridspec
    #from matplotlib import patches as mpatches
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax1.yaxis.grid(linestyle=':', color='0.5')
    #ax1.set_ylabel('# images')
    #years = np.arange(output['dates'][0].year, output['dates'][-1].year+1)
    #im_counts = dict([])
    #total_sum = 0
    #for year in years:
    #    im_counts[str(year)] = dict([])
    #    for satname in np.unique(output['satname']):
    #        idx_year = [_.year == year for _ in output['dates']]
    #        idx_satname = [_ == satname for _ in output['satname']]
    #        idx = np.logical_and(idx_year, idx_satname)
    #        im_counts[str(year)][satname] = sum(idx)
    #        total_sum = total_sum + sum(idx)
    #        if satname == 'L8': 
    #            barcolor = 'C0'
    #            ax1.bar(year, height=sum(idx), color=barcolor)                     
    #        elif satname == 'S2': 
    #            barcolor = 'C1'
    #            ax1.bar(year, height=sum(idx), color=barcolor, bottom=im_counts[str(year)]['L8'])                                     
    #blue = mpatches.Patch(color='C0', label='L8')
    #orange = mpatches.Patch(color='C1', label='S2')
    #ax1.legend(handles=[blue, orange], loc=2) 
    #average = total_sum/len(years)
    #plt.title('%d images, %.2f images / year' % (len(output['dates']), average))
    
    
    
    #%% 4. Shoreline analysis
    
    # if you have already mapped the shorelines, load the output.pkl file
#    filepath = os.path.join(inputs['filepath'], sitename)
#    with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
#        output = pickle.load(f) 
    
    # now we have to define cross-shore transects over which to quantify the shoreline changes
    # each transect is defined by two points, its origin and a second point that defines its orientation
    # the parameter transect length determines how far from the origin the transect will span
    settings['transect_length'] = 300 
    
    # there are 3 options to create the transects:
    # - option 1: draw the shore-normal transects along the beach
    # - option 2: load the transect coordinates from a .kml file
    # - option 3: create the transects manually by providing the coordinates
    # - option 4: load transects from pre-made pickle file
    # - option 5: calculate transects emanating from single origin point (e.g. for islands)
    
    # option 1: draw origin of transect first and then a second point to define the orientation
    #transects = SDS_transects.draw_transects(output, settings)
        
    # option 2: load the transects from a KML file
    #kml_file = 'NARRA_transects.kml'
    #transects = SDS_transects.load_transects_from_kml(kml_file)
    
    # option 3: create the transects by manually providing the coordinates of two points 
    #transects = dict([])
    #transects['Transect 1'] = np.array([[342836, 6269215], [343315, 6269071]])
    #transects['Transect 2'] = np.array([[342482, 6268466], [342958, 6268310]])
    #transects['Transect 3'] = np.array([[342185, 6267650], [342685, 6267641]])
    
    # option 4: load transects from pre-made pickle file
    #filepath = os.path.join(inputs['filepath'], sitename)
    #with open(os.path.join(filepath, sitename + '_transects' + '.pkl'), 'rb') as f:
    #    transects = pickle.load(f)
    
    #option 5: transects from single origin point
    ang_start = 0 
    ang_end = 360
    ang_step = 1 #degree step for calculating transects 
    settings['heading'] = np.array(list(range(ang_start,ang_end,ang_step)))
           
    transects = SDS_transects.calc_island_transects(settings)
    
    # intersect the transects with the 2D shorelines to obtain time-series of cross-shore distance
    settings['along_dist'] = 10
    
    #add some print out to show percentage of shorelines processed 
    cross_distance = SDS_transects.compute_intersection(output, transects, settings) 
    
       
    # plot the time-series
    #from matplotlib import gridspec
    #fig = plt.figure()
    #gs = gridspec.GridSpec(len(cross_distance),1)
    #gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
    #for i,key in enumerate(cross_distance.keys()):
    #    ax = fig.add_subplot(gs[i,0])
    #    ax.grid(linestyle=':', color='0.5')
    #    ax.set_ylim([-75,75])
    #    if not i == len(cross_distance.keys()):
    #        ax.set_xticks = []
    #    ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-^', markersize=6)
    #    ax.set_ylabel('distance [m]', fontsize=12)
    #    ax.text(0.5,0.95,'Transect ' + key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
    #            va='top', transform=ax.transAxes, fontsize=14)
    #mng = plt.get_current_fig_manager()                                         
    #mng.window.showMaximized()    
    #fig.set_size_inches([15.76,  8.52])
    
    #%% 5. tide correction for transects and sand polygon
    
    #load tide if already processed
    #filepath = 'P:\CUTTLER_CoastSat\CoastSat\data'
    #with open(os.path.join(filepath, 'ExTide.pkl'),'rb') as f:
    #    tide = pickle.load(f) 
    
    #process tide data
    tide_file = 'E:\Dropbox\Pilbara Island Remote Sensing\TideData\ExGulf_Tides.txt'
    tide, output_corrected = SDS_tools.process_tide_data(tide_file, output)
    
    cross_distance_corrected = SDS_tools.tide_correct(cross_distance,tide,settings['zref'],settings['beach_slope'])
        
    #Calculate tidally corrected sand_polygon 
    
    output_corrected = SDS_tools.tide_correct_sand_polygon(cross_distance_corrected, output_corrected, settings)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
