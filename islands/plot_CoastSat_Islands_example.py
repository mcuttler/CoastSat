# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:07:29 2019

@author: 00084142
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# image processing modules
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology

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
import pickle
#import simplekml

# own modules
from coastsat import SDS_tools, SDS_preprocess
#%%
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
       
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'testing')
    if not os.path.exists(filepath_jpg):      
            os.makedirs(filepath_jpg)
            
   filepath = os.path.join(inputs['filepath'], sitename)
    with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
        output = pickle.load(f) 
    #%%
#    print('Mapping shorelines:')

    # loop through satellite list
#    for satname in metadata.keys():
        satname = 'S2'

        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = output['filename']

        # initialise some variables
#        output_timestamp = []  # datetime at which the image was acquired (UTC time)
#        output_shoreline = []  # vector of shoreline points
#        output_filename = []   # filename of the images from which the shorelines where derived
#        output_cloudcover = [] # cloud cover of the images
#        output_geoaccuracy = []# georeferencing accuracy of the images
#        output_idxkeep = []    # index that were kept during the analysis (cloudy images are skipped)
#        # sand fields    
#        output_sand_area = []       #area of sandy pixles identified from classification 
#        output_sand_perimeter = []  #perimieter of sandy pixels
#        output_sand_centroid = []   #coordinates center of mass of sandy pixels
#        output_sand_points = []     #coordinates of sandy pixels
        

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
#        for i in range(len(filenames)):
#            print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')
            i = 295
            # get image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
#            fn = output['filename'][i]
            # preprocess image (cloud mask + pansharpening/downsampling)
            im_ms, georef, cloud_mask, im_extra, imQA = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]['epsg'][i]
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            # skip image if cloud cover is above threshold
#            if cloud_cover > settings['cloud_thresh']:
#                continue

            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = SDS_island_shorelines.classify_image_NN(im_ms, im_extra, cloud_mask,
                                    min_beach_area_pixels, clf)
            # if the classifier does not detect sand pixels skip this image
#            if sum(sum(im_labels[:,:,0])) == 0:
#                continue

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
            
            #######################################################################################
            ####################################################################################### 
            

            # if a reference shoreline is provided, only map the contours that are within a distance
            # of the reference shoreline. For this, first create a buffer around the ref shoreline
            im_ref_buffer = np.ones(cloud_mask.shape).astype(bool)
            if 'reference_shoreline' in settings.keys():
                ref_sl = settings['reference_shoreline']
                # convert to pixel coordinates
                ref_sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(ref_sl, settings['output_epsg'],
                                                                                image_epsg)[:,:-1], georef)
                ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)
                # create binary image of the reference shoreline
                im_binary = np.zeros(cloud_mask.shape)
                for j in range(len(ref_sl_pix_rounded)):
                    im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
                im_binary = im_binary.astype(bool)
                # dilate the binary image to create a buffer around the reference shoreline
                se = morphology.disk(max_dist_ref_pixels)
                im_ref_buffer = morphology.binary_dilation(im_binary, se)

            # extract water line contours
            # if there aren't any sandy pixels, use find_wl_contours1 (traditional method),
            # otherwise use find_wl_contours2 (enhanced method with classification)
            # use try/except structure for long runs
#            try:
#                if sum(sum(im_labels[:,:,0])) == 0 :
#                    # compute MNDWI (SWIR-Green normalized index) grayscale image
#                    im_mndwi = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
#                    # find water contours on MNDWI grayscale image
#                    contours_mwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
#                else:
#                    # use classification to refine threshold and extract sand/water interface
#                    contours_wi, contours_mwi = find_wl_contours2(im_ms, im_labels,
#                                                cloud_mask, buffer_size_pixels, im_ref_buffer)
#            except:
#                print('Could not map shoreline for this image: ' + filenames[i])
##                continue
#
#            # process water contours into shorelines
#            shoreline = SDS_island_shorelines.process_shoreline(contours_mwi, georef, image_epsg, settings)
#            
#            if settings['check_detection_sand_poly']:
#                date = filenames[i][:18]
#                skip_image = show_detection_sand_poly(im_ms, cloud_mask, im_binary_sand, im_binary_sand_closed, 
#                                                      sand_contours, settings, date, satname)
#                # if user decides to skip the image
#                if skip_image:
#                    continue
                
         
            # visualise the mapped shorelines, there are two options:
            # if settings['check_detection'] = True, show the detection to the user for accept/reject
            # if settings['save_figure'] = True, save a figure for each mapped shoreline
#            elif settings['check_detection'] or settings['save_figure']:
#                date = filenames[i][:19]
#                skip_image = show_detection(im_ms, cloud_mask, im_labels, shoreline,
#                                            image_epsg, georef, settings, date, satname)
#                # if the user decides to skip the image, continue and do not save the mapped shoreline
#                if skip_image:
#                    continue
 #%%
 
#    sitename = settings['inputs']['sitename']
#    filepath_data = settings['inputs']['filepath']
#    
#    # subfolder where the .jpg file is stored if the user accepts the shoreline detection 
#    filepath_sandpoly = os.path.join(filepath_data, sitename, 'jpg_files', 'sand_polygons')
#    if not os.path.exists(filepath_sandpoly):      
#        os.makedirs(filepath_sandpoly)
        
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    im_class = np.copy(im_RGB)
    cmap = cm.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5]
    colours[1,:] = np.array([204/255,1,1,1])
    colours[2,:] = np.array([0,91/255,1,1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]
    
    # according to the image shape, decide whether it is better to have the images in the subplot
    # in different rows or different columns
    fig = plt.figure()
    if im_RGB.shape[1] > 2*im_RGB.shape[0]:
        # vertical subplots
        gs = gridspec.GridSpec(3, 1)
        gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[2,0])
    else: 
        # horizontal subplots
        gs = gridspec.GridSpec(1, 3)
        gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
                                     
    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.axis('off')
#    btn_keep = plt.text(0, 0.9, 'keep', size=16, ha="left", va="top",
#                           transform=ax1.transAxes,
#                           bbox=dict(boxstyle="square", ec='k',fc='w'))   
#    btn_skip = plt.text(1, 0.9, 'skip', size=16, ha="right", va="top",
#                           transform=ax1.transAxes,
#                           bbox=dict(boxstyle="square", ec='k',fc='w'))
    ax1.set_title('Eva Island', fontweight='bold', fontsize=16)

    # create image 2 (sandy pixels)
#    ax2.imshow(im_binary_sand,cmap='gray')
#    ax2.axis('off')
#    ax2.set_title('sand pixels', fontweight='bold')
    
    # create image 3 (closed sand polygon)
    ax3.imshow(im_RGB)
    ax3.axis('off')
    ax3.set_title('Island area polygon', fontweight='bold',fontsize=16)
    
    ax2.imshow(im_class)   
    ax2.axis('off')
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title('Automated image classification', fontweight='bold', fontsize=16)
    
    #plot sand contours on each sub plot
    for k in range(len(sand_contours)):
                ax3.plot(sand_contours[k][:,1], sand_contours[k][:,0], 'k-', linewidth=2.5)
                ax2.plot(sand_contours[k][:,1], sand_contours[k][:,0], 'k-', linewidth=2.5)
#                ax1.plot(sand_contours[k][:,1], sand_contours[k][:,0], 'k--', linewidth=1.5)

    
    fig.set_size_inches([12.53, 9.3])
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()