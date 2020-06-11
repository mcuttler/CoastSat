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
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'jpg_shorelines')
    if not os.path.exists(filepath_jpg):      
            os.makedirs(filepath_jpg)
    #%%

        satname = 'S2'

        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = output['filename']

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
#            i = 29
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

                       
 
        
            im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    
            fig = plt.figure()
            # create image 1 (RGB)
            plt.imshow(im_RGB)
            plt.axis('off')
            plt.title('Eva Island', fontweight='bold', fontsize=16)

   
            #plot sand contours on each sub plot
            for k in range(len(sand_contours)):
                plt.plot(sand_contours[k][:,1], sand_contours[k][:,0], 'r--', linewidth=2)

    
            fig.set_size_inches([9,9])    
            plt.rcParams['savefig.jpeg_quality'] = 100
            date = output['dates'][i].strftime('%Y_%m_%d')
            fig.savefig(os.path.join(filepath_jpg,sitename + '_' +
                             date + '_' + satname + '.jpg'), dpi=150) 
            plt.close()
    