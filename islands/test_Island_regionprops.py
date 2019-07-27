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
        
        output_sand_area = []       #area of sandy pixles identified from classification 
        output_sand_perimeter = []  #perimieter of sandy pixels
        output_sand_centroid = []   #coordinates center of mass of sandy pixels
        output_sand_points = []     #coordinates of sandy pixels
        output_sand_eccentricity = []
        output_sand_orientation = []

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
            
            fig = plt.figure()
            im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            plt.imshow(im_RGB)
            plt.title(output['dates'][i].strftime('%Y %m %d'))
    
            for props in regions:                       
                y0, x0 = props.centroid
                orientation = props.orientation
                x1 = x0 + math.cos(orientation) * 0.5* props.major_axis_length
                y1 = y0 - math.sin(orientation) * 0.5* props.major_axis_length
                x2 = x0 - math.cos(orientation) * 0.5* props.major_axis_length
                y2 = y0 + math.sin(orientation) * 0.5* props.major_axis_length

                plt.plot((x0, x1), (y0, y1), '-b', linewidth=2.5)
                plt.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
                plt.plot(x0, y0, '.r', markersize=15)
                plt.text(50,25,'Orientation = ' + str(round(np.degrees(sand_orientation),2)), color = 'white',fontweight = 'bold')
                plt.savfig(                        )  
                plt.close(fig)
            

#plt.figure()
#plt.plot(output['dates'],output_sand_eccentricity)
#plt.ylabel('Eccentricity')
#plt.title(sitename)
#
#plt.figure()
#plt.plot(output['dates'],np.degrees(np.array(output_sand_orientation)))
#plt.ylabel('Orientation')
#plt.title(sitename)
#%%
import math
im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
fig = plt.figure()
plt.imshow(im_RGB)
plt.title(output['dates'][i].strftime('%Y %m %d'))

for props in regions:                       
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5* props.major_axis_length
    y1 = y0 - math.sin(orientation) * 0.5* props.major_axis_length
#    x2 = x0 - math.sin(orientation) * 0.5* props.minor_axis_length
#    y2 = y0 - math.cos(orientation) * 0.5* props.minor_axis_length
    plt.plot((x0, x1), (y0, y1), '-b', linewidth=2.5)
    plt.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
    plt.plot(x0, y0, '.r', markersize=15)
plt.axis('off')   
#plt.imshow(im_binary_sand_closed)        

    