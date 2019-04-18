# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:26:17 2019

@author: cuttl
"""
#%% Edits to add to SDS_shoreline.extract_shoreline

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):      
            os.makedirs(filepath_jpg)
    
    print('Mapping shorelines:')

    # loop through satellite list
    #for satname in metadata.keys():
        satname = 'S2'    
        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']

        # initialise some variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points 
        output_filename = []   # filename of the images from which the shorelines where derived
        output_cloudcover = [] # cloud cover of the images 
        output_geoaccuracy = []# georeferencing accuracy of the images
        output_idxkeep = []    # index that were kept during the analysis (cloudy images are skipped)
        # sand fields    
        output_sand_area = []       #area of sandy pixles identified from classification 
        output_sand_perimeter = []  #perimieter of sandy pixels
        output_sand_centroid = []   #coordinates center of mass of sandy pixels
        output_sand_points = []     #coordinates of sandy pixels
        
        # load classifiers and convert settings['min_beach_area'] and settings['buffer_size'] 
        # from metres to pixels
        if satname in ['L5','L7','L8']:
            clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_Landsat.pkl'))
            pixel_size = 15
        elif satname == 'S2':
            clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_S2.pkl'))
            pixel_size = 10
        buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
        
        # loop through the images
        #for i in range(len(filenames)):
            i=5
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
            #if cloud_cover > settings['cloud_thresh']:
            #    continue
            
            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = SDS_shoreline.classify_image_NN(im_ms, im_extra, cloud_mask,
                                    min_beach_area_pixels, clf)
            # if the classifier does not detect sand pixels skip this image
#            if sum(sum(im_labels[:,:,0])) == 0:
#                continue

            #######################################################################################
            # SAND POLYGONS (kilian)
            #######################################################################################
            #######################################################################################

            # create binary image with True where the sand pixels and non-classified pixels are
            im_binary_sand = (im_classif == 1)
            # fill the interior of the ring of sand around the island
            im_binary_sand_closed = morphology.remove_small_holes(im_binary_sand, area_threshold=3000, connectivity=1)
            # vectorise the contours
            sand_contours = measure.find_contours(im_binary_sand_closed, 0.5)
            
            # if several contours, it means there is a gap --> merge sand and other pixels
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
            # calculate the attributes
            sand_area = sand_polygon.area
            sand_perimeter = sand_polygon.exterior.length
            sand_centroid = np.array(sand_polygon.centroid.coords)
            sand_points = np.array(sand_polygon.exterior.coords)

            #######################################################################################
            ####################################################################################### 
            
               
            # extract water line contours
            # if there aren't any sandy pixels, use find_wl_contours1 (traditional method), 
            # otherwise use find_wl_contours2 (enhanced method with classification)
            try: # use try/except structure for long runs
                if sum(sum(im_labels[:,:,0])) == 0 :
                    # compute MNDWI (SWIR-Green normalized index) grayscale image
                    im_mndwi = nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                    # find water contourson MNDWI grayscale image
                    contours_mwi = find_wl_contours1(im_mndwi, cloud_mask)
                else:
                    # use classification to refine threshold and extract sand/water interface
                    is_reference_sl = 'reference_shoreline' in settings.keys()
                    contours_wi, contours_mwi = find_wl_contours2(im_ms, im_labels, 
                                                cloud_mask, buffer_size_pixels, is_reference_sl)
            except:
                continue
            
            # process water contours into shorelines
            shoreline = process_shoreline(contours_mwi, georef, image_epsg, settings)
            
            if settings['check_detection_sand_poly']:
                date = filenames[i][:18]
                skip_image = SDS_shoreline.show_detection_sand_poly(im_ms, cloud_mask, im_binary_sand, im_binary_sand_closed, 
                                                      sand_contours, settings, date, satname)
                # if user decides to skip the image
                if skip_image:
                    continue
                
            elif settings['check_detection']:
                date = filenames[i][:18]
                skip_image = show_detection(im_ms, cloud_mask, im_labels, shoreline,
                                            image_epsg, georef, settings, date, satname)
                # if user decides to skip the image
                if skip_image:
                    continue
            
            # append to output variables
            output_timestamp.append(metadata[satname]['dates'][i])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
            output_idxkeep.append(i)
            # sand fields
            output_sand_area.append(sand_area)
            output_sand_perimeter.append(sand_perimeter)
            output_sand_centroid.append(sand_centroid)
            output_sand_points.append(sand_points)
            
        # create dictionnary of output
        output[satname] = {
                'dates': output_timestamp,
                'shorelines': output_shoreline,
                'filename': output_filename,
                'cloud_cover': output_cloudcover,
                'geoaccuracy': output_geoaccuracy,
                'idx': output_idxkeep,
                'sand_area': output_sand_area,
                'sand_perimeter': output_sand_perimeter,
                'sand_centroid': output_sand_centroid,
                'sand_points': output_sand_points,
                }
    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = SDS_tools.merge_output(output)
    
    # save outputput structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)
        
    # save output as kml for GIS applications
    kml = simplekml.Kml()
    for i in range(len(output['shorelines'])):
        if len(output['shorelines'][i]) == 0:
            continue
        sl = output['shorelines'][i]
        date = output['dates'][i]
        newline = kml.newlinestring(name= date.strftime('%Y-%m-%d %H:%M:%S'))
        newline.coords = sl
        newline.description = satname + ' shoreline' + '\n' + 'acquired at ' + date.strftime('%H:%M:%S') + ' UTC'
    kml.save(os.path.join(filepath, sitename + '_output.kml'))  
    
    # save sand polygons as kml
    kml = simplekml.Kml()
    for i in range(len(output['sand_points'])):
        if len(output['sand_points'][i]) == 0:
            continue
        sl = output['sand_points'][i]
        date = output['dates'][i]
        newline = kml.newpolygon(name=date.strftime('%Y-%m-%d %H:%M:%S'), outerboundaryis=sl)
        newline.description = satname + ' shoreline' + '\n' + 'acquired at ' + date.strftime('%H:%M:%S') + ' UTC'
    kml.save(os.path.join(filepath, sitename + '_sand_polygons.kml')) 
        
    return output

#%%
sand_poly_pix = np.zeros([1,2])

for i,coords in enumerate(sand_contours):
    sand_poly_pix = np.concatenate((sand_poly_pix,coords))

#get rid of first zeros
sand_poly_pix = sand_poly_pix[1:len(sand_poly_pix),:]    
sand_poly = SDS_tools.convert_pix2world(sand_poly_pix,georef)
sand_poly_WGS84 = SDS_tools.convert_epsg(sand_poly,28350,4326)
# save output as kml for QGIS applications
        kml = simplekml.Kml()
        
        #for i,x in enumerate(sand_poly):
        #    kmlxy
        kml.newlinestring(name="Sand_poly", description="contour of sand area",
                        coords=sand_poly)  
        kml.save('D:\Projects\HANSEN_UWA-UNSW_Linkage\Analysis\CoastSat\CoastSat\data\EVA\sand_poly_test.kml')
# save output as kml for GoogleEarth applications
        kml84 = simplekml.Kml()
        
        #for i,x in enumerate(sand_poly):
        #    kmlxy
        kml84.newlinestring(name="Sand_poly", description="contour of sand area",
                        coords=sand_poly_WGS84)  
        kml84.save('D:\Projects\HANSEN_UWA-UNSW_Linkage\Analysis\CoastSat\CoastSat\data\EVA\sand_poly_test_WGS84.kml')

           
#%% random plotting
                plt.plot(output['dates'],output['sand_area'],'b-x')


for i, coords in enumerate(output['sand_centroid']):
    plt.plot(coords[0],coords[1],'.')


import scipy
scipy.stats.linregress(test,output['sand_area'])

import time
test = []
for i, dum in enumerate(output['dates']):
    test.append(time.mktime(dum.timetuple()))

for i,centroid in enumerate(output['sand_centroid']):
    plt.plot(test[i],centroid[0],'r.')

<<<<<<< HEAD
#%%
#Calculate when image is L8 or S2 and plot in different colors
dum = output['satname']
test=np.array([])
test2=np.array([])
for i in range(len(dum)):
    test.append(dum[i]=='L8')
    test2.append(dum[i]=='S2')

plt.plot(output['dates'],output['sand_area'],'b.-')
plt.plot(output['dates'][test],output['sand_area'][test],'r.')
            

#%% check eva sat codes
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.plot(output['dates'],output['sand_area'],'k-',linewidth=0.5)

for i,sat in enumerate(output['satname']):
    if sat == 'L8':
        plt.plot(output['dates'][i],output['sand_area'][i],'r.')
       # plt.plot(output['dates'][i],sand_area2[i],'r*')
    else:
        plt.plot(output['dates'][i],output['sand_area'][i],'b.')
        #plt.plot(output['dates'][i],sand_area2[i],'b*')

plt.grid()
plt.ylabel('sub-aerial sand area (m^2)')

red_dot = mlines.Line2D([], [], color='red', marker='.',
                          markersize=15, label='L8')

blue_dot = mlines.Line2D([], [], color='blue', marker='.',
                          markersize=15, label='S2')

plt.legend(handles=[red_dot,blue_dot])
plt.title('Fly Island')


#%% re calculate sand_area for L8 satellites using 30m pix
sand_area = output['sand_area']
sand_area2 = np.array(output['sand_area'],dtype=float)
for i,satname in enumerate(output['satname']):
    if satname == 'L8':
        sand_area2[i]=output['sand_area'][i]*2

        
