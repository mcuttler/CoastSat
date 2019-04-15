# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:26:17 2019

@author: cuttl
"""
#%% Edits to add to SDS_shoreline.extract_shoreline

    sitename = settings['inputs']['sitename']
    
    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(os.getcwd(), 'data', sitename, 'jpg_files', 'detection')
    try:
        os.makedirs(filepath_jpg)
    except:
        print('')
    
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
        output_sand_area=[]
        output_sand_contours=[]
        output_sand_points=[]
        output_sand_centroid=[]
           
        # convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels
        #if satname in ['L5','L7','L8']:
        #    pixel_size = 15
        #elif satname == 'S2':
            pixel_size = 10
        buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
        
        # loop through the images
        #sand_contours = []
        for i in range(len(filenames)-340):
            
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
                                    min_beach_area_pixels, satname)
            
            #calculate sand area, sand perimeter and save
            im_sand2 = im_classif==1
            
            #vectorize sand pixels
            
            rows,cols = im_sand2.shape
            sand_pix = np.array([[-999,-999]],dtype=float)
            
            for ii in range(0,rows-1):    
                for jj in range(0,cols-1):
                    if im_sand2[ii,jj]==True:
                        dum = np.array([[ii,jj]],dtype=float)
                        sand_pix = np.concatenate((sand_pix,dum),axis=0)
                        
            rows,cols = sand_pix.shape           
            sand_pix = sand_pix[1:rows,:]
                        
            sand_area = sum(im_classif[im_sand2])*pixel_size
            
            if sand_area>0:         
                #conver to real world coordinates
                sand_world = SDS_tools.convert_pix2world(sand_pix,georef)
               
                #from image_epsg to output_epsg
                sand_points = SDS_tools.convert_epsg(sand_world, image_epsg, settings['output_epsg'])
              
                #calculate centroid coordinates
                xCenter = np.sum(sand_points[:,0])/len(sand_points[:,0])
                yCenter = np.sum(sand_points[:,1])/len(sand_points[:,1])
                sand_centroid = np.array([xCenter,yCenter])
                
                #calculate inner and outer bounds of sand polygon
                sand_contours_dum = measure.find_contours(im_sand2, 0.99)
                min_contour_len = 50
                sand_contours=[]
                for D in range(len(sand_contours_dum)):
                    if len(sand_contours_dum[D])>min_contour_len:
                        sand_contours.append(sand_contours_dum[D])
            else:
                sand_points = np.zeros([])
                sand_centroid = np.zeros([])
                sand_contours = np.zeros([])
            
      
            
            # fill and save outputput structure
            output_timestamp.append(metadata[satname]['dates'][i])
            #output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
            output_idxkeep.append(i)
            output_sand_area.append(sand_area)
            output_sand_contours.append(sand_contours)
            output_sand_points.append(sand_points)
            output_sand_centroid.append(sand_centroid)
            
        output[satname] = {
                'timestamp': output_timestamp,
                'shoreline': output_shoreline,
                'filename': output_filename,
                'cloudcover': output_cloudcover,
                'geoaccuracy': output_geoaccuracy,
                'idxkeep': output_idxkeep,
                'sand_area': output_sand_area,
                'sand_contours': output_sand_contours,
                'sand_points': output_sand_points,
                'sand_centroid': output_sand_centroid
                }

    # add some metadata
    output['meta'] = {
            'timestamp': 'UTC time',
            'shoreline': 'coordinate system epsg : ' + str(settings['output_epsg']),
            'cloudcover': 'calculated on the cropped image',
            'geoaccuracy': 'RMSE error based on GCPs',
            'idxkeep': 'indices of the images that were kept to extract a shoreline',
            'sand_area': 'area of sandy pixels',
            'sand_contours': 'boundaryes of sandy pixels',
            'sand_points': 'real world coordinates of sandy pixels',
            'sand_centroid': 'coordinates for center of mass of sandy pixels'
            }
    
    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = SDS_tools.merge_output(output)

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
