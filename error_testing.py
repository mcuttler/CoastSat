# -*- coding: utf-8 -*-
"""
FIND GDAL ERROR!
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# image processing modules
import skimage.transform as transform
import skimage.morphology as morphology
import sklearn.decomposition as decomposition
import skimage.exposure as exposure

# other modules
from osgeo import gdal, osr
from pylab import ginput
import pickle
import geopandas as gpd
from shapely import geometry

# own modules
from coastsat import SDS_tools

#settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)


#%%
            sitename = settings['inputs']['sitename']
            filepath_data = settings['inputs']['filepath']
    
    # check if reference shoreline already exists in the corresponding folder
            filepath = os.path.join(filepath_data, sitename)
            filename = sitename + '_reference_shoreline.pkl'

            satname = 'S2'
            filepath = SDS_tools.get_filepath(settings['inputs'],satname)
            filenames = metadata[satname]['filenames']
            
            i=1

            
            # read image
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            im_ms, georef, cloud_mask, im_extra, imQA = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            
            # calculate cloud cover
            cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
           
                       
            # rescale image intensity for display purposes
            im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            
            # plot the image RGB on a figure
            fig = plt.figure()
            fig.set_size_inches([18,9])
            fig.set_tight_layout(True)
            plt.axis('off')
            plt.imshow(im_RGB)
            
            # decide if the image if good enough for digitizing the shoreline
            plt.title('click <keep> if image is clear enough to digitize the shoreline.\n' +
                      'If not (too cloudy) click on <skip> to get another image', fontsize=14)
            keep_button = plt.text(0, 0.9, 'keep', size=16, ha="left", va="top",
                                   transform=plt.gca().transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))   
            skip_button = plt.text(1, 0.9, 'skip', size=16, ha="right", va="top",
                                   transform=plt.gca().transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            mng = plt.get_current_fig_manager()                                         
            mng.window.showMaximized()
            
            # let user click on the image once
            pt_input = ginput(n=1, timeout=1e9, show_clicks=False)
            pt_input = np.array(pt_input)
            
            # if clicks next to <skip>, show another image
            if pt_input[0][0] > im_ms.shape[1]/2:
                plt.close()
                
            
            else:
                # remove keep and skip buttons
                keep_button.set_visible(False)
                skip_button.set_visible(False)
                # create two new buttons
                add_button = plt.text(0, 0.9, 'add', size=16, ha="left", va="top",
                                       transform=plt.gca().transAxes,
                                       bbox=dict(boxstyle="square", ec='k',fc='w'))   
                end_button = plt.text(1, 0.9, 'end', size=16, ha="right", va="top",
                                       transform=plt.gca().transAxes,
                                       bbox=dict(boxstyle="square", ec='k',fc='w')) 
                
                # add multiple reference shorelines (until user clicks on <end> button)
                pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                geoms = []
                while 1:
                    add_button.set_visible(False)
                    end_button.set_visible(False) 
                    # update title (instructions)
                    plt.title('Click points along the shoreline (enough points to capture the beach curvature).\n' +
                              'Start at one end of the beach.\n' + 'When finished digitizing, click <ENTER>',
                              fontsize=14)     
                    plt.draw()
                    
                    # let user click on the shoreline
                    pts = ginput(n=50000, timeout=1e9, show_clicks=True)
                    pts_pix = np.array(pts)       
                    # convert pixel coordinates to world coordinates
                    pts_world = SDS_tools.convert_pix2world(pts_pix[:,[1,0]], georef) 
                    
                    # interpolate between points clicked by the user (1m resolution)
                    pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                    for k in range(len(pts_world)-1):
                        pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
                        xvals = np.arange(0,pt_dist)
                        yvals = np.zeros(len(xvals))
                        pt_coords = np.zeros((len(xvals),2))
                        pt_coords[:,0] = xvals
                        pt_coords[:,1] = yvals
                        phi = 0
                        deltax = pts_world[k+1,0] - pts_world[k,0]
                        deltay = pts_world[k+1,1] - pts_world[k,1]
                        phi = np.pi/2 - np.math.atan2(deltax, deltay)
                        tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
                        pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0) 
                    pts_world_interp = np.delete(pts_world_interp,0,axis=0)
                    
                    # save as geometry (to create .geojson file later)
                    geoms.append(geometry.LineString(pts_world_interp))
                    
                    # convert to pixel coordinates and plot
                    pts_pix_interp = SDS_tools.convert_world2pix(pts_world_interp, georef)
                    pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
                    plt.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
                    plt.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
                    plt.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')
                    
                    # update title and buttons
                    add_button.set_visible(True)
                    end_button.set_visible(True) 
                    plt.title('click <add> to digitize another shoreline or <end> to finish and save the shoreline(s)',
                              fontsize=14)     
                    plt.draw()      
                    
                    # let the user click again (<add> another shoreline or <end>)
                    pt_input = ginput(n=1, timeout=1e9, show_clicks=False)
                    pt_input = np.array(pt_input) 
                    
                    # if user clicks on <end>, save the points and break the loop
                    if pt_input[0][0] > im_ms.shape[1]/2: 
                        add_button.set_visible(False)
                        end_button.set_visible(False)                                                                         
                        plt.title('Reference shoreline saved as ' + sitename + '_reference_shoreline.pkl and ' + sitename + '_reference_shoreline.geojson')
                        plt.draw()
                        ginput(n=1, timeout=3, show_clicks=False)
                        plt.close() 
                        break
                        
                    
                pts_sl = np.delete(pts_sl,0,axis=0)     
                # convert world image coordinates to user-defined coordinate system
#                image_epsg = metadata[satname]['epsg'][i]
#                pts_coords = SDS_tools.convert_epsg(pts_sl, image_epsg, settings['output_epsg'])
                
 #%%
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # transform points
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    elif type(points) is np.ndarray:
        points_converted = coordTransform.TransformPoints(points)
    else:
        raise Exception('invalid input type')              
  
#%%

class GdalErrorHandler(object):
    def __init__(self):
        self.err_level=gdal.CE_None
        self.err_no=0
        self.err_msg=''

    def handler(self, err_level, err_no, err_msg):
        self.err_level=err_level
        self.err_no=err_no
        self.err_msg=err_msg

if __name__=='__main__':

    err=GdalErrorHandler()
    handler=err.handler # Note don't pass class method directly or python segfaults
                        # due to a reference counting bug 
                        # http://trac.osgeo.org/gdal/ticket/5186#comment:4

    gdal.PushErrorHandler(handler)
    gdal.UseExceptions() #Exceptions will get raised on anything >= gdal.CE_Failure

    try:
        gdal.Error(gdal.CE_Warning,1,'Test warning message')
    except Exception as e:
        print('Operation raised an exception')
        raise
    else:
        if err.err_level >= gdal.CE_Warning:
            print('Operation raised an warning')
            raise RuntimeError(err.err_level, err.err_no, err.err_msg)
    finally:
        gdal.PopErrorHandler()
        
        

