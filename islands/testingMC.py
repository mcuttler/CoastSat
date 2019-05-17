# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:26:17 2019

@author: cuttl
"""
#%% tide correction
import pandas as pd
from datetime import tzinfo, timedelta, datetime, timezone
import numpy
import pickle

#load EVA DATA
filepath = 'P:\CUTTLER_CoastSat\CoastSat\data\EVA'
with open(os.path.join(filepath, 'EVA_output.pkl'), 'rb') as f:
        output = pickle.load(f) 
        
tideraw = pd.read_csv('E:\Dropbox\Pilbara Island Remote Sensing\TideData\ExGulf_Tides.txt',sep='\t')
tideraw = tideraw.values    
             
tide = dict([])
tidetime = []
ztide = []

for i,row in enumerate(tideraw):
    #convert tide time to UTC from WA local time (UTC+8 hrs)
    dumtime = datetime(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])) - timedelta(hours = 8)   
    dumtimestamp = datetime.timestamp(dumtime)
    
    tidetime.append(dumtimestamp)
    ztide.append(row[-1])
    del dumtime, i, row
    

tide['time'] = numpy.array([tidetime])
tide['time'] = numpy.reshape(tide['time'],-1)
tide['ztide'] = numpy.array([ztide])
tide['ztide'] = numpy.reshape(tide['ztide'],-1)

image_dates = []
#parse tide for overlapping times with images
for i, date in enumerate(output['dates']):
    image_dates.append(datetime.timestamp(date))
    del i, date
    
ind_min = [] 
for i, date in enumerate(image_dates):
    dum = numpy.absolute(date-tide['time'])
    #find index of min value
    ind_min.append(numpy.argmin(dum))
    del i, date

tide_out = dict([])
tide_out['time_image'] = tide['time'][ind_min]
tide_out['ztide_image']=tide['ztide'][ind_min] - 1.412

time_image_UTC = []
for i,time in enumerate(tide_out['time_image']):
    time_image_UTC.append(datetime.timestamp(datetime.fromtimestamp(time)-timedelta(hours=8)))

tide_out['time_image_UTC']=time_image_UTC
#save tide data
filepath = 'P:\CUTTLER_CoastSat\CoastSat\data'
with open(os.path.join(filepath, 'ExTide.pkl'), 'wb') as f:
        pickle.dump(tide_out, f) 
#%%
    tide_file = 'E:\Dropbox\Pilbara Island Remote Sensing\TideData\ExGulf_Tides.txt'
    # import data from tide file
    tideraw = pd.read_csv(tide_file, sep='\t')
    tideraw = tideraw.values

    #create tide_data dictionary    
    tide_data = {'dates': [], 'tide': []}

    for i,row in enumerate(tideraw):
        #convert tide time to UTC from WA local time (UTC+8 hrs) - assumes text file has time in Local time
        dumtime = datetime(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))
        dumtime = datetime.timestamp(dumtime)
        dumtime = datetime.fromtimestamp(dumtime,tz=timezone.utc)
        tide_data['dates'].append(dumtime)
        tide_data['tide'].append(row[-1])
    
    # extract tide heights corresponding to shoreline detections
    tide = []
    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start
    
    for i,date in enumerate(output['dates']):
        print('\rCalculating tides: %d%%' % int((i+1)*100/len(output['dates'])), end='')
        tide.append(tide_data['tide'][find(min(item for item in tide_data['dates'] if item > date), tide_data['dates'])])
    
    tide = np.array(tide)
#%%
    cross_distance_corrected = dict([])
    #Check that length of tide time series is same as SDS timeseries
    #Cross distance should have at least 1 transect
    if len(cross_distance['1'])==len(tide['ztide_image']):
        #Cyclone through all transects
        for key,transect in cross_distance.items():
            transect_corrected = []   
          
            for i,ztide in enumerate(tide['ztide_image']):
                delX = (zref-ztide)/beta             
                transect_corrected.append(transect[i]+delX)
                
            transect_corrected = np.array(transect_corrected)
            cross_distance_corrected[key] = transect_corrected        
    else:
        print('ERROR - time series not same lenght!')
#%%
        for i, sand in enumerate(output['sand_points']):
            plt.plot(sand[:,0],sand[:,1],'.')
        
        for key, transect in transects.items():
            plt.plot(transect[:,0],transect[:,1],'b-')
        
        plt.grid()
    
    #%%
    
from matplotlib import gridspec
fig = plt.figure()
gs = gridspec.GridSpec(len(cross_distance),1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i,key in enumerate(cross_distance.keys()):
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([-400,400])
    if not i == len(cross_distance.keys()):
        ax.set_xticks = []
    ax.plot(output['dates'], cross_distance[key], '-^', markersize=6)
    ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95,'Transect ' + key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
            va='top', transform=ax.transAxes, fontsize=14)
mng = plt.get_current_fig_manager()                                         
mng.window.showMaximized()    
fig.set_size_inches([15.76,  8.52])

#%% Transect calculation

    if settings['check_detection_sand_poly']:
        shorelines = output['sand_points']
    else:
        shorelines = output['shorelines']
    
    along_dist = settings['along_dist']
    
    # initialise variables
    chainage_mtx = np.zeros((len(shorelines),len(transects),6))
    idx_points = []
    
    #for i in range(len(shorelines)):

        sl = shorelines[i]
        idx_points_all = []
        
        for j,key in enumerate(list(transects.keys())): 
            
            # compute rotation matrix
            X0 = transects[key][0,0]
            Y0 = transects[key][0,1]
            temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0,Y0])
            p2 = transects[key][-1,:]
            d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km 
            logic_close = np.logical_and(d_line <= along_dist, d_origin <= 1000)
            idx_close = SDS_transects.find_indices(logic_close, lambda e: e == True)
            idx_points_all.append(idx_close)
            
            # in case there are no shoreline points close to the transect 
            if not idx_close:
                chainage_mtx[i,j,:] = np.tile(np.nan,(1,6))
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                                   [Y0]]), (1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot, xy_close)
                    
                # compute mean, median, max, min and std of chainage position
                n_points = len(xy_rot[0,:])
                mean_cross = np.nanmean(xy_rot[0,:])
                median_cross = np.nanmedian(xy_rot[0,:])
                max_cross = np.nanmax(xy_rot[0,:])
                min_cross = np.nanmin(xy_rot[0,:])
                std_cross = np.nanstd(xy_rot[0,:])
                # store all statistics
                chainage_mtx[i,j,:] = np.array([mean_cross, median_cross, max_cross,
                            min_cross, n_points, std_cross])
    
        # store the indices of the shoreline points that were used
        idx_points.append(idx_points_all)
     
    # format into dictionnary
    chainage = dict([])
    chainage['mean'] = chainage_mtx[:,:,0]
    chainage['median'] = chainage_mtx[:,:,1]
    chainage['max'] = chainage_mtx[:,:,2]
    chainage['min'] = chainage_mtx[:,:,3]
    chainage['npoints'] = chainage_mtx[:,:,4]
    chainage['std'] = chainage_mtx[:,:,5]
    chainage['idx_points'] = idx_points
        
    # only return the median
    cross_dist = dict([])
    for j,key in enumerate(list(transects.keys())): 
        cross_dist[key] = chainage['median'][:,j]    
        