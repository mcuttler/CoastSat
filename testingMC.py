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
