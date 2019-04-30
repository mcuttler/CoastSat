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
tide_out['ztide_image']=tide['ztide'][ind_min]

time_image_UTC = []
for i,time in enumerate(tide_out['time_image']):
    time_image_UTC.append(datetime.timestamp(datetime.fromtimestamp(time)-timedelta(hours=8)))

tide_out['time_image_UTC']=time_image_UTC
#save tide data
filepath = 'P:\CUTTLER_CoastSat\CoastSat\data'
with open(os.path.join(filepath, 'ExTide.pkl'), 'wb') as f:
        pickle.dump(tide_out, f) 


