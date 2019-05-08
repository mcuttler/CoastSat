# -*- coding: utf-8 -*-
"""
Compare output across all Pilbara Islands

M Cuttler 
May 2019
"""
#%% import all islands tide_corrected data
import os
import pickle
import matplotlib.pyplot as plt

islands = ['EVA','FLY','OBSERVATION','ASHBURTON','LOCKER','Y']
filepath_data = os.path.join(os.getcwd(), 'data')

all_islands = dict([])
for i,island in enumerate(islands):
#    print(island)
    filepath = os.path.join(filepath_data, island)
    with open(os.path.join(filepath, island + '_output_tide_corrected' + '.pkl'), 'rb') as f:
        all_islands[island] = pickle.load(f)
        
##Isolate S2 and save new output 
#output_S2 = dict([])
#for i,island in enumerate(all_islands):
#    output_cloud_cover = []
#    output_dates = []
#    output_filename = []
#    output_geoaccuracy = []
#    output_idx = []
#    output_sand_area = []
#    output_sand_centroid = []
#    output_sand_perimeter=[]
#    output_sand_points=[]
#    output_satname=[]
#    output_shorelines=[]
#    output_tide=[]
#    output_sand_points_corrected=[]
#    output_sand_area_corrected=[]
#    output_sand_perimeter_corrected=[]
#    output_sand_centroid_corrected=[]
#    output_sand_points_poly_corrected = []    
#
#    for j,sat in enumerate(all_islands[island]['satname']):      
#        if sat=='S2':
#             output_cloud_cover.append(all_islands[island]['cloud_cover'][j])
#             output_dates.append(all_islands[island]['dates'][j])
#             output_filename.append(all_islands[island]['filename'][j])
#             output_geoaccuracy.append(all_islands[island]['geoaccuracy'][j])
#             output_idx.append(all_islands[island]['idx'][j])
#             output_sand_area.append(all_islands[island]['sand_area'][j])
#             output_sand_centroid.append(all_islands[island]['sand_centroid'][j])
#             output_sand_perimeter.append(all_islands[island]['sand_perimeter'][j])
#             output_sand_points.append(all_islands[island]['sand_points'][j])
#             output_satname.append(all_islands[island]['satname'][j])
#             output_shorelines.append(all_islands[island]['shorelines'][j])
#             output_tide.append(all_islands[island]['tide'][j])
#             output_sand_points_corrected.append(all_islands[island]['sand_points_corrected'][j])
#             output_sand_area_corrected.append(all_islands[island]['sand_area_corrected'][j])
#             output_sand_perimeter_corrected.append(all_islands[island]['sand_perimeter_corrected'][j])
#             output_sand_centroid_corrected.append(all_islands[island]['sand_centroid_corrected'][j])
#             output_sand_points_poly_corrected.append(all_islands[island]['sand_points_poly_corrected'][j])
#        else:
#             continue
#    output_S2[island] = {'cloud_cover': output_cloud_cover,
#             'dates': output_dates,
#             'filename': output_filename,
#             'geoaccuracy': output_geoaccuracy,
#             'idx': output_idx,
#             'sand_area': output_sand_area,
#             'sand_centroid': output_sand_centroid,
#             'sand_perimeter': output_sand_perimeter,
#             'sand_points': output_sand_points,
#             'satname': output_satname,
#             'shorelines': output_shorelines,
#             'tide': output_tide,
#             'sand_points_corrected': output_sand_points_corrected,
#             'sand_area_corrected': output_sand_area_corrected,
#             'sand_perimeter_corrected': output_sand_perimeter_corrected,
#             'sand_centroid_corrected': output_sand_centroid_corrected,
#             'sand_points_poly_corrected': output_sand_points_poly_corrected}
#    
#    
#with open(os.path.join(filepath_data, 'PilbaraIslands_output_S2' + '.pkl'), 'wb') as f:
#    pickle.dump(output_S2,f)

#%% Write all output to CSV files for further processing
import os
import pandas as pd
import pickle

with open('P:\CUTTLER_CoastSat\CoastSat\data\PilbaraIslands_output_S2.pkl', 'rb') as f: 
    all_islands = pickle.load(f)
    
filepath = 'P:\CUTTLER_CoastSat\CoastSat\data'

for i,island in enumerate(all_islands):    
    csv_path = os.path.join(filepath,island, island + '_S2_tide_corrected.csv')    
    data_out = pd.DataFrame.from_dict(all_islands[island])     
    data_out.to_csv(csv_path)

#%% plot figure comparing island area change
    
import os
import pickle
import matplotlib.pyplot as plt
import math

with open('P:\CUTTLER_CoastSat\CoastSat\data\PilbaraIslands_output_S2.pkl', 'rb') as f: 
    all_islands_S2 = pickle.load(f)


islands = ['EVA','FLY','OBSERVATION','ASHBURTON','LOCKER','Y']
c = ['b','r','g','y','k','c']
for i,island in enumerate(all_islands):
    plt.figure()
    plt.plot(all_islands[island]['dates'],all_islands[island]['sand_area_corrected'],c[i]+'.-')
    plt.grid()
    plt.title(islands[i])   
    plt.xlabel('Date')
    plt.ylabel('Tide-corrected island area (m^2)')

#%% plot comparison of islands using L8 and S2 imagery

for i,island in enumerate(all_islands):
    fig = plt.figure()
    plt.plot(all_islands[island]['dates'], all_islands[island]['sand_area_corrected'],'b',label = 'L8+S2')
    plt.plot(all_islands_S2[island]['dates'], all_islands_S2[island]['sand_area_corrected'],'r',label = 'S2 only')
    plt.legend()
    plt.title(island)
    fig.set_size_inches([8,  4])
    figpath = 'E:\Dropbox\Pilbara Island Remote Sensing\CoastSAT\Figures\L8vS2'
    figpath = os.path.join(figpath, island + '_L8vS2.png')
    plt.savefig(

    
    
    

    
#plt.savefig('E:\Dropbox\Pilbara Island Remote Sensing\CoastSAT\Figures\All_islands_area_tide_corrected.png')

#%% plot only S2 data

plt.figure()
c = ['b','r','g','y','k','c']
for i,island in enumerate(all_islands):
    #create plotting variables based on satellite
    dates = []
    sand_area = []
    for j,sat in enumerate(all_islands[island]['satname']):      
        if sat=='S2':
            dates.append(all_islands[island]['dates'][j])
            sand_area.append(all_islands[island]['sand_area_corrected'][j])
        else:
            continue
    plt.plot(dates, sand_area,c[i]+'-')
    
plt.grid()
plt.legend(islands)

plt.xlabel('Date')
plt.ylabel('Tide-corrected island area (m^2)')

plt.savefig('E:\Dropbox\Pilbara Island Remote Sensing\CoastSAT\Figures\All_islands_area_S2_tide_corrected.png')


#%% 


