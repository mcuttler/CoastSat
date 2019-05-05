# -*- coding: utf-8 -*-
"""
Compare output across all Pilbara Islands

M Cuttler 
May 2019
"""
#%% import all islands tide_corrected data
import os

islands = ['EVA','FLY','OBSERVATION','ASHBURTON','LOCKER','Y']
filepath_data = os.path.join(os.getcwd(), 'data')

all_islands = dict([])
for i,island in enumerate(islands):
#    print(island)
    filepath = os.path.join(filepath_data, island)
    with open(os.path.join(filepath, island + '_output_tide_corrected' + '.pkl'), 'rb') as f:
        all_islands[island] = pickle.load(f)

#%% plot figure comparing island area change
import matplotlib.pyplot as plt

plt.figure()
c = ['b','r','g','y','k','c']
for i,island in enumerate(all_islands):
    plt.plot(all_islands[island]['dates'],all_islands[island]['sand_area_corrected'],c[i]+'-')
    
plt.grid()
plt.legend()
plt.legend(islands)

plt.xlabel('Date')
plt.ylabel('Tide-corrected island area (m^2)')

plt.savefig('E:\Dropbox\Pilbara Island Remote Sensing\CoastSAT\Figures\All_islands_area.png')

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

plt.savefig('E:\Dropbox\Pilbara Island Remote Sensing\CoastSAT\Figures\All_islands_area_S2.png')








