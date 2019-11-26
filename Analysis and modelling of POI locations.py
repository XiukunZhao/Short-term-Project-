#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:25:43 2017

This file is used to solve the problem at *. Generally, the file consists of four parts as follows: 
1. Cleanup data: Some suspicious records need to be labelled and deleted. There are two types of suspicious records,
                 including the records with identical 'geoinfo' and 'timest' and the records with strange 'geoinfo'.
2. Label: Four POI locations are in 'POIList.csv', but the first two are same, so the first POI is deleted.
          All requests are assigned to the three POI locations based on minimum distances.
3. Analysis: For each POI location, average and standard deviation of distances, radius, and density are calculated 
             and stored in the matrix 'POI_matrix'. A figure shows three circles which are centred at POI locations
             and the corresponding records.
4. Model: Design two mathematics models to map the density of POI locations into a scale from -10 to 10.
          Model 1 is a discrete model with boundary [0, max radius], which does not consider -Inf. 
          Model 2 is a continuous model considering Inf situation, where tanh function is used to approach Inf part. 

"""

import pandas as pd
import collections
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.legend_handler import HandlerLine2D

raw_data = pd.read_csv('EQdata/data/DataSample.csv', sep=',', encoding="ISO-8859-1") 
# print(raw_data.columns)
raw_data = raw_data.rename(columns={' TimeSt': 'TimeSt'})
timest = raw_data.TimeSt
lat = raw_data.Latitude
long = raw_data.Longitude


### Cleanup data ###

raw_data['label'] = np.zeros(len(timest))      # 'label': to indicate whether the record is suspicious   
timest_same = [el for el, count in collections.Counter(timest).items() if count > 1]
lat_same = [el for el, count in collections.Counter(lat).items() if count > 1]
long_same = [el for el, count in collections.Counter(long).items() if count > 1]
print(raw_data['Country'].value_counts())      # information of raw_data.Country ('CA') means we only need to 
                                               # consider the land mass of Canada (42N-83N, 53W-141W)
for i in range(len(raw_data)):
    if (timest[i] in timest_same) and (lat[i] in lat_same) and (long[i] in long_same):
        raw_data.loc[i,'label'] = 1            # label 1: suspicious (same 'geoinfo' and 'timest')
    elif (lat[i]>83) or (lat[i]<42) or (long[i]>-53) or (long[i]<-141):
        raw_data.loc[i,'label'] = 2            # label 2: suspicious (not belong to CA)

suspicious_data = raw_data[raw_data['label']!=0]   
my_data = raw_data[raw_data['label']==0]     
my_data = my_data.drop('label', 1)             # my_data without suspicious records


POI = pd.read_csv('EQdata/data/POIList.csv') 
POI = POI.rename(columns={' Latitude': 'Latitude'})

fig, ax = plt.subplots(1, figsize=(20,10),dpi=80)     # figure 1: to project all records (including suspicious records) to a map
m = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=70,llcrnrlon=-150,urcrnrlon=150,lat_ts=20,resolution='c')
m.drawcoastlines()
m.drawmapboundary()
xs, ys = m(list(suspicious_data["Longitude"].astype(float)), list(suspicious_data["Latitude"].astype(float)))
plt.scatter(xs,ys,10,marker='o',color='red')          # red: suspicious records
x, y = m(list(my_data["Longitude"].astype(float)), list(my_data["Latitude"].astype(float)))
plt.scatter(x,y,10,marker='o',color='green')          # green: useful records
xc, yc = m(list(POI["Longitude"].astype(float)), list(POI["Latitude"].astype(float)))
plt.scatter(xc,yc,60,marker='o',color='yellow')       # yellow: POI locations


#### Label ###
 
POI = POI.drop_duplicates(['Latitude','Longitude'],keep='last')
n_center = len(POI)
n_data = len(my_data)
dist_matrix = np.zeros((n_data,n_center))
for i in range(n_center):
    dist_matrix[:,i] = np.sqrt((my_data['Latitude']-POI.loc[i+1,'Latitude'])**2+(my_data['Longitude']-POI.loc[i+1,'Longitude'])**2)
my_data['Label'] = np.argmin(dist_matrix, axis=1)   # Label: 0 (assigned to POI2), 1 (assigned to POI3), 2 (assigned to POI4)


#### Analysis ###

POI2_data = my_data.loc[my_data['Label']==0]
POI2_dist = np.sqrt((POI2_data['Latitude']-POI.loc[1,'Latitude'])**2+(POI2_data['Longitude']-POI.loc[1,'Longitude'])**2)
POI3_data = my_data.loc[my_data['Label']==1]
POI3_dist = np.sqrt((POI3_data['Latitude']-POI.loc[2,'Latitude'])**2+(POI3_data['Longitude']-POI.loc[2,'Longitude'])**2)
POI4_data = my_data.loc[my_data['Label']==2]
POI4_dist = np.sqrt((POI4_data['Latitude']-POI.loc[3,'Latitude'])**2+(POI4_data['Longitude']-POI.loc[3,'Longitude'])**2)
POI_dist = np.array([POI2_dist,POI3_dist,POI4_dist])

POI_matrix = np.zeros((4,n_center))          # POI_matrix: store calculated mean, SD, radius, and density (row)
                                             # POI2, POI3, POI4 (column)
for i in range(n_center):
    POI_matrix[0,i] = np.mean(POI_dist[i])   # average of distances between POI location and assigned records
    POI_matrix[1,i] = np.std(POI_dist[i])    # standard deviation of distance between POI location and assigned records
    POI_matrix[2,i] = np.max(POI_dist[i])    # radius of the circle including POI location and assigned records
    POI_matrix[3,i] = len(POI_dist[i])/(np.pi*(POI_matrix[2,i]**2))  # density of the circle including POI location and assigned records


fig = plt.figure(2, figsize=(10,10),dpi=80)   # figure 2: POI locations and all records
plt.axis([-150,-50,0,100])
plt.scatter(POI2_data['Longitude'],POI2_data['Latitude'],label='records',color='#a020f0')
plt.scatter(POI3_data['Longitude'],POI3_data['Latitude'],label='records',color='#ff69b4')
plt.scatter(POI4_data['Longitude'],POI4_data['Latitude'],label='records',color='#ffa500')
plt.scatter(POI['Longitude'],POI['Latitude'],label='POI',color='r')
ax = fig.add_subplot(1,1,1)
circ1 = plt.Circle((1.5+POI.loc[1,'Longitude']/100,POI.loc[1,'Latitude']/100), radius=POI_matrix[2,0]/100, color='#9900d3', fill=False,transform=ax.transAxes)
ax.add_patch(circ1)
circ2 = plt.Circle((1.5+POI.loc[2,'Longitude']/100,POI.loc[2,'Latitude']/100), radius=POI_matrix[2,1]/100, color='#ff1493', fill=False,transform=ax.transAxes)
ax.add_patch(circ2)
circ3 = plt.Circle((1.5+POI.loc[3,'Longitude']/100,POI.loc[3,'Latitude']/100), radius=POI_matrix[2,2]/100, color='#ff6347', fill=False,transform=ax.transAxes)
ax.add_patch(circ3)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('POI locations and the corresponding records')


### Model ###

# Model 1 (discrete model)
POI2_radii = np.arange(1, math.ceil(POI_matrix[2,0]), math.floor(POI_matrix[2,0])/math.floor(POI_matrix[2,1]))
POI3_radii = np.arange(1, math.ceil(POI_matrix[2,1]))
POI4_radii = np.arange(1, math.ceil(POI_matrix[2,2]), math.floor(POI_matrix[2,2])/math.floor(POI_matrix[2,1]))

POI_density = np.zeros((len(POI2_radii),n_center))
for j in range(len(POI2_radii)):
    POI_density[j,0] = sum(i<POI2_radii[j] for i in POI2_dist)/(np.pi*(POI2_radii[j]**2))
    POI_density[j,1] = sum(i<POI3_radii[j] for i in POI3_dist)/(np.pi*(POI3_radii[j]**2)) 
    POI_density[j,2] = sum(i<POI4_radii[j] for i in POI4_dist)/(np.pi*(POI4_radii[j]**2))
    
d = np.reshape(POI_density.T,len(POI2_radii)*n_center)
factor = (max(d)-min(d))/20     # linear scaling 
d1 = d/factor-(max(d/factor)-10)
density_scale = d1-np.mean(d1)
d_max = np.max(density_scale)
d_min = np.abs(np.min(density_scale))

for j in range(len(density_scale)):     # arrage density distribution in order to make model to be sensitive around mean
   if density_scale[j]>0:
       density_scale[j] = density_scale[j]/d_max*10
   else:
       density_scale[j] = density_scale[j]/d_min*10    

fig = plt.figure(3)    # figure 3: to show population of POIs
plt.subplot(311)
plt.plot(POI2_radii,density_scale[0:20])
plt.ylabel('density of POI2')
plt.xlabel('radius')
plt.title('Population of POI locations')
plt.subplot(312)
plt.plot(POI3_radii,density_scale[20:40])
plt.ylabel('density of POI3')
plt.xlabel('radius')
plt.subplot(313)
plt.plot(POI3_radii,density_scale[40:60])
plt.ylabel('density of POI4')
plt.xlabel('radius')


# Model 2 (continuous model)
r_max = math.ceil(max(POI_matrix[2,:]))
POI_radii = np.arange(0.1, r_max, 0.1)
POI_d = np.zeros((len(POI_radii),n_center))
for j in range(len(POI_radii)):
    POI_d[j,0] = sum(i<POI_radii[j] for i in POI2_dist)/(np.pi*(POI_radii[j]**2))
    POI_d[j,1] = sum(i<POI_radii[j] for i in POI3_dist)/(np.pi*(POI_radii[j]**2)) 
    POI_d[j,2] = sum(i<POI_radii[j] for i in POI4_dist)/(np.pi*(POI_radii[j]**2))

NN = len(POI_radii)*n_center    
d_c = np.reshape(POI_d.T,NN)

p = 0.95           
limit_n = math.ceil(NN*p)
d_c.sort()
limit = d_c[limit_n]

a = [el for el in d_c if (el > 0) and (el<limit)]
e = (1-p)/2*20
A = a-(a[-1]+a[0])/2
f = 20*p/(a[-1]-a[0])
B = A*f

dd_c = np.reshape(POI_d.T,NN)
D = dd_c-(a[-1]+a[0])/2
DD = D*f
D1 = DD[0:249]
xx = np.arctanh(abs(min(D1))/10)
for j in range(len(POI_radii)):
    if D1[j]>9.5:
       D1[j] = np.tanh(D1[j])*p*10
    elif D1[j]<-9.5:
       D1[j] = -np.tanh(D1[j])*p*10

k = np.arange(POI_radii[-1],100, 0.1)
y = POI_radii[-1]/100-xx
Y = -np.tanh((k/100-y))*10  
     
fig = plt.figure(4)      # POI2 density
line1, = plt.plot(POI_radii,D1,label="finite part")
plt.plot(k,Y,label="infinite part")
plt.ylabel('density of POI2')
plt.xlabel('radius')
plt.title('Population of POI2')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

D2 = DD[249:249*2]
xx = np.arctanh(abs(min(D2))/10)
for j in range(len(POI_radii)):
    if D2[j]>9.5:
       D2[j] = np.tanh(D2[j])*p*10
    elif D2[j]<-9.5:
       D2[j] = -np.tanh(D2[j])*p*10

k = np.arange(POI_radii[-1],100, 0.1)
y = POI_radii[-1]/100-xx
Y = -np.tanh((k/100-y))*10    
   
fig = plt.figure(5)    # POI3 density
line1, = plt.plot(POI_radii,D2,label="finite part")
plt.plot(k,Y,label="infinite part")
plt.ylabel('density of POI3')
plt.xlabel('radius')
plt.title('Population of POI3')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
       
D3 = DD[249*2:249*3]
xx = np.arctanh(abs(min(D3))/10)
for j in range(len(POI_radii)):
    if D3[j]>9.5:
       D3[j] = np.tanh(D3[j])*p*10
    elif D3[j]<-9.5:
       D3[j] = np.tanh(D3[j])*p*10

k = np.arange(POI_radii[-1],100, 0.1)
y = POI_radii[-1]/100-xx
Y = -np.tanh((k/100-y))*10       
fig = plt.figure(6)    # POI4 density
line1, = plt.plot(POI_radii,D3,label="finite part")
plt.plot(k,Y,label="infinite part")
plt.ylabel('density of POI4')
plt.xlabel('radius')
plt.title('Population of POI4')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})






