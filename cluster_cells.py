#!/usr/bin/env python

#
# Last modified: 26 May 2016
# Author: Darrick Lee <y.l.darrick@gmail.com>
# This file performs clustering on cells. The type of clustering performed can be specified.
#
# type = 1: k-means clustering
# type = 2: agglomerative clustering
#
# Required Packages: scikit-learn
#
# Tutorial: https://github.com/jakevdp/sklearn_pycon2015
#
# Ty

import os
import csv
import subprocess
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.image as MPIMG
from shlex import split
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from optparse import OptionParser

inputFile = None
frame = -1
outFile = 'out.csv'
numCluster = 2
imgDotSize = 5
clusterDotSize = 12
pixLength = 0.8
clusterType = 1

# HARD CODED FILE NAMES
# CODE CURRENTLY WORKS FOR FRAME = 10
greenInput = 'Experimental Data/Green Channel/AllMyExptMyCells.csv' # Green cells data file
greenVelocity = 'Experimental Data/Green Channel/Green_Velocity_Frame010.csv' # Green cells velocity at frame 10
greenImageFolder = 'Experimental Data/Green Channel/OutlineCells/OutlineCells010.png' # Green cells image folder
redInput = 'Experimental Data/Red Channel/AllMyExptMyCells.csv' # Red cells data file
redVelocity = 'Experimental Data/Red Channel/Red_Velocity_Frame010.csv' # Red cells velocity at frame 10
redImageFolder = 'Experimental Data/Red Channel/OutlineCells/OutlineCells010.png' # Red cells image folder
frame = 10;

clusterFeat_X = 'Centroid X (um)'
clusterFeat_Y = 'Speed (um/min)'


## OPTIONS CURRENTLY COMMENTED OUT AS FILE/FOLDERS ARE HARD CODED AS ABOVE
# parser = OptionParser()
# parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to csv file containing data", metavar="INPUT")
# parser.add_option("-f", "--frame", action="store", type="int", dest="frame", help="frame to analyze", metavar="FRAME")
# parser.add_option("-c", "--cluster", action="store", type="int", dest="cluster", help="number of clusters", metavar="CLUSTER")
# parser.add_option("-t", "--type", action="store", type="int", dest="clusterType", help="what type of clustering to perform", metavar="TYPE")

# # Options parsing
# (options, args) = parser.parse_args()
# if options.inputFile:
# 	inputFile = options.inputFile
# if options.frame:
# 	frame = options.frame
# if options.cluster:
# 	numCluster = options.cluster
# if options.clusterType:
# 	if options.clusterType == 1 or options.clusterType == 2:
# 		clusterType = options.clusterType;
# 	else:
# 		print("Error: Invalid cluster type.\n")
# 		print("type = 1: k-means clustering\n")
# 		print("type = 2: agglomerative clustering\n")
# 		sys.exit(1)

# # Set up
# inputFileName = os.path.splitext(inputFile)[0]

# # Generate image file path
# frame3c = "%03d"%frame
# imagePath = imageFolder + imageName + frame3c + ".png"

imagePath = 'Images/'

## PARSE DATA FILES ######################################################
fh_red = open(redInput, "rb")
lines_red = fh_red.readlines()
linereader_red = csv.DictReader(lines_red)
featDict_red =dict()

for row in linereader_red:
	if(int(row['Metadata_FrameNumber']) == frame):

		cell_id = int(row['ObjectNumber'])
		area = float(row['AreaShape_Area'])
		center_x = float(row['AreaShape_Center_X'])
		center_y = float(row['AreaShape_Center_Y'])
		eccentricity = float(row['AreaShape_Eccentricity'])
		major_axis = float(row['AreaShape_MajorAxisLength'])
		minor_axis = float(row['AreaShape_MinorAxisLength'])
		perimeter = float(row['AreaShape_Perimeter'])

		featDict_red[cell_id] = [center_x, center_y, area, perimeter, major_axis, minor_axis, eccentricity]

fh_green = open(greenInput, "rb")
lines_green = fh_green.readlines()
linereader_green = csv.DictReader(lines_green)
featDict_green =dict()

for row in linereader_green:
	if(int(row['Metadata_FrameNumber']) == frame):

		cell_id = int(row['ObjectNumber'])
		area = float(row['AreaShape_Area'])
		center_x = float(row['AreaShape_Center_X'])
		center_y = float(row['AreaShape_Center_Y'])
		eccentricity = float(row['AreaShape_Eccentricity'])
		major_axis = float(row['AreaShape_MajorAxisLength'])
		minor_axis = float(row['AreaShape_MinorAxisLength'])
		perimeter = float(row['AreaShape_Perimeter'])

		featDict_green[cell_id] = [center_x, center_y, area, perimeter, major_axis, minor_axis, eccentricity]


## PARSE VELOCITY FILES ##################################################
fh_vred = open(redVelocity, "rb")
lines_vred = fh_vred.readlines()
linereader_vred = csv.DictReader(lines_vred)
featList_red = []

for row in linereader_vred:
	cell_id = int(row['ObjectNumber'])
	velocity_x = float(row['Velocity_X'])
	velocity_y = float(row['Velocity_Y'])
	speed = NP.sqrt(velocity_x**2 + velocity_y**2)

	featList_red.append([cell_id] + featDict_red[cell_id] + [velocity_x, velocity_y, speed])

fh_vgreen = open(greenVelocity, "rb")
lines_vgreen = fh_vgreen.readlines()
linereader_vgreen = csv.DictReader(lines_vgreen)
featList_green = []

for row in linereader_vgreen:
	cell_id = int(row['ObjectNumber'])
	velocity_x = float(row['Velocity_X'])
	velocity_y = float(row['Velocity_Y'])
	speed = NP.sqrt(velocity_x**2 + velocity_y**2)

	featList_green.append([cell_id] + featDict_green[cell_id] + [velocity_x, velocity_y, speed])


## K-MEANS CLUSTERING #####################################################

# Perform k-means clustering on red cells
est_red = KMeans(numCluster)
featList_red = NP.array(featList_red)
X_red = featList_red[:,[1,10]]

est_red.fit(X_red)
y_red = est_red.predict(X_red)

# If cluster center 0 is larger than cluster center 1, then flip 0 and 1
est_red_cc = est_red.cluster_centers_
if NP.linalg.norm(est_red_cc[0,:]) > NP.linalg.norm(est_red_cc[1,:]):
	y_red_orig = NP.copy(y_red)
	y_red[y_red_orig == 0] = 1
	y_red[y_red_orig == 1] = 0

# Perform k-means clustering on green cells
est_green = KMeans(numCluster)
featList_green = NP.array(featList_green)
X_green = featList_green[:,[1,10]]

est_green.fit(X_green)
y_green = est_green.predict(X_green)

# If cluster center 0 is larger than cluster center 1, then flip 0 and 1
est_green_cc = est_green.cluster_centers_
if NP.linalg.norm(est_green_cc[0,:]) > NP.linalg.norm(est_green_cc[1,:]):
	y_green_orig = NP.copy(y_green)
	y_green[y_green_orig == 0] = 1
	y_green[y_green_orig == 1] = 0


## AGGLOMERATIVE CLUSTERING ###############################################
aggl_red = AgglomerativeClustering(numCluster)
X_aggl_red = featList_red[:,3:]
y_aggl_red = aggl_red.fit_predict(X_aggl_red)

aggl_green = AgglomerativeClustering(numCluster)
X_aggl_green = featList_green[:,3:]
y_aggl_green = aggl_green.fit_predict(X_aggl_green)

# If cluster center 0 is larger than cluster center 1, then flip 0 and 1
aggl_cc = est_green.cluster_centers_
if NP.linalg.norm(est_green_cc[0,:]) > NP.linalg.norm(est_green_cc[1,:]):
	y_green_orig = NP.copy(y_green)
	y_green[y_green_orig == 0] = 1
	y_green[y_green_orig == 1] = 0

## PLOT KMEANS RESULTS ####################################################
numFigs = 0
colors = PLT.cm.Paired(NP.linspace(0,1,numCluster))

## RED PLOTS ###########################
# Cluster Plot
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)
for i, c in enumerate(colors):
	PLT.scatter(X_red[y_red==i, 0]*0.8, X_red[y_red==i, 1]*0.8*0.2, color=c, edgecolor='black', s=clusterDotSize)

PLT.ylim((0,12))
PLT.xlabel(clusterFeat_X)
PLT.ylabel(clusterFeat_Y)
PLT.savefig('RedKMeansCluster' + '_Frame010.png', bbox_inches='tight', dpi = 400)

# Image Plot
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)

img = MPIMG.imread(redImageFolder)
PLT.imshow(img)
center = featList_red[:,[1,2]]

for i, c in enumerate(colors):
	PLT.scatter(center[y_red==i, 0], center[y_red==i, 1], color=c, edgecolor='black', s=imgDotSize);
PLT.axis('off')

PLT.savefig('RedKMeansIMG' + '_Frame010.png', bbox_inches='tight', dpi = 400)

## GREEN PLOTS #########################
# Cluster Plot
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)
for i, c in enumerate(colors):
	PLT.scatter(X_green[y_green!=i, 0]*0.8, X_green[y_green!=i, 1]*0.8*0.2, color=c, edgecolor='black', s=clusterDotSize)

PLT.ylim((0,12))
PLT.xlabel(clusterFeat_X)
PLT.ylabel(clusterFeat_Y)
PLT.savefig('GreenKMeansCluster'+ '_Frame010.png', bbox_inches='tight', dpi = 400)

# Image Plot
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)

img = MPIMG.imread(greenImageFolder)
PLT.imshow(img)
center = featList_green[:,[1,2]]

for i, c in enumerate(colors):
	PLT.scatter(center[y_green!=i, 0], center[y_green!=i, 1], color=c, edgecolor='black', s=imgDotSize);
PLT.axis('off')

PLT.savefig('GreenKMeansIMG' + '_Frame010.png', pad_inches=0.0, bbox_inches='tight', dpi = 400)



## PLOT AGGLOMERATIVE RESULTS ################################################
## RED PLOTS #########################
# Cluster Plot
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)
PLT.ylim((0,12))
for i, c in enumerate(colors):
	PLT.scatter(X_red[y_aggl_red!=i, 0]*0.8, X_green[y_aggl_red!=i, 1]*0.8*0.2, color=c, edgecolor='black', s=clusterDotSize)
PLT.xlabel(clusterFeat_X)
PLT.ylabel(clusterFeat_Y)
PLT.savefig('RedAgglomerativeCluster' + '_Frame010.png', bbox_inches='tight', dpi = 400)

# Image Plot
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)

img = MPIMG.imread(redImageFolder)
PLT.imshow(img)
center = featList_red[:,[1,2]]

for i, c in enumerate(colors):
	PLT.scatter(center[y_aggl_red!=i, 0], center[y_aggl_red!=i, 1], color=c, edgecolor='black', s=imgDotSize);
PLT.axis('off')

PLT.savefig('RedAgglomerativeIMG' + '_Frame010.png', bbox_inches='tight', dpi = 400)


## GREEN PLOTS #########################
# Cluster Plot
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)
PLT.ylim((0,12))
for i, c in enumerate(colors):
	PLT.scatter(X_green[y_aggl_green!=i, 0]*0.8, X_green[y_aggl_green!=i, 1]*0.8*0.2, color=c, edgecolor='black', s=clusterDotSize)
PLT.xlabel(clusterFeat_X)
PLT.ylabel(clusterFeat_Y)
PLT.savefig('GreenAgglomerativeCluster' + '_Frame010.png', bbox_inches='tight', dpi = 400)

# Image Plot
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)

img = MPIMG.imread(greenImageFolder)
PLT.imshow(img)
center = featList_green[:,[1,2]]

for i, c in enumerate(colors):
	PLT.scatter(center[y_aggl_green!=i, 0], center[y_aggl_green!=i, 1], color=c, edgecolor='black', s=imgDotSize);
PLT.axis('off')

PLT.savefig('GreenAgglomerativeIMG' + '_Frame010.png', bbox_inches='tight', dpi = 400)

