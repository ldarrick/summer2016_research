#!/usr/bin/env python

#
# Last modified: June 9 2016
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

import os
import csv
import subprocess
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.image as MPIMG
from shlex import split
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from optparse import OptionParser
from track_cells import get_future_features, calc_velocity

inputFile = None
frame = -1
numCluster = 2
numFV = 2
clusterType = 1

featureNames = ['AreaShape_Area', 'AreaShape_Perimeter', 
	'AreaShape_Eccentricity', 'AreaShape_MajorAxisLength', 
	'AreaShape_MinorAxisLength', 'AreaShape_Orientation']
imageName = 'OutlineCells'
imgDotSize = 5
clusterDotSize = 12
pixLength = 0.8

def clusterImagePlot(X, y, center, imgFile, colors, outFolder, fname, pxlabel, pylabel, pcacomp=0,featNames=0):
	# Image parameters
	imgDotSize = 5
	clusterDotSize = 12

	# Cluster Plot
	fig = PLT.figure()
	for i, c in enumerate(colors):
		PLT.scatter(X[y==i, 0], X[y==i, 1], color=c, edgecolor='black', s=clusterDotSize)

	if pcacomp!=0:
		xvector = pcacomp[0]
		yvector = pcacomp[1]
		numFeat = len(xvector)

		for i in range(numFeat):
			plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
			          color='r', width=0.005, head_width=0.05)
			plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
			         featNames[i], color='r')


	PLT.xlabel(pxlabel)
	PLT.ylabel(pylabel)
	PLT.savefig(outFolder + 'CL_' + fname, bbox_inches='tight', dpi = 400)

	# Image Plot
	fig = PLT.figure()
	fig.patch.set_alpha(0)

	img = MPIMG.imread(imgFile)
	PLT.imshow(img)

	for i, c in enumerate(colors):
		PLT.scatter(center[y==i, 0], center[y==i, 1], color=c, edgecolor='black', s=imgDotSize);
	PLT.axis('off')

	PLT.savefig(outFolder + 'IMG_' + fname, bbox_inches='tight', dpi = 400)

# Options Parsing
parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to csv file containing data", metavar="INPUT")
parser.add_option("-I", "--imagefolder", action="store", type="string", dest="imageFolder", help="folder where OutlineCell images are stored", metavar="IMGFOLDER")
parser.add_option("-f", "--frame", action="store", type="int", dest="frame", help="frame to analyze", metavar="FRAME")
parser.add_option("-n", "--numFrameV", action="store", type="int", dest="numFV", help="number of future/previous frame to use to calculate velocity", metavar="NUMFRAMEVELOCITY")
parser.add_option("-c", "--cluster", action="store", type="int", dest="cluster", help="number of clusters", metavar="CLUSTER")
parser.add_option("-a","--aggregate", action="store_true", dest="aggregate", default=False, help="use aggregate statistics for features")

# Options parsing
(options, args) = parser.parse_args()
if options.inputFile:
	inputFile = options.inputFile
if options.imageFolder:
	imageFolder = options.imageFolder
if options.frame:
	frame = options.frame
if options.numFV:
	numFV = options.numFV
if options.aggregate:
	aggregate = True
else:
	aggregate = False
if options.cluster:
	numCluster = options.cluster

# Generate image file path
frame3c = "%03d"%frame
imagePath = imageFolder + imageName + frame3c + ".png"

# Output Folder Names
AP_folder = 'AP/'
PCA_folder = 'SA/'
SA_folder = 'clusterSA/'
full_folder = 'clusterFull/'

# Extract features from csv file
time = frame - numFV
cutoff = numFV*2
cID = []
aXY = []
lifetime = []
features = []
numFrames = numFV*2 +1
numFeatures = len(featureNames)

get_future_features(inputFile, time, cutoff, featureNames, cID, aXY, lifetime, features)

# Keep only the cells that are tracked for all frames
for i, cell in reversed(list(enumerate(cID))):
	if len(cell) != numFrames:
		cID.pop(i)
		aXY.pop(i)
		lifetime.pop(i)
		features.pop(i)
numCells = len(lifetime)

aXY_np = NP.array(aXY)
# Calculate the velocity
velocity = calc_velocity(aXY,numFV,numCells)
speed = NP.sqrt(NP.sum(velocity**2,1))
displacement = NP.sqrt(NP.sum((aXY_np[:,0,:]-aXY_np[:,-1,:])**2,1))

## BUILD FEATURE VECTOR ###################################################

featList = []
featList.append([a[numFV][0] for a in aXY]) # x-centroid
featList.append([a[numFV][1] for a in aXY]) # y-centroid
featList.append([v[0] for v in velocity]) # x-velocity
featList.append([v[1] for v in velocity]) # y-velocity
featList.append(speed) # speed
featList.append(displacement) # total displacement

# Add all other features 
for i in range(numFeatures):
	featList.append([f[numFV][i] for f in features])

fullFeatureNames = ['Centroid_X', 'Centroid_Y', 'Velocity_X','Velocity_Y', 'Speed','Displacement']+featureNames
featListOriginal = NP.array(featList)
featList = NP.array(featList)
numFeat = len(featList)

# Transpose and scale parameters
featListOriginal[[0,1,5,7,9,10],:] = featListOriginal[[0,1,5,7,9,10],:]*0.8
featListOriginal[[2,3,4],:] = featListOriginal[[2,3,4],:]*0.8*0.2
featListOriginal[6,:] = featListOriginal[6,:]*0.8*0.8
featListOriginal = NP.transpose(featListOriginal)

## STANDARDIZE FEATURES ###################################################

# Don't standardize the centroids
for k in range(2,numFeat):
	featList[k] = (featList[k] - NP.mean(featList[k]))/NP.sqrt(NP.var(featList[k]))

# Transpose the feature list to use in clustering
featList = NP.transpose(featList)
feat_aggl = FeatureAgglomeration(2)
feat_aggl.fit(featList[:,2:])

## AGGLOMERATIVE CLUSTERING ###############################################

aggl_all = AgglomerativeClustering(numCluster)
X_All = featList[:,2:]
y_aggl_All = aggl_all.fit_predict(X_All)

X_XS = featListOriginal[:,[0,4]]
X_XA = featListOriginal[:,[0,6]]
X_AP = featListOriginal[:,[6,7]]
X_SA = featListOriginal[:,[4,6]]


## PLOT RESULTS ####################################################
numFigs = 0
center = featList[:,[0,1]]
colors = PLT.cm.Paired(NP.linspace(0,1,numCluster))

clusterImagePlot(X_XS, y_aggl_All, center, imagePath, colors, AP_folder, 'AgglomerativeXS', 'X Location (um)', 'Speed (um/min)')
clusterImagePlot(X_XA, y_aggl_All, center, imagePath, colors, AP_folder, 'AgglomerativeXA', ' Location (um)', 'Area (um^2)')
clusterImagePlot(X_AP, y_aggl_All, center, imagePath, colors, AP_folder, 'AgglomerativeAP', 'Area (um^2)', 'Perimeter (um)')
clusterImagePlot(X_SA, y_aggl_All, center, imagePath, colors, AP_folder, 'AgglomerativeSA', 'Speed (um/min)', 'Area (um^2)')