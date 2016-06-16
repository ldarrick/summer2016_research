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

def clusterImagePlot(X, y, center, colors, outFolder, fname, pxlabel, pylabel, xlim,ylim):
	# Image par ameters
	imgDotSize = 5
	clusterDotSize = 12

	# Cluster Plot
	fig = PLT.figure()
	for i, c in enumerate(colors):
		PLT.scatter(X[y==i, 0], X[y==i, 1], color=c, edgecolor='black', s=clusterDotSize)

	axes = PLT.gca()
	axes.set_xlim(xlim)
	axes.set_ylim(ylim)

	PLT.xlabel(pxlabel)
	PLT.ylabel(pylabel)
	PLT.savefig(outFolder + fname, bbox_inches='tight', dpi = 400)

	## USED FOR PCA
	# sc = 15
	# xvector = pca.components_[0]
	# yvector = pca.components_[1]
	# features = ['velocity_x','velocity_y','speed','displacement','area','perim','eccen','major','minor','orient'];
	# for i in range(len(xvector)):
	# 	PLT.arrow(0, 0, xvector[i]*sc, yvector[i]*sc,
	# 	          color='g', width=0.005, head_width=0.05)
	# 	PLT.text(xvector[i]*sc*1.1, yvector[i]*sc*1.1,
	# 	         features[i], color='g')

	# Image Plot
	fig = PLT.figure()
	fig.patch.set_alpha(0)
	imgFile = greenImageFolder = 'Experimental Data/Red Channel/OutlineCells/OutlineCells100.png'
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
parser.add_option("-b", "--bins", action="store", type="int", dest="bins", help="number of bins", metavar="BINS")
parser.add_option("-a","--aggregate", action="store_true", dest="aggregate", default=False, help="use aggregate statistics for features")
parser.add_option("-o", "--output", action="store", type="string", dest="outFile", help="path to where output folders will be stored", metavar="OUTPUT")

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
if options.bins:
	numBins = options.bins
if options.outFile:
	outputFolder = options.outFile

# Generate image file path
frame3c = "%03d"%frame
# imagePath = imageFolder + imageName + frame3c + ".png"

# Output Folder Names
XA_folder = outputFolder + 'XA/'
XS_folder = outputFolder + 'XS/'
AP_folder = outputFolder + 'AP/'
SA_folder = outputFolder + 'SA/'
PCA_folder = outputFolder + 'PCA/'

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

aggl_all = AgglomerativeClustering(2)
X_All = featList[:,2:]
y2 = aggl_all.fit_predict(X_All)

## PCA ###############################################################

pca_model = PCA(2)
X_PCA = pca_model.fit_transform(X_All)
print(pca_model.explained_variance_ratio_)

## SPLIT INTO numBINS ###############################################
percentiles = NP.floor(NP.linspace(0,100,numBins+1))
percentiles = percentiles[:-1]

bins = NP.percentile(featList[:,0],list(percentiles))

y = NP.digitize(featList[:,0],bins)-1

X_XS = featListOriginal[:,[0,4]]
X_XA = featListOriginal[:,[0,6]]
X_AP = featListOriginal[:,[6,7]]
X_SA = featListOriginal[:,[4,6]]

## PLOT RESULTS ####################################################
numFigs = 0
center = featList[:,[0,1]]
colorspace = NP.linspace(0,1,numBins+1)[:-1]
colors = PLT.cm.hsv(colorspace)

# clusterImagePlot(X_XS, y, center, colors, XS_folder, 'XS_'+frame3c, 'X Location (um)', 'Speed (um/min)',[0,1200],[-0.5,3])
# clusterImagePlot(X_XA, y, center, colors, XA_folder, 'XA_'+frame3c, 'X Location (um)', 'Area (um^2)',[0,1200],[0,1400])
# clusterImagePlot(X_AP, y, center, colors, AP_folder, 'AP_'+frame3c, 'Area (um^2)', 'Perimeter (um)',[0,1400],[0,250])
# clusterImagePlot(X_SA, y, center, colors, SA_folder, 'SA_'+frame3c, 'Speed (um/min)', 'Area (um^2)',[-0.5,3],[0,1400])
clusterImagePlot(X_PCA, y, center, colors, PCA_folder, 'PCA_'+frame3c, 'PCA 1', 'PCA 2',[-10,10],[-10,10])