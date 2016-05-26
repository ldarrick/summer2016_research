#!/usr/bin/env python

#
# Last modified: 25 May 2016
# Author: Darrick Lee <y.l.darrick@gmail.com>
# This file performs k-means clustering on the cells.
# 
# Note: This code assumes that the csv file has been truncated, and only has the following columns:
# Image Number, Object Number, Area, Center_x, Center_y, Eccentricity, Major Axis Length, Minor Axis Length, Perimeter

import subprocess
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.image as MPIMG
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from optparse import OptionParser

inputFile = None
frame = -1
outFile = 'out.csv'
numCluster = 2

# Name of images
imageFolder = 'OutlineCells/'
imageName = 'OutlineCells'

parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to csv file containing data", metavar="INPUT")
parser.add_option("-f", "--frame", action="store", type="int", dest="frame", help="frame to analyze", metavar="HEIGHT")
parser.add_option("-o", "--output", action="store", type="string", dest="outFile", help="file name of output truncated file", metavar="OUTFILE")
parser.add_option("-c", "--cluster", action="store", type="int", dest="cluster", help="number of clusters", metavar="CLUSTER")

# Options parsing
(options, args) = parser.parse_args()
if options.inputFile:
	inputFile = options.inputFile
if options.frame:
	frame = options.frame
if options.outFile:
	outFile = options.outFile
if options.cluster:
	numCluster = options.cluster

# Generate image file path
frame3c = "%03d"%frame
imagePath = imageFolder + imageName + frame3c + ".png"

# Truncate the csv file
subprocess.call(['./trunc_csv.sh',inputFile, str(frame), outFile])

# Parse truncated csv file
cells = open(outFile).read().split('\n')[1:-1]
featList = []

for cell in cells:
	fields = cell.split(',')

	cell_id = int(fields[1])
	area = float(fields[2])
	center_x = float(fields[3])
	center_y = float(fields[4])
	eccentricity = float(fields[5])
	major_axis = float(fields[6])
	minor_axis = float(fields[7])
	perimeter = float(fields[8])

	featList.append(NP.array([cell_id, center_x, center_y, area, perimeter, major_axis, minor_axis, eccentricity]))

# Perform area-perimeter (AP) clustering
est_AP = KMeans(numCluster)
featList = NP.array(featList)
X_AP = featList[:,[3,4]]

est_AP.fit(X_AP)
y_AP = est_AP.predict(X_AP)

# Perform PCA clustering
X_prePCA = featList[:,[3,4,5,6,7]]
pca = PCA(2)
X_PCA = pca.fit_transform(X_prePCA)

est_PCA = KMeans(numCluster)
est_PCA.fit(X_PCA)
y_PCA = est_PCA.predict(X_PCA)

score = accuracy_score(y_PCA, y_AP)*100
print('The two clusters agree on {0:.2f}% of points.'.format(score))

# Plot Results

# AP Cluster Plot
numFigs = 0
PLT.figure(numFigs)
PLT.scatter(X_AP[:, 0], X_AP[:, 1], c=y_AP, s=8, cmap='rainbow');
PLT.savefig(inputFile + frame3c + '_ClusterAP.png', bbox_inches='tight', dpi = 400)

# AP Cluster Result on Image
numFigs += 1
PLT.figure(numFigs)
img = MPIMG.imread(imagePath)
PLT.imshow(img)
center = featList[:,[1,2]]
PLT.scatter(center[:, 0], center[:, 1], c=y_AP, s=8, cmap='rainbow');
PLT.savefig(inputFile + frame3c + '_ImageAP.png', bbox_inches='tight', dpi = 400)

# PCA Cluster Plot
numFigs += 1
PLT.figure(numFigs)
PLT.scatter(X_PCA[:, 0], X_PCA[:, 1], c=y_PCA, s=8, cmap='rainbow');
PLT.savefig(inputFile + frame3c + '_ClusterPCA.png', bbox_inches='tight', dpi = 400)

# PCA Cluster Result on Image
numFigs += 1
PLT.figure(numFigs)
img = MPIMG.imread(imagePath)
PLT.imshow(img)
PLT.scatter(center[:, 0], center[:, 1], c=y_PCA, s=8, cmap='rainbow');
PLT.savefig(inputFile + frame3c + '_ImagePCa.png', bbox_inches='tight', dpi = 400)





