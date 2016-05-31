#!/usr/bin/env python

#
# Last modified: 26 May 2016
# Author: Darrick Lee <y.l.darrick@gmail.com>
# This file performs k-means clustering on the cells.
#
# Required Packages: scikit-learn
#
# Tutorial: https://github.com/jakevdp/sklearn_pycon2015

import os
import subprocess
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.image as MPIMG
from shlex import split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from optparse import OptionParser

inputFile = None
frame = -1
outFile = 'out.csv'
numCluster = 2
dotSize = 5

# Name of image folder and file
imageFolder = 'OutlineCellsGreen/'
imageName = 'OutlineCells'

parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to csv file containing data", metavar="INPUT")
parser.add_option("-f", "--frame", action="store", type="int", dest="frame", help="frame to analyze", metavar="FRAME")
parser.add_option("-c", "--cluster", action="store", type="int", dest="cluster", help="number of clusters", metavar="CLUSTER")

# Options parsing
(options, args) = parser.parse_args()
if options.inputFile:
	inputFile = options.inputFile
if options.frame:
	frame = options.frame
if options.cluster:
	numCluster = options.cluster

tempTruncFile = 'input_trunc_temp.csv'
inputFileName = os.path.splitext(inputFile)[0]

# Generate image file path
frame3c = "%03d"%frame
imagePath = imageFolder + imageName + frame3c + ".png"

# Truncate the csv file
# p1 = subprocess.Popen(split('cat ' + inputFile), stdout=subprocess.PIPE)
# p2 = subprocess.Popen(split('cut -d, -f2,5,9,10,11,13,17,23,25'), stdin=p1.stdout, stdout=subprocess.PIPE)
# p3 = subprocess.Popen(split('grep ^[^,]*,' + frame3c))

subprocess.call(['./trunc_csv.sh', inputFile, frame3c, tempTruncFile])

# Parse truncated csv file
cells = open(tempTruncFile).read().split('\n')[1:-1]
featList = []

for cell in cells:
	fields = cell.split(',')

	cell_id = int(fields[0])
	area = float(fields[2])
	center_x = float(fields[3])
	center_y = float(fields[4])
	eccentricity = float(fields[5])
	major_axis = float(fields[6])
	minor_axis = float(fields[7])
	perimeter = float(fields[8])

	featList.append(NP.array([cell_id, center_x, center_y, area, perimeter, major_axis, minor_axis, eccentricity]))

# Perform perimeter-eccentricity (PE) clustering
est_PE = KMeans(numCluster)
featList = NP.array(featList)
X_PE = featList[:,[4,7]]

est_PE.fit(X_PE)
y_PE = est_PE.predict(X_PE)

# If cluster center 0 is larger than cluster center 1, then flip 0 and 1
est_PE_cc = est_PE.cluster_centers_
if NP.linalg.norm(est_PE_cc[0,:]) > NP.linalg.norm(est_PE_cc[1,:]):
	y_PE_orig = NP.copy(y_PE)
	y_PE[y_PE_orig == 0] = 1
	y_PE[y_PE_orig == 1] = 0

# Perform area-perimeter (AP) clustering
est_AP = KMeans(numCluster)
featList = NP.array(featList)
X_AP = featList[:,[3,4]]

est_AP.fit(X_AP)
y_AP = est_AP.predict(X_AP)

# If cluster center 0 is larger than cluster center 1, then flip 0 and 1
est_AP_cc = est_AP.cluster_centers_
if NP.linalg.norm(est_AP_cc[0,:]) > NP.linalg.norm(est_AP_cc[1,:]):
	y_AP_orig = NP.copy(y_AP)
	y_AP[y_AP_orig == 0] = 1
	y_AP[y_AP_orig == 1] = 0

score = accuracy_score(y_AP, y_PE)*100
print('The two clusters agree on {0:.2f}% of points.'.format(score))

# Plot Results
# PE Cluster Plot
numFigs = 0
colors = PLT.cm.Paired(NP.linspace(0,1,numCluster))

fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)
for i, c in enumerate(colors):
	PLT.scatter(X_PE[y_PE==i, 0], X_PE[y_PE==i, 1], color=c, edgecolor='black', s=dotSize)

PLT.xlabel('Perimeter')
PLT.ylabel('Eccentricity')
PLT.savefig(inputFileName + frame3c + '_ClusterPE.png', bbox_inches='tight', dpi = 400)

# PE Cluster Result on Image
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)

img = MPIMG.imread(imagePath)
PLT.imshow(img)
center = featList[:,[1,2]]

for i, c in enumerate(colors):
	PLT.scatter(center[y_PE==i, 0], center[y_PE==i, 1], color=c, edgecolor='black', s=dotSize);
PLT.axis('off')
lgd = PLT.legend(["Smaller Cells","Larger Cells"], 
	bbox_to_anchor=(0.0, 1.1, 1.0, 1.5), loc=3, ncol=2, mode="expand", borderaxespad=0.2, fancybox=True, shadow=True)
PLT.savefig(inputFileName + frame3c + '_ImagePE.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 400)

# AP Cluster Plot
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)
for i, c in enumerate(colors):
	PLT.scatter(X_AP[y_AP==i, 0], X_AP[y_AP==i, 1], color=c, edgecolor='black', s=dotSize)
PLT.xlabel('Area')
PLT.ylabel('Perimeter')
PLT.savefig(inputFileName + frame3c + '_ClusterAP.png', bbox_inches='tight', dpi = 400)

# AP Cluster Result on Image
numFigs += 1
fig = PLT.figure(numFigs)
fig.patch.set_alpha(0)

PLT.imshow(img)
for i, c in enumerate(colors):
	PLT.scatter(center[y_AP==i, 0], center[y_AP==i, 1], color=c, edgecolor='black', s=dotSize);
PLT.axis('off')
lgd = PLT.legend(["Smaller Cells","Larger Cells"],
	bbox_to_anchor=(0.0, 1.1, 1.0, 1.5), loc=3, ncol=2, mode="expand", borderaxespad=0.2, fancybox=True, shadow=True)
PLT.savefig(inputFileName + frame3c + '_ImageAP.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 400)

os.remove(tempTruncFile)