#!/usr/bin/env python

#
# Last modified: June 16 2016
# Author: Darrick Lee <y.l.darrick@gmail.com>
# This file creates a .csv file for the experimental data
#

import sys
import csv
import time
import datetime
import numpy as NP
from optparse import OptionParser
from track_cells import get_future_features, calc_velocity
from scipy.signal import savgol_filter

inputFile = None
frame = -1
numFV = -1
outFile = None
numMov = -1

# Options Parsing
parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to csv file containing data", metavar="INPUT")
parser.add_option("-f", "--frame", action="store", type="int", dest="frame", help="frame to analyze", metavar="FRAME")
parser.add_option("-n", "--numFrameV", action="store", type="int", dest="numFV", help="number of future/previous frame to use to calculate velocity", metavar="NUMFRAMEVELOCITY")
parser.add_option("-o", "--output", action="store", type="string", dest="outFile", help="path to where output folders will be stored", metavar="OUTPUT")
parser.add_option("-m", "--movie", action="store", type="int", dest="movie", help="create feature vector of a movie with aggregate statistics", metavar="MOVNUM")

# Options parsing
(options, args) = parser.parse_args()
if options.inputFile:
	inputFile = options.inputFile
if options.frame:
	frame = options.frame
if options.numFV:
	numFV = options.numFV
if options.outFile:
	outFile = options.outFile
if options.movie:
	numMov = options.movie

# Unit conversion functions
def conv_distance(x):
	conv_factor = 0.8
	units = 'um'
	return NP.array(x)*conv_factor, units

def conv_speed(x):
	conv_factor = 0.2*0.8
	units = 'um/min'
	return NP.array(x)*conv_factor, units

def conv_area(x):
	conv_factor = 0.64
	units = 'um^2'
	return NP.array(x)*conv_factor, units

# Features to keep track of
featureNames = ['AreaShape_Area', 'AreaShape_Perimeter', 
	'AreaShape_Eccentricity', 'AreaShape_MajorAxisLength', 
	'AreaShape_MinorAxisLength', 'AreaShape_Orientation']

# Names for the features in the output csv file
outFeatureNames = ['Area', 'Perimeter', 'Eccentricity',
	'MajorAxis', 'MinorAxis', 'Orientation']

# Parameters for cell tracking
# Note that we start numFV frames behind the current frame so that we can use the SG filter
# for the given window size
frame_start = frame - numFV
cutoff = numFV*2
cID = []
aXY = []
lifetime = []
features = []

if numMov == -1:
	numFrames = numFV*2 +1
else:
	numFrames = numFV*2+1+numMov

if numMov == -1:
	get_future_features(inputFile, frame_start, cutoff, featureNames, cID, aXY, lifetime, features)
else:
	# If we want to obtain features for a movie, we need to track the cells for longer
	# Keep in mind we still want to pad the tracks by numFV on each end to get the right
	# velocity values
	get_future_features(inputFile, frame_start, cutoff+numMov, featureNames, cID, aXY, lifetime, features)

# Keep only the cells that are tracked for all frames
for i, cell in reversed(list(enumerate(cID))):
	if len(cell) != numFrames:
		cID.pop(i)
		aXY.pop(i)
		lifetime.pop(i)
		features.pop(i)
numCells = len(lifetime)

aXY_np = NP.array(aXY)

# Calculate the velocity and speed and put these into features
# Note that this calculate the velocity for every cell and every frame so
# the array shapes match those in features
velocity_x = savgol_filter(aXY_np[:,:,0],numFrames,3,deriv=1,axis=1)
velocity_y = savgol_filter(aXY_np[:,:,1],numFrames,3,deriv=1,axis=1)
speed = NP.sqrt(velocity_x**2 + velocity_y**2)

for i, c in enumerate(features):
	for j, f in enumerate(c):
		f.append(velocity_x[i][j])
		f.append(velocity_y[i][j])
		f.append(speed[i][j])

numFeatures = len(features[0][0][:])

## BUILD FEATURE VECTOR ###################################################

# Location list contains cell_id, location x and location y
# These are NOT considered features
# Note that for both frame-by-frame and movie options, the cell ID and location
# are taken at the first frame
locationList = []
locationList.append([int(a[numFV]) for a in cID]) # cell id
locationList.append([a[numFV][0] for a in aXY]) # x-centroid
locationList.append([a[numFV][1] for a in aXY]) # y-centroid
locationNames = ['Cell_ID', 'Location_X', 'Location_Y']

# The base name of the features (these will be modified for the movie to include which statistic)
featNamesBase = ['Displacement']+ outFeatureNames + ['Velocity_X', 'Velocity_Y', 'Speed']

if numMov == -1:
	# Complete velocity calculations
	displacement = NP.sqrt(NP.sum((aXY_np[:,0,:]-aXY_np[:,-1,:])**2,1))

	# Begin creating feature vector
	featListStd = []
	featListStd.append(displacement) # total displacement

	# Add all other features 
	for i in range(numFeatures):
		featListStd.append([float(f[numFV][i]) for f in features])

	featNamesStd = list(featNamesBase)
	featNamesOrig = list(featNamesBase)
	featListOrig = list(featListStd) 
	numFeat = len(featListOrig) # Does not include location variables

	# Select the index of elements to scale
	dist_index = [0,2,4,5]
	speed_index = [7,8,9]
	area_index = [1]

else:
	# From frame to frame+numMov
	# Note that this does not take into account the frames used for padding
	displacement = NP.sqrt(NP.sum((aXY_np[:,numFV,:]-aXY_np[:,-numFV,:])**2,1))

	# Begin creating feature vector
	featListStd = []
	featNamesStd = []
	featListStd.append(displacement) # This is first as it doesn't have any statistics 
	featNamesStd.append('Displacement')
	features = NP.array(features)

	for i in range(numFeatures):
		cur_feat = [(f[numFV:-numFV,i]).astype(float) for f in features]

		# Calculate statistics for every feature
		featListStd.append(NP.max(NP.abs(cur_feat),1)) # Max magnitude
		featListStd.append(NP.min(NP.abs(cur_feat),1)) # Min magnitude
		featListStd.append(NP.mean(cur_feat,1)) # Mean
		featListStd.append(NP.median(cur_feat,1)) # Median
		featListStd.append(NP.sqrt(NP.var(cur_feat,1))) # Standard deviation

		# Create feature name list
		featNamesStd.append('Max_Magnitude_' + featNamesBase[i+1])
		featNamesStd.append('Min_Magnitude_' + featNamesBase[i+1])
		featNamesStd.append('Mean_' + featNamesBase[i+1])
		featNamesStd.append('Median_' + featNamesBase[i+1])
		featNamesStd.append('SD_' + featNamesBase[i+1])

	featNamesOrig = list(featNamesStd)
	featListOrig = list(featListStd) 
	numFeat = len(featNamesOrig)

	# Select the index of elements to scale
	dist_index = [0,6,7,8,9,10,16,17,18,19,20,21,22,23,24,25]
	speed_index = [31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
	area_index = [1,2,3,4,5]

for i in dist_index:
	featListOrig[i][:], dist_unit = conv_distance(featListOrig[i][:])
	featNamesOrig[i] = featNamesOrig[i] + '_(' + dist_unit + ')'

for i in speed_index:
	featListOrig[i][:], speed_unit = conv_speed(featListOrig[i][:])
	featNamesOrig[i] = featNamesOrig[i] + '_(' + speed_unit + ')'

for i in area_index:
	featListOrig[i][:], area_unit = conv_area(featListOrig[i][:])
	featNamesOrig[i] = featNamesOrig[i] + '_(' + area_unit + ')'

## STANDARDIZE FEATURES ###################################################

featListStd = NP.array(featListStd)
# Don't standardize the centroids
for k in range(numFeat):
	featListStd[k] = (featListStd[k] - NP.mean(featListStd[k]))/NP.sqrt(NP.var(featListStd[k]))
featListStd = list(featListStd)

## CREATE CSV FILE ########################################################

# Put all features into a single list
featList = NP.transpose(locationList + featListStd + featListOrig)
featNames = locationNames + featNamesStd + featNamesOrig

with open(outFile, 'wb') as csvfile:
    featWriter = csv.writer(csvfile)

    # Write the comments
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    
    featWriter.writerow(['# Experimental Data Feature List'])
    featWriter.writerow(['# Date Created: ' + timestamp])

    if numMov == -1:
    	featWriter.writerow(['# Mode: Single Frame'])
    else:
    	featWriter.writerow(['# Mode: Movie (using ' + str(numMov) + ' frames)'])

    featWriter.writerow(['# Identifier (Cell ID) Column: 0'])
    featWriter.writerow(['# Location Columns: 1-2'])
    featWriter.writerow(['# Standardized Feature Columns: 3-' + str(2+numFeat)])
    featWriter.writerow(['# Original Feature Columns: ' + str(3+numFeat) + '-' + str(2+2*numFeat)])
    featWriter.writerow(['# Number of Tracked Cells: ' + str(numCells)])

    featWriter.writerow([])
    featWriter.writerow(featNames)

    for row in featList:
    	featWriter.writerow(row)
