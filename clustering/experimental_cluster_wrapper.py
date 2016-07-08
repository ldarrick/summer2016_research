#!/usr/bin/env python

#
# Last modified: June 30 2016
# Author: Darrick Lee <y.l.darrick@gmail.com>
# This is a wrapper to create the feature csv files from experimental data and then
# perform various clustering techniques
#
# Single frame standardized columns: 3-12
# Movie standardized columns: 3-48
#

import os
import subprocess
from optparse import OptionParser
from featurecluster import FeatureCluster

outFolder = ''

# Options Parsing
parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to csv file containing data", metavar="INPUT")
parser.add_option("--start", action="store", type="int", dest="frame_start", help="first frame to analyze", metavar="FRAME_START")
parser.add_option("--end", action="store", type="int", dest="frame_end", help="last frame to analyze", metavar="FRAME_END")
parser.add_option("-n", "--numFrameV", action="store", type="int", dest="numFV", help="number of future/previous frame to use to calculate velocity", metavar="NUMFRAMEVELOCITY")
parser.add_option("-o", "--output", action="store", type="string", dest="outFolder", help="path to where output folders will be stored", metavar="OUTPUT")
parser.add_option("-m", "--movie", action="store_true", dest="movie", help="perform clustering on aggregate statistics", metavar="MOVIE")

# Options parsing
(options, args) = parser.parse_args()
if options.inputFile:
	inputFile = options.inputFile
if options.frame_start:
	frame_start = options.frame_start
if options.frame_end:
	frame_end = options.frame_end
if options.numFV:
	numFV = options.numFV
if options.movie:
	movie = True
else:
	movie = False
if options.outFolder:
	outFolder = options.outFolder

## INITIALIZE FOLDERS #########################################################
csvFolder = outFolder + 'csvOutput/'
agglAPFolder = outFolder + 'agglomerative_AP/'
agglXSFolder = outFolder + 'agglomerative_XS/'
agglPCAFolder = outFolder + 'agglomerative_PCA/'
featTreeFolder = outFolder + 'featTree/'
biplotFolder = outFolder + 'biPlot/'

if not os.path.isdir(csvFolder):
	os.mkdir(csvFolder)
if not os.path.isdir(agglAPFolder):
	os.mkdir(agglAPFolder)
if not os.path.isdir(agglXSFolder):
	os.mkdir(agglXSFolder)
if not os.path.isdir(agglPCAFolder):
	os.mkdir(agglPCAFolder)
if not os.path.isdir(featTreeFolder):
	os.mkdir(featTreeFolder)
if not os.path.isdir(biplotFolder):
	os.mkdir(biplotFolder)


## CREATE CSV FILE AND PERFORM CLUSTERING #####################################

if movie:
	movLength = frame_end - frame_start
	csvName = 'f' + str(frame_start) + '_m' + str(movLength) + '.csv'
	csvFilePath = csvFolder + csvName

	subprocess.call(['python', 'experimentalcsv.py',
		'-i', inputFile,
		'-f', str(frame_start),
		'-n', str(numFV),
		'-o', csvFilePath,
		'-m', str(movLength)])

	# Perform clustering
	FC = FeatureCluster(csvFilePath)
	M_clist = range(3,49)
	M_plist = [[1,46], [52, 57],[-1,-2]]
	M_pathlist = [agglXSFolder, agglAPFolder, agglPCAFolder]

	FC.AgglomerativeClustering(M_clist)
	FC.FeatureAgglomeration(M_clist)
	FC.PCA(M_clist)

	FC.PlotAgglomerative(M_plist, M_pathlist)
	FC.PlotFeatureTree(featTreeFolder)
	FC.PlotPCA(path=biplotFolder)

else:

	for frame in range(frame_start, frame_end+1):

		# Create csv file
		csvName = 'f' + str(frame) + '_singleframe.csv'
		csvFilePath = csvFolder + csvName

		subprocess.call(['python', 'experimentalcsv.py',
			'-i', inputFile,
			'-f', str(frame),
			'-n', str(numFV),
			'-o', csvFilePath])

		# Perform clustering
		FC = FeatureCluster(csvFilePath)
		SF_clist = range(3,13)
		SF_plist = [[1,22], [14,15], [-1,-2]] # XS, AP, PCA
		SF_pathlist = [agglXSFolder, agglAPFolder, agglPCAFolder]

		FC.AgglomerativeClustering(SF_clist)
		FC.FeatureAgglomeration(SF_clist)
		FC.PCA(SF_clist)

		FC.PlotAgglomerative(SF_plist,SF_pathlist)
		FC.PlotFeatureTree(featTreeFolder)
		FC.PlotPCA(path=biplotFolder)



