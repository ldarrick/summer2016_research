#!/usr/bin/env python

#
# Last modified: 13 June 2016
# Author: Darrick Lee <y.l.darrick@gmail.com>
# A wrapper to execute cluster cells for several different frames.
# 

import subprocess
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", action="store", type="string", dest="inputFile", help="path to csv file containing data", metavar="INPUT")
parser.add_option("--start", action="store", type="int", dest="start", help="first time to process", metavar="START")
parser.add_option("--end", action="store", type="int", dest="end", help="last time to process", metavar="END")
parser.add_option("-n", "--numFrameV", action="store", type="int", dest="numFV", help="number of future/previous frame to use to calculate velocity", metavar="NUMFRAMEVELOCITY")
parser.add_option("-b", "--bins", action="store", type="int", dest="bins", help="number of bins", metavar="BINS")
parser.add_option("-o", "--output", action="store", type="string", dest="outFile", help="path to where output folders will be stored", metavar="OUTPUT")

# Options parsing
(options, args) = parser.parse_args()
if options.inputFile:
	inputFile = options.inputFile
if options.start:
	start_frame = options.start
if options.end:
	end_frame = options.end
if options.numFV:
	numFV = options.numFV
if options.bins:
	numBins = options.bins
if options.outFile:
	outputFolder = options.outFile


for i in range(start_frame,end_frame+1):
	subprocess.call(['python', 'cluster_cells.py',
		'-i', inputFile,
		'-f', str(i),
		'-n', str(numFV),
		'-b', str(numBins),
		'-o', outputFolder])
