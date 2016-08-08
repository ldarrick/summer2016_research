#!/usr/bin/env python

#
# Last modified: 1 Aug 2016
# Author: Darrick Lee <y.l.darrick@gmail.com>
# This script merges csv files for red and green cells. It adds an extra column to the
# data which classifies the cells as 0 for red and 1 for green.
# 
# NOTE: This script assumes that the red and green csv files have identical columns in the same order.

import csv
from optparse import OptionParser

redFile = ''
greenFile = ''
outputFile = ''

# Options Parsing
parser = OptionParser()
parser.add_option("-r", "--red", action="store", type="string", dest="redFile", help="path to red csv file", metavar="REDCSV")
parser.add_option("-g", "--green", action="store", type="string", dest="greenFile", help="path to green csv file", metavar="GREENCSV")
parser.add_option("-o", "--output", action="store", type="string", dest="outputFile", help="path to output csv file", metavar="OUTPUT")

# Options parsing
(options, args) = parser.parse_args()
if options.redFile:
	redFile = options.redFile
if options.greenFile:
	greenFile = options.greenFile
if options.outputFile:
	outputFile = options.outputFile

# Begin to merge csv files
with open(outputFile, 'wb') as outputcsvfile:
    outputWriter = csv.writer(outputcsvfile)

    # First, add the red cells
    with open(redFile, 'rb') as redcsvfile:
    	redReader = csv.reader(redcsvfile)

        # Add the column names
    	for i, row in enumerate(redReader):
            if (i==0):
                row.append('Label_RG')
            else:
                row.append('0')

            outputWriter.writerow(row)

    # Next, add the green cells
    with open(greenFile, 'rb') as greencsvfile:
    	greenReader = csv.reader(greencsvfile)

    	for i, row in enumerate(greenReader):
            if (i!=0):
                row.append('1')
                outputWriter.writerow(row)

