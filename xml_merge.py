#!/usr/bin/env python

# Example usage: python xml_merge.py --file1 CellSorting_PIF_complete_alt_11_18_2015_10_20_02/simulation_cs_complete_results.xml --file2 CellSorting_cc3d_05_04_2016_11_55_50/simulation_cs_complete_results.xml -o output3.xml --t1start 0 --t1end 4500 --t2start 0 --t2end 999 -p

# Note: The first XML file (xml1 is taken to have the "true" value at the merge point)
# This means that the relative error is calculated relative to this and also this is the
# value that is kept in the final merged xml file.

import sys
import numpy as NP
import lxml.etree as ET
import matplotlib.pyplot as PLT
from optparse import OptionParser

# Tolerance for merge
tol_area_relerr = 0.1
tol_perim_relerr = 0.1
tol_centroid_err = 1

parser = OptionParser()

parser.add_option("--file1", action="store", type="string", dest="inputfile1", help="first XML file to merge", metavar="FILE")
parser.add_option("--file2", action="store", type="string", dest="inputfile2", help="second XML file to merge", metavar="FILE")
parser.add_option("-o","--out", action="store", type="string", dest="outputfile", help="output XML file", metavar="FILE")
parser.add_option("--t1start", action="store", type="int", dest="t1start", help="start time of first file", metavar="INT")
parser.add_option("--t1end", action="store", type="int", dest="t1end", help="end time of first file", metavar="INT")
parser.add_option("--t2start", action="store", type="int", dest="t2start", help="start time of second file", metavar="INT")
parser.add_option("--t2end", action="store", type="int", dest="t2end", help="end time of second file", metavar="INT")
parser.add_option("-p","--plot", action="store_true", dest="ploterror", default=False, help="plot error magnitudes at merge time")

(options, args) = parser.parse_args()

inputfile1 = options.inputfile1
inputfile2 = options.inputfile2
outputfile = options.outputfile
t1start = options.t1start
t1end = int(options.t1end)
t2start = int(options.t2start)
t2end = int(options.t2end)

if options.ploterror:
	ploterror = 1
else:
	ploterror = 0

infile1 = open(inputfile1,'r')
infile2 = open(inputfile2,'r')
outfile = open(outputfile,'w')

xml1 = ET.parse(infile1)
xml2 = ET.parse(infile2)

root = xml1.getroot()

# Remove all elements from xml1 not within range
for time in xml1.getiterator('time'):
	time_num = int(time.get('t'))

	if(time_num < t1start or time_num > t1end):
		root.remove(time)

	if(time_num == t1end):
		time1end = time;

# Look through elements from xml2
for time in xml2.getiterator('time'):
	time_num = int(time.get('t'))

	# Check that the last time element from xml1 matches first from xml2
	if(time_num == t2start):

		t1_cells = time1end.getchildren()
		t2_cells = time.getchildren()
		num_cell = len(t1_cells)
		cell_range = range(num_cell)

		area_err = NP.zeros(num_cell)
		perim_err = NP.zeros(num_cell)
		centroid_err = NP.zeros(num_cell)

		for (cell1, cell2, num) in zip(t1_cells, t2_cells, cell_range):
			area_err_abs = abs(float(cell1.get('area')) - float(cell2.get('area')))
			perim_err_abs = abs(float(cell1.get('perimeter')) - float(cell2.get('perimeter')))

			# Note cell1 is used as the "true" value
			area_err[num] = area_err_abs/(float(cell1.get('area')))
			perim_err[num] = perim_err_abs/(float(cell1.get('perimeter')))

			x1 = float(cell1.get('x'))
			y1 = float(cell1.get('y'))
			x2 = float(cell2.get('x'))
			y2 = float(cell2.get('y'))

			centroid_err[num] = NP.sqrt((x1-x2)**2 + (y1-y2)**2)

		if ploterror:
			PLT.figure(1)
			PLT.plot(cell_range,area_err)
			PLT.xlabel('Cell')
			PLT.ylabel('Relative error in area')
			PLT.savefig('area_err.png', bbox_inches='tight', dpi = 400)

			PLT.figure(2)
			PLT.plot(cell_range,perim_err)
			PLT.xlabel('Cell')
			PLT.ylabel('Relative error in perimeter')
			PLT.savefig('perim_err.png', bbox_inches='tight', dpi = 400)

			PLT.figure(3)
			PLT.plot(cell_range,centroid_err)
			PLT.xlabel('Cell')
			PLT.ylabel('Absolute error in centroid')
			PLT.savefig('centroid_err.png', bbox_inches='tight', dpi = 400)

		max_area_relerr = NP.max(area_err)
		max_perim_relerr = NP.max(perim_err)
		max_centroid_err = NP.max(centroid_err)

		if(max_area_relerr > tol_area_relerr or max_perim_relerr > tol_perim_relerr or max_centroid_err > tol_centroid_err):
			if ploterror:
				PLT.show()

			sys.stderr.write('State at t1end does not match state at t2start')
			sys.exit()

	if(time_num > t2start and time_num <= t2end):
		time_num_new = t1end + (time_num - t2start)

		time.set('t',str(time_num_new))
		root.append(time)


xml1.write(outfile)

infile1.close()
infile2.close()
outfile.close()

if ploterror:
	PLT.show()