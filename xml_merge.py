#!/usr/bin/env python

# Example usage: python xml_merge.py --file1 CellSorting_PIF_complete_alt_11_18_2015_10_20_02/simulation_cs_complete_results.xml --file2 CellSorting_cc3d_05_04_2016_11_55_50/simulation_cs_complete_results.xml -o output3.xml --t1start 0 --t1end 4500 --t2start 0 --t2end 999 -p

# Note: The first XML file (xml1 is taken to have the "true" value at the merge point)
# This means that the relative error is calculated relative to this and also this is the
# value that is kept in the final merged xml file.

import sys
import numpy as NP
import lxml.etree as ET
import matplotlib.pyplot as PLT
import statsmodels.api as SM
from scipy import stats
from optparse import OptionParser

# Parameter for 95% confidence in KS-test
# Source: https://www.webdepot.umontreal.ca/Usagers/angers/MonDepotPublic/STT3500H10/Critical_KS.pdf
c_alpha = 1.36
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
parser.add_option("-f","--force", action="store_true", dest="force", default=False, help="force merge regardless of errors at merge point")

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

if options.force:
	forcemerge = 1
else:
	forcemerge = 0

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

		# TO DO: add assert here to check if the number of cells is the same

		cell_range = range(num_cell)

		area1 = NP.zeros(num_cell)
		area2 = NP.zeros(num_cell)
		perim1 = NP.zeros(num_cell)
		perim2 = NP.zeros(num_cell)

		centroid_err = NP.zeros(num_cell)

		for (cell1, cell2, num) in zip(t1_cells, t2_cells, cell_range):
			area1[num] = float(cell1.get('area'))
			area2[num] = float(cell2.get('area'))
			perim1[num] = float(cell1.get('perimeter'))
			perim2[num] = float(cell2.get('perimeter'))

			x1 = float(cell1.get('x'))
			y1 = float(cell1.get('y'))
			x2 = float(cell2.get('x'))
			y2 = float(cell2.get('y'))

			centroid_err[num] = NP.sqrt((x1-x2)**2 + (y1-y2)**2)

		# Calculate ECDFs
		area1_ecdf = SM.distributions.ECDF(area1)
		area2_ecdf = SM.distributions.ECDF(area2)
		min_area = min(NP.concatenate([area1,area2]))
		max_area = max(NP.concatenate([area1,area2]))
		area_range = NP.linspace(min_area, max_area, 100)
		area1_ecdf_pt = area1_ecdf(area_range)
		area2_ecdf_pt = area2_ecdf(area_range)

		perim1_ecdf = SM.distributions.ECDF(perim1)
		perim2_ecdf = SM.distributions.ECDF(perim2)
		min_perim = min(NP.concatenate([perim1, perim2]))
		max_perim = max(NP.concatenate([perim1, perim2]))
		perim_range = NP.linspace(min_perim, max_perim, 100)
		perim1_ecdf_pt = perim1_ecdf(perim_range)
		perim2_ecdf_pt = perim2_ecdf(perim_range)

		# KS Test: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov.E2.80.93Smirnov_test
		(D_area, p_area) = stats.ks_2samp(area1, area2)
		(D_perim, p_perim) = stats.ks_2samp(perim1, perim2)
		ks_threshold = c_alpha*NP.sqrt(2.0/num_cell)

		print("Area and Perimeter statistics at merge time:")
		print("Area: ks-stat: {0}, p-value: {1}".format(D_area,p_area))
		print("Perimeter: ks-stat: {0}, p-value: {1}".format(D_perim,p_perim))
		print("ks-value threshold at 95% confidence: {0}".format(ks_threshold))	

		if ploterror:
			PLT.figure(1)
			PLT.plot(cell_range,centroid_err)
			PLT.xlabel('Cell')
			PLT.ylabel('Absolute error in centroid')
			PLT.savefig('centroid_err.png', bbox_inches='tight', dpi = 400)

			PLT.figure(2)
			PLT.step(area_range, area1_ecdf_pt)
			PLT.step(area_range, area2_ecdf_pt)
			PLT.xlabel('Area')
			PLT.ylabel('CDF')
			PLT.title('CDF of Area at merge point')
			PLT.legend(["XML1", "XML2"], loc=4)
			PLT.savefig('ecdf_area.png', bbox_inches='tight', dpi = 400)

			PLT.figure(3)
			PLT.step(perim_range, perim1_ecdf_pt)
			PLT.step(perim_range, perim2_ecdf_pt)
			PLT.xlabel('Perimeter')
			PLT.ylabel('CDF')
			PLT.title('CDF of Perimeter at merge point')
			PLT.legend(["XML1", "XML2"], loc=4)
			PLT.savefig('ecdf_perim.png', bbox_inches='tight', dpi = 400)

		max_centroid_err = NP.max(centroid_err)

		if not forcemerge:
			if (D_area > ks_threshold or D_perim > ks_threshold or max_centroid_err > tol_centroid_err):
				if ploterror:
					PLT.show()

				sys.stderr.write('State at t1end does not match state at t2start either according to the KS test with 95% confidence or cells are too far.\n')
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