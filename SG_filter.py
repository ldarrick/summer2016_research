#!/usr/bin/env python

#
# Last modified: 14 June 2016
# Authors: Darrick Lee <yldarrick@gmail.com>
# Description: Implementation of the Savitzky-Golay filter on a list of position vectors.
#
# Input:
#	pos - List of position vectors
#	n - window size for S-G filter
#	deriv - which derivative to return
#	pl - boolean for plotting
#

import matplotlib.pyplot as PLT
from scipy.signal import savgol_filter

def SG_filter(pos, n, deriv, pl=False):
	npos = len(pos)

	X = [cur_pos[0] for cur_pos in pos]
	Y = [cur_pos[1] for cur_pos in pos]

	filteredX = savgol_filter(X,n,3,deriv=deriv,mode='nearest')
	filteredY = savgol_filter(Y,n,3,deriv=deriv,mode='nearest')

	if pl:
		PLT.figure()
		PLT.plot(X,Y)
		PLT.plot(filteredX,filteredY,'o')
		PLT.show()

	return zip(filteredX, filteredY)
	










