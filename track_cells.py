#!/usr/bin/env python

#
# Last modified: 9 June 2016
# Authors: Dhananjay Bhaskar <dbhaskar92@gmail.com>, Darrick Lee <yldarrick@gmail.com>
# Description: 
#

import csv
import numpy as NP
from scipy.special import binom

# Helper function						
def get_future_features(inputFile, time, cutoff, features, cID_future, aXY_future, 
	lifetime_future, features_future):

	# Initialize frame list
	frame_future = []
	cell_lastID = []
	max_tracked_frame = 0

	cur_frame = 0

	with open(inputFile) as csvh:
		
		hreader = csv.DictReader(csvh)

		for data in hreader:

			cur_frame = int(data['Metadata_FrameNumber'])

			if cur_frame < time:
				continue

			elif cur_frame > time + cutoff:
				# Quit if frame is above the cutoff
				break
		
			elif cur_frame == time:
				cID_future.append([int(data['ObjectNumber'])])
				aXY_future.append([(float(data['AreaShape_Center_X']), float(data['AreaShape_Center_Y']))])
				lifetime_future.append(int(data['TrackObjects_Lifetime_30']))
				frame_future.append(time)
				cell_lastID.append(int(data['ObjectNumber']))

				featureVector = [];
				for feat in features:
					featureVector.append(float(data[feat]))

				features_future.append([featureVector])

				max_tracked_frame = max(frame_future)
	
			else:

				if cur_frame > max_tracked_frame + 2:
					# The current frame is ahead of max tracked frame by at least 1,
					# so there are no longer any tracked cells
					break

				parentid = int(data['TrackObjects_ParentObjectNumber_30'])
				lifetime = int(data['TrackObjects_Lifetime_30'])

				id_lifetime_frame = zip(cell_lastID,lifetime_future,frame_future)
			
				# Check if the parent id of the current cell was tracked
				if (parentid, lifetime-1, cur_frame-1) in id_lifetime_frame:
					pid_index = id_lifetime_frame.index((parentid, lifetime-1, cur_frame-1))

					# Append current frame features to cell
					cID_future[pid_index].append(int(data['ObjectNumber']))
					aXY_future[pid_index].append((float(data['AreaShape_Center_X']), float(data['AreaShape_Center_Y'])))
					lifetime_future[pid_index] = lifetime
					frame_future[pid_index] = cur_frame

					featureVector = [];
					for feat in features:
						featureVector.append(float(data[feat]))

					features_future[pid_index].append(featureVector)

					# Update cell_lastID and max_tracked_frame
					cell_lastID[pid_index] = int(data['ObjectNumber'])
					max_tracked_frame = max(frame_future)
			

def calc_velocity(aXY, numFV, numCells):

	n = numFV
	N = numFV*2 + 1
	m = (N-3)/2
	M = (N-1)/2

	L = 1/float(2**(2*m+1))

	# Change to array so that it's easier to work with
	# First index: cell
	# Second index: frame
	# Third index: x/y
	aXY_np = NP.array(aXY)

	velocity = NP.zeros((numCells,2))

	for k in range(1,M+1):
		ck = L*(binom(2*m,m-k+1) - binom(2*m,m-k-1))
		velocity += ck*(aXY_np[:,numFV+k,:] - aXY_np[:,numFV-k,:])

	return velocity



