import csv
import os
import pandas as pd

import numpy as np


# if 5 past times are taken into consideration x = [1, 5, 2] (5 segments, 2 coordinates) , y (1, 3, 2) if 3 future times should be calculated
# T_x = 5, T_y = 3 --> x = [1, 5, 2] y = [1, 3, 2] 

def create_data(file, T_x, T_y):
	'''
	create data splits and returns x and y arrays

	file: the csv file with data
	T_x: amount of past time segments per sample
	T_y: amount of future time segments per sample

	returns
	X: numpy array with x and y coordinates of past samples
	Y: numpy array with x and y coordinates of corresponding future samples
	'''
	df = pd.read_csv(file,header=0)

	# only consider vehicles for now
	df_vehicle = df[df['Actor Type'] == 'Vehicle']

	# get total timesegments (== Simulation time)
	total_timesegments = len(df_vehicle[df_vehicle['Actor ID'] == df_vehicle['Actor ID'][0]]['Time'])

	# get unique vehicles
	unique_vehicles = df_vehicle['Actor ID'].unique()
	nr_vechicles = len(unique_vehicles)
	# save locations at each timesegment of each unique vehicle
	locations = np.zeros((nr_vechicles,total_timesegments,2))
	i = 0
	for vehicles in unique_vehicles:
		relative_x = df_vehicle[df_vehicle['Actor ID'] == vehicles]['X']
		relative_y = df_vehicle[df_vehicle['Actor ID'] == vehicles]['Y']
		locations[i] = np.transpose(np.array([relative_x,relative_y]))
		i += 1
		
	# amount of time segments in total segments per vehicle
	segments = total_timesegments - T_x - T_y + 1
	X  = np.zeros((segments*nr_vechicles,T_x,2))
	y  = np.zeros((segments*nr_vechicles,T_y,2))
	# create segments by shifting over locations using i:i+T_x and i+T_x:i+T_x+T_y respectively
	segments_iterator = 0 # to iterate over segment lengths
	for j in range(0,nr_vechicles):
		X_temp = np.array([(locations[j][i:i+T_x] - locations[j][i]) for i in range(0,segments)])
		# print(np.shape(X_temp))
		y_temp = np.array([(locations[j][i+T_x:i+T_x+T_y] - locations[j][i]) for i in range(0,segments)])
		X[j*segments:j*segments+segments] = X_temp
		y[j*segments:j*segments+segments] = y_temp
		segments_iterator += segments
		
	return X, y
if __name__== "__main__":
	# Get file name (location)
	cwd = os.getcwd()
	file1 = cwd + '/data/Coordinates_T30_run_1.csv'
	# file2 = cwd + '/LSTM/relative_coordinates_T20_2.csv'
	
	X, y = create_data(file1,3,2)
	