# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Richard Camuccio
# 9 Jun 2018
#
# Last update: 10 Jun 2018
#
# CAP
# CCD analysis pipeline
#

from astropy.io import fits
import argparse
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Command line argument parser
def do_argparse():

	parser = argparse.ArgumentParser()

	parser.add_argument("-i", default=os.getcwd(), metavar="DIR", dest="input_path")

	args = parser.parse_args()

	input_dir = args.input_path

	return input_dir

def main():

	import time
	start = time.time()

	# Define input directory
	input_dir = do_argparse()	

	# Create bias frame list
	bias_dir = input_dir + "bias"
	bias_frame_list = []

	os.chdir(bias_dir)

	for frame in glob.glob("*.fit"):
		bias_frame_list.append(frame)

	# Calculate readout noise using bias frames
	test_bias_1 = bias_frame_list[0]
	test_bias_1_fits = fits.open(str(bias_dir) + "/" + str(test_bias_1))
	test_bias_1_array = test_bias_1_fits[0].data
	test_bias_1_array = test_bias_1_array.astype(float)

	test_bias_2 = bias_frame_list[1]
	test_bias_2_fits = fits.open(str(bias_dir) + "/" + str(test_bias_2))
	test_bias_2_array = test_bias_2_fits[0].data
	test_bias_2_array = test_bias_2_array.astype(float)

	diff_bias_array = np.subtract(test_bias_1_array, test_bias_2_array)
	diff_mean = np.mean(diff_bias_array)
	diff_median = np.median(diff_bias_array)
	diff_std = np.std(diff_bias_array)

	print()
	print("Difference between frames " + str(test_bias_1) + " and " + str(test_bias_2) + ":")
	print("Mean: " + str(round(diff_mean, 3)))
	print("Median: " + str(round(diff_mean, 3)))
	print("STD: " + str(round(diff_std, 3)))

	readout_noise = diff_std / math.sqrt(2)

	print("Readout noise: " + str(round(readout_noise, 3)) + " ADU")

	# Create dark frame list
	dark_dir = input_dir + "dark"
	dark_frame_list = []
	
	os.chdir(dark_dir)

	for frame in glob.glob("*.fit"):
		dark_frame_list.append(frame)


	# Create flat frame list
	flat_dir = input_dir + "flat"
	flat_frame_list = []

	os.chdir(flat_dir)

	for frame in glob.glob("*.fit"):
		flat_frame_list.append(frame)


	end = time.time()
	print()
	print(str(end - start) + " seconds to complete.")

main()

"""
bias = fits.open("20180606/bias/bias-avg.fit")
flatfield = fits.open("20180606/flat/flatfield.fit")

bias_array = bias[0].data
flatfield_array = flatfield[0].data

f_bias_array = bias_array.flatten()
f_flat_array = flatfield_array.flatten()

bias_xrange = (994, 1083)
flatfield_xrange = (0.95, 1.05)
nbins = 500
nbins_bias = np.arange(min(f_bias_array), max(f_bias_array) + 1, 1)

font = {"fontname":"Monospace",
		"size":10}

plt.hist(f_bias_array, range=bias_xrange, bins=nbins_bias, histtype="step")
plt.title("Histogram of bias-avg.fit\nnbins=" + str(len(nbins_bias)), **font)
plt.xticks(**font)
plt.yticks(**font)
plt.show()

plt.hist(f_flat_array, range=flatfield_xrange, bins=nbins, histtype="step")
plt.title("Histogram of flatfield.fit\nnbins=" + str(nbins), **font)
plt.xticks(**font)
plt.yticks(**font)
plt.show()
"""