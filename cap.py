# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Richard Camuccio
# 9 Jun 2018
#
# Last update: 19 Jun 2018
#
# CAP
# CCD analysis pipeline
#

from astropy.io import fits
import argparse
import ccdproc
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

def do_combine(frame_list, log):

	print("<STATUS> Combining frames in list", frame_list, "...")
	print()
	combined_frame = ccdproc.combine(frame_list, method="median", unit="adu")

	return combined_frame

def do_stat(array, name, log):

	mean = np.mean(array)
	log.write("Mean of " + str(name) + " = " + str(round(mean, 3)) + "\n")

	median = np.median(array)
	log.write("Median of " + str(name) + " = " + str(round(median, 3)) + "\n")

	stddev = np.std(array)
	log.write("STD of " + str(name) + " = " + str(round(stddev, 3)) + "\n")

	minimum = np.min(array)
	log.write("Min of " + str(name) + " = " + str(round(minimum, 3)) + "\n")

	maximum = np.max(array)
	log.write("Max of " + str(name) + " = " + str(round(maximum, 3)) + "\n")

	length = np.size(array)
	log.write("NPIX of " + str(name) + " = " + str(length) + "\n\n")

	return mean, median, stddev, minimum, maximum, length

def main():

	import time
	start = time.time()

	input_dir = do_argparse()
	log = open("log.txt", "w")	


	#
	# BIAS ANALYSIS
	#
	bias_dir = input_dir + "/bias"
	bias_list = []

	os.chdir(bias_dir)

	for frame in glob.glob("*.fit"):
		bias_list.append(frame)

	bias_1 = bias_list[0]
	bias_1_fits = fits.open(str(bias_dir) + "/" + str(bias_1))
	bias_1_array = bias_1_fits[0].data
	bias_1_array = bias_1_array.astype(float)

	bias_2 = bias_list[1]
	bias_2_fits = fits.open(str(bias_dir) + "/" + str(bias_2))
	bias_2_array = bias_2_fits[0].data
	bias_2_array = bias_2_array.astype(float)

	print("<STATUS> Subtracting frames " + str(bias_1) + " and " + str(bias_2) + " ...")
	print()
	diff_bias_array = np.subtract(bias_1_array, bias_2_array)

	diff_stddev = do_stat(diff_bias_array, "difference bias array", log)[2]
	readout_noise = diff_stddev / math.sqrt(2)
	log.write("Calculated readout noise from biases = " + str(round(readout_noise, 3)) + " ADU\n\n")

	master_bias = do_combine(bias_list, log)
	#print("<STATUS> Writing master bias to disk ...")
	#print()
	#ccdproc.fits_ccddata_writer(master_bias, str(bias_dir) + "/master-bias.fit", overwrite=True)

	#
	# HISTOGRAM OF BIAS
	#
	master_bias_data = master_bias[0].data
	f_bias_array = master_bias_data.flatten()

	bias_xrange = (994, 1083)
	nbins = np.arange(min(f_bias_array), max(f_bias_array) + 1, 1)

	font = {"fontname":"Monospace",
		"size":10}

	plt.hist(f_bias_array, range=bias_xrange, bins=nbins, histtype="step")
	plt.title("Histogram of bias-avg.fit\nnbins=" + str(len(nbins)), **font)
	plt.xticks(**font)
	plt.yticks(**font)
	plt.savefig(str(bias_dir) + "/bias-hist.png")


	#
	# DARK ANALYSIS
	#
	dark_dir = input_dir + "/dark"
	dark_list = []
	
	os.chdir(dark_dir)

	for frame in glob.glob("*.fit"):
		dark_list.append(frame)

	dark_1 = dark_list[0]
	dark_1_fits = fits.open(str(dark_dir) + "/" + str(dark_1))
	dark_1_array = dark_1_fits[0].data
	dark_1_array = dark_1_array.astype(float)
	dark_min_bias_1_array = np.subtract(dark_1_array, master_bias)

	dark_2 = dark_list[1]
	dark_2_fits = fits.open(str(dark_dir) + "/" + str(dark_2))
	dark_2_array = dark_2_fits[0].data
	dark_2_array = dark_2_array.astype(float)
	dark_min_bias_2_array = np.subtract(dark_2_array, master_bias)

	print("<STATUS> Subtracting frames " + str(dark_1) + " and " + str(dark_2) + " ...")
	print()
	diff_dark_array = np.subtract(dark_min_bias_1_array, dark_min_bias_2_array)

	do_stat(dark_min_bias_1_array, "dark 1 minus master bias", log)
	do_stat(dark_min_bias_2_array, "dark 2 minus master bias", log)
	do_stat(diff_dark_array, "difference dark array", log)

	#print("<STATUS> Writing differenced dark frame 1 to disk ...")
	#print()
	#dark_min_bias_1 = ccdproc.CCDData(dark_min_bias_1_array, unit="adu")
	#ccdproc.fits_ccddata_writer(dark_min_bias_1, str(dark_dir) + "/dark-min-bias-1.fit", overwrite=True)

	#print("<STATUS> Writing differenced dark frame 2 to disk ...")
	#print()
	#dark_min_bias_2 = ccdproc.CCDData(dark_min_bias_2_array, unit="adu")
	#ccdproc.fits_ccddata_writer(dark_min_bias_2, str(dark_dir) + "/dark-min-bias-2.fit", overwrite=True)

	dark_sum = np.add(dark_min_bias_1_array, dark_min_bias_2_array)
	dark_1_exptime = dark_1_fits[0].header['EXPTIME']
	dark_2_exptime = dark_2_fits[0].header['EXPTIME']
	total_exp = dark_1_exptime + dark_2_exptime

	dark_current_array = dark_sum / total_exp

	#print("<STATUS> Writing dark current to disk ...")
	#print()
	#dark_current = ccdproc.CCDData(dark_current_array, unit="adu")
	#ccdproc.fits_ccddata_writer(dark_current, str(dark_dir) + "/dark-current.fit", overwrite=True)


	#
	# FLAT ANALYSIS
	# 
	flat_dir = input_dir + "/flat"
	flat_half_dir = flat_dir + "/flat_half"
	flat_linear_dir = flat_dir + "/flat_linear"

	flat_half_list = []

	os.chdir(flat_half_dir)

	for frame in glob.glob("*.fit"):
		flat_half_list.append(frame)

	#
	# CREATE AVERAGED FLAT
	#
	flat_half_fits = fits.open(flat_half_list[0])
	flat_half_exptime = flat_half_fits[0].header['EXPTIME']
	flat_half_combined = do_combine(flat_half_list, log)

	#print("<STATUS> Writing combined half-saturation flat to disk ...")
	#print()
	#flat_half = ccdproc.CCDData(flat_half_combined, unit="adu")
	#ccdproc.fits_ccddata_writer(flat_half, str(flat_half_dir) + "/flat-half-combined.fit", overwrite=True)

	#
	# CREATE FLATFIELD
	#
	flat_min_dark_array = flat_half_combined - (dark_current_array * flat_half_exptime)
	flat_avg_value = np.mean(flat_min_dark_array)
	flatfield_array = flat_min_dark_array / flat_avg_value

	do_stat(flatfield_array, "flatfield", log)

	#print("<STATUS> Writing flatfield to disk ...")
	#print()
	flatfield = ccdproc.CCDData(flatfield_array, unit="adu")
	#ccdproc.fits_ccddata_writer(flatfield, str(flat_half_dir) + "/flatfield.fit", overwrite=True)

	#
	# HISTOGRAM OF FLATFIELD
	#
	flat_ff_array = flatfield_array.flatten()

	flatfield_xrange = (0.95, 1.05)
	nbins = 500

	font = {"fontname":"Monospace",
			"size":10}

	plt.hist(flat_ff_array, range=flatfield_xrange, bins=nbins, histtype="step")
	plt.title("Histogram of flatfield.fit\nnbins=" + str(nbins), **font)
	plt.xticks(**font)
	plt.yticks(**font)
	plt.savefig(str(flat_dir) + "/flat_half/flat-hist.png")



	for x in os.walk(flat_linear_dir):
		print(x[0])
		print()

	log.close()

	end = time.time()
	print(str(end - start) + " seconds to complete.")


main()