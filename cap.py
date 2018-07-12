# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Richard Camuccio
# 9 Jun 2018
#
# Last update: 12 Jul 2018
#
# CAP
# CCD analysis pipeline
#

from astropy.io import fits
from scipy.optimize import curve_fit
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

def main():

	import time

	start = time.time()
	print("<STATUS> Starting clock ...")

	input_dir = do_argparse()

	log = open("log.txt", "w")	
	print("<STATUS> Opening data log ...")


	#
	# BIAS ANALYSIS
	#
	bias_dir = input_dir + "bias"
	print("<STATUS> Bias directory defined as", str(bias_dir), "...")

	bias_list = []

	os.chdir(bias_dir)
	print("<STATUS> Changing directory to", str(bias_dir), "...")

	for frame in glob.glob("*.fit"):
		bias_list.append(frame)
		print("<STATUS> Appending", str(frame), "to bias list ...")

	print("<STATUS> Bias list defined as", str(bias_list), "...")

	# Define 1st bias
	bias_1 = bias_list[0]
	print("<STATUS> Reading test bias 1 as", str(bias_1), "...", type(bias_1))

	bias_1_fits = fits.open(str(bias_dir) + "/" + str(bias_1))
	print("<STATUS> Opening", str(bias_1), "as FITS ...", type(bias_1_fits))

	bias_1_array = bias_1_fits[0].data
	print("<STATUS> Reading", str(bias_1), "data ...", type(bias_1_array))

	bias_1_array = bias_1_array.astype(float)
	print("<STATUS> Converting", str(bias_1), "data to float ...", type(bias_1_array))

	# Define 2nd bias
	bias_2 = bias_list[1]
	print("<STATUS> Reading test bias 2 as", str(bias_2), "...", type(bias_2))

	bias_2_fits = fits.open(str(bias_dir) + "/" + str(bias_2))
	print("<STATUS> Opening", str(bias_2), "as FITS ...", type(bias_2_fits))

	bias_2_array = bias_2_fits[0].data
	print("<STATUS> Reading", str(bias_2), "data ...", type(bias_2_array))

	bias_2_array = bias_2_array.astype(float)
	print("<STATUS> Converting", str(bias_2), "data to float ...", type(bias_2_array))

	# Subtract two bias frames from list
	diff_bias_array = np.subtract(bias_1_array, bias_2_array)
	print("<STATUS> Subtracting", str(bias_1), "and", str(bias_2), "...", type(diff_bias_array))

	# Calculate readout noise from difference bias frames
	diff_stddev = np.std(diff_bias_array)
	print("<STATUS> Calculating standard deviation from difference ...")

	readout_noise = diff_stddev / math.sqrt(2)
	print("<STATUS> Calcuating readout noise ...")

	log.write("Bias RN = " + str(readout_noise) + " ADU\n\n")
	print("<STATUS> Writing result to log ...")

	# Median combine bias frames
	master_bias = ccdproc.combine(bias_list, method="median", unit="adu")
	print("<STATUS> Creating master bias ...", type(master_bias))

	# Histogram of bias
	master_bias_array = master_bias[0].data
	print("<STATUS> Reading master bias data ...", type(master_bias_array))

	f_bias_array = master_bias_array.flatten()
	print("<STATUS> Flattening array ...")

	bias_min = np.min(f_bias_array)
	print("<STATUS> Calculating master bias minimum ...")

	bias_max = np.max(f_bias_array)
	print("<STATUS> Calculating master bias maximum ...")

	print("<STATUS> Producing histogram of master bias ...")
	bias_xrange = (bias_min - 10, bias_max + 10)
	nbins = np.arange(bias_min, bias_max + 1, 1)
	font = {"fontname":"Monospace",
		"size":10}
	plt.hist(f_bias_array, range=bias_xrange, bins=nbins, histtype="step")
	plt.title("Histogram of Master Bias\nnbins=" + str(len(nbins)), **font)
	plt.xticks(**font)
	plt.yticks(**font)
	plt.savefig(str(input_dir) + "/bias-hist.png")


	#
	# DARK ANALYSIS
	#
	dark_dir = input_dir + "dark"
	print("<STATUS> Dark directory defined as", str(dark_dir), "...")

	dark_list = []
	
	os.chdir(dark_dir)
	print("<STATUS> Changing directory to", str(dark_dir), "...")

	for frame in glob.glob("*.fit"):
		dark_list.append(frame)
		print("<STATUS> Appending", str(frame), "to dark list ...")

	# Dark frame 1
	dark_1 = dark_list[0]
	print("<STATUS> Reading dark 1 as", str(dark_1), "...", type(dark_1))

	dark_1_fits = fits.open(str(dark_dir) + "/" + str(dark_1))
	print("<STATUS> Opening", str(dark_1), "as FITS...", type(dark_1_fits))

	dark_1_array = dark_1_fits[0].data
	print("<STATUS> Reading", str(dark_1), "data...", type(dark_1_array))

	dark_1_array = dark_1_array.astype(float)
	print("<STATUS> Converting", str(dark_1), "data to float...", type(dark_1_array))

	dark_min_bias_1_array = np.subtract(dark_1_array, master_bias)
	print("<STATUS> Subtracting bias from", str(dark_1), "...", type(dark_min_bias_1_array))

	# Dark frame 2
	dark_2 = dark_list[1]
	print("<STATUS> Reading dark 2 as", str(dark_2), "...", type(dark_2))

	dark_2_fits = fits.open(str(dark_dir) + "/" + str(dark_2))
	print("<STATUS> Opening", str(dark_1), "as FITS...", type(dark_2_fits))

	dark_2_array = dark_2_fits[0].data
	print("<STATUS> Reading", str(dark_2), "data...", type(dark_2_array))

	dark_2_array = dark_2_array.astype(float)
	print("<STATUS> Converting", str(dark_2), "data to float...", type(dark_2_array))

	dark_min_bias_2_array = np.subtract(dark_2_array, master_bias)
	print("<STATUS> Subtracting bias from", str(dark_2), "...", type(dark_min_bias_2_array))

	# Dark sum
	dark_sum = np.add(dark_min_bias_1_array, dark_min_bias_2_array)
	print("<STATUS> Adding corrected", str(dark_1), "and corrected", str(dark_2), "...")

	dark_1_exptime = dark_1_fits[0].header["EXPTIME"]
	print("<STATUS> Reading dark 1 exposure time ...", str(dark_1_exptime), "s")

	dark_2_exptime = dark_2_fits[0].header["EXPTIME"]
	print("<STATUS> Reading dark 2 exposure time ...", str(dark_2_exptime), "s")

	total_exp = dark_1_exptime + dark_2_exptime
	print("<STATUS> Adding exposure times...", str(total_exp), "s")

	# Dark current
	dark_current_array = dark_sum / total_exp
	print("<STATUS> Creating dark current frame ...", type(dark_current_array))


	#
	# FLAT ANALYSIS
	# 
	flat_dir = input_dir + "flat"
	print("<STATUS> Flat directory defined as", str(flat_dir), "...")

	flat_half_dir = flat_dir + "/flat_half"
	print("<STATUS> Half-saturation flat directory defined as", str(flat_half_dir), "...")

	flat_linear_dir = flat_dir + "/flat_linear"
	print("<STATUS> Linear flat directory defined as", str(flat_linear_dir), "...")

	flat_half_list = []

	os.chdir(flat_half_dir)
	print("<STATUS> Changing directory to", str(flat_half_dir), "...")

	for frame in glob.glob("*.fit"):
		flat_half_list.append(frame)
		print("<STATUS> Appending", str(frame), "to half-saturation flat list ...")

	flat_half_fits = fits.open(flat_half_list[0])
	print("<STATUS> Opening", str(flat_half_list[0]), "as FITS...", type(flat_half_fits))

	flat_half_exptime = flat_half_fits[0].header["EXPTIME"]
	print("<STATUS> Reading", str(flat_half_list[0]), "exposure time...", str(flat_half_exptime), "s")

	flat_half_combined = ccdproc.combine(flat_half_list, method="median", unit="adu")
	print("<STATUS> Creating median-combined flat ...", type(flat_half_combined))

	flat_min_dark_array = flat_half_combined - (dark_current_array * flat_half_exptime)
	print("<STATUS> Subtracting scaled dark from combined flat ...", type(flat_min_dark_array))

	flat_avg_value = np.mean(flat_min_dark_array)
	print("<STATUS> Calculating mean of dark-subtracted flat ...", str(flat_avg_value), "ADU")

	flatfield_array = flat_min_dark_array / flat_avg_value
	print("<STATUS> Creating flatfield array ... ", type(flatfield_array))

	flatfield = ccdproc.CCDData(flatfield_array, unit="adu")
	print("<STATUS> Converting flatfield array to CCDData object ...", type(flatfield))

	# Histogram of flatfield
	flat_ff_array = flatfield_array.flatten()
	print("<STATUS> Flattening array ...")

	print("<STATUS> Producing histogram of flatfield ...")
	nbins = 500
	font = {"fontname":"Monospace",
			"size":10}

	plt.hist(flat_ff_array, bins=nbins, histtype="step")
	plt.xlim(0.95, 1.05)
	plt.title("Histogram of Flatfield\nnbins=" + str(nbins), **font)
	plt.xticks(**font)
	plt.yticks(**font)
	plt.savefig(str(input_dir) + "/flat-hist.png")

	# Initialize lists for (sum) mean vs (diff) variance plot
	sum_mean_list = []
	sum_std_list = []
	diff_std_list = []

	# Initialize lists for linearity plot
	int_time_list = []
	mean_val_list = []
	stddev_list = []

	for x in os.walk(flat_linear_dir):

		os.chdir(x[0])
		print("<STATUS> Changing directory to", str(x[0]), "...")

		data_list = []

		for frame in glob.glob("*.fit"):
	
			frame_fits = fits.open(frame)
			print("<STATUS> Opening", str(frame), "as FITS...", type(frame_fits))

			frame_exptime = frame_fits[0].header["EXPTIME"]
			print("<STATUS> Reading", str(frame), "exposure time...", str(frame_exptime), "s")

			if frame_exptime not in int_time_list:
				int_time_list.append(frame_exptime)

			frame_array = frame_fits[0].data
			print("<STATUS> Reading", str(frame), "data...", type(frame_array))

			frame_array = frame_array - master_bias_array - (dark_current_array * frame_exptime)
			print("<STATUS> Calibrating", str(frame), "with master bias and dark current at", str(frame_exptime), "s ...", type(frame_array))

			data_list.append(frame_array)
			print("<STATUS> Appending calibrated", str(frame), "to data list...")

		if len(data_list) != 0:

			data1 = data_list[0]
			data2 = data_list[1]

			data_sum = data1 + data2
			data_diff = data1 - data2

			# Data for mean vs variance plot
			if frame_exptime < 80:
				
				sum_mean = np.mean(data_sum)
				sum_mean_list.append(sum_mean)

				sum_std = np.std(data_sum)
				sum_std_list.append(sum_std)

				diff_std = np.std(data_diff)
				diff_std_list.append(diff_std)

			# Data for linearity measurement
			data_mean = np.mean(data_sum / 2)
			mean_val_list.append(data_mean)

			data1_std = np.std(data1)
			data2_std = np.std(data2)

			data_std = math.sqrt(data1_std**2 + data2_std**2)
			stddev_list.append(data_std)

	# Calculate variance of difference frames
	diff_var_list = []
	for value in diff_std_list:
		value = value**2
		diff_var_list.append(value)

	# Convert lists to arrays
	sum_mean_array = np.asarray(sum_mean_list)
	diff_var_array = np.asarray(diff_var_list)
	sum_std_array = np.asarray(sum_std_list)

	# Define model fn (linear: y = mx + b)
	def f(x, m, b):
		return (m*x) + b

	#
	# Unweighted fit calculation
	popt, pcov = curve_fit(f, diff_var_array, sum_mean_array)
	yfit = f(diff_var_array, *popt)

	print()
	print("(Unweighted) fit parameters:", popt)
	log.write("(Unweighted) fit parameters: " + str(popt) + "\n\n")

	print("(Unweighted) covariance matrix:", pcov)
	log.write("(Unweighted) covariance matrix: " + str(pcov) + "\n\n")

	# Slope
	m = popt[0]
	delta_m = math.sqrt(pcov[0][0])

	# y-intercept
	b = popt[1]
	delta_b = math.sqrt(pcov[1][1])

	# Gain
	g = 1/m
	delta_g = (g**2)*delta_m

	# Readout noise
	sig = math.sqrt(g*b/(-2))
	delta_sig = math.sqrt(((sig**2)/4)*((delta_b/b)**2-(delta_g/g)**2))

	R = sig/g
	delta_R = R*math.sqrt((delta_sig/sig)**2 + (delta_g/g)**2)

	print("m =", "%.4f" % m, "+/-", "%.4f" % delta_m)
	log.write("m = " + str(m) + " +/- " + str(delta_m) + "\n\n")

	print("b =", "%.4f" % b, "+/-", "%.4f" % delta_b)
	log.write("b = " + str(b) + " +/- " + str(delta_b) + "\n\n")

	print("g = (", "%.4f" % g, "+/-", "%.4f" % delta_g, ") ADU/e")
	log.write("g = (" + str(g) + " +/- " + str(delta_g) + ") ADU/e\n\n")

	print("RN = (", "%.4f" % sig, "+/-", "%.4f" % delta_sig, ") ADU")
	log.write("RN = (" + str(sig) + " +/- " + str(delta_sig) + ") ADU\n\n")

	print("RN = (", "%.4f" % R, "+/-", "%.4f" % delta_R, ") e")
	log.write("RN = (" + str(R) + " +/- " + str(delta_R) + ") e\n\n")

	print()

	#
	# Weighted fit calculation
	popt2, pcov2 = curve_fit(f, diff_var_array, sum_mean_array, sigma=sum_std_array, absolute_sigma=True)
	yfit2 = f(diff_var_array, *popt2)

	print("(Weighted) fit parameters:", popt2)
	log.write("(Weighted) fit parameters: " + str(popt2) + "\n\n")

	print("(Weighted) covariance matrix:", pcov2)
	log.write("(Weighted) covariance matrix: " + str(pcov2) + "\n\n")

	# Slope
	m = popt2[0]
	delta_m = math.sqrt(pcov2[0][0])

	# y-intercept
	b = popt2[1]
	delta_b = math.sqrt(pcov2[1][1])

	# Gain
	g = 1/m
	delta_g = (g**2)*delta_m

	# Readout noise
	sig = math.sqrt(g*b/(-2))
	delta_sig = math.sqrt(((sig**2)/4)*((delta_b/b)**2-(delta_g/g)**2))

	R = sig/g
	delta_R = R*math.sqrt((delta_sig/sig)**2 + (delta_g/g)**2)

	print("m =", "%.4f" % m, "+/-", "%.4f" % delta_m)
	log.write("m = " + str(m) + " +/- " + str(delta_m) + "\n\n")

	print("b =", "%.4f" % b, "+/-", "%.4f" % delta_b)
	log.write("b = " + str(b) + " +/- " + str(delta_b) + "\n\n")

	print("g = (", "%.4f" % g, "+/-", "%.4f" % delta_g, ") ADU/e")
	log.write("g = (" + str(g) + " +/- " + str(delta_g) + ") ADU/e\n\n")

	print("RN = (", "%.4f" % sig, "+/-", "%.4f" % delta_sig, ") ADU")
	log.write("RN = (" + str(sig) + " +/- " + str(delta_sig) + ") ADU\n\n")

	print("RN = (", "%.4f" % R, "+/-", "%.4f" % delta_R, ") e")
	log.write("RN = (" + str(R) + " +/- " + str(delta_R) + ") e\n\n")

	print()
	
	# Plotting
	plt.clf()
	font = {"fontname":"Monospace",
			"size":10}
	plt.errorbar(diff_var_array, sum_mean_array, yerr=sum_std_array, fmt="o", linewidth=0.5, markersize=0.5, capsize=2, capthick=0.5)
	plt.plot(diff_var_array, yfit, "--", linewidth=0.5, label="Unweighted fit")
	plt.plot(diff_var_array, yfit2, "--", linewidth=0.5, label="Weighted fit")
	plt.title("Mean Pixel Value vs Variance \n CTMO SBIG ST-8300M", **font)
	plt.xlabel("Difference Frame Variance (ADU$^2$)", **font)
	plt.ylabel("Sum Frame Mean (ADU)", **font)
	plt.xticks(**font)
	plt.yticks(**font)
	plt.legend()
	plt.savefig(str(input_dir) + "mean_vs_variance.png", dpi=300)

	plt.clf()
	font = {"fontname":"Monospace",
			"size":10}
	plt.errorbar(int_time_list, mean_val_list, yerr=stddev_list, fmt="o", linewidth=0.5, markersize=0.5, capsize=2, capthick=0.5)
	plt.title("Mean Pixel Value vs Integration Time \n CTMO SBIG ST-8300M", **font)
	plt.xlabel("Integration Time (s)", **font)
	plt.ylabel("Mean Frame Value (ADU)", **font)
	plt.xticks(**font)
	plt.yticks(**font)
	plt.savefig(str(input_dir) + "linearity.png", dpi=300)

	log.close()

	end = time.time()
	print(str(end - start) + " seconds to complete.")


main()