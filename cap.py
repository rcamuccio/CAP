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
import matplotlib.pyplot as plt
import numpy as np

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