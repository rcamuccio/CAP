# CAP

CAP is a CCD analysis pipeline. 

The script is run from the terminal as:

    python cap.py -i <directory>

where the directory of choice contains subdirectories for bias, dark, and flat FITS frames. The default directory is the current working directory.

The pipeline works on a series of bias, dark, and flat FITS frames in order to produce the following:

* Histograms of averaged bias and flatfield frames
* Calculation of CCD chip gain, readout noise, and saturation limit
* Plots of the CCD linearity

CAP is currently under construction. Above features will be added over time.
