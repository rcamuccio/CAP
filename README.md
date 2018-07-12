# CAP

CAP is a CCD analysis pipeline. 

Usage:

    $ python cap.py -i /path/to/directory

where the target directory assumes the following (specific) tree structure:

```
root_path
    |--- bias
    |    |--- bias-001.fit
    |    |--- bias-002.fit
    |    |--- ...
    |
    |--- dark
    |    |--- dark-001.fit
    |    |--- dark-002.fit
    |
    |--- flat
    |    |--- flat_half
    |         |--- flat_half-001.fit
    |         |--- flat_half-002.fit
    |         |--- ...
    |    |--- flat_linear
    |         |--- 0_1
    |              |--- flat-001_0_1.fit
    |              |--- flat-002_0_1.fit
    |         |--- 0_2
    |              |--- flat-001_0_2.fit
    |              |--- flat-002_0_2.fit
    |         |--- ...
    |         |--- 1_0
    |              |--- ...
    |         |--- 2_0
    |         |--- 10_0
    |         |--- 20_0
    |         |--- ...
```

The default directory is the current working directory.

The pipeline works on a series of bias, dark, and flat FITS frames in order to produce the following:

* Histograms of averaged bias and flatfield frames
* Calculation of CCD chip gain and readout noise
* Plots of the CCD linearity

Contact richard.camuccio01 {at} utrgv.edu for questions.
