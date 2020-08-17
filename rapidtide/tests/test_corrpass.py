#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from __future__ import print_function, division

import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.correlate as tide_corr
import rapidtide.stats as tide_stats
import rapidtide.io as tide_io
import rapidtide.calcsimfunc as tide_calcsimfunc
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.resample as tide_resample
import rapidtide.helper_classes as tide_classes
from rapidtide.tests.utils import mse

import matplotlib.pyplot as plt
from rapidtide.tests.utils import get_test_data_path, get_test_target_path, get_test_temp_path, get_examples_path, get_rapidtide_root, get_scripts_path, create_dir
import os


def test_calcsimfunc(debug=False, display=False):
    # make the lfo filter
    lfofilter = tide_filt.noncausalfilter(filtertype='lfo')

    # make some data
    oversampfactor = 2
    numvoxels = 100
    numtimepoints = 500
    tr = 0.72
    Fs = 1.0 / tr
    init_fmri_x = np.linspace(0.0, numtimepoints, numtimepoints, endpoint=False) * tr
    oversampfreq = oversampfactor * Fs
    os_fmri_x = np.linspace(0.0, numtimepoints * oversampfactor, numtimepoints * oversampfactor) * (1.0 / oversampfreq)

    theinputdata = np.zeros((numvoxels, numtimepoints), dtype=np.float)
    meanval = np.zeros((numvoxels), dtype=np.float)

    testfreq = 0.075
    msethresh = 1e-3

    # make the starting regressor
    sourcedata = np.sin(2.0 * np.pi * testfreq * os_fmri_x)
    numpasses = 1

    # make the timeshifted data
    shiftstart = -5.0
    shiftend = 5.0
    voxelshifts = np.linspace(shiftstart, shiftend, numvoxels, endpoint=False)
    for i in range(numvoxels):
        theinputdata[i, :] = np.sin(2.0 * np.pi * testfreq * (init_fmri_x - voxelshifts[i]))
        

    if display:
        plt.figure()
        plt.plot(sourcedata)
        plt.show()
    genlagtc = tide_resample.fastresampler(os_fmri_x, sourcedata)

    thexcorr = tide_corr.fastcorrelate(sourcedata, sourcedata)
    xcorrlen = len(thexcorr)
    xcorr_x = np.linspace(0.0, xcorrlen, xcorrlen, endpoint=False) * tr - (xcorrlen * tr) / 2.0 + tr / 2.0

    if display:
        plt.figure()
        plt.plot(xcorr_x, thexcorr)
        plt.show()


    corrzero = xcorrlen // 2
    lagmin = -10.0
    lagmax = 10.0
    lagmininpts = int((-lagmin * oversampfreq) - 0.5)
    lagmaxinpts = int((lagmax * oversampfreq) + 0.5)

    searchstart = int(np.round(corrzero + lagmin / tr))
    searchend = int(np.round(corrzero + lagmax / tr))
    numcorrpoints = lagmaxinpts + lagmininpts
    corrout = np.zeros((numvoxels, numcorrpoints), dtype=np.float)
    lagmask = np.zeros((numvoxels), dtype=np.float)
    failimage = np.zeros((numvoxels), dtype=np.float)
    lagtimes = np.zeros((numvoxels), dtype=np.float)
    lagstrengths = np.zeros((numvoxels), dtype=np.float)
    lagsigma = np.zeros((numvoxels), dtype=np.float)
    gaussout = np.zeros((numvoxels, numcorrpoints), dtype=np.float)
    windowout = np.zeros((numvoxels, numcorrpoints), dtype=np.float)
    R2 = np.zeros((numvoxels), dtype=np.float)
    lagtc = np.zeros((numvoxels, numtimepoints), dtype=np.float)

    optiondict = {
        'numestreps':        10000,
        'interptype':        'univariate',
        'showprogressbar':   debug,
        'detrendorder':      3,
        'windowfunc':        'hamming',
        'corrweighting':     'None',
        'nprocs':            1,
        'widthlimit':        1000.0,
        'bipolar':           False,
        'fixdelay':          False,
        'peakfittype':       'gauss',
        'lagmin':            lagmin,
        'lagmax':            lagmax,
        'absminsigma':       0.25,
        'absmaxsigma':       25.0,
        'edgebufferfrac':    0.0,
        'lthreshval':        0.0,
        'uthreshval':        1.1,
        'debug':             False,
        'enforcethresh':     True,
        'lagmod':            1000.0,
        'searchfrac':        0.5,
        'mp_chunksize':      1000,
        'oversampfactor':    oversampfactor,
        'despeckle_thresh':  5.0,
        'zerooutbadfit':     False,
        'permutationmethod': 'shuffle',
        'hardlimit':         True
    }

    theprefilter = tide_filt.noncausalfilter('lfo')
    thecorrelator = tide_classes.correlator(Fs=oversampfreq,
                                         ncprefilter=theprefilter,
                                         detrendorder=optiondict['detrendorder'],
                                         windowfunc=optiondict['windowfunc'],
                                         corrweighting=optiondict['corrweighting'])

    thefitter = tide_classes.simfunc_fitter(lagmod=optiondict['lagmod'],
                                             lthreshval=optiondict['lthreshval'],
                                             uthreshval=optiondict['uthreshval'],
                                             bipolar=optiondict['bipolar'],
                                             lagmin=optiondict['lagmin'],
                                             lagmax=optiondict['lagmax'],
                                             absmaxsigma=optiondict['absmaxsigma'],
                                             absminsigma=optiondict['absminsigma'],
                                             debug=optiondict['debug'],
                                             peakfittype=optiondict['peakfittype'],
                                             zerooutbadfit=optiondict['zerooutbadfit'],
                                             searchfrac=optiondict['searchfrac'],
                                             enforcethresh=optiondict['enforcethresh'],
                                             hardlimit=optiondict['hardlimit'])

    if debug:
        print(optiondict)

    thecorrelator.setlimits(lagmininpts, lagmaxinpts)
    thecorrelator.setreftc(sourcedata)
    dummy, trimmedcorrscale, dummy = thecorrelator.getfunction()
    thefitter.setcorrtimeaxis(trimmedcorrscale)

    for thenprocs in [1, -1]:
        for i in range(numpasses):
            voxelsprocessed_cp, theglobalmaxlist, trimmedcorrscale \
                = tide_calcsimfunc.correlationpass(theinputdata,
                                                   sourcedata,
                                                   thecorrelator,
                                                   init_fmri_x,
                                                   os_fmri_x,
                                                   lagmininpts,
                                                   lagmaxinpts,
                                                   corrout,
                                                   meanval,
                                                   nprocs=thenprocs,
                                                   oversampfactor=optiondict['oversampfactor'],
                                                   interptype=optiondict['interptype'],
                                                   showprogressbar=optiondict['showprogressbar'],
                                                   chunksize=optiondict['mp_chunksize'])

            if display:
                plt.figure()
                plt.plot(trimmedcorrscale, corrout[numvoxels // 2, :], 'k')
                plt.show()

            voxelsprocessed_fc \
                = tide_simfuncfit.fitcorr(genlagtc,
                                          init_fmri_x,
                                          lagtc,
                                          trimmedcorrscale,
                                          thefitter,
                                          corrout,
                                          lagmask, failimage, lagtimes, lagstrengths, lagsigma,
                                          gaussout, windowout, R2,
                                          nprocs=optiondict['nprocs'],
                                          fixdelay=optiondict['fixdelay'],
                                          showprogressbar=optiondict['showprogressbar'],
                                          chunksize=optiondict['mp_chunksize'],
                                          despeckle_thresh=optiondict['despeckle_thresh']
                                          )
            if display:
                plt.figure()
                plt.plot(voxelshifts, 'k')
                plt.plot(lagtimes, 'r')
                plt.show()

            if debug:
                for i in range(numvoxels):
                    print(voxelshifts[i], lagtimes[i], lagstrengths[i], lagsigma[i], failimage[i])

            assert mse(voxelshifts, lagtimes) < msethresh

if __name__ == '__main__':
    test_calcsimfunc(debug=True, display=True)