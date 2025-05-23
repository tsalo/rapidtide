#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
#
#
import argparse
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

import rapidtide.correlate as tide_corr
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resamp
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="showstxcorr",
        description="Plots the data in text files.",
        allow_abbrev=False,
    )

    pf.addreqinputtextfile(parser, "infilename1", onecol=True)
    pf.addreqinputtextfile(parser, "infilename2", onecol=True)
    parser.add_argument(
        "outfilename",
        type=str,
        action="store",
        help="The root name of the output files.",
    )

    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        metavar="FREQ",
        type=lambda x: pf.is_float(parser, x),
        help=(
            "Set the sample rate of the data file to FREQ. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )
    sampling.add_argument(
        "--sampletime",
        dest="samplerate",
        action="store",
        metavar="TSTEP",
        type=lambda x: pf.invert_float(parser, x),
        help=(
            "Set the sample rate of the data file to 1.0/TSTEP. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )
    # add optional arguments
    parser.add_argument(
        "--corrthresh",
        dest="corrthresh",
        metavar="THRESH",
        type=lambda x: pf.is_float(parser, x),
        help=("Cross correlation magnitude threshold to accept a delay value (default is 0.5)."),
        default=0.5,
    )
    parser.add_argument(
        "--windowwidth",
        dest="windowwidth",
        metavar="WINDOWWIDTH",
        type=lambda x: pf.is_float(parser, x),
        help="Use a window width of WINDOWWIDTH seconds (default is 50.0s).",
        default=50.0,
    )
    parser.add_argument(
        "--stepsize",
        dest="stepsize",
        metavar="STEPSIZE",
        type=lambda x: pf.is_float(parser, x),
        help=(
            "Timestep between subsequent measurements (default is 25.0s).  "
            "Will be rounded to the nearest sample time."
        ),
        default=25.0,
    )
    parser.add_argument(
        "--starttime",
        dest="starttime",
        metavar="START",
        type=float,
        help="Start plotting at START seconds (default is 0.0).",
        default=0.0,
    )
    parser.add_argument(
        "--duration",
        dest="duration",
        metavar="DURATION",
        type=float,
        help="Amount of data, in seconds, to process after starttime (default is the entire timecourse).",
        default=1000000.0,
    )
    parser.add_argument(
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Do not plot the data (for noninteractive use)"),
        default=True,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help=("Print out more debugging information"),
        default=False,
    )
    pf.addsearchrangeopts(parser, details=True, defaultmin=-15.0, defaultmax=15.0)
    pf.addtimerangeopts(parser)

    preproc = parser.add_argument_group()
    preproc.add_argument(
        "--detrendorder",
        dest="detrendorder",
        action="store",
        type=int,
        metavar="ORDER",
        help=("Set order of trend removal (0 to disable, default is 1 - linear). "),
        default=1,
    )
    # add window options
    pf.addwindowopts(parser)

    # Filter arguments
    pf.addfilteropts(parser, filtertarget="timecourses", details=True)

    # Preprocessing options
    preproc = parser.add_argument_group("Preprocessing options")
    preproc.add_argument(
        "--corrweighting",
        dest="corrweighting",
        action="store",
        type=str,
        choices=["None", "phat", "liang", "eckart"],
        help=("Method to use for cross-correlation " "weighting. Default is  None. "),
        default="None",
    )
    preproc.add_argument(
        "--invert",
        dest="invert",
        action="store_true",
        help=("Invert second timecourse prior to correlation. "),
        default=False,
    )
    preproc.add_argument(
        "--label",
        dest="label",
        metavar="LABEL",
        action="store",
        type=str,
        help=("Label for the delay value. "),
        default="None",
    )
    return parser


def printthresholds(pcts, thepercentiles, labeltext):
    print(labeltext)
    for i in range(0, len(pcts)):
        print("\tp <", 1.0 - thepercentiles[i], ": ", pcts[i])


def showstxcorr(args):
    # get the command line parameters
    verbose = True
    matrixoutput = False

    # finish up processing arguments, do initial filter setup
    args, theprefilter = pf.postprocessfilteropts(args)
    args = pf.postprocesssearchrangeopts(args)
    args = pf.postprocesstimerangeopts(args)

    if args.display:
        import matplotlib as mpl

        mpl.use("TkAgg")
        import matplotlib.pyplot as plt

    # check that required arguments are set
    if args.samplerate == "auto":
        print("samplerate must be set")
        sys.exit()
    sampletime = 1.0 / args.samplerate

    # read in the files and get everything trimmed to the right length
    stepsize = sampletime * np.round(args.stepsize / sampletime)

    # Now update the lower limit of the filter.
    init_lowerstop, init_lowerpass, init_upperpass, init_upperstop = theprefilter.getfreqs()
    window_lowestfreq = 1.0 / args.windowwidth
    if window_lowestfreq > init_lowerpass:
        print(
            f"resetting lower limit of filter from {init_lowerpass} to {window_lowestfreq}Hz for window length {args.windowwidth}s"
        )
        theprefilter.settype("arb")
        theprefilter.setfreqs(
            window_lowestfreq,
            window_lowestfreq,
            init_upperpass,
            init_upperstop,
        )

    startpoint = max([int(args.starttime * args.samplerate), 0])
    inputdata1 = tide_io.readvec(args.infilename1)
    numpoints = len(inputdata1)
    inputdata2 = tide_io.readvec(args.infilename2)
    endpoint1 = min(
        [
            startpoint + int(args.duration * args.samplerate),
            int(len(inputdata1)),
            int(len(inputdata2)),
        ]
    )
    endpoint2 = min(
        [int(args.duration * args.samplerate), int(len(inputdata1)), int(len(inputdata2))]
    )
    trimmeddata = np.zeros((2, numpoints), dtype="float")
    trimmeddata[0, :] = inputdata1[startpoint:endpoint1]
    trimmeddata[1, :] = inputdata2[0:endpoint2]

    # band limit the regressors if that is needed
    if theprefilter.gettype() != "None":
        if verbose:
            print("filtering to ", theprefilter.gettype(), " band")

    thedims = trimmeddata.shape
    tclen = thedims[1]
    numcomponents = thedims[0]
    reformdata = np.reshape(trimmeddata, (numcomponents, tclen))

    print("preprocessing all timecourses")
    for component in range(0, numcomponents):
        filtereddata = tide_math.corrnormalize(
            theprefilter.apply(args.samplerate, reformdata[component, :]),
            windowfunc="None",
            detrendorder=args.detrendorder,
        )
        reformdata[component, :] = tide_math.stdnormalize(
            tide_fit.detrend(tide_math.stdnormalize(filtereddata), order=args.detrendorder)
        )

    xcorr_x = np.r_[0.0:tclen] * sampletime - (tclen * sampletime) / 2.0
    searchstart = int(int(tclen) // 2 + (args.lagmin * args.samplerate))
    searchend = int(int(tclen) // 2 + (args.lagmax * args.samplerate))
    xcorr_x_trim = xcorr_x[searchstart:searchend]
    if args.invert:
        flipfac = -1.0
    else:
        flipfac = 1.0

    # now that we have all the information, we have a couple of places to go:
    # We are either doing short term correlations or full timecourse
    # We are either doing two time courses from different (or the same) files, or we are doing more than 2

    if matrixoutput:
        # find the lengths of the outputfiles
        print("finding timecourse lengths")
        times, corrpertime, ppertime = tide_corr.shorttermcorr_1D(
            reformdata[0, :],
            reformdata[0, :],
            sampletime,
            args.windowwidth,
            samplestep=int(stepsize // sampletime),
            windowfunc=args.windowfunc,
            detrendorder=0,
        )
        plength = len(times)
        times, xcorrpertime, Rvals, delayvals, valid = tide_corr.shorttermcorr_2D(
            reformdata[0, :],
            reformdata[0, :],
            sampletime,
            args.windowwidth,
            laglimits=[args.lagmin, args.lagmax],
            samplestep=int(stepsize // sampletime),
            weighting=args.corrweighting,
            windowfunc=args.windowfunc,
            detrendorder=0,
            displayplots=False,
        )
        xlength = len(times)

        # now allocate the output arrays
        print("allocating data arrays")
        Rvals = np.zeros((numcomponents, numcomponents, 1, xlength), dtype="float")
        delayvals = np.zeros((numcomponents, numcomponents, 1, xlength), dtype="float")
        valid = np.zeros((numcomponents, numcomponents, 1, xlength), dtype="float")
        corrpertime = np.zeros((numcomponents, numcomponents, 1, plength), dtype="float")
        ppertime = np.zeros((numcomponents, numcomponents, 1, plength), dtype="float")

        # do the correlations
        for component1 in range(0, numcomponents):
            print("correlating with component", component1)
            for component2 in range(0, numcomponents):
                (
                    times,
                    corrpertime[component1, component2, 0, :],
                    ppertime[component1, component2, 0, :],
                ) = tide_corr.shorttermcorr_1D(
                    reformdata[component1, :],
                    flipfac * reformdata[component2, :],
                    sampletime,
                    args.windowwidth,
                    samplestep=int(stepsize // sampletime),
                    windowfunc=args.windowfunc,
                    detrendorder=0,
                )
                (
                    times,
                    xcorrpertime,
                    Rvals[component1, component2, 0, :],
                    delayvals[component1, component2, 0, :],
                    valid[component1, component2, 0, :],
                ) = tide_corr.shorttermcorr_2D(
                    reformdata[component1, :],
                    flipfac * reformdata[component2, :],
                    sampletime,
                    args.windowwidth,
                    laglimits=[args.lagmin, args.lagmax],
                    samplestep=int(stepsize // sampletime),
                    weighting=args.corrweighting,
                    windowfunc=args.windowfunc,
                    detrendorder=0,
                    displayplots=False,
                )

        outputaffine = np.eye(4)
        # input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(inputfilename)
        init_img = nib.Nifti1Image(corrpertime, outputaffine)
        init_hdr = init_img.header.copy()
        init_sizes = init_hdr["pixdim"].copy()
        init_sizes[4] = sampletime
        init_hdr["toffset"] = times[0]
        tide_io.savetonifti(corrpertime, init_hdr, args.outfilename + "_pearsonR")
        tide_io.savetonifti(ppertime, init_hdr, args.outfilename + "_corrp")
        tide_io.savetonifti(Rvals, init_hdr, args.outfilename + "_maxxcorr")
        tide_io.savetonifti(delayvals, init_hdr, args.outfilename + "_delayvals")
        tide_io.savetonifti(valid, init_hdr, args.outfilename + "_valid")
        rows = []
        cols = []
        for i in range(numcomponents):
            rows.append("region " + str(i + 1))
            cols.append("region " + str(i + 1))
        for segment in range(plength):
            df = pd.DataFrame(data=corrpertime[:, :, 0, 0], columns=cols)
            df.insert(0, "sources", pd.Series(rows))
            df.to_csv(
                args.outfilename + "_seg_" + str(segment).zfill(4) + "_pearsonR.csv",
                index=False,
            )
            df = pd.DataFrame(data=ppertime[:, :, 0, 0], columns=cols)
            df.insert(0, "sources", pd.Series(rows))
            df.to_csv(
                args.outfilename + "_seg_" + str(segment).zfill(4) + "_corrp.csv",
                index=False,
            )
        for segment in range(xlength):
            df = pd.DataFrame(data=Rvals[:, :, 0, 0], columns=cols)
            df.insert(0, "sources", pd.Series(rows))
            df.to_csv(
                args.outfilename + "_seg_" + str(segment).zfill(4) + "_maxxcorr.csv",
                index=False,
            )
            df = pd.DataFrame(data=delayvals[:, :, 0, 0], columns=cols)
            df.insert(0, "sources", pd.Series(rows))
            df.to_csv(
                args.outfilename + "_seg_" + str(segment).zfill(4) + "_delayvals.csv",
                index=False,
            )
            df = pd.DataFrame(data=valid[:, :, 0, 0], columns=cols)
            df.insert(0, "sources", pd.Series(rows))
            df.to_csv(
                args.outfilename + "_seg_" + str(segment).zfill(4) + "_valid.csv",
                index=False,
            )

    else:
        times, corrpertime, ppertime = tide_corr.shorttermcorr_1D(
            reformdata[0, :],
            flipfac * reformdata[1, :],
            sampletime,
            args.windowwidth,
            samplestep=int(stepsize // sampletime),
            windowfunc=args.windowfunc,
            detrendorder=0,
        )
        times, xcorrpertime, Rvals, delayvals, valid = tide_corr.shorttermcorr_2D(
            reformdata[0, :],
            flipfac * reformdata[1, :],
            sampletime,
            args.windowwidth,
            laglimits=[args.lagmin, args.lagmax],
            samplestep=int(stepsize // sampletime),
            weighting=args.corrweighting,
            windowfunc=args.windowfunc,
            detrendorder=0,
            displayplots=False,
        )
        tide_io.writenpvecs(corrpertime, args.outfilename + "_pearson.txt")
        tide_io.writenpvecs(ppertime, args.outfilename + "_pvalue.txt")
        tide_io.writenpvecs(Rvals, args.outfilename + "_Rvalue.txt")
        tide_io.writenpvecs(delayvals, args.outfilename + "_delay.txt")
        tide_io.writenpvecs(valid, args.outfilename + "_mask.txt")

        # threshold by corrthresh
        validlocs = np.where(np.fabs(Rvals) > args.corrthresh)

        # do a polynomial fit to the delay function
        thefit = (
            Polynomial.fit(
                times[validlocs],
                delayvals[validlocs],
                7,
                w=(Rvals[validlocs] * Rvals[validlocs]),
            )
            .convert()
            .coef[::-1]
        )
        smoothdelayvals = np.poly1d(thefit)(times)

        hirestimeaxis = np.linspace(0, sampletime * endpoint2, num=endpoint2, endpoint=False)
        highresdelayvals = np.poly1d(thefit)(hirestimeaxis)
        if args.debug:
            print("len(hirestimeaxis):", len(hirestimeaxis))
            print("len(highresdelayvals):", len(highresdelayvals))
            print("len(reformdata[1, :]):", len(reformdata[1, :]))

        timewarped = tide_resamp.timewarp(
            hirestimeaxis,
            reformdata[1, :],
            highresdelayvals,
            debug=args.debug,
        )
        tide_io.writenpvecs(timewarped, args.outfilename + "_timewarped.txt")
        tide_io.writenpvecs(highresdelayvals, args.outfilename + "_hiresdelayvals.txt")

        if args.display:
            # timeaxis = np.r_[0.0:len(filtereddata1)] * sampletime
            fig, ax1 = plt.subplots()
            ax1.plot(times, corrpertime, "k")
            ax1.set_ylabel("Pearson R", color="k")
            ax2 = ax1.twinx()
            ax2.plot(times, ppertime, "r")
            ax2.set_ylabel("p value", color="r")
            fig, ax3 = plt.subplots()
            ax3.plot(times[validlocs], Rvals[validlocs], "k")
            ax3.set_ylabel("Xcorr max R", color="k")
            ax4 = ax3.twinx()
            ax4.plot(times[validlocs], delayvals[validlocs], marker=".", linestyle="None")
            ax4.plot(times, smoothdelayvals, "g")
            ax4.set_ylabel("Delay (s)", color="r")
            # fig, ax5 = plt.subplots()
            # ax5.set_ylabel("Delay vs xcorr max R", color="k")
            # ax5.plot(Rvals, delayvals, "r", marker=".", linestyle="None")
            # ax2.set_yscale('log')
            plt.show()
