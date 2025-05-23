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

import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io


def _get_parser():
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="variabilityizer",
        description="Transform a nifti fmri file into a temporal variability file.",
        allow_abbrev=False,
    )
    parser.add_argument("inputfilename", type=str, help="The name of the input nifti file.")
    parser.add_argument("outputfilename", type=str, help="The name of the output nifti file.")
    parser.add_argument(
        "windowlength",
        type=float,
        help="The size of the temporal window in seconds.",
    )
    return parser


def cvttovariability(windowhalfwidth, data):
    themean = np.mean(data)
    if themean > 0.0:
        thestd = data * 0.0
        for i in range(windowhalfwidth):
            thestd[i] = np.std(data[: i + windowhalfwidth + 1])
            thestd[-(i + 1)] = np.std(data[-(i + 1) - windowhalfwidth :])
        for i in range(windowhalfwidth, len(data) - windowhalfwidth):
            thestd[i] = np.std(data[i - windowhalfwidth : i + windowhalfwidth + 1])
        return thestd + themean
    else:
        return data


def variabilityizer(args):
    # get the input TR
    inputtr_fromfile, numinputtrs = tide_io.fmritimeinfo(args.inputfilename)
    print("input data: ", numinputtrs, " timepoints, tr = ", inputtr_fromfile)

    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    if input_hdr.get_xyzt_units()[1] == "msec":
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    winsize = int(np.round(args.windowlength / tr))
    winsize += 1 - (winsize % 2)  # make odd
    windowhalfwidth = winsize // 2
    print(f"window size in trs = {2 * windowhalfwidth + 1} ({2 * windowhalfwidth * tr} seconds)")

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    stdtcs = np.zeros((xsize, ysize, numslices, timepoints), dtype="float")

    # cycle over all voxels
    print("now cycling over all voxels")
    for zloc in range(numslices):
        print("processing slice ", zloc)
        for yloc in range(ysize):
            for xloc in range(xsize):
                stdtcs[xloc, yloc, zloc, :] = cvttovariability(
                    windowhalfwidth, input_data[xloc, yloc, zloc, :]
                )

    # now do the ones with other numbers of time points
    tide_io.savetonifti(stdtcs, input_hdr, args.outputfilename)
