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

import numpy as np

import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for filttc
    """
    parser = argparse.ArgumentParser(
        prog="filttc",
        description=("Filter timecourse data in text files"),
        allow_abbrev=False,
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "inputfile")
    pf.addreqoutputtextfile(parser, "outputfile")

    # add optional arguments
    freq_group = parser.add_mutually_exclusive_group()
    freq_group.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        metavar="FREQ",
        help=(
            "Timecourses in file have sample "
            "frequency FREQ (default is 1.0Hz) "
            "NB: --samplerate and --sampletstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )
    freq_group.add_argument(
        "--sampletstep",
        dest="samplerate",
        action="store",
        type=lambda x: pf.invert_float(parser, x),
        metavar="TSTEP",
        help=(
            "Timecourses in file have sample "
            "timestep TSTEP (default is 1.0s) "
            "NB: --samplerate and --sampletstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )

    # Filter arguments
    pf.addfilteropts(parser, filtertarget="timecourses")

    # Normalization arguments
    pf.addnormalizationopts(parser, normtarget="timecourses", defaultmethod="None")

    parser.add_argument(
        "--normfirst",
        dest="normfirst",
        action="store_true",
        help=("Normalize before filtering, rather than after."),
        default=False,
    )
    parser.add_argument(
        "--demean",
        dest="demean",
        action="store_true",
        help=("Demean before filtering."),
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )

    # Miscellaneous options

    return parser


def filttc(args):
    args, thefilter = pf.postprocessfilteropts(args)

    # read in data
    (
        samplerate,
        starttime,
        colnames,
        invecs,
        compressed,
        filetype,
    ) = tide_io.readvectorsfromtextfile(args.inputfile)

    if samplerate is None:
        if args.samplerate == "auto":
            print(
                "sample rate must be specified, either by command line arguments or in the file header."
            )
            sys.exit()
        else:
            samplerate = args.samplerate
    else:
        if args.samplerate != "auto":
            samplerate = args.samplerate

    print("about to filter")
    numvecs = invecs.shape[0]
    if numvecs == 1:
        print("there is 1 timecourse")
    else:
        print("there are", numvecs, "timecourses")
    print("samplerate is", samplerate)
    outvecs = invecs * 0.0
    for i in range(numvecs):
        if args.normfirst:
            outvecs[i, :] = thefilter.apply(
                samplerate, tide_math.normalize(invecs[i, :], method=args.normmethod)
            )
        else:
            outvecs[i, :] = tide_math.normalize(
                thefilter.apply(samplerate, invecs[i, :]), method=args.normmethod
            )
        if args.demean:
            outvecs[i, :] -= np.mean(outvecs[i, :])

    tide_io.writevectorstotextfile(
        outvecs,
        args.outputfile,
        samplerate=samplerate,
        starttime=starttime,
        columns=colnames,
        compressed=compressed,
        filetype=filetype,
    )
    # tide_io.writenpvecs(outvecs, args.outputfile)
