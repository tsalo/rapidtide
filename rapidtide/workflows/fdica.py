#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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
import pyfftw
import copy

fftpack = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()
import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_valid_file


def _get_parser():
    """
    Argument parser for fdica
    """
    parser = argparse.ArgumentParser(
        prog="fdica",
        description="Fit a spatial template to a 3D or 4D NIFTI file.",
        usage="%(prog)s fdica datafile datamask outputroot [options]",
    )

    # Required arguments
    parser.add_argument(
        "datafile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 or 4 dimensional nifti file to fit.",
    )
    parser.add_argument(
        "datamask",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 dimensional nifti file voxel mask (must match datafile).",
    )
    parser.add_argument("outputroot", type=str, help="The root name for all output files.")

    parser.add_argument(
        "--ncomp",
        metavar="NCOMP",
        dest="ncomponents",
        type=int,
        help="Return NCOMP components",
        default=-1,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Output additional debugging information.",
        default=False,
    )
    return parser


def fdica(
    datafile,
    datamask,
    outputroot,
    ncomponents=-1,
    lowerfreq=0.009,
    upperfreq=0.15,
    debug=False,
):

    # read in data
    print("reading in data arrays")
    (
        datafile_img,
        datafile_data,
        datafile_hdr,
        datafiledims,
        datafilesizes,
    ) = tide_io.readfromnifti(datafile)
    (
        datamask_img,
        datamask_data,
        datamask_hdr,
        datamaskdims,
        datamasksizes,
    ) = tide_io.readfromnifti(datamask)

    xsize = datafiledims[1]
    ysize = datafiledims[2]
    numslices = datafiledims[3]
    timepoints = datafiledims[4]

    if datafile_hdr.get_xyzt_units()[1] == "msec":
        fmritr = datafilesizes[4] / 1000.0
    else:
        fmritr = datafilesizes[4]
    nyquist = 0.5 / fmritr
    hzperpoint = nyquist / timepoints
    print(f"nyquist: {nyquist}Hz, hzperpoint: {hzperpoint}Hz")

    # figure out what bins we will retain
    lowerbin = int(np.floor(lowerfreq / hzperpoint))
    upperbin = int(np.ceil(upperfreq / hzperpoint))
    trimmedsize = upperbin - lowerbin + 1
    print(f"will retain points {lowerbin} to {upperbin}")

    # check dimensions
    print("checking dimensions")
    if not tide_io.checkspacedimmatch(datafiledims, datamaskdims):
        print("input mask spatial dimensions do not match image")
        exit()
    if datamaskdims[4] != 1:
        print("specify a 3d data mask")
        sys.exit()

    # create arrays
    print("allocating arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    rs_datafile = datafile_data.reshape((numspatiallocs, timepoints))
    rs_datamask = datamask_data.reshape(numspatiallocs)
    rs_datamask_bin = np.where(rs_datamask > 0.9, 1.0, 0.0)
    savearray = np.zeros(xsize, ysize, numslices, trimmedsize)
    rs_savearray = savearray.reshape(numspatiallocs, trimmedsize)

    # select the voxels to process
    voxelstofit = np.where(rs_datamask_bin > 0.5)
    procvoxels = rs_datafile[voxelstofit, :]

    # calculating FFT
    print("calculating forward FFT")
    complexfftdata = fftpack.fft(procvoxels, axis=1)

    # trim the data
    trimmeddata = complexfftdata[:, lowerbin : min(upperbin + 1, timepoints)]

    # convert to polar
    magdata = np.absolute(trimmeddata)
    phasedata = np.unwrap(np.angle(trimmeddata))

    saveheader = copy.deepcopy(datafile_hdr)
    saveheader["dim"][4] = trimmedsize
    saveheader["pixdim"][4] = hzperpoint

    rs_savearray[voxelstofit, :] = magdata
    tide_io.savetonifti(
        savearray.reshape((xsize, ysize, numslices, trimmedsize)),
        saveheader,
        outputroot + "_mag",
    )
    rs_savearray[voxelstofit, :] = phasedata
    tide_io.savetonifti(
        savearray.reshape((xsize, ysize, numslices, trimmedsize)),
        saveheader,
        outputroot + "_phase",
    )
    icainput = np.vstack(magdata, phasedata)


def main():

    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    print(args)

    fdica(
        args.datafile,
        args.datamask,
        args.outputroot,
        ncomponents=args.ncomponents,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()