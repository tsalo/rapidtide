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
#
#
#       $Author: frederic $
#       $Date: 2016/06/14 12:04:50 $
#       $Id: linfit,v 1.4 2016/06/14 12:04:50 frederic Exp $
#
from __future__ import print_function, division
import sys
import argparse
from .parser_funcs import is_valid_file, invert_float, is_float

import numpy as np
import rapidtide.io as tide_io
import rapidtide.filter as tide_filt
from sklearn.decomposition import FastICA, PCA, SparsePCA


def _get_parser():
    """
    Argument parser for temporaldecomp
    """
    parser = argparse.ArgumentParser(
        prog="temporaldecomp",
        description="Perform PCA or ICA decomposition on a data file in the time dimension.",
        usage="%(prog)s datafile outputroot [options]",
    )

    # Required arguments
    parser.add_argument(
        "datafile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 or 4 dimensional nifti file to fit",
    )
    parser.add_argument("outputroot", help="The root name for the output nifti files")

    # Optional arguments
    parser.add_argument(
        "--dmask",
        dest="datamaskname",
        type=lambda x: is_valid_file(parser, x),
        action="store",
        metavar="DATAMASK",
        help=("Use DATAMASK to specify which voxels in the data to use."),
        default=None,
    )
    parser.add_argument(
        "--ncomp",
        dest="ncomp",
        type=lambda x: is_float(parser, x),
        action="store",
        metavar="NCOMPS",
        help=(
            "The number of PCA/ICA components to return (default is to estimate the number)."
        ),
        default=-1.0,
    )
    parser.add_argument(
        "--smooth",
        dest="sigma",
        type=lambda x: is_float(parser, x),
        action="store",
        metavar="SIGMA",
        help=("Spatially smooth the input data with a SIGMA mm kernel."),
        default=0.0,
    )
    parser.add_argument(
        "--type",
        dest="decomptype",
        action="store",
        type=str,
        choices=["pca", "sparse", "ica"],
        help=("Type of decomposition to perform. Default is pca."),
        default="pca",
    )
    parser.add_argument(
        "--nodemean",
        dest="demean",
        action="store_false",
        help=("Do not demean data prior to decomposition."),
        default=True,
    )
    parser.add_argument(
        "--novarnorm",
        dest="varnorm",
        action="store_false",
        help=("Do not variance normalize data prior to decomposition."),
        default=True,
    )

    return parser


def temporaldecomp(
    datafile,
    outputroot,
    datamaskname=None,
    decomptype="pca",
    pcacomponents=0.5,
    icacomponents=None,
    varnorm=True,
    demean=True,
    sigma=0.0,
):

    # save the command line
    tide_io.writevec([" ".join(sys.argv)], outputroot + "_commandline.txt")

    # read in data
    print("reading in data arrays")
    (
        datafile_img,
        datafile_data,
        datafile_hdr,
        datafiledims,
        datafilesizes,
    ) = tide_io.readfromnifti(datafile)
    if datamaskname is not None:
        (
            datamask_img,
            datamask_data,
            datamask_hdr,
            datamaskdims,
            datamasksizes,
        ) = tide_io.readfromnifti(datamaskname)

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(datafiledims)
    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(datafilesizes)

    # check dimensions
    if datamaskname is not None:
        print("checking mask dimensions")
        if not tide_io.checkspacedimmatch(datafiledims, datamaskdims):
            print("input mask spatial dimensions do not match image")
            exit()
        if not (
            tide_io.checktimematch(datafiledims, datamaskdims) or datamaskdims[4] == 1
        ):
            print("input mask time dimension does not match image")
            exit()

    # smooth the data
    if sigma > 0.0:
        print("smoothing data")
        for i in range(timepoints):
            datafile_data[:, :, :, i] = tide_filt.ssmooth(
                xdim, ydim, slicethickness, sigma, datafile_data[:, :, :, i]
            )

    # allocating arrays
    print("reshaping arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    rs_datafile = datafile_data.reshape((numspatiallocs, timepoints))

    print("masking arrays")
    if datamaskname is not None:
        if datamaskdims[4] == 1:
            proclocs = np.where(datamask_data.reshape(numspatiallocs) > 0.9)
        else:
            proclocs = np.where(
                np.mean(datamask_data.reshape((numspatiallocs, timepoints)), axis=1)
                > 0.9
            )
            rs_mask = datamask_data.reshape((numspatiallocs, timepoints))[proclocs, :]
            rs_mask = np.where(rs_mask > 0.9, 1.0, 0.0)[0]
    else:
        datamaskdims = [1, xsize, ysize, numslices, 1]
        themaxes = np.max(rs_datafile, axis=1)
        themins = np.min(rs_datafile, axis=1)
        thediffs = (themaxes - themins).reshape(numspatiallocs)
        proclocs = np.where(thediffs > 0.0)
    procdata = rs_datafile[proclocs, :][0]
    print(rs_datafile.shape, procdata.shape)

    # normalize the individual images
    if demean:
        print("demeaning array")
        themean = np.mean(procdata, axis=1)
        print("shape of mean", themean.shape)
        for i in range(procdata.shape[0]):
            procdata[i, :] -= themean[i]

    if varnorm:
        print("variance normalizing array")
        thevar = np.var(procdata, axis=1)
        print("shape of mean", themean.shape)
        for i in range(procdata.shape[0]):
            procdata[i, :] /= thevar[i]
        procdata = np.nan_to_num(procdata)

    # applying mask
    if datamaskdims[4] > 1:
        procdata *= rs_mask

    # now perform the decomposition
    if decomptype == "ica":
        print("performing ica decomposition")
        if icacomponents is None:
            print("will return all significant components")
        else:
            print("will return", icacomponents, "components")
        thefit = FastICA(n_components=icacomponents).fit(
            procdata
        )  # Reconstruct signals
        if icacomponents is None:
            thecomponents = thefit.components_[:]
            print(thecomponents.shape[1], "components found")
        else:
            thecomponents = thefit.components_[0:icacomponents]
            print("returning first", thecomponents.shape[1], "components found")
    else:
        print("performing pca decomposition")
        if pcacomponents < 1.0:
            print(
                "will return the components accounting for",
                pcacomponents * 100.0,
                "% of the variance",
            )
        else:
            print("will return", pcacomponents, "components")
        if decomptype == "pca":
            thepca = PCA(n_components=pcacomponents)
        else:
            thepca = SparsePCA(n_components=pcacomponents)
        thefit = thepca.fit(procdata)
        thetransform = thepca.transform(procdata)
        theinvtrans = thepca.inverse_transform(thetransform)
        if pcacomponents < 1.0:
            thecomponents = thefit.components_[:]
            thesingvals = thefit.singular_values_[:]
            print("returning", thecomponents.shape[1], "components")
        else:
            thecomponents = thefit.components_[0:pcacomponents]
            thesingvals = thefit.singular_values_[:pcacomponents]

        # save the eigenvalues
        print(
            "variance explained by component:", 100.0 * thefit.explained_variance_ratio_
        )
        tide_io.writenpvecs(
            100.0 * thefit.explained_variance_ratio_,
            outputroot + "_explained_variance_pct.txt",
        )

        # save the components
        print("writing component timecourses")
        tide_io.writenpvecs(thecomponents, outputroot + "_components.txt")

        # save the singular values
        print("writing singular values")
        tide_io.writenpvecs(np.transpose(thesingvals), outputroot + "_singvals.txt")

        # save the coefficients
        print("writing out the coefficients")
        coefficients = thetransform
        print("coefficients shape:", coefficients.shape)
        theheader = datafile_hdr
        theheader["dim"][4] = coefficients.shape[1]
        tempout = np.zeros((numspatiallocs, coefficients.shape[1]), dtype="float")
        tempout[proclocs, :] = coefficients[:, :]
        tide_io.savetonifti(
            tempout.reshape((xsize, ysize, numslices, coefficients.shape[1])),
            datafile_hdr,
            outputroot + "_coefficients",
        )

        # save the dimensionality reduced data
        invtransformeddata = theinvtrans
        print("writing fit data")
        theheader = datafile_hdr
        theheader["dim"][4] = invtransformeddata.shape[1]
        tempout = np.zeros((numspatiallocs, invtransformeddata.shape[1]), dtype="float")
        tempout[proclocs, :] = invtransformeddata[:, :]
        tide_io.savetonifti(
            tempout.reshape((xsize, ysize, numslices, invtransformeddata.shape[1])),
            datafile_hdr,
            outputroot + "_fit",
        )


def getparameters():
    try:
        args = vars(_get_parser().parse_args())
    except SystemExit:
        _get_parser().print_help()
        raise
    print()
    print("before postprocessing")
    print(args)

    if args["ncomp"] < 0.0:
        args["pcacomponents"] = 0.5
        args["icacomponents"] = None
    elif args["ncomp"] < 1.0:
        args["pcacomponents"] = inputnum
        args["icacomponents"] = None
    else:
        args["pcacomponents"] = int(a)
        args["icacomponents"] = pcacomponents

    return args


def main():
    args = getparameters()

    temporaldecomp(
        args["datafile"],
        args["outputroot"],
        datamaskname=args["datamaskname"],
        decomptype=args["decomptype"],
        pcacomponents=args["pcacomponents"],
        icacomponents=args["icacomponents"],
        varnorm=args["varnorm"],
        demean=args["demean"],
        sigma=args["sigma"],
    )


if __name__ == "__main__":
    main()
