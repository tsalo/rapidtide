#!/usr/bin/env python
#
#   Copyright 2016 Blaise Frederick
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
# $Author: frederic $
# $Date: 2016/07/11 14:50:43 $
# $Id: rapidtide,v 1.161 2016/07/11 14:50:43 frederic Exp $
#
#
#

from __future__ import print_function, division

import bisect
import gc

import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc
import rapidtide.util as tide_util


def onecorrfitx(thetc, corrscale, optiondict, disablethresholds=False,
                displayplots=False, initiallag=None,
                rt_floatset=np.float64, rt_floattype='float64'):
    """
    optiondict : dict
        Dictionary with fields: despeckle_thresh, widthlimit, lthreshval,
        bipolar, fixdelay, findmaxtype, lagmin, lagmax, absmaxsigma,
        edgebufferfrac, uthreshval, debug, gaussrefine, searchfrac,
        fastgauss, enforcethresh, zerooutbadfit, lagmod, hardlimit,
        fixeddelayvalue, showprogressbar, nprocs, mp_chunksize
    """
    if initiallag is not None:
        maxguess = initiallag
        useguess = True
        widthlimit = optiondict['despeckle_thresh']
    else:
        maxguess = 0.0
        useguess = False
        widthlimit = optiondict['widthlimit']

    if disablethresholds:
        thethreshval = 0.0
    else:
        thethreshval = optiondict['lthreshval']

    flipfac = None
    if optiondict['bipolar']:
        if max(thetc) < -1.0 * min(thetc):
            flipfac = rt_floatset(-1.0)
        else:
            flipfac = rt_floatset(1.0)
    else:
        flipfac = rt_floatset(1.0)

    if not optiondict['fixdelay']:
        if optiondict['findmaxtype'] == 'gauss':
            (maxindex, maxlag, maxval, maxsigma,
             maskval, failreason,
             peakstart, peakend) = tide_fit.findmaxlag_gauss_rev(
                 corrscale,
                 thetc,
                 optiondict['lagmin'],
                 optiondict['lagmax'],
                 widthlimit,
                 absmaxsigma=optiondict['absmaxsigma'],
                 edgebufferfrac=optiondict['edgebufferfrac'],
                 threshval=thethreshval,
                 uthreshval=optiondict['uthreshval'],
                 debug=optiondict['debug'],
                 refine=optiondict['gaussrefine'],
                 bipolar=optiondict['bipolar'],
                 maxguess=maxguess,
                 useguess=useguess,
                 searchfrac=optiondict['searchfrac'],
                 fastgauss=optiondict['fastgauss'],
                 enforcethresh=optiondict['enforcethresh'],
                 zerooutbadfit=optiondict['zerooutbadfit'],
                 lagmod=optiondict['lagmod'],
                 hardlimit=optiondict['hardlimit'],
                 displayplots=displayplots)
        else:
            (maxindex, maxlag, maxval, maxsigma,
             maskval, failreason,
             peakstart, peakend) = tide_fit.findmaxlag_quad(
                corrscale,
                flipfac * thetc,
                optiondict['lagmin'], optiondict['lagmax'], widthlimit,
                edgebufferfrac=optiondict['edgebufferfrac'],
                threshval=optiondict['lthreshval'],
                uthreshval=optiondict['uthreshval'],
                debug=optiondict['debug'],
                refine=optiondict['gaussrefine'],
                maxguess=maxguess,
                useguess=useguess,
                fastgauss=optiondict['fastgauss'],
                enforcethresh=optiondict['enforcethresh'],
                zerooutbadfit=optiondict['zerooutbadfit'],
                lagmod=optiondict['lagmod'],
                displayplots=displayplots)
            maxval *= flipfac
    else:
        # do something different
        failreason = np.int16(0)
        maxlag = rt_floatset(optiondict['fixeddelayvalue'])
        maxindex = np.int16(bisect.bisect_left(corrscale, optiondict['fixeddelayvalue']))
        maxval = rt_floatset(flipfac * thetc[maxindex])
        maxsigma = rt_floatset(1.0)
        maskval = np.uint16(1)

    return (maxindex, maxlag, maxval, maxsigma,
            maskval, peakstart, peakend, failreason)


def _procOneVoxelFitcorrx(vox, corrtc, corrscale, genlagtc, initial_fmri_x,
                          optiondict, displayplots=False, initiallag=None,
                          rt_floatset=np.float64, rt_floattype='float64'):
    (maxindex, maxlag, maxval, maxsigma,
     maskval, peakstart, peakend,
     failreason) = onecorrfitx(corrtc, corrscale, optiondict,
                               displayplots=displayplots,
                               initiallag=initiallag,
                               rt_floatset=rt_floatset,
                               rt_floattype=rt_floattype)

    if maxval > 0.3:
        displayplots = False

    # question - should maxlag be added or subtracted?  As of 10/18, it is subtracted
    #  potential answer - tried adding, results are terrible.
    thelagtc = rt_floatset(genlagtc.yfromx(initial_fmri_x - maxlag))

    # now tuck everything away in the appropriate output array
    volumetotalinc = 0
    thewindowout = rt_floatset(0.0 * corrtc)
    thewindowout[peakstart:peakend + 1] = 1.0
    if (maskval == 0) and optiondict['zerooutbadfit']:
        thetime = rt_floatset(0.0)
        thestrength = rt_floatset(0.0)
        thesigma = rt_floatset(0.0)
        thegaussout = 0.0 * corrtc
        theR2 = rt_floatset(0.0)
    else:
        volumetotalinc = 1
        thetime = rt_floatset(np.fmod(maxlag, optiondict['lagmod']))
        thestrength = rt_floatset(maxval)
        thesigma = rt_floatset(maxsigma)
        thegaussout = rt_floatset(0.0 * corrtc)
        thewindowout = rt_floatset(0.0 * corrtc)
        if (not optiondict['fixdelay']) and (maxsigma != 0.0):
            thegaussout = rt_floatset(tide_fit.gauss_eval(corrscale, [maxval, maxlag, maxsigma]))
        else:
            thegaussout = rt_floatset(0.0)
            thewindowout = rt_floatset(0.0)
        theR2 = rt_floatset(thestrength * thestrength)

    return vox, volumetotalinc, thelagtc, thetime, thestrength, thesigma, thegaussout, \
           thewindowout, theR2, maskval, failreason


def fitcorrx(genlagtc,
            initial_fmri_x,
            lagtc,
            slicesize,
            corrscale,
            lagmask,
            failimage,
            lagtimes,
            lagstrengths,
            lagsigma,
            corrout,
            meanval,
            gaussout,
            windowout,
            R2,
            optiondict,
            initiallags=None,
            rt_floatset=np.float64,
            rt_floattype='float64'):
    displayplots = False
    inputshape = np.shape(corrout)
    if initiallags is None:
        themask = None
    else:
        themask = np.where(initiallags > -1000000.0, 1, 0)
    reportstep = 1000
    volumetotal, ampfails, lagfails, windowfails, widthfails, edgefails, fitfails = 0, 0, 0, 0, 0, 0, 0
    FML_BADAMPLOW = np.uint16(0x01)
    FML_BADAMPNEG = np.uint16(0x02)
    FML_BADSEARCHWINDOW = np.uint16(0x04)
    FML_BADWIDTH = np.uint16(0x08)
    FML_BADLAG = np.uint16(0x10)
    FML_HITEDGE = np.uint16(0x20)
    FML_FITFAIL = np.uint16(0x40)
    FML_INITFAIL = np.uint16(0x80)
    zerolagtc = rt_floatset(genlagtc.yfromx(initial_fmri_x))
    sliceoffsettime = 0.0

    if optiondict['nprocs'] > 1:
        # define the consumer function here so it inherits most of the arguments
        def fitcorr_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    if initiallags is None:
                        thislag = None
                    else:
                        thislag = initiallags[val]
                    outQ.put(_procOneVoxelFitcorrx(val,
                                                  corrout[val, :],
                                                  corrscale,
                                                  genlagtc,
                                                  initial_fmri_x,
                                                  optiondict,
                                                  displayplots=displayplots,
                                                  initiallag=thislag,
                                                  rt_floatset=rt_floatset,
                                                  rt_floattype=rt_floattype))
                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(fitcorr_consumer,
                                                inputshape, themask,
                                                nprocs=optiondict['nprocs'],
                                                showprogressbar=True,
                                                chunksize=optiondict['mp_chunksize'])

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            volumetotal += voxel[1]
            lagtc[voxel[0], :] = voxel[2]
            lagtimes[voxel[0]] = voxel[3]
            lagstrengths[voxel[0]] = voxel[4]
            lagsigma[voxel[0]] = voxel[5]
            gaussout[voxel[0], :] = voxel[6]
            windowout[voxel[0], :] = voxel[7]
            R2[voxel[0]] = voxel[8]
            lagmask[voxel[0]] = voxel[9]
            failimage[voxel[0]] = voxel[10] & 0x3f
        if (FML_BADAMPLOW | FML_BADAMPNEG) & voxel[10]:
            ampfails += 1
        if FML_BADSEARCHWINDOW & voxel[10]:
            windowfails += 1
        if FML_BADWIDTH & voxel[10]:
            widthfails += 1
        if FML_BADLAG & voxel[10]:
            lagfails += 1
        if FML_HITEDGE & voxel[10]:
            edgefails += 1
        if (FML_FITFAIL | FML_INITFAIL) & voxel[10]:
            fitfails += 1
        del data_out
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete')
            if themask is None:
                dothisone = True
                thislag = None
            elif themask[vox] > 0:
                dothisone = True
                thislag = initiallags[vox]
            else:
                dothisone = False
                thislag = None
            if dothisone:
                dummy, \
                volumetotalinc, \
                lagtc[vox, :], \
                lagtimes[vox], \
                lagstrengths[vox], \
                lagsigma[vox], \
                gaussout[vox, :], \
                windowout[vox, :], \
                R2[vox], \
                lagmask[vox], \
                failreason = \
                    _procOneVoxelFitcorrx(vox,
                                         corrout[vox, :],
                                         corrscale,
                                         genlagtc,
                                         initial_fmri_x,
                                         optiondict,
                                         displayplots=displayplots,
                                         initiallag=thislag,
                                         rt_floatset=rt_floatset,
                                         rt_floattype=rt_floattype)
                volumetotal += volumetotalinc
                if (FML_BADAMPLOW | FML_BADAMPNEG) & failreason:
                    ampfails += 1
                if FML_BADSEARCHWINDOW & failreason:
                    windowfails += 1
                if FML_BADWIDTH & failreason:
                    widthfails += 1
                if FML_BADLAG & failreason:
                    lagfails += 1
                if FML_HITEDGE & failreason:
                    edgefails += 1
                if (FML_FITFAIL | FML_INITFAIL) & failreason:
                    fitfails += 1
    print('\nCorrelation fitted in ' + str(volumetotal) + ' voxels')
    print('\tampfails=', ampfails,
          ' lagfails=', lagfails,
          ' windowfails=', windowfails,
          ' widthfail=', widthfails,
          ' edgefail=', edgefails,
          ' fitfail=', fitfails)

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal

#### old style correlation fitting below this point

def onecorrfit(corrfunc,
               corrscale,
               optiondict,
               displayplots=False,
               initiallag=None,
               rt_floatset=np.float64,
               rt_floattype='float64'
               ):
    if initiallag is not None:
        maxguess = initiallag
        useguess = True
        widthlimit = optiondict['despeckle_thresh']
    else:
        maxguess = 0.0
        useguess = False
        widthlimit = optiondict['widthlimit']

    if optiondict['bipolar']:
        if max(corrfunc) < -1.0 * min(corrfunc):
            flipfac = rt_floatset(-1.0)
        else:
            flipfac = rt_floatset(1.0)
    else:
        flipfac = rt_floatset(1.0)

    if not optiondict['fixdelay']:
        if optiondict['findmaxtype'] == 'gauss':
            maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = tide_fit.findmaxlag_gauss(
                corrscale,
                flipfac * corrfunc,
                optiondict['lagmin'], optiondict['lagmax'], widthlimit,
                edgebufferfrac=optiondict['edgebufferfrac'],
                threshval=optiondict['lthreshval'],
                uthreshval=optiondict['uthreshval'],
                debug=optiondict['debug'],
                refine=optiondict['gaussrefine'],
                maxguess=maxguess,
                useguess=useguess,
                fastgauss=optiondict['fastgauss'],
                enforcethresh=optiondict['enforcethresh'],
                zerooutbadfit=optiondict['zerooutbadfit'],
                lagmod=optiondict['lagmod'],
                displayplots=displayplots)
        else:
            maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = tide_fit.findmaxlag_quad(
                corrscale,
                flipfac * corrfunc,
                optiondict['lagmin'], optiondict['lagmax'], widthlimit,
                edgebufferfrac=optiondict['edgebufferfrac'],
                threshval=optiondict['lthreshval'],
                uthreshval=optiondict['uthreshval'],
                debug=optiondict['debug'],
                refine=optiondict['gaussrefine'],
                maxguess=maxguess,
                useguess=useguess,
                fastgauss=optiondict['fastgauss'],
                enforcethresh=optiondict['enforcethresh'],
                zerooutbadfit=optiondict['zerooutbadfit'],
                lagmod=optiondict['lagmod'],
                displayplots=displayplots)
        maxval *= flipfac
    else:
        # do something different
        failreason = np.int16(0)
        maxlag = rt_floatset(optiondict['fixeddelayvalue'])
        maxindex = np.int16(bisect.bisect_left(corrscale, optiondict['fixeddelayvalue']))
        maxval = rt_floatset(flipfac * corrfunc[maxindex])
        maxsigma = rt_floatset(1.0)
        maskval = np.uint16(1)

    return maxindex, maxlag, maxval, maxsigma, maskval, failreason


def _procOneVoxelFitcorr(vox,
                         corrtc,
                         corrscale,
                         genlagtc,
                         initial_fmri_x,
                         optiondict,
                         displayplots,
                         initiallag=None,
                         rt_floatset=np.float64,
                         rt_floattype='float64'
                         ):
    maxindex, maxlag, maxval, maxsigma, maskval, failreason = onecorrfit(corrtc,
                                                                         corrscale,
                                                                         optiondict,
                                                                         displayplots=displayplots,
                                                                         initiallag=initiallag,
                                                                         rt_floatset=rt_floatset,
                                                                         rt_floattype=rt_floattype)

    if maxval > 0.3:
        displayplots = False

    # question - should maxlag be added or subtracted?  As of 10/18, it is subtracted
    #  potential answer - tried adding, results are terrible.
    thelagtc = rt_floatset(genlagtc.yfromx(initial_fmri_x - maxlag))

    # now tuck everything away in the appropriate output array
    volumetotalinc = 0
    if (maskval == 0) and optiondict['zerooutbadfit']:
        thetime = rt_floatset(0.0)
        thestrength = rt_floatset(0.0)
        thesigma = rt_floatset(0.0)
        thegaussout = 0.0 * corrtc
        theR2 = rt_floatset(0.0)
    else:
        volumetotalinc = 1
        thetime = rt_floatset(np.fmod(maxlag, optiondict['lagmod']))
        thestrength = rt_floatset(maxval)
        thesigma = rt_floatset(maxsigma)
        if (not optiondict['fixdelay']) and (maxsigma != 0.0):
            thegaussout = rt_floatset(tide_fit.gauss_eval(corrscale, [maxval, maxlag, maxsigma]))
        else:
            thegaussout = rt_floatset(0.0)
        theR2 = rt_floatset(thestrength * thestrength)

    return vox, volumetotalinc, thelagtc, thetime, thestrength, thesigma, thegaussout, theR2, maskval, failreason


def fitcorr(genlagtc, initial_fmri_x, lagtc, slicesize, corrscale,
            lagmask, lagtimes, lagstrengths, lagsigma,
            corrout, meanval, gaussout, R2, optiondict, initiallags=None,
            rt_floatset=np.float64, rt_floattype='float64'):
    displayplots = False
    inputshape = np.shape(corrout)
    if initiallags is None:
        themask = None
    else:
        themask = np.where(initiallags > -1000000.0, 1, 0)
    volumetotal, ampfails, lagfails, widthfails, edgefails, fitfails = 0, 0, 0, 0, 0, 0
    reportstep = 1000
    zerolagtc = rt_floatset(genlagtc.yfromx(initial_fmri_x))

    if optiondict['nprocs'] > 1:
        # define the consumer function here so it inherits most of the arguments
        def fitcorr_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    if initiallags is None:
                        thislag = None
                    else:
                        thislag = initiallags[val]
                    outQ.put(
                        _procOneVoxelFitcorr(val,
                                             corrout[val, :],
                                             corrscale, genlagtc,
                                             initial_fmri_x,
                                             optiondict,
                                             displayplots,
                                             initiallag=thislag,
                                             rt_floatset=rt_floatset,
                                             rt_floattype=rt_floattype))

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(fitcorr_consumer,
                                                inputshape, themask,
                                                nprocs=optiondict['nprocs'],
                                                showprogressbar=True,
                                                chunksize=optiondict['mp_chunksize'])


        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            volumetotal += voxel[1]
            lagtc[voxel[0], :] = voxel[2]
            lagtimes[voxel[0]] = voxel[3]
            lagstrengths[voxel[0]] = voxel[4]
            lagsigma[voxel[0]] = voxel[5]
            gaussout[voxel[0], :] = voxel[6]
            R2[voxel[0]] = voxel[7]
            lagmask[voxel[0]] = voxel[8]
        data_out = []
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete')
            if themask is None:
                dothisone = True
                thislag = None
            elif themask[vox] > 0:
                dothisone = True
                thislag = initiallags[vox]
            else:
                dothisone = False
            if dothisone:
                dummy, \
                volumetotalinc, \
                lagtc[vox, :], \
                lagtimes[vox], \
                lagstrengths[vox], \
                lagsigma[vox], \
                gaussout[vox, :], \
                R2[vox], \
                lagmask[vox], \
                failreason = \
                    _procOneVoxelFitcorr(vox,
                                         corrout[vox, :],
                                         corrscale,
                                         genlagtc,
                                         initial_fmri_x,
                                         optiondict,
                                         displayplots,
                                         initiallag=thislag,
                                         rt_floatset=rt_floatset,
                                         rt_floattype=rt_floattype)

                volumetotal += volumetotalinc
    print('\nCorrelation fitted in ' + str(volumetotal) + ' voxels')
    print('\tampfails=', ampfails, ' lagfails=', lagfails,
          ' widthfail=', widthfails, ' edgefail=', edgefails,
          ' fitfail=', fitfails)

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal
