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
# $Author: frederic $
# $Date: 2016/07/12 13:50:29 $
# $Id: tide_funcs.py,v 1.4 2016/07/12 13:50:29 frederic Exp $
#
from __future__ import print_function, division

import time

import numpy as np
import scipy as sp
from scipy import fftpack, signal
import pylab as pl
import sys
import bisect

import rapidtide.util as tide_util
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit

# ---------------------------------------- Global constants -------------------------------------------
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------


try:
    from numba import jit

    numbaexists = True
except ImportError:
    numbaexists = False

donotusenumba = False

try:
    import pyfftw

    pyfftwexists = True
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()
except ImportError:
    pyfftwexists = False


def conditionaljit():
    def resdec(f):
        if (not numbaexists) or donotusenumba:
            return f
        return jit(f)

    return resdec


def conditionaljit2():
    def resdec(f):
        if (not numbaexists) or donotusenumba or donotbeaggressive:
            return f
        return jit(f)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- Resampling and time shifting functions -------------------------------------------
'''
class congrid:
    def __init__(self, timeaxis, width, method='gauss', circular=True, upsampleratio=100, doplot=False, debug=False):
        self.upsampleratio = upsampleratio
        self.initstep = timeaxis[1] - timeaxis[0]
        self.initstart = timeaxis[0]
        self.initend = timeaxis[-1]
        self.hiresstep = self.initstep / np.float64(self.upsampleratio)
        if method == 'gauss':
            fullwidth = 2.355 * width
        fullwidthpts = int(np.round(fullwidth / self.hiresstep, 0))
        fullwidthpts += ((fullwidthpts % 2) - 1)
        self.hires_x = np.linspace(-fullwidth / 2.0, fullwidth / 2.0, numpts = fullwidthpts, endpoint=True)
        if method == 'gauss':
            self.hires_y = tide_fit.gauss_eval(self.hires_x, np.array([1.0, 0.0, width])
        if debug:
            print(self.hires_x)
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('congrid convolution function')
            pl.plot(self.hires_x, self.hires_y)
            pl.legend(('input', 'hires'))
            pl.show()

    def gridded(xvals, yvals):
        if len(xvals) != len(yvals):
            print('x and y vectors do not match - aborting')
            return None
        for i in range(len(xvals)):
        outindices = ((newtimeaxis - self.hiresstart) // self.hiresstep).astype(int)
'''

congridyvals = {}


def congrid(xaxis, loc, val, width, kernel='kaiser', debug=False):
    """
    Perform a convolution gridding operation with a Kaiser-Bessel or Gaussian kernel of width 'width'

    Parameters
    ----------
    xaxis: array-like
        The target axis for resampling
    loc: float
        The location, in x-axis units, of the sample to be gridded
    val: float
        The value to be gridded
    width: float
        The width of the gridding kernel
    kernel: {'old', 'gauss', 'kaiser'}, optional
        The type of convolution gridding kernel.  Default is 'kaiser'.
    debug: bool, optional
        When True, output additional information about the gridding process

    Returns
    -------
    vals: array-like
        The input value, convolved with the gridding kernel, projected on to x axis points
    weights: array-like
        The values of convolution kernel, projected on to x axis points (used for normalization)
    indices: array-like
        The indices along the x axis where the vals and weights fall.

    Notes
    -----
    See  IEEE TRANSACTIONS ON MEDICAL IMAGING. VOL. IO.NO. 3, SEPTEMBER 1991

    """
    optsigma = np.array([0.4241, 0.4927, 0.4839, 0.5063, 0.5516, 0.5695, 0.5682, 0.5974])
    optbeta  = np.array([1.9980, 2.3934, 3.3800, 4.2054, 4.9107, 5.7567, 6.6291, 7.4302])
    xstep = xaxis[1] - xaxis[0]
    if loc < xaxis[0] or loc > xaxis[-1]:
        print('loc', loc, 'not in range', xaxis[0], xaxis[-1])

    # choose the smoothing kernel based on the width
    if kernel != 'old':
        if not (1.5 <= width <= 5.0) or (np.fmod(width, 0.5) > 0.0):
            print('congrid: width is', width)
            print('congrid: width must be a half-integral value between 1.5 and 5.0 inclusive')
            sys.exit()
        else:
            kernelindex = int((width - 1.5) // 0.5)

    # find the closest grid point to the target location, calculate relative offsets from this point
    center = tide_util.valtoindex(xaxis, loc, toleft=False)
    offset = np.fmod(np.round((loc - xaxis[center]) / xstep, 3), 1.0)  # will vary from 0.0 to 1.0
    if not (0.0 <= offset < 1.0):
        print('(loc, xstep, center, offset):', loc, xstep, center, offset)
        print('xaxis:', xaxis)
        sys.exit()
    offsetkey = str(offset)

    if kernel == 'old':
        widthinpts = int(np.round(width * 4.6 / xstep))
        widthinpts -= widthinpts % 2 - 1
        try:
            yvals = congridyvals[offsetkey]
        except KeyError:
            if debug:
                print('new key:', offsetkey)
            xvals = np.linspace(-xstep * (widthinpts // 2), xstep * (widthinpts // 2), num=widthinpts,
                                endpoint=True) + offset
            congridyvals[offsetkey] = tide_fit.gauss_eval(xvals, np.array([1.0, 0.0, width]))
            yvals = congridyvals[offsetkey]
        startpt = int(center - widthinpts // 2)
        indices = range(startpt, startpt + widthinpts)
        indices = np.remainder(indices, len(xaxis))
        return val * yvals, yvals, indices
    else:
        offsetinpts = center + offset
        startpt = int(np.ceil(offsetinpts - width / 2.0))
        endpt = int(np.floor(offsetinpts + width / 2.0))
        indices = np.remainder(range(startpt, endpt + 1), len(xaxis))
        try:
            yvals = congridyvals[offsetkey]
        except KeyError:
            if debug:
                print('new key:', offsetkey)
            xvals = xaxis[indices] + offset * xstep
            if kernel == 'gauss':
                sigma = optsigma[kernelindex]
                congridyvals[offsetkey] = tide_fit.gauss_eval(xvals, np.array([1.0, 0.0, sigma]))
            elif kernel == 'kaiser':
                beta = optbeta[kernelindex]
                congridyvals[offsetkey] = tide_fit.kaiserbessel_eval(xvals, np.array([beta, width / 2.0]))
            else:
                print('illegal kernel value in congrid - exiting')
                sys.exit()
            yvals = congridyvals[offsetkey]
        return val * yvals, yvals, indices


class fastresampler:
    def __init__(self, timeaxis, timecourse, padvalue=30.0, upsampleratio=100, doplot=False, debug=False,
                 method='univariate'):
        self.upsampleratio = upsampleratio
        self.padvalue = padvalue
        self.initstep = timeaxis[1] - timeaxis[0]
        self.initstart = timeaxis[0]
        self.initend = timeaxis[-1]
        self.hiresstep = self.initstep / np.float64(self.upsampleratio)
        self.hires_x = np.arange(timeaxis[0] - self.padvalue, self.initstep * len(timeaxis) + self.padvalue,
                                 self.hiresstep)
        self.hiresstart = self.hires_x[0]
        self.hiresend = self.hires_x[-1]
        if method == 'poly':
            self.hires_y = 0.0 * self.hires_x
            self.hires_y[int(self.padvalue // self.hiresstep) + 1:-(int(self.padvalue // self.hiresstep) + 1)] = \
                signal.resample_poly(timecourse, np.int(self.upsampleratio * 10), 10)
        elif method == 'fourier':
            self.hires_y = 0.0 * self.hires_x
            self.hires_y[int(self.padvalue // self.hiresstep) + 1:-(int(self.padvalue // self.hiresstep) + 1)] = \
                signal.resample(timecourse, self.upsampleratio * len(timeaxis))
        else:
            self.hires_y = doresample(timeaxis, timecourse, self.hires_x, method=method)
        self.hires_y[:int(self.padvalue // self.hiresstep)] = self.hires_y[int(self.padvalue // self.hiresstep)]
        self.hires_y[-int(self.padvalue // self.hiresstep):] = self.hires_y[-int(self.padvalue // self.hiresstep)]
        if debug:
            print('fastresampler __init__:')
            print('    padvalue:, ', self.padvalue)
            print('    initstep, hiresstep:', self.initstep, self.hiresstep)
            print('    initial axis limits:', self.initstart, self.initend)
            print('    hires axis limits:', self.hiresstart, self.hiresend)

        # self.hires_y[:int(self.padvalue // self.hiresstep)] = 0.0
        # self.hires_y[-int(self.padvalue // self.hiresstep):] = 0.0
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('fastresampler initial timecourses')
            pl.plot(timeaxis, timecourse, self.hires_x, self.hires_y)
            pl.legend(('input', 'hires'))
            pl.show()

    def yfromx(self, newtimeaxis, doplot=False, debug=False):
        if debug:
            print('fastresampler: yfromx called with following parameters')
            print('    padvalue:, ', self.padvalue)
            print('    initstep, hiresstep:', self.initstep, self.hiresstep)
            print('    initial axis limits:', self.initstart, self.initend)
            print('    hires axis limits:', self.hiresstart, self.hiresend)
            print('    requested axis limits:', newtimeaxis[0], newtimeaxis[-1])
        outindices = ((newtimeaxis - self.hiresstart) // self.hiresstep).astype(int)
        if debug:
            print('len(self.hires_y):', len(self.hires_y))
        try:
            out_y = self.hires_y[outindices]
        except IndexError:
            print('')
            print('indexing out of bounds in fastresampler')
            print('    padvalue:, ', self.padvalue)
            print('    initstep, hiresstep:', self.initstep, self.hiresstep)
            print('    initial axis limits:', self.initstart, self.initend)
            print('    hires axis limits:', self.hiresstart, self.hiresend)
            print('    requested axis limits:', newtimeaxis[0], newtimeaxis[-1])
            sys.exit()
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('fastresampler timecourses')
            pl.plot(self.hires_x, self.hires_y, newtimeaxis, out_y)
            pl.legend(('hires', 'output'))
            pl.show()
        return out_y


def doresample(orig_x, orig_y, new_x, method='cubic', padlen=0):
    """

    Parameters
    ----------
    orig_x
    orig_y
    new_x
    method
    padlen

    Returns
    -------

    """
    pad_y = tide_filt.padvec(orig_y, padlen=padlen)
    tstep = orig_x[1] - orig_x[0]
    if padlen > 0:
        pad_x = np.concatenate((np.arange(orig_x[0] - padlen * tstep, orig_x[0], tstep),
                                orig_x,
                                np.arange(orig_x[-1] + tstep, orig_x[-1] + tstep * (padlen + 1), tstep)))
    else:
        pad_x = orig_x
    if padlen > 0:
        print('padlen=', padlen)
        print('tstep=', tstep)
        print(pad_x)
    if method == 'cubic':
        cj = signal.cspline1d(pad_y)
        return tide_filt.unpadvec(
            np.float64(signal.cspline1d_eval(cj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])), padlen=padlen)
    elif method == 'quadratic':
        qj = signal.qspline1d(pad_y)
        return tide_filt.unpadvec(
            np.float64(signal.qspline1d_eval(qj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])), padlen=padlen)
    elif method == 'univariate':
        interpolator = sp.interpolate.UnivariateSpline(pad_x, pad_y, k=3, s=0)  # s=0 interpolates
        return tide_filt.unpadvec(np.float64(interpolator(new_x)), padlen=padlen)
    else:
        print('invalid interpolation method')
        return None


def arbresample(inputdata, init_freq, final_freq, intermed_freq=0.0, method='univariate', debug=False, decimate=False):
    """

    Parameters
    ----------
    inputdata
    init_freq
    final_freq
    intermed_freq
    method
    debug

    Returns
    -------

    """
    if decimate:
        if final_freq > init_freq:
            # upsample only
            return upsample(inputdata, init_freq, final_freq, method=method, debug=debug)
        elif final_freq < init_freq:
            # downsampling, so upsample by an amount that allows integer decimation
            intermed_freq = final_freq * np.ceil(init_freq / final_freq)
            q = int(intermed_freq // final_freq)
            if debug:
               print('going from', init_freq, 'to', final_freq, ': upsampling to', intermed_freq, 'Hz, then decimating by,', q)
            if intermed_freq == init_freq:
                upsampled = inputdata
            else:
                upsampled = upsample(inputdata, init_freq, intermed_freq, method=method, debug=debug)
            return signal.decimate(upsampled, q)
        else:
            return inputdata
    else:
        if intermed_freq <= 0.0:
            intermed_freq = np.max([2.0 * init_freq, 2.0 * final_freq])
        orig_x = sp.linspace(0.0, 1.0 / init_freq * len(inputdata), num=len(inputdata), endpoint=False)
        if debug:
            print('arbresample:', len(orig_x), len(inputdata), init_freq, final_freq, intermed_freq)
        return dotwostepresample(orig_x, inputdata, intermed_freq, final_freq, method=method, debug=debug)


def upsample(inputdata, Fs_init, Fs_higher, method='univariate', debug=False):
    starttime = time.time()
    if Fs_higher <= Fs_init:
        print('upsample: target frequency must be higher than initial frequency')
        sys.exit()

    # upsample
    orig_x = sp.linspace(0.0, 1.0 / Fs_init * len(inputdata), num=len(inputdata), endpoint=False)
    endpoint = orig_x[-1] - orig_x[0]
    ts_higher = 1.0 / Fs_higher
    numresamppts = int(endpoint // ts_higher + 1)
    upsampled_x = np.arange(0.0, ts_higher * numresamppts, ts_higher)
    upsampled_y = doresample(orig_x, inputdata, upsampled_x, method=method)
    if debug:
        print('upsampling took', time.time() - starttime, 'seconds')
    return upsampled_y


def dotwostepresample(orig_x, orig_y, intermed_freq, final_freq, method='univariate', debug=False):
    """

    Parameters
    ----------
    orig_x
    orig_y
    intermed_freq
    final_freq
    method
    debug

    Returns
    -------
    resampled_y

    """
    if intermed_freq <= final_freq:
        print('intermediate frequency must be higher than final frequency')
        sys.exit()

    # upsample
    starttime = time.time()
    endpoint = orig_x[-1] - orig_x[0]
    init_freq = len(orig_x) / endpoint
    intermed_ts = 1.0 / intermed_freq
    numresamppts = int(endpoint // intermed_ts + 1)
    intermed_x = np.arange(0.0, intermed_ts * numresamppts, intermed_ts)
    intermed_y = doresample(orig_x, orig_y, intermed_x, method=method)
    if debug:
        print('upsampling took', time.time() - starttime, 'seconds')

    # antialias and ringstop filter
    starttime = time.time()
    aafilterfreq = np.min([final_freq, init_freq]) / 2.0
    aafilter = tide_filt.noncausalfilter(filtertype='arb', usebutterworth=False, debug=debug)
    aafilter.setarb(0.0, 0.0, 0.95 * aafilterfreq, aafilterfreq)
    antialias_y = aafilter.apply(intermed_freq, intermed_y)
    if debug:
        print('antialiasing took', time.time() - starttime, 'seconds')

    # downsample
    starttime = time.time()
    final_ts = 1.0 / final_freq
    numresamppts = np.ceil(endpoint / final_ts) + 1
    final_x = np.arange(0.0, final_ts * numresamppts, final_ts)
    resampled_y = doresample(intermed_x, antialias_y, final_x, method=method)
    if debug:
        print('downsampling took', time.time() - starttime, 'seconds')
    return resampled_y


def calcsliceoffset(sotype, slicenum, numslices, tr, multiband=1):
    """

    Parameters
    ----------
    sotype
    slicenum
    numslices
    tr
    multiband

    Returns
    -------

    """
    # Slice timing correction
    # 0 : None
    # 1 : Regular up (0, 1, 2, 3, ...)
    # 2 : Regular down
    # 3 : Use slice order file
    # 4 : Use slice timings file
    # 5 : Standard Interleaved (0, 2, 4 ... 1, 3, 5 ... )
    # 6 : Siemens Interleaved (0, 2, 4 ... 1, 3, 5 ... for odd number of slices)
    # (1, 3, 5 ... 0, 2, 4 ... for even number of slices)
    # 7 : Siemens Multiband Interleaved

    # default value of zero
    slicetime = 0.0

    # None
    if sotype == 0:
        slicetime = 0.0

    # Regular up
    if type == 1:
        slicetime = slicenum * (tr / numslices)

    # Regular down
    if sotype == 2:
        slicetime = (numslices - slicenum - 1) * (tr / numslices)

    # Slice order file not supported - do nothing
    if sotype == 3:
        slicetime = 0.0

    # Slice timing file not supported - do nothing
    if sotype == 4:
        slicetime = 0.0

    # Standard interleave
    if sotype == 5:
        if (slicenum % 2) == 0:
            # even slice number
            slicetime = (tr / numslices) * (slicenum / 2)
        else:
            # odd slice number
            slicetime = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)

    # Siemens interleave format
    if sotype == 6:
        if (numslices % 2) == 0:
            # even number of slices - slices go 1,3,5,...,0,2,4,...
            if (slicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numslices) * (numslices / 2 + slicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numslices) * ((slicenum - 1) / 2)
        else:
            # odd number of slices - slices go 0,2,4,...,1,3,5,...
            if (slicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numslices) * (slicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)

    # Siemens multiband interleave format
    if sotype == 7:
        numberofshots = numslices / multiband
        modslicenum = slicenum % numberofshots
        if (numberofshots % 2) == 0:
            # even number of shots - slices go 1,3,5,...,0,2,4,...
            if (modslicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numberofshots) * (numberofshots / 2 + modslicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numberofshots) * ((modslicenum - 1) / 2)
        else:
            # odd number of slices - slices go 0,2,4,...,1,3,5,...
            if (modslicenum % 2) == 0:
                # even slice number
                slicetime = (tr / numberofshots) * (modslicenum / 2)
            else:
                # odd slice number
                slicetime = (tr / numberofshots) * ((numberofshots + 1) / 2 + (modslicenum - 1) / 2)
    return slicetime


# NB: a positive value of shifttrs delays the signal, a negative value advances it
# timeshift using fourier phase multiplication
def timeshift(inputtc, shifttrs, padtrs, doplot=False):
    """

    Parameters
    ----------
    inputtc
    shifttrs
    padtrs
    doplot

    Returns
    -------

    """
    # set up useful parameters
    thelen = np.shape(inputtc)[0]
    thepaddedlen = thelen + 2 * padtrs
    imag = 1.j

    # initialize variables
    preshifted_y = np.zeros(thepaddedlen, dtype='float')  # initialize the working buffer (with pad)
    weights = np.zeros(thepaddedlen, dtype='float')  # initialize the weight buffer (with pad)

    # now do the math
    preshifted_y[padtrs:padtrs + thelen] = inputtc[:]  # copy initial data into shift buffer
    weights[padtrs:padtrs + thelen] = 1.0  # put in the weight vector
    revtc = inputtc[::-1]  # reflect data around ends to
    preshifted_y[0:padtrs] = revtc[-padtrs:]  # eliminate discontinuities
    preshifted_y[padtrs + thelen:] = revtc[0:padtrs]

    # finish initializations
    fftlen = np.shape(preshifted_y)[0]

    # create the phase modulation timecourse
    initargvec = (np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(fftlen)) - np.pi)
    if len(initargvec) > fftlen:
        initargvec = initargvec[:fftlen]
    argvec = np.roll(initargvec * shifttrs, -int(fftlen // 2))
    modvec = np.cos(argvec) - imag * np.sin(argvec)

    # process the data (fft->modulate->ifft->filter)
    fftdata = fftpack.fft(preshifted_y)  # do the actual shifting
    shifted_y = fftpack.ifft(modvec * fftdata).real

    # process the weights
    w_fftdata = fftpack.fft(weights)  # do the actual shifting
    shifted_weights = fftpack.ifft(modvec * w_fftdata).real

    if doplot:
        xvec = range(0, thepaddedlen)  # make a ramp vector (with pad)
        print("shifttrs:", shifttrs)
        print("offset:", padtrs)
        print("thelen:", thelen)
        print("thepaddedlen:", thepaddedlen)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial vector')
        pl.plot(xvec, preshifted_y)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Initial and shifted vector')
        pl.plot(xvec, preshifted_y, xvec, shifted_y)

        pl.show()

    return ([shifted_y[padtrs:padtrs + thelen], shifted_weights[padtrs:padtrs + thelen], shifted_y,
             shifted_weights])
