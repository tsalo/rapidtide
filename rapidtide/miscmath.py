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

import numpy as np
from scipy import fftpack

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False

try:
    from numba import jit

    numbaexists = True
except ImportError:
    numbaexists = False

try:
    import nibabel as nib

    nibabelexists = True
except ImportError:
    nibabelexists = False

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


# --------------------------- Spectral analysis functions ---------------------------------------
def phase(mcv):
    r"""Return phase of complex numbers.

    Parameters
    ----------
    mcv : complex array
        A complex vector

    Returns
    -------
    phase : float array
        The phase angle of the numbers, in radians

    """
    return np.arctan2(mcv.imag, mcv.real)


def polarfft(invec, samplerate):
    """

    Parameters
    ----------
    invec
    samplerate

    Returns
    -------

    """
    if np.shape(invec)[0] % 2 == 1:
        thevec = invec[:-1]
    else:
        thevec = invec
    spec = fftpack.fft(tide_filt.hamming(np.shape(thevec)[0]) * thevec)[0:np.shape(thevec)[0] // 2]
    magspec = abs(spec)
    phspec = phase(spec)
    maxfreq = samplerate / 2.0
    freqs = np.arange(0.0, maxfreq, maxfreq / (np.shape(spec)[0]))
    return freqs, magspec, phspec


def complex_cepstrum(x):
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
        unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
        return unwrapped, ndelay

    spectrum = fftpack.fft(x)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = fftpack.ifft(log_spectrum).real

    return ceps, ndelay


def real_cepstrum(x):
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    return fftpack.ifft(np.log(np.abs(fftpack.fft(x)))).real


# --------------------------- miscellaneous math functions -------------------------------------------------
def thederiv(y):
    """

    Parameters
    ----------
    y

    Returns
    -------

    """
    dyc = [0.0] * len(y)
    dyc[0] = (y[0] - y[1]) / 2.0
    for i in range(1, len(y) - 1):
        dyc[i] = (y[i + 1] - y[i - 1]) / 2.0
    dyc[-1] = (y[-1] - y[-2]) / 2.0
    return dyc


def primes(n):
    """

    Parameters
    ----------
    n

    Returns
    -------

    """
    # found on stackoverflow: https://stackoverflow.com/questions/16996217/prime-factorization-list
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def largestfac(n):
    """

    Parameters
    ----------
    n

    Returns
    -------

    """
    return primes(n)[-1]


# --------------------------- Normalization functions -------------------------------------------------
def znormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    return stdnormalize(vector)


@conditionaljit()
def stdnormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    demeaned = vector - np.mean(vector)
    sigstd = np.std(demeaned)
    if sigstd > 0.0:
        return demeaned / sigstd
    else:
        return demeaned


def varnormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    demeaned = vector - np.mean(vector)
    sigvar = np.var(demeaned)
    if sigvar > 0.0:
        return demeaned / sigvar
    else:
        return demeaned


def pcnormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    sigmean = np.mean(vector)
    if sigmean > 0.0:
        return vector / sigmean - 1.0
    else:
        return vector


def ppnormalize(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    demeaned = vector - np.mean(vector)
    sigpp = np.max(demeaned) - np.min(demeaned)
    if sigpp > 0.0:
        return demeaned / sigpp
    else:
        return demeaned


@conditionaljit()
def corrnormalize(thedata, dodetrend, windowfunc='hamming'):
    """

    Parameters
    ----------
    thedata
    dodetrend : :obj:`bool`
        Whether to detrend timeseries or not.
    windowfunc : {'hamming', None}, optional
        Window function to use. If None, no windowing is done.

    Returns
    -------

    """
    # detrend first
    if dodetrend:
        intervec = stdnormalize(tide_fit.detrend(thedata, demean=True))
    else:
        intervec = stdnormalize(thedata)

    # then window
    if windowfunc:
        return stdnormalize(tide_filt.windowfunction(
            np.shape(thedata)[0], type=windowfunc) * intervec) / np.sqrt(np.shape(thedata)[0])
    else:
        return stdnormalize(intervec) / np.sqrt(np.shape(thedata)[0])


def rms(vector):
    """

    Parameters
    ----------
    vector

    Returns
    -------

    """
    return np.sqrt(np.mean(np.square(vector)))


def envdetect(Fs, inputdata, cutoff=0.25):
    """

    Parameters
    ----------
    Fs : float
        Sample frequency in Hz.
    inputdata : float array
        Data to be envelope detected
    cutoff : float
        Highest possible modulation frequency

    Returns
    -------
    envelope : float array
        The envelope function

    """
    demeaned = inputdata - np.mean(inputdata)
    sigabs = abs(demeaned)
    theenvbpf = tide_filt.noncausalfilter(filtertype='arb')
    theenvbpf.setarb(0.0, 0.0, cutoff, 1.1 * cutoff)
    return theenvbpf.apply(Fs, sigabs)
