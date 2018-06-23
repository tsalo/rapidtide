"""
Spectral analysis functions
"""
import numpy as np
from scipy import fftpack


def phase(mcv):
    return np.arctan2(mcv.imag, mcv.real)


def polarfft(invec, samplerate):
    if np.shape(invec)[0] % 2 == 1:
        thevec = invec[:-1]
    else:
        thevec = invec
    spec = fftpack.fft(hamming(np.shape(thevec)[0]) * thevec)[0:np.shape(thevec)[0] // 2]
    magspec = abs(spec)
    phspec = phase(spec)
    maxfreq = samplerate / 2.0
    freqs = np.arange(0.0, maxfreq, maxfreq / (np.shape(spec)[0]))
    return freqs, magspec, phspec


def _unwrap(phase):
    samples = phase.shape[-1]
    unwrapped = np.unwrap(phase)
    center = (samples + 1) // 2
    if samples == 1:
        center = 0
    ndelay = np.array(np.round(unwrapped[...,center]/np.pi))
    unwrapped -= np.pi * ndelay[...,None] * np.arange(samples) / center
    return unwrapped, ndelay


def complex_cepstrum(x):
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    spectrum = fftpack.fft(x)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = fftpack.ifft(log_spectrum).real

    return ceps, ndelay


def real_cepstrum(x):
    # adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
    return fftpack.ifft(np.log(np.abs(fftpack.fft(x)))).real
