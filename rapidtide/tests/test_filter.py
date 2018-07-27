#!/usr/bin/env python
from __future__ import print_function, division

from rapidtide.utils import valtoindex
from rapidtide.filter import noncausalfilter
import numpy as np
import scipy as sp
import pylab as plt

def spectralfilterprops(thefilter, debug=False):
    lowerstop, lowerpass, upperpass, upperstop = thefilter['filter'].getfreqlimits()
    lowerstopindex = valtoindex(thefilter['frequencies'], lowerstop)
    lowerpassindex = valtoindex(thefilter['frequencies'], lowerpass, toleft=False)
    upperpassindex = valtoindex(thefilter['frequencies'], upperpass)
    upperstopindex = np.min([valtoindex(thefilter['frequencies'], upperstop, toleft=False), len(thefilter['frequencies']) - 1])
    if debug:
        print('target freqs:', lowerstop, lowerpass, upperpass, upperstop)
        print('actual freqs:', thefilter['frequencies'][lowerstopindex],
                            thefilter['frequencies'][lowerpassindex],
                            thefilter['frequencies'][upperpassindex],
                            thefilter['frequencies'][upperstopindex])
    response = {}

    passbandmean = np.mean(thefilter['transferfunc'][lowerpassindex:upperpassindex])
    passbandmax = np.max(thefilter['transferfunc'][lowerpassindex:upperpassindex])
    passbandmin = np.min(thefilter['transferfunc'][lowerpassindex:upperpassindex])

    response['passbandripple'] = (passbandmax - passbandmin)/passbandmean

    if lowerstopindex > 2:
        response['lowerstopmean'] = np.mean(thefilter['transferfunc'][0:lowerstopindex])/passbandmean
        response['lowerstopmax'] = np.max(np.abs(thefilter['transferfunc'][0:lowerstopindex]))/passbandmean
    else:
        response['lowerstopmean'] = 0.0
        response['lowerstopmax'] = 0.0

    if len(thefilter['transferfunc']) - upperstopindex > 2:
        response['upperstopmean'] = np.mean(thefilter['transferfunc'][upperstopindex:-1])/passbandmean
        response['upperstopmax'] = np.max(np.abs(thefilter['transferfunc'][upperstopindex:-1]))/passbandmean
    else:
        response['upperstopmean'] = 0.0
        response['upperstopmax'] = 0.0
    return response


def eval_filterprops(sampletime=0.72, tclengthinsecs=300.0, numruns=100, display=False):
    tclen = int(tclengthinsecs // sampletime)
    print('Testing transfer function:')
    lowestfreq = 1.0/(sampletime * tclen)
    nyquist = 0.5/sampletime
    print('    sampletime=',sampletime,', timecourse length=',tclengthinsecs, 's,  possible frequency range:',lowestfreq, nyquist)
    timeaxis = np.arange(0.0, 1.0 * tclen) * sampletime

    overall = np.random.normal(size=tclen)
    nperseg = np.min([tclen, 256])
    f, dummy = sp.signal.welch(overall, fs=1.0/sampletime, nperseg=nperseg)

    allfilters = []

    # construct all the filters
    for filtertype in ['lfo', 'resp', 'cardiac']:
        testfilter = noncausalfilter(filtertype=filtertype)
        lstest, lptest, uptest, ustest = testfilter.getfreqlimits()
        if lptest < nyquist:
            allfilters.append(
                {
                    'name': filtertype + ' brickwall',
                    'filter': noncausalfilter(filtertype=filtertype),
                })
            allfilters.append(
                {
                    'name': filtertype + ' trapezoidal',
                    'filter': noncausalfilter(filtertype=filtertype, usetrapfftfilt=True),
                })

    # calculate the transfer functions for the filters
    for index in range(0, len(allfilters)):
        psd_raw = 0.0 * dummy
        psd_filt = 0.0 * dummy
        for i in range(0,numruns):
            inputsig = np.random.normal(size=tclen)
            outputsig = allfilters[index]['filter'].apply(1.0/sampletime, inputsig)
            f, raw = sp.signal.welch(inputsig, fs=1.0/sampletime, nperseg=nperseg)
            f, filt = sp.signal.welch(outputsig, fs=1.0/sampletime, nperseg=nperseg)
            psd_raw += raw
            psd_filt += filt
        allfilters[index]['frequencies'] = f
        allfilters[index]['transferfunc'] = psd_filt / psd_raw

    # show transfer functions
    if display:
        legend = []
        plt.figure()
        plt.hold(True)
        plt.ylim([-1.1, 1.1 * len(allfilters)])
        offset = 0.0
        for thefilter in allfilters:
            plt.plot(thefilter['frequencies'], thefilter['transferfunc'] + offset)
            legend.append(thefilter['name'])
            offset += 1.1
        plt.legend(legend)
        plt.show()

    # test transfer function responses
    for thefilter in allfilters:
        response = spectralfilterprops(thefilter)
        print('    Evaluating', thefilter['name'], 'transfer function')
        assert response['passbandripple'] < 0.4
        assert response['lowerstopmax'] < 1e4
        assert response['lowerstopmean'] < 1e4
        assert response['upperstopmax'] < 1e4
        assert response['upperstopmean'] < 1e4

    # construct some test waveforms for end effects
    testwaves = []
    testwaves.append({
        'name':        'constant high',
        'timeaxis':    1.0 * timeaxis,
        'waveform':    np.ones((tclen), dtype='float'),
        })
    testwaves.append({
        'name':        'white noise',
        'timeaxis':    1.0 * timeaxis,
        'waveform':    0.3 * np.random.normal(size=tclen),
        })
    
    scratch = timeaxis * 0.0
    scratch[int(tclen / 5):int(2 * tclen / 5)] = 1.0
    scratch[int(3 * tclen / 5):int(4 * tclen / 5)] = 1.0
    testwaves.append({
        'name':        'block regressor',
        'timeaxis':    1.0 * timeaxis,
        'waveform':    1.0 * scratch,
        })

    # show the end effects waveforms
    if display:
        legend = []
        plt.figure()
        plt.hold(True)
        plt.ylim([-2.2, 2.2 * len(testwaves)])
        offset = 0.0
        for thewave in testwaves:
            for thefilter in allfilters:
                plt.plot(thewave['timeaxis'], offset + thefilter['filter'].apply(1.0/sampletime, thewave['waveform']))
                legend.append(thewave['name'] + ': '+ thefilter['name'])
                offset += 1.1
            #plt.plot(thewave['timeaxis'], thewave['waveform'] + offset)
            #legend.append(thewave['name'])
            #offset += 2.2
            plt.legend(legend)
            plt.show()


def testfilterprops(display=False):
    eval_filterprops(sampletime=0.72, tclengthinsecs=300.0, numruns=100, display=display)
    eval_filterprops(sampletime=2.0,  tclengthinsecs=300.0, numruns=100, display=display)
    eval_filterprops(sampletime=0.1,  tclengthinsecs=1000.0, numruns=10, display=display)

def main():
    testfilterprops(display=True)

if __name__ == '__main__':
    main()
