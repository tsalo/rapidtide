import numpy as np
import multiprocessing as mp

import rapidtide.stats as tide_stats
import rapidtide.miscmath as tide_math


def numpy2shared(inarray, thetype):
    thesize = inarray.size
    theshape = inarray.shape
    if thetype == np.float64:
        inarray_shared = mp.RawArray('d', inarray.reshape(thesize))
    else:
        inarray_shared = mp.RawArray('f', inarray.reshape(thesize))
    inarray = np.frombuffer(inarray_shared, dtype=thetype, count=thesize)
    inarray.shape = theshape
    return inarray, inarray_shared, theshape


def getglobalsignal(indata, optiondict, includemask=None, excludemask=None,
                    rt_floatset=np.float32):
    # mask to interesting voxels
    if optiondict['globalmaskmethod'] == 'mean':
        themask = tide_stats.makemask(np.mean(indata, axis=1), optiondict['corrmaskthreshpct'])
    elif optiondict['globalmaskmethod'] == 'variance':
        themask = tide_stats.makemask(np.var(indata, axis=1), optiondict['corrmaskthreshpct'])
    if optiondict['nothresh']:
        themask *= 0
        themask += 1
    if includemask is not None:
        themask = themask * includemask
    if excludemask is not None:
        themask = themask * excludemask

    # add up all the voxels
    globalmean = rt_floatset(indata[0, :])
    thesize = np.shape(themask)
    numvoxelsused = 0
    for vox in range(0, thesize[0]):
        if themask[vox] > 0.0:
            numvoxelsused += 1
            if optiondict['meanscaleglobal']:
                themean = np.mean(indata[vox, :])
                if themean != 0.0:
                    globalmean = globalmean + indata[vox, :] / themean - 1.0
            else:
                globalmean = globalmean + indata[vox, :]
    print()
    print('used ', numvoxelsused, ' voxels to calculate global mean signal')
    return tide_math.stdnormalize(globalmean)


def allocshared(theshape, thetype):
    thesize = int(1)
    for element in theshape:
        thesize *= int(element)
    if thetype == np.float64:
        outarray_shared = mp.RawArray('d', thesize)
    else:
        outarray_shared = mp.RawArray('f', thesize)
    outarray = np.frombuffer(outarray_shared, dtype=thetype, count=thesize)
    outarray.shape = theshape
    return outarray, outarray_shared, theshape
