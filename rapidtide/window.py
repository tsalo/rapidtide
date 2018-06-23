"""
Window functions
"""

BHwindows = {}
def blackmanharris(length):
    #return a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)
    try:
        return BHwindows[str(length)]
    except:
        argvec = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(length))
        a0 = 0.35875
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168
        BHwindows[str(length)] = a0 - a1 * np.cos(argvec) + a2 * np.cos(2.0 * argvec) - a3 * np.cos(3.0 * argvec)
        print('initialized Blackman-Harris window for length', length)
        return BHwindows[str(length)]

hannwindows = {}
def hann(length):
    #return 0.5 * (1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi))
    try:
        return hannwindows[str(length)]
    except:
        hannwindows[str(length)] = 0.5 * (1.0 - np.cos(np.arange(0.0, 1.0, 1.0 / float(length)) * 2.0 * np.pi))
        print('initialized hann window for length', length)
        return hannwindows[str(length)]


hammingwindows = {}
def hamming(length):
#   return 0.54 - 0.46 * np.cos((np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi)
    try:
        return hammingwindows[str(length)]
    except:
        hammingwindows[str(length)] = 0.54 - 0.46 * np.cos((np.arange(0.0, float(length), 1.0) / float(length)) * 2.0 * np.pi)
        print('initialized hamming window for length', length)
        return hammingwindows[str(length)]

def windowfunction(length, type='hamming'):
    if type == 'hamming':
        return hamming(length)
    elif type == 'hann':
        return hann(length)
    elif type == 'blackmanharris':
        return blackmanharris(length)
    elif type == 'None':
        return np.ones(length)
    else:
        print('illegal window function')
        sys.exit()


def envdetect(vector, filtwidth=3.0):
    demeaned = vector - np.mean(vector)
    sigabs = abs(demeaned)
    return dolptrapfftfilt(1.0, 1.0 / (2.0 * filtwidth), 1.1 / (2.0 * filtwidth), sigabs)
