"""
Non-NIFTI file I/O functions
"""


def checkifparfile(filename):
    if filename.endswith(".par"):
        return True
    else:
        return False


def readvecs(inputfilename):
    thefile = open(inputfilename, 'r')
    lines = thefile.readlines()
    numvecs = len(lines[0].split())
    inputvec = np.zeros((numvecs, MAXLINES), dtype='float64')
    numvals = 0
    for line in lines:
        numvals += 1
        thetokens = line.split()
        for vecnum in range(0, numvecs):
            inputvec[vecnum, numvals - 1] = np.float64(thetokens[vecnum])
    return 1.0 * inputvec[:, 0:numvals]


def readvec(inputfilename):
    inputvec = np.zeros(MAXLINES, dtype='float64')
    numvals = 0
    with open(inputfilename, 'r') as thefile:
        lines = thefile.readlines()
        for line in lines:
            numvals += 1
            inputvec[numvals - 1] = np.float64(line)
    return 1.0 * inputvec[0:numvals]


def readlabels(inputfilename):
    inputvec = []
    with open(inputfilename, 'r') as thefile:
        lines = thefile.readlines()
        for line in lines:
            inputvec.append(line.rstrip())
    return inputvec


def writedict(thedict, outputfile, lineend=''):
    if lineend == 'mac':
        thelineending = '\r'
        openmode = 'wb'
    elif lineend == 'win':
        thelineending = '\r\n'
        openmode = 'wb'
    elif lineend == 'linux':
        thelineending = '\n'
        openmode = 'wb'
    else:
        thelineending = '\n'
        openmode = 'w'
    with open(outputfile, openmode) as FILE:
        for key, value in sorted(thedict.items()):
            FILE.writelines(str(key) + ':\t' + str(value) + thelineending)


def writevec(thevec, outputfile, lineend=''):
    if lineend == 'mac':
        thelineending = '\r'
        openmode = 'wb'
    elif lineend == 'win':
        thelineending = '\r\n'
        openmode = 'wb'
    elif lineend == 'linux':
        thelineending = '\n'
        openmode = 'wb'
    else:
        thelineending = '\n'
        openmode = 'w'
    with open(outputfile, openmode) as FILE:
        for i in thevec:
            FILE.writelines(str(i) + thelineending)


# rewritten to guarantee file closure, combines writenpvec and writenpvecs
def writenpvecs(thevecs, outputfile, lineend=''):
    theshape = np.shape(thevecs)
    if lineend == 'mac':
        thelineending = '\r'
        openmode = 'wb'
    elif lineend == 'win':
        thelineending = '\r\n'
        openmode = 'wb'
    elif lineend == 'linux':
        thelineending = '\n'
        openmode = 'wb'
    else:
        thelineending = '\n'
        openmode = 'w'
    with open(outputfile, openmode) as FILE:
        if thevecs.ndim == 2:
            for i in range(0, theshape[1]):
                for j in range(0, theshape[0]):
                    FILE.writelines(str(thevecs[j, i]) + '\t')
                FILE.writelines(thelineending)
        else:
            for i in range(0, theshape[0]):
                FILE.writelines(str(thevecs[i]) + thelineending)
