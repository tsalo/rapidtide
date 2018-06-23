"""
Functions for reading in and writing out nifti files.
"""
import os
import sys

import nibabel as nib



def readfromnifti(inputfile):
    if os.path.isfile(inputfile):
        inputfilename = inputfile
    elif os.path.isfile(inputfile + '.nii.gz'):
        inputfilename = inputfile + '.nii.gz'
    elif os.path.isfile(inputfile + '.nii'):
        inputfilename = inputfile + '.nii'
    else:
        print('nifti file', inputfile, 'does not exist')
        sys.exit()
    nim = nib.load(inputfilename)
    nim_data = nim.get_data()
    nim_hdr = nim.get_header()
    thedims = nim_hdr['dim'].copy()
    thesizes = nim_hdr['pixdim'].copy()
    return nim, nim_data, nim_hdr, thedims, thesizes


# dims are the array dimensions along each axis
def parseniftidims(thedims):
    return thedims[1], thedims[2], thedims[3], thedims[4]


# sizes are the mapping between voxels and physical coordinates
def parseniftisizes(thesizes):
    return thesizes[1], thesizes[2], thesizes[3], thesizes[4]


def savetonifti(thearray, theheader, thepixdim, thename):
    outputaffine = theheader.get_best_affine()
    qaffine, qcode = theheader.get_qform(coded=True)
    saffine, scode = theheader.get_sform(coded=True)
    if theheader['magic'] == 'n+2':
        output_nifti = nib.Nifti2Image(thearray, outputaffine, header=theheader)
        suffix = '.nii'
    else:
        output_nifti = nib.Nifti1Image(thearray, outputaffine, header=theheader)
        suffix = '.nii.gz'
    output_nifti.set_qform(qaffine, code=int(qcode))
    output_nifti.set_sform(saffine, code=int(scode))
    output_nifti.to_filename(thename + suffix)
    output_nifti = None


def checkifnifti(filename):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        return True
    else:
        return False


def checkiftext(filename):
    if filename.endswith(".txt"):
        return True
    else:
        return False


def getniftiroot(filename):
    if filename.endswith(".nii"):
        return filename[:-4]
    elif filename.endswith(".nii.gz"):
        return filename[:-7]
    else:
        return filename


def fmritimeinfo(niftifilename):
    nim = nib.load(niftifilename)
    hdr = nim.get_header()
    thedims = hdr['dim']
    thesizes = hdr['pixdim']
    if hdr.get_xyzt_units()[1] == 'msec':
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    timepoints = thedims[4]
    return tr, timepoints


def checkspacematch(dims1, dims2):
    for i in range(1, 4):
        if dims1[i] != dims2[i]:
            print("File spatial voxels do not match")
            print("dimension ", i, ":", dims1[i], "!=", dims2[i])
            return False
        else:
            return True


def checktimematch(dims1, dims2, numskip1, numskip2):
    if (dims1[4] - numskip1) != (dims2[4] - numskip2):
        print("File numbers of timepoints do not match")
        print("dimension ", 4, ":", dims1[4],
              "(skip ", numskip1, ") !=",
              dims2[4],
              " (skip ", numskip2, ")")
        return False
    else:
        return True
