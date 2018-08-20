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

import sys
import time
import json
import warnings

import argparse
import platform
import numpy as np
import nibabel as nib

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False

import rapidtide.io as tide_io
import rapidtide.fit as tide_fit
import rapidtide.util as tide_util
import rapidtide.filter as tide_filt
import rapidtide.stats as tide_stats
import rapidtide.miscmath as tide_math
import rapidtide.correlate as tide_corr
import rapidtide.glmpass as tide_glmpass
import rapidtide.corrfit as tide_corrfit
import rapidtide.corrpass as tide_corrpass
import rapidtide.resample as tide_resample
import rapidtide.nullcorrpass as tide_nullcorr
from rapidtide.workflows.parser_funcs import (is_valid_file, invert_float, is_float)
from rapidtide.workflows.workflow_utils import (numpy2shared, getglobalsignal, allocshared)


def _get_parser():
    """
    Argument parser for rapidtide
    """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('in_file',
                        type=lambda x: is_valid_file(parser, x),
                        help='The input data file (BOLD fmri file or NIRS)')
    parser.add_argument('prefix',
                        help='The root name for the output files')

    # Macros
    macros = parser.add_argument_group('Macros').add_mutually_exclusive_group()
    macros.add_argument('--venousrefine',
                        dest='venousrefine',
                        action='store_true',
                        help=('This is a macro that sets --lagminthresh=2.5, '
                              '--lagmaxthresh=6.0, --ampthresh=0.5, and '
                              '--refineupperlag to bias refinement towards '
                              'voxels in the draining vasculature for an '
                              'fMRI scan.'),
                        default=False)
    macros.add_argument('--nirs',
                        dest='nirs',
                        action='store_true',
                        help=('This is a NIRS analysis - this is a macro that '
                              'sets --nothresh, --preservefiltering, '
                              '--refineprenorm=var, --ampthresh=0.7, and '
                              '--lagminthresh=0.1.'),
                        default=False)

    # Preprocessing options
    preproc = parser.add_argument_group('Preprocessing options')
    realtr = preproc.add_mutually_exclusive_group()
    realtr.add_argument('--datatstep',
                        dest='realtr',
                        action='store',
                        metavar='TSTEP',
                        type=lambda x: is_float(parser, x),
                        help=('Set the timestep of the data file to TSTEP. '
                              'This will override the TR in an '
                              'fMRI file. NOTE: if using data from a text '
                              'file, for example with NIRS data, using one '
                              'of these options is mandatory.'),
                        default='auto')
    realtr.add_argument('--datafreq',
                        dest='realtr',
                        action='store',
                        metavar='FREQ',
                        type=lambda x: invert_float(parser, x),
                        help=('Set the timestep of the data file to 1/FREQ. '
                              'This will override the TR in an '
                              'fMRI file. NOTE: if using data from a text '
                              'file, for example with NIRS data, using one '
                              'of these options is mandatory.'),
                        default='auto')
    preproc.add_argument('-a',
                         dest='antialias',
                         action='store_false',
                         help='Disable antialiasing filter',
                         default=True)
    preproc.add_argument('--invert',
                         dest='invertregressor',
                         action='store_true',
                         help=('Invert the sign of the regressor before '
                               'processing'),
                         default=False)
    preproc.add_argument('--interptype',
                         dest='interptype',
                         action='store',
                         type=str,
                         choices=['univariate', 'cubic', 'quadratic'],
                         help=("Use specified interpolation type. Options "
                               "are 'cubic','quadratic', and 'univariate' "
                               "(default)."),
                         default='univariate')
    preproc.add_argument('--offsettime',
                         dest='offsettime',
                         action='store',
                         type=float,
                         metavar='OFFSETTIME',
                         help='Apply offset OFFSETTIME to the lag regressors',
                         default=0.)
    preproc.add_argument('--butterorder',
                         dest='butterorder',
                         action='store',
                         type=int,
                         metavar='ORDER',
                         help=('Use butterworth filter for band splitting '
                               'instead of trapezoidal FFT filter and set '
                               'filter order to ORDER.'),
                         default=None)

    filttype = preproc.add_mutually_exclusive_group()
    filttype.add_argument('-F', '--arb',
                          dest='arbvec',
                          action='store',
                          nargs='+',
                          type=lambda x: is_float(parser, x),
                          metavar=('LOWERFREQ UPPERFREQ',
                                   'LOWERSTOP UPPERSTOP'),
                          help=('Filter data and regressors from LOWERFREQ to '
                                'UPPERFREQ. LOWERSTOP and UPPERSTOP can also '
                                'be specified, or will be calculated '
                                'automatically'),
                          default=None)
    filttype.add_argument('--filtertype',
                          dest='filtertype',
                          action='store',
                          type=str,
                          choices=['arb', 'vlf', 'lfo', 'resp', 'cardiac'],
                          help=('Filter data and regressors to specific band'),
                          default='vlf')
    filttype.add_argument('-V', '--vlf',
                          dest='filtertype',
                          action='store_const',
                          const='vlf',
                          help=('Filter data and regressors to VLF band'),
                          default='vlf')
    filttype.add_argument('-L', '--lfo',
                          dest='filtertype',
                          action='store_const',
                          const='lfo',
                          help=('Filter data and regressors to LFO band'),
                          default='vlf')
    filttype.add_argument('-R', '--resp',
                          dest='filtertype',
                          action='store_const',
                          const='resp',
                          help=('Filter data and regressors to respiratory '
                                'band'),
                          default='vlf')
    filttype.add_argument('-C', '--cardiac',
                          dest='filtertype',
                          action='store_const',
                          const='cardiac',
                          help=('Filter data and regressors to cardiac band'),
                          default='vlf')

    preproc.add_argument('-N', '--numnull',
                         dest='numestreps',
                         action='store',
                         type=int,
                         metavar='NREPS',
                         help=('Estimate significance threshold by running '
                               'NREPS null correlations (default is 10000, '
                               'set to 0 to disable)'),
                         default=10000)
    preproc.add_argument('--skipsighistfit',
                         dest='dosighistfit',
                         action='store_false',
                         help=('Do not fit significance histogram with a '
                               'Johnson SB function'),
                         default=True)

    wfunc = preproc.add_mutually_exclusive_group()
    wfunc.add_argument('--windowfunc',
                       dest='windowfunc',
                       action='store',
                       type=str,
                       choices=['hamming', 'hann', 'blackmanharris', 'None'],
                       help=('Window funcion to use prior to correlation. '
                             'Options are hamming (default), hann, '
                             'blackmanharris, and None'),
                       default='hamming')
    wfunc.add_argument('--nowindow',
                       dest='windowfunc',
                       action='store_const',
                       const='None',
                       help='Disable precorrelation windowing',
                       default='hamming')

    preproc.add_argument('-f', '--spatialfilt',
                         dest='gausssigma',
                         action='store',
                         type=float,
                         metavar='GAUSSSIGMA',
                         help=('Spatially filter fMRI data prior to analysis '
                               'using GAUSSSIGMA in mm'),
                         default=0.)
    preproc.add_argument('-M', '--globalmean',
                         dest='useglobalref',
                         action='store_true',
                         help=('Generate a global mean regressor and use that '
                               'as the reference regressor'),
                         default=False)
    preproc.add_argument('--globalmaskmethod',
                         dest='globalmaskmethod',
                         action='store',
                         type=str,
                         choices=['mean', 'variance'],
                         help=('Use value type to mask voxels prior to '
                               'generating global mean'),
                         default='mean')
    preproc.add_argument('--meanscale',
                         dest='meanscaleglobal',
                         action='store_true',
                         help=('Mean scale regressors during global mean '
                               'estimation'),
                         default=False)
    preproc.add_argument('--slicetimes',
                         dest='slicetimes',
                         action='store',
                         type=lambda x: is_valid_file(parser, x),
                         metavar='FILE',
                         help=('Apply offset times from FILE to each slice in '
                               'the dataset'),
                         default=None)
    preproc.add_argument('--numskip',
                         dest='preprocskip',
                         action='store',
                         type=int,
                         metavar='SKIP',
                         help=('SKIP TRs were previously deleted during '
                               'preprocessing (default is 0)'),
                         default=0)
    preproc.add_argument('--nothresh',
                         dest='nothresh',
                         action='store_false',
                         help=('Disable voxel intensity threshold (especially '
                               'useful for NIRS data)'),
                         default=True)

    # Correlation options
    corr = parser.add_argument_group('Correlation options')
    corr.add_argument('--oversampfac',
                      dest='oversampfactor',
                      action='store',
                      type=int,
                      metavar='OVERSAMPFAC',
                      help=('Oversample the fMRI data by the following '
                            'integral factor (default is 2)'),
                      default=2)
    corr.add_argument('--regressor',
                      dest='regressorfile',
                      action='store',
                      type=lambda x: is_valid_file(parser, x),
                      metavar='FILE',
                      help=('Read probe regressor from file FILE (if none '
                            'specified, generate and use global regressor)'),
                      default=None)

    reg_group = corr.add_mutually_exclusive_group()
    reg_group.add_argument('--regressorfreq',
                           dest='inputfreq',
                           action='store',
                           type=lambda x: is_float(parser, x),
                           metavar='FREQ',
                           help=('Probe regressor in file has sample '
                                 'frequency FREQ (default is 1/tr) '
                                 'NB: --regressorfreq and --regressortstep) '
                                 'are two ways to specify the same thing'),
                           default='auto')
    reg_group.add_argument('--regressortstep',
                           dest='inputfreq',
                           action='store',
                           type=lambda x: invert_float(parser, x),
                           metavar='TSTEP',
                           help=('Probe regressor in file has sample '
                                 'frequency FREQ (default is 1/tr) '
                                 'NB: --regressorfreq and --regressortstep) '
                                 'are two ways to specify the same thing'),
                           default='auto')

    corr.add_argument('--regressorstart',
                      dest='inputstarttime',
                      action='store',
                      type=float,
                      metavar='START',
                      help=('The time delay in seconds into the regressor '
                            'file, corresponding in the first TR of the fMRI '
                            'file (default is 0.0)'),
                      default=0.)

    cc_group = corr.add_mutually_exclusive_group()
    cc_group.add_argument('--corrweighting',
                          dest='corrweighting',
                          action='store',
                          type=str,
                          choices=['none', 'phat', 'liang', 'eckart'],
                          help=('Method to use for cross-correlation '
                                'weighting.'),
                          default='none')
    cc_group.add_argument('--nodetrend',
                          dest='dodetrend',
                          action='store_false',
                          help='Disable linear trend removal',
                          default=True)

    mask_group = corr.add_mutually_exclusive_group()
    mask_group.add_argument('--corrmaskthresh',
                            dest='corrmaskthreshpct',
                            action='store',
                            type=float,
                            metavar='PCT',
                            help=('Do correlations in voxels where the mean '
                                  'exceeds this percentage of the robust max '
                                  '(default is 1.0)'),
                            default=1.0)
    mask_group.add_argument('--corrmask',
                            dest='corrmaskname',
                            action='store',
                            type=lambda x: is_valid_file(parser, x),
                            metavar='FILE',
                            help=('Only do correlations in voxels in FILE '
                                  '(if set, corrmaskthresh is ignored).'),
                            default=None)

    # Correlation fitting options
    corr_fit = parser.add_argument_group('Correlation fitting options')

    fixdelay = corr_fit.add_mutually_exclusive_group()
    fixdelay.add_argument('-Z',
                          dest='fixeddelayvalue',
                          action='store',
                          type=float,
                          metavar='DELAYTIME',
                          help=("Don't fit the delay time - set it to "
                                "DELAYTIME seconds for all voxels"),
                          default=None)
    fixdelay.add_argument('-r',
                          dest='lag_extrema',
                          action='store',
                          nargs=2,
                          type=int,
                          metavar=('LAGMIN', 'LAGMAX'),
                          help=('Limit fit to a range of lags from LAGMIN to '
                                'LAGMAX'),
                          default=(-30.0, 30.0))

    corr_fit.add_argument('--sigmalimit',
                          dest='widthlimit',
                          action='store',
                          type=float,
                          metavar='SIGMALIMIT',
                          help=('Reject lag fits with linewidth wider than '
                                'SIGMALIMIT Hz'),
                          default=100.0)
    corr_fit.add_argument('--bipolar',
                          dest='bipolar',
                          action='store_true',
                          help=('Bipolar mode - match peak correlation '
                                'ignoring sign'),
                          default=False)
    corr_fit.add_argument('--nofitfilt',
                          dest='zerooutbadfit',
                          action='store_false',
                          help=('Do not zero out peak fit values if fit '
                                'fails'),
                          default=True)
    corr_fit.add_argument('--maxfittype',
                          dest='findmaxtype',
                          action='store',
                          type=str,
                          choices=['gauss', 'quad'],
                          help=("Method for fitting the correlation peak "
                                "(default is 'gauss'). 'quad' uses a "
                                "quadratic fit.  Faster but not as well "
                                "tested"),
                          default='gauss')
    corr_fit.add_argument('--despecklepasses',
                          dest='despeckle_passes',
                          action='store',
                          type=int,
                          metavar='PASSES',
                          help=('Detect and refit suspect correlations to '
                                'disambiguate peak locations in PASSES '
                                'passes'),
                          default=0)
    corr_fit.add_argument('--despecklethresh',
                          dest='despeckle_thresh',
                          action='store',
                          type=int,
                          metavar='VAL',
                          help=('Refit correlation if median discontinuity '
                                'magnitude exceeds VAL (default is 5s)'),
                          default=5)

    # Regressor refinement options
    reg_ref = parser.add_argument_group('Regressor refinement options')
    reg_ref.add_argument('--refineprenorm',
                         dest='refineprenorm',
                         action='store',
                         type=str,
                         choices=['None', 'mean', 'var', 'std', 'invlag'],
                         help=("Apply TYPE prenormalization to each "
                               "timecourse prior to refinement. Valid "
                               "weightings are 'None', 'mean' (default), "
                               "'var', and 'std'"),
                         default='mean')
    reg_ref.add_argument('--refineweighting',
                         dest='refineweighting',
                         action='store',
                         type=str,
                         choices=['None', 'NIRS', 'R', 'R2'],
                         help=("Apply TYPE weighting to each timecourse prior "
                               "to refinement. Valid weightings are "
                               "'None', 'NIRS', 'R', and 'R2' (default)"),
                         default='R2')
    reg_ref.add_argument('--passes',
                         dest='passes',
                         action='store',
                         type=int,
                         metavar='PASSES',
                         help=('Set the number of processing passes to '
                               'PASSES'),
                         default=1)
    reg_ref.add_argument('--includemask',
                         dest='includemaskname',
                         action='store',
                         type=lambda x: is_valid_file(parser, x),
                         metavar='FILE',
                         help=('Only use voxels in NAME for global regressor '
                               'generation and regressor refinement'),
                         default=None)
    reg_ref.add_argument('--excludemask',
                         dest='excludemaskname',
                         action='store',
                         type=lambda x: is_valid_file(parser, x),
                         metavar='FILE',
                         help=('Do not use voxels in NAME for global '
                               'regressor generation and regressor '
                               'refinement'),
                         default=None)
    reg_ref.add_argument('--lagminthresh',
                         dest='lagminthresh',
                         action='store',
                         metavar='MIN',
                         type=float,
                         help=('For refinement, exclude voxels with delays '
                               'less than MIN (default is 0.5s)'),
                         default=0.5)
    reg_ref.add_argument('--lagmaxthresh',
                         dest='lagmaxthresh',
                         action='store',
                         metavar='MAX',
                         type=float,
                         help=('For refinement, exclude voxels with delays '
                               'greater than MAX (default is 5s)'),
                         default=5.0)
    reg_ref.add_argument('--ampthresh',
                         dest='ampthresh',
                         action='store',
                         metavar='AMP',
                         type=float,
                         help=('or refinement, exclude voxels with '
                               'correlation coefficients less than AMP '
                               '(default is 0.3)'),
                         default=0.3)
    reg_ref.add_argument('--sigmathresh',
                         dest='sigmathresh',
                         action='store',
                         metavar='SIGMA',
                         type=float,
                         help=('For refinement, exclude voxels with widths '
                               'greater than SIGMA (default is 100s)'),
                         default=100.0)
    reg_ref.add_argument('--refineoffset',
                         dest='refineoffset',
                         action='store_true',
                         help=('Bipolar mode - match peak correlation '
                               'ignoring sign'),
                         default=False)
    reg_ref.add_argument('--psdfilter',
                         dest='psdfilter',
                         action='store_true',
                         help=('Apply a PSD weighted Wiener filter to '
                               'shifted timecourses prior to refinement'),
                         default=False)

    gauss_ref = reg_ref.add_mutually_exclusive_group()
    gauss_ref.add_argument('--nogaussrefine',
                           dest='gaussrefine',
                           action='store_false',
                           help=('Will not use gaussian correlation peak '
                                 'refinement'),
                           default=True)
    gauss_ref.add_argument('--fastgauss',
                           dest='fastgauss',
                           action='store_true',
                           help=('Will use alternative fast gauss refinement '
                                 '(does not work well)'),
                           default=False)

    refine = reg_ref.add_mutually_exclusive_group()
    refine.add_argument('--refineupperlag',
                        dest='lagmaskside',
                        action='store_const',
                        const='upper',
                        help=('Only use positive lags for regressor '
                              'refinement'),
                        default='both')
    refine.add_argument('--refinelowerlag',
                        dest='lagmaskside',
                        action='store_const',
                        const='lower',
                        help=('Only use negative lags for regressor '
                              'refinement'),
                        default='both')
    reg_ref.add_argument('--refinetype',
                         dest='refinetype',
                         action='store',
                         type=str,
                         choices=['avg', 'pca', 'ica', 'weightedavg'],
                         help=('Method with which to derive refined '
                               'regressor.'),
                         default='avg')

    # Output options
    output = parser.add_argument_group('Output options')
    output.add_argument('--limitoutput',
                        dest='savelagregressors',
                        action='store_false',
                        help=("Don't save some of the large and rarely used "
                              "files"),
                        default=True)
    output.add_argument('--savelags',
                        dest='savecorrtimes',
                        action='store_true',
                        help='Save a table of lagtimes used',
                        default=False)
    output.add_argument('--histlen',  # was -h
                        dest='histlen',
                        action='store',
                        type=int,
                        metavar='HISTLEN',
                        help=('Change the histogram length to HISTLEN '
                              '(default is 100)'),
                        default=100)
    output.add_argument('--timerange',
                        dest='timerange',
                        action='store',
                        nargs=2,
                        type=int,
                        metavar=('START', 'END'),
                        help=('Limit analysis to data between timepoints '
                              'START and END in the fmri file'),
                        default=(-1, 10000000))
    output.add_argument('--glmsourcefile',
                        dest='glmsourcefile',
                        action='store',
                        type=lambda x: is_valid_file(parser, x),
                        metavar='FILE',
                        help=('Regress delayed regressors out of FILE instead '
                              'of the initial fmri file used to estimate '
                              'delays'),
                        default=None)
    output.add_argument('--noglm',
                        dest='doglmfilt',
                        action='store_false',
                        help=('Turn off GLM filtering to remove delayed '
                              'regressor from each voxel (disables output of '
                              'fitNorm)'),
                        default=True)
    output.add_argument('--preservefiltering',
                        dest='preservefiltering',
                        action='store_true',
                        help="Don't reread data prior to GLM",
                        default=False)

    # Miscellaneous options
    misc = parser.add_argument_group('Miscellaneous options')
    misc.add_argument('--noprogressbar',
                      dest='showprogressbar',
                      action='store_false',
                      help='Will disable progress bars',
                      default=True)
    misc.add_argument('--wiener',
                      dest='dodeconv',
                      action='store_true',
                      help=('Do Wiener deconvolution to find voxel transfer '
                            'function'),
                      default=False)
    misc.add_argument('--usesp',
                      dest='internalprecision',
                      action='store_const',
                      const='single',
                      help=('Use single precision for internal calculations '
                            '(may be useful when RAM is limited)'),
                      default='double')
    misc.add_argument('--cifti',
                      dest='isgrayordinate',
                      action='store_true',
                      help='Data file is a converted CIFTI',
                      default=False)
    misc.add_argument('--simulate',
                      dest='fakerun',
                      action='store_true',
                      help='Simulate a run - just report command line options',
                      default=False)
    misc.add_argument('-d',
                      dest='displayplots',
                      action='store_true',
                      help='Display plots of interesting timecourses',
                      default=False)
    misc.add_argument('--nonumba',
                      dest='nonumba',
                      action='store_true',
                      help='Disable jit compilation with numba',
                      default=False)
    misc.add_argument('--nosharedmem',
                      dest='sharedmem',
                      action='store_false',
                      help=('Disable use of shared memory for large array '
                            'storage'),
                      default=True)
    misc.add_argument('--memprofile',
                      dest='memprofile',
                      action='store_true',
                      help=('Enable memory profiling for debugging - '
                            'warning: this slows things down a lot.'),
                      default=False)
    misc.add_argument('--mklthreads',
                      dest='mklthreads',
                      action='store',
                      type=int,
                      metavar='NTHREADS',
                      help=('Use no more than NTHREADS worker threads in '
                            'accelerated numpy calls.'),
                      default=1)
    misc.add_argument('--nprocs',
                      dest='nprocs',
                      action='store',
                      type=int,
                      metavar='NPROCS',
                      help=('Use NPROCS worker processes for multiprocessing. '
                            'Setting NPROCS to less than 1 sets the number of '
                            'worker processes to n_cpus - 1.'),
                      default=1)
    # TODO: Also set theprefilter.setdebug(True)
    misc.add_argument('--debug',
                      dest='debug',
                      action='store_true',
                      help=('Enable additional information output'),
                      default=False)

    # Experimental options (not fully tested, may not work)
    experimental = parser.add_argument_group('Experimental options (not fully '
                                             'tested, may not work)')
    # TODO: Also set shiftall to True, although shiftall is set to True anyway
    experimental.add_argument('--cleanrefined',
                              dest='cleanrefined',
                              action='store_true',
                              help=('Perform additional processing on refined '
                                    'regressor to remove spurious '
                                    'components.'),
                              default=False)
    experimental.add_argument('--dispersioncalc',
                              dest='dodispersioncalc',
                              action='store_true',
                              help=('Generate extra data during refinement to '
                                    'allow calculation of dispersion.'),
                              default=False)
    experimental.add_argument('--acfix',
                              dest='fix_autocorrelation',
                              action='store_true',
                              help=('Perform a secondary correlation to '
                                    'disambiguate peak location (enables '
                                    '--accheck). Experimental.'),
                              default=False)
    experimental.add_argument('--tmask',
                              dest='tmaskname',
                              action='store',
                              type=lambda x: is_valid_file(parser, x),
                              metavar='FILE',
                              help=('Only correlate during epochs specified '
                                    'in MASKFILE (NB: each line of FILE '
                                    'contains the time and duration of an '
                                    'epoch to include'),
                              default=None)
    exp_group = experimental.add_mutually_exclusive_group()
    exp_group.add_argument('--prewhiten',
                           dest='doprewhiten',
                           action='store_true',
                           help='Prewhiten and refit data',
                           default=False)
    exp_group.add_argument('--saveprewhiten',
                           dest='saveprewhiten',
                           action='store_true',
                           help=('Save prewhitened data (turns prewhitening '
                                 'on)'),
                           default=False)
    experimental.add_argument('--AR',
                              dest='armodelorder',
                              action='store',
                              type=int,
                              help='Set AR model order (default is 1)',
                              default=1)
    return parser


def rapidtide_workflow(in_file, prefix, venousrefine=False, nirs=False,
                       realtr='auto', antialias=True, invertregressor=False,
                       interptype='univariate', offsettime=0.,
                       butterorder=None, arbvec=None, filtertype='arb',
                       numestreps=10000, dosighistfit=True,
                       windowfunc='hamming', gausssigma=0.,
                       useglobalref=False, meanscaleglobal=False,
                       globalmaskmethod='mean',
                       slicetimes=None, preprocskip=0, nothresh=True,
                       oversampfactor=2, regressorfile=None, inputfreq=1.,
                       inputstarttime=0., corrweighting='none',
                       dodetrend=True, corrmaskthreshpct=1.,
                       corrmaskname=None, fixeddelayvalue=None,
                       lag_extrema=(-30.0, 30.0), widthlimit=100.,
                       bipolar=False, zerooutbadfit=True, findmaxtype='gauss',
                       despeckle_passes=0, despeckle_thresh=5,
                       refineprenorm='mean', refineweighting='R2', passes=1,
                       includemaskname=None, excludemaskname=None,
                       lagminthresh=0.5, lagmaxthresh=5., ampthresh=0.3,
                       sigmathresh=100.,
                       gaussrefine=True, fastgauss=False,
                       refineoffset=False, psdfilter=False,
                       lagmaskside='both', refinetype='avg',
                       savelagregressors=True, savecorrtimes=False,
                       histlen=100, timerange=(-1, 10000000),
                       glmsourcefile=None, doglmfilt=True,
                       preservefiltering=False, showprogressbar=True,
                       dodeconv=False, internalprecision='double',
                       isgrayordinate=False, fakerun=False, displayplots=False,
                       nonumba=False, sharedmem=True, memprofile=False,
                       mklthreads=1,
                       nprocs=1, debug=False, cleanrefined=False,
                       dodispersioncalc=False, fix_autocorrelation=False,
                       tmaskname=None, doprewhiten=False, saveprewhiten=False,
                       armodelorder=1, offsettime_total=None,
                       ampthreshfromsig=False, nohistzero=False,
                       fixdelay=False, usebutterworthfilter=False):
    """
    Run the full rapidtide workflow.
    """
    # set the number of MKL threads to use
    if mklexists:
        mkl.set_num_threads(mklthreads)

    # Initialize the memory usage file
    if not memprofile:
        memfile = prefix + '_memusage.csv'
        tide_util.logmem(None, file=memfile)
    else:
        memfile = None

    # start the clock!
    timings = [['Start', time.time(), None, None]]

    # TODO: Probably drop these. TS
    addedskip = 0
    lagmin, lagmax = lag_extrema
    verbose = True
    lthreshval = 0.
    uthreshval = 1.
    absmaxsigma = 100.
    edgebufferfrac = 0.
    lagmod = 1000.
    dolagmod = True
    dodemean = True
    filtorder = 6
    padseconds = 30.0
    enforcethresh = True
    trapezoidalfftfilter = True
    check_autocorrelation = True
    outputprecision = 'single'
    mp_chunksize = 50000
    acwidth = 0.0
    saveglmfiltered = True
    if isgrayordinate:
        datatype = 'cifti'

    # set internal precision
    if internalprecision == 'double':
        print('setting internal precision to double')
        rt_floattype = 'float64'
        rt_floatset = np.float64
    else:
        print('setting internal precision to single')
        rt_floattype = 'float32'
        rt_floatset = np.float32

    # set the output precision
    if outputprecision == 'double':
        print('setting output precision to double')
        rt_outfloattype = 'float64'
        rt_outfloatset = np.float64
    else:
        print('setting output precision to single')
        rt_outfloattype = 'float32'
        rt_outfloatset = np.float32

    theprefilter = tide_filt.noncausalfilter()
    theprefilter.setbutter(bool(butterorder), filtorder)
    theprefilter.setdebug(debug)
    theprefilter.settype(filtertype)
    if filtertype != 'vlf':
        despeckle_thresh = np.max([despeckle_thresh,
                                   0.5 / theprefilter.getfreqlimits()[2]])

    if filtertype == 'arb':
        if len(arbvec) == 2:
            arb_lower, arb_upper = arbvec
            arb_lowerstop = arb_lower * 0.9
            arb_upperstop = arb_upper * 1.1
        elif len(arbvec) == 4:
            arb_lower, arb_upper, arb_lowerstop, arb_upperstop = arbvec
        else:
            raise ValueError('If arb filtering is used, arbvec must contain 2 '
                             'or 4 items.')
        theprefilter.setarb(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)

    # open the fmri datafile
    tide_util.logmem('before reading in fmri data', file=memfile)
    if tide_io.checkiftext(in_file):
        print('input file is text - all I/O will be to text files')
        datatype = 'text'
        if gausssigma > 0:
            gausssigma = 0.
            print('gaussian spatial filter disabled for text input files')

        nim_data = tide_io.readvecs(in_file)
        theshape = np.shape(nim_data)
        xsize = theshape[0]
        ysize = 1
        numslices = 1
        timepoints = theshape[1]
        thesizes = [0, int(xsize), 1, 1, int(timepoints)]
        numspatiallocs = int(xsize)
        slicesize = numspatiallocs
    else:
        nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(in_file)
        if nim_hdr['intent_code'] == 3002:
            print('input file is CIFTI')
            datatype = 'cifti'
            timepoints = nim_data.shape[4]
            numspatiallocs = nim_data.shape[5]
            slicesize = numspatiallocs
        else:
            print('input file is NIFTI')
            datatype = 'nifti'
            xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
            numspatiallocs = int(xsize) * int(ysize) * int(numslices)
            slicesize = numspatiallocs / int(numslices)
        xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
    tide_util.logmem('after reading in fmri data', file=memfile)

    # correct some fields if necessary
    if datatype == 'cifti':
        fmritr = 0.72  # this is wrong and is a hack until I can parse CIFTI XML
    elif datatype == 'text':
        if realtr <= 0.0:
            raise ValueError('for text file data input, you must use the -t '
                             'option to set the timestep')
    else:
        if nim_hdr.get_xyzt_units()[1] == 'msec':
            fmritr = thesizes[4] / 1000.0
        else:
            fmritr = thesizes[4]

    if realtr > 0.0:
        fmritr = realtr

    fmrifreq = 1. / fmritr

    # check to see if we need to adjust the oversample factor
    if oversampfactor < 0:
        oversampfactor = int(np.ceil(fmritr // 0.5))
        print('oversample factor set to', oversampfactor)

    oversamptr = fmritr / oversampfactor
    if verbose:
        print('fmri data: {0} timepoints, tr = {1}, oversamptr = '
              '{2}'.format(timepoints, fmritr, oversamptr))
    print('{0} spatial locations, {1} timepoints'.format(numspatiallocs, timepoints))
    timings.append(['Finish reading fmrifile', time.time(), None, None])

    # if the user has specified start and stop points, limit check, then use these numbers
    validstart, validend = tide_util.startendcheck(timepoints, timerange[0], timerange[1])
    if abs(lagmin) > (validend - validstart + 1) * fmritr / 2.0:
        raise ValueError('magnitude of lagmin exceeds {0} - '
                         'invalid'.format((validend - validstart + 1) * fmritr / 2.0))

    if abs(lagmax) > (validend - validstart + 1) * fmritr / 2.0:
        raise ValueError('magnitude of lagmax exceeds {0} - '
                         'invalid'.format((validend - validstart + 1) * fmritr / 2.0))

    if gausssigma > 0:
        print('applying gaussian spatial filter to timepoints {0} to '
              '{1}'.format(validstart, validend))
        reportstep = 10
        for i in range(validstart, validend + 1):
            if (i % reportstep == 0 or i == validend) and showprogressbar:
                tide_util.progressbar(i - validstart + 1, timepoints,
                                      label='Percent complete')
            nim_data[:, :, :, i] = tide_filt.ssmooth(xdim, ydim, slicethickness, gausssigma,
                                                     nim_data[:, :, :, i])
        timings.append(['End 3D smoothing', time.time(), None, None])
        print()

    # reshape the data and trim to a time range, if specified.
    # Check for special case of no trimming to save RAM
    if (validstart == 0) and (validend == timepoints):
        fmri_data = nim_data.reshape((numspatiallocs, timepoints))
    else:
        fmri_data = nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1]
        timepoints = validend - validstart + 1

    # read in the optional masks
    tide_util.logmem('before setting masks', file=memfile)
    internalincludemask = None
    internalexcludemask = None
    if includemaskname is not None:
        if datatype == 'text':
            theincludemask = tide_io.readvecs(includemaskname).astype('int16')
            theshape = np.shape(nim_data)
            theincludexsize = theshape[0]
            if not theincludexsize == xsize:
                raise ValueError('Dimensions of include mask do not match the '
                                 'fmri data - exiting')
        else:
            (nimincludemask, theincludemask, nimincludemask_hdr,
             theincludemaskdims, theincludmasksizes) = tide_io.readfromnifti(
                includemaskname)
            if not tide_io.checkspacematch(nimincludemask_hdr, nim_hdr):
                raise ValueError('Dimensions of include mask do not match the '
                                 'fmri data - exiting')
        internalincludemask = theincludemask.reshape(numspatiallocs)

    if excludemaskname is not None:
        if datatype == 'text':
            theexcludemask = tide_io.readvecs(excludemaskname).astype('int16')
            theexcludemask = 1.0 - theexcludemask
            theshape = np.shape(nim_data)
            theexcludexsize = theshape[0]
            if not theexcludexsize == xsize:
                raise ValueError('Dimensions of exclude mask do not match the '
                                 'fmri data - exiting')
        else:
            (nimexcludemask,
             theexcludemask,
             nimexcludemask_hdr,
             theexcludemaskdims,
             theexcludmasksizes) = tide_io.readfromnifti(excludemaskname)
            theexcludemask = 1.0 - theexcludemask
            if not tide_io.checkspacematch(nimexcludemask_hdr, nim_hdr):
                raise ValueError('Dimensions of exclude mask do not match the '
                                 'fmri data - exiting')
        internalexcludemask = theexcludemask.reshape(numspatiallocs)
    tide_util.logmem('after setting masks', file=memfile)

    # find the threshold value for the image data
    tide_util.logmem('before selecting valid voxels', file=memfile)
    threshval = tide_stats.getfracval(fmri_data[:, addedskip:], 0.98) / 25.0
    if corrmaskname is not None:
        if datatype == 'text':
            corrmask = tide_io.readvecs(corrmaskname).astype('int16')
            theshape = np.shape(nim_data)
            corrxsize = theshape[0]
            if not corrxsize == xsize:
                raise ValueError('Dimensions of correlation mask do not match '
                                 'the fmri data - exiting')
        else:
            (nimcorrmask,
             thecorrmask,
             nimcorrmask_hdr,
             corrmaskdims,
             theincludmasksizes) = tide_io.readfromnifti(corrmaskname)
            if not tide_io.checkspacematch(nimcorrmask_hdr, nim_hdr):
                raise ValueError('Dimensions of correlation mask do not match '
                                 'the fmri data - exiting')
            corrmask = np.uint16(np.where(thecorrmask > 0, 1, 0).reshape(numspatiallocs))
    else:
        corrmask = np.uint16(tide_stats.makemask(np.mean(fmri_data[:, addedskip:], axis=1),
                                                 threshpct=corrmaskthreshpct))

    if nothresh:
        corrmask *= 0
        corrmask += 1
        threshval = -10000000.0

    if verbose:
        print('image threshval = {0}'.format(threshval))

    validvoxels = np.where(corrmask > 0)[0]
    numvalidspatiallocs = np.shape(validvoxels)[0]
    print('validvoxels shape = {0}'.format(numvalidspatiallocs))
    fmri_data_valid = fmri_data[validvoxels, :] + 0.0
    print('original size = {0}, trimmed size = {1}'.format(np.shape(fmri_data),
                                                           np.shape(fmri_data_valid)))
    if internalincludemask is not None:
        internalincludemask_valid = 1.0 * internalincludemask[validvoxels]
        del internalincludemask
        print('internalincludemask_valid has size: {0}'.format(internalincludemask_valid.size))
    else:
        internalincludemask_valid = None

    if internalexcludemask is not None:
        internalexcludemask_valid = 1.0 * internalexcludemask[validvoxels]
        del internalexcludemask
        print('internalexcludemask_valid has size: {0}'.format(internalexcludemask_valid.size))
    else:
        internalexcludemask_valid = None
    tide_util.logmem('after selecting valid voxels', file=memfile)

    # move fmri_data_valid into shared memory
    if sharedmem:
        print('moving fmri data to shared memory')
        timings.append(['Start moving fmri_data to shared memory', time.time(), None, None])
        if memprofile:
            numpy2shared_func = profile(numpy2shared, precision=2)
        else:
            tide_util.logmem('before fmri data move', file=memfile)
            numpy2shared_func = numpy2shared
        (fmri_data_valid,
         fmri_data_valid_shared,
         fmri_data_valid_shared_shape) = numpy2shared_func(fmri_data_valid,
                                                           rt_floatset)
        timings.append(['End moving fmri_data to shared memory', time.time(), None, None])

    # get rid of memory we aren't using
    tide_util.logmem('before purging full sized fmri data', file=memfile)
    del fmri_data
    del nim_data
    tide_util.logmem('after purging full sized fmri data', file=memfile)

    # read in the timecourse to resample
    timings.append(['Start of reference prep', time.time(), None, None])
    if regressorfile is None:
        print('no regressor file specified - will use the global mean regressor')
        useglobalref = True

    temp_dict = {'despeckle_thresh': despeckle_thresh,
                 'widthlimit': widthlimit,
                 'lthreshval': lthreshval,
                 'bipolar': bipolar,
                 'fixdelay': fixdelay,
                 'findmaxtype': findmaxtype,
                 'lagmin': lagmin,
                 'lagmax': lagmax,
                 'absmaxsigma': absmaxsigma,
                 'edgebufferfrac': edgebufferfrac,
                 'uthreshval': uthreshval,
                 'debug': debug,
                 'gaussrefine': gaussrefine,
                 'fastgauss': fastgauss,
                 'enforcethresh': enforcethresh,
                 'zerooutbadfit': zerooutbadfit,
                 'lagmod': lagmod,
                 'fixeddelayvalue': fixeddelayvalue,
                 'showprogressbar': showprogressbar,
                 'nprocs': nprocs,
                 'mp_chunksize': mp_chunksize,
                 'globalmaskmethod': globalmaskmethod,
                 'corrmaskthreshpct': corrmaskthreshpct,
                 'nothresh': nothresh,
                 'meanscaleglobal': meanscaleglobal,
                 'dodetrend': dodetrend,
                 'windowfunc': windowfunc,
                 'corrweighting': corrweighting,
                 'numestreps': numestreps,
                 'oversampfactor': oversampfactor,
                 'interptype': interptype}

    if useglobalref:
        inputfreq = 1.0 / fmritr
        inputperiod = 1.0 * fmritr
        inputstarttime = 0.0
        inputvec = getglobalsignal(fmri_data_valid, temp_dict,
                                   includemask=internalincludemask_valid,
                                   excludemask=internalexcludemask_valid,
                                   rt_floatset=rt_floatset)
        preprocskip = 0
    else:
        if inputfreq is None:
            print('no regressor frequency specified - defaulting to 1/tr')
            inputfreq = 1.0 / fmritr
        if inputstarttime is None:
            print('no regressor start time specified - defaulting to 0.0')
            inputstarttime = 0.0
        inputperiod = 1.0 / inputfreq
        inputvec = tide_io.readvec(regressorfile)
    numreference = len(inputvec)
    print('regressor start time, end time, and step, {0}, {1}, '
          '{2}'.format(inputstarttime, inputstarttime + numreference * inputperiod,
                       inputperiod))

    if verbose:
        print('input vector length, {0}, input freq, {1}, input start time, '
              '{2}'.format(len(inputvec), inputfreq, inputstarttime))

    reference_x = np.arange(0.0, numreference) * inputperiod - (inputstarttime + offsettime)

    # Print out initial information
    if verbose:
        print('there are {0} points in the original regressor'.format(numreference))
        print('the timepoint spacing is {0}'.format(1.0 / inputfreq))
        print('the input timecourse start time is {0}'.format(inputstarttime))

    # generate the time axes
    skiptime = fmritr * (preprocskip + addedskip)
    print('first fMRI point is at {0} seconds relative to time origin'.format(skiptime))
    initial_fmri_x = np.arange(0.0, timepoints - addedskip) * fmritr + skiptime
    os_fmri_x = np.arange(0.0, (timepoints - addedskip) * oversampfactor - (
            oversampfactor - 1)) * oversamptr + skiptime

    if verbose:
        print(np.shape(os_fmri_x)[0])
        print(np.shape(initial_fmri_x)[0])

    # Clip the data
    if not useglobalref and False:
        clipstart = bisect.bisect_left(reference_x, os_fmri_x[0] - 2.0 * lagmin)
        clipend = bisect.bisect_left(reference_x, os_fmri_x[-1] + 2.0 * lagmax)
        print('clip indices=', clipstart, clipend, reference_x[clipstart],
              reference_x[clipend], os_fmri_x[0], os_fmri_x[-1])

    # generate the comparison regressor from the input timecourse
    # correct the output time points
    # check for extrapolation
    if os_fmri_x[0] < reference_x[0]:
        print('WARNING: extrapolating {0} seconds of data at beginning of '
              'timecourse'.format(os_fmri_x[0] - reference_x[0]))
    if os_fmri_x[-1] > reference_x[-1]:
        print('WARNING: extrapolating {0} seconds of data at end of '
              'timecourse'.format(os_fmri_x[-1] - reference_x[-1]))

    # invert the regressor if necessary
    if invertregressor:
        invertfac = -1.0
    else:
        invertfac = 1.0

    # detrend the regressor if necessary
    if dodetrend:
        reference_y = invertfac * tide_fit.detrend(inputvec[0:numreference],
                                                   demean=dodemean)
    else:
        reference_y = invertfac * (inputvec[0:numreference] - np.mean(inputvec[0:numreference]))

    # write out the reference regressor prior to filtering
    tide_io.writenpvecs(reference_y, prefix + '_reference_origres_prefilt.txt')

    # band limit the regressor if that is needed
    print('filtering to {0} band'.format(theprefilter.gettype()))
    reference_y_classfilter = theprefilter.apply(inputfreq, reference_y)
    reference_y = reference_y_classfilter

    # write out the reference regressor used
    tide_io.writenpvecs(tide_math.stdnormalize(reference_y),
                        prefix + '_reference_origres.txt')

    # filter the input data for antialiasing
    if antialias:
        if trapezoidalfftfilter:
            print('applying trapezoidal antialiasing filter')
            reference_y_filt = tide_filt.dolptrapfftfilt(inputfreq,
                                                         0.25 * fmrifreq,
                                                         0.5 * fmrifreq,
                                                         reference_y,
                                                         padlen=int(inputfreq * padseconds),
                                                         debug=debug)
        else:
            print('applying brickwall antialiasing filter')
            reference_y_filt = tide_filt.dolpfftfilt(inputfreq,
                                                     0.5 * fmrifreq,
                                                     reference_y,
                                                     padlen=int(inputfreq * padseconds),
                                                     debug=debug)
        reference_y = rt_floatset(reference_y_filt.real)

    warnings.filterwarnings('ignore', 'Casting*')

    if fakerun:
        return None

    # write out the resampled reference regressors
    if dodetrend:
        resampnonosref_y = tide_fit.detrend(
            tide_resample.doresample(reference_x, reference_y, initial_fmri_x,
                                     method=interptype), demean=dodemean)
        resampref_y = tide_fit.detrend(
            tide_resample.doresample(reference_x, reference_y, os_fmri_x,
                                     method=interptype), demean=dodemean)
    else:
        resampnonosref_y = tide_resample.doresample(reference_x, reference_y,
                                                    initial_fmri_x,
                                                    method=interptype)
        resampref_y = tide_resample.doresample(reference_x, reference_y,
                                               os_fmri_x, method=interptype)

    # prepare the temporal mask
    if bool(tmaskname):
        tmask_y = maketmask(tmaskname, reference_x, rt_floatset(reference_y))
        tmaskos_y = tide_resample.doresample(reference_x, tmask_y, os_fmri_x,
                                             method=interptype)
        tide_io.writenpvecs(tmask_y, prefix + '_temporalmask.txt')
        resampnonosref_y *= tmask_y
        thefit, R = tide_fit.mlregress(tmask_y, resampnonosref_y)
        resampnonosref_y -= thefit[0, 1] * tmask_y
        resampref_y *= tmaskos_y
        thefit, R = tide_fit.mlregress(tmaskos_y, resampref_y)
        resampref_y -= thefit[0, 1] * tmaskos_y

    if passes > 1:
        nonosrefname = '_reference_fmrires_pass1.txt'
        osrefname = '_reference_resampres_pass1.txt'
    else:
        nonosrefname = '_reference_fmrires.txt'
        osrefname = '_reference_resampres.txt'

    tide_io.writenpvecs(tide_math.stdnormalize(resampnonosref_y),
                        prefix + nonosrefname)
    tide_io.writenpvecs(tide_math.stdnormalize(resampref_y),
                        prefix + osrefname)
    timings.append(['End of reference prep', time.time(), None, None])

    corrtr = oversamptr
    if verbose:
        print('corrtr=', corrtr)

    numccorrlags = 2 * oversampfactor * (timepoints - addedskip) - 1
    corrscale = np.arange(0.0, numccorrlags) * corrtr - (numccorrlags * corrtr) / 2.0 + (oversampfactor - 0.5) * corrtr
    corrorigin = numccorrlags // 2 + 1
    lagmininpts = int((-lagmin / corrtr) - 0.5)
    lagmaxinpts = int((lagmax / corrtr) + 0.5)
    if verbose:
        print('corrorigin at point ', corrorigin, corrscale[corrorigin])
        print('corr range from {0} ({1}) to {2} '
              '({3})'.format(corrorigin - lagmininpts,
                             corrscale[corrorigin - lagmininpts],
                             corrorigin + lagmaxinpts,
                             corrscale[corrorigin + lagmaxinpts]))

    if savecorrtimes:
        tide_io.writenpvecs(corrscale[corrorigin-lagmininpts:corrorigin+lagmaxinpts],
                            prefix + '_corrtimes.txt')

    # allocate all of the data arrays
    tide_util.logmem('before main array allocation', file=memfile)
    if datatype == 'text':
        nativespaceshape = xsize
        nativearmodelshape = (xsize, armodelorder)
    elif datatype == 'cifti':
        nativespaceshape = (1, 1, 1, 1, numspatiallocs)
        nativearmodelshape = (1, 1, 1, armodelorder, numspatiallocs)
    else:
        nativespaceshape = (xsize, ysize, numslices)
        nativearmodelshape = (xsize, ysize, numslices, armodelorder)
    internalspaceshape = numspatiallocs
    internalarmodelshape = (numspatiallocs, armodelorder)
    internalvalidspaceshape = numvalidspatiallocs
    internalvalidarmodelshape = (numvalidspatiallocs, armodelorder)
    meanval = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagtimes = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagstrengths = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagsigma = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagmask = np.zeros(internalvalidspaceshape, dtype='uint16')
    R2 = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    outmaparray = np.zeros(internalspaceshape, dtype=rt_floattype)
    outarmodelarray = np.zeros(internalarmodelshape, dtype=rt_floattype)
    tide_util.logmem('after main array allocation', file=memfile)

    corroutlen = np.shape(corrscale[corrorigin - lagmininpts:corrorigin + lagmaxinpts])[0]
    if datatype == 'text':
        nativecorrshape = (xsize, corroutlen)
    else:
        if datatype == 'cifti':
            nativecorrshape = (1, 1, 1, corroutlen, numspatiallocs)
        else:
            nativecorrshape = (xsize, ysize, numslices, corroutlen)
    internalcorrshape = (numspatiallocs, corroutlen)
    internalvalidcorrshape = (numvalidspatiallocs, corroutlen)
    print('allocating memory for correlation arrays', internalcorrshape, internalvalidcorrshape)
    if sharedmem:
        corrout, _, _ = allocshared(internalvalidcorrshape, rt_floatset)
        gaussout, _, _ = allocshared(internalvalidcorrshape, rt_floatset)
        outcorrarray, _, _ = allocshared(internalcorrshape, rt_floatset)
    else:
        corrout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        gaussout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        outcorrarray = np.zeros(internalcorrshape, dtype=rt_floattype)
    tide_util.logmem('after correlation array allocation', file=memfile)

    if datatype == 'text':
        nativefmrishape = (xsize, np.shape(initial_fmri_x)[0])
    elif datatype == 'cifti':
        nativefmrishape = (1, 1, 1, np.shape(initial_fmri_x)[0], numspatiallocs)
    else:
        nativefmrishape = (xsize, ysize, numslices, np.shape(initial_fmri_x)[0])
    internalfmrishape = (numspatiallocs, np.shape(initial_fmri_x)[0])
    internalvalidfmrishape = (numvalidspatiallocs, np.shape(initial_fmri_x)[0])
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    tide_util.logmem('after lagtc array allocation', file=memfile)

    if passes > 1:
        if sharedmem:
            shiftedtcs, _, _ = allocshared(internalvalidfmrishape, rt_floatset)
            weights, _, _ = allocshared(internalvalidfmrishape, rt_floatset)
        else:
            shiftedtcs = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
            weights = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
        tide_util.logmem('after refinement array allocation', file=memfile)
    if sharedmem:
        outfmriarray, _, _ = allocshared(internalfmrishape, rt_floatset)
    else:
        outfmriarray = np.zeros(internalfmrishape, dtype=rt_floattype)

    # prepare for fast resampling
    padvalue = max((-lagmin, lagmax)) + 30.0
    # print('setting up fast resampling with padvalue =',padvalue)
    numpadtrs = int(padvalue // fmritr)
    padvalue = fmritr * numpadtrs
    genlagtc = tide_resample.fastresampler(reference_x, reference_y, padvalue=padvalue)

    # cycle over all voxels
    refine = True
    if verbose:
        print('refine is set to {0}'.format(refine))
    edgebufferfrac = max([edgebufferfrac, 2.0 / np.shape(corrscale)[0]])
    if verbose:
        print('edgebufferfrac set to {0}'.format(edgebufferfrac))

    fft_fmri_data = None
    sidelobe_dict = {}
    for thepass in range(1, passes + 1):
        # initialize the pass
        if passes > 1:
            print('\n\n*********************')
            print('Pass number {0}'.format(thepass))

        referencetc = tide_math.corrnormalize(resampref_y, dodetrend,
                                              windowfunc=windowfunc)
        nonosreferencetc = tide_math.corrnormalize(resampnonosref_y, dodetrend,
                                                   windowfunc=windowfunc)
        oversampfreq = oversampfactor / fmritr

        # Step -1 - check the regressor for periodic components in the passband
        dolagmod = True
        doreferencenotch = False
        if check_autocorrelation:
            print('checking reference regressor autocorrelation properties')
            lagmod = 1000.0
            lagindpad = corrorigin - 2 * np.max((lagmininpts, lagmaxinpts))
            acmininpts = lagmininpts + lagindpad
            acmaxinpts = lagmaxinpts + lagindpad
            thexcorr, _ = tide_corrpass.onecorrelation(
                referencetc,
                oversampfreq,
                corrorigin,
                acmininpts,
                acmaxinpts,
                theprefilter,
                referencetc,
                temp_dict)
            outputarray = np.asarray([corrscale[corrorigin - acmininpts:corrorigin + acmaxinpts], thexcorr])
            tide_io.writenpvecs(outputarray, prefix + '_referenceautocorr_pass' + str(thepass) + '.txt')
            thelagthresh = np.max((abs(lagmin), abs(lagmax)))
            theampthresh = 0.1
            print('searching for sidelobes with amplitude > {0} with abs(lag) '
                  '< {1}s'.format(theampthresh, thelagthresh))
            sidelobetime, sidelobeamp = tide_corr.autocorrcheck(
                corrscale[corrorigin - acmininpts:corrorigin + acmaxinpts],
                thexcorr,
                acampthresh=theampthresh,
                aclagthresh=thelagthresh,
                prewindow=bool(windowfunc),
                dodetrend=dodetrend)
            if sidelobetime is not None:
                passsuffix = '_pass' + str(thepass + 1)
                sidelobe_dict['acsidelobelag' + passsuffix] = sidelobetime
                despeckle_thresh = np.max([despeckle_thresh, sidelobetime / 2.0])
                sidelobe_dict['acsidelobeamp' + passsuffix] = sidelobeamp
                print('\n\nWARNING: autocorrcheck found bad sidelobe at {0} '
                      'seconds ({1}Hz)...'.format(sidelobetime, 1.0 / sidelobetime))
                tide_io.writenpvecs(np.array([sidelobetime]),
                                    prefix + '_autocorr_sidelobetime' + passsuffix + '.txt')
                if fix_autocorrelation:
                    print('Removing sidelobe')
                    if dolagmod:
                        print('subjecting lag times to modulus')
                        lagmod = sidelobetime / 2.0
                    if doreferencenotch:
                        print('removing spectral component at sidelobe frequency')
                        acstopfreq = 1.0 / sidelobetime
                        acfixfilter = tide_filt.noncausalfilter(debug=debug)
                        acfixfilter.settype('arb_stop')
                        acfixfilter.setarb(acstopfreq * 0.9, acstopfreq * 0.95,
                                           acstopfreq * 1.05, acstopfreq * 1.1)
                        cleaned_referencetc = tide_math.stdnormalize(acfixfilter.apply(fmrifreq, referencetc))
                        cleaned_nonosreferencetc = tide_math.stdnormalize(acfixfilter.apply(fmrifreq, nonosreferencetc))
                        tide_io.writenpvecs(cleaned_referencetc,
                                            prefix + '_cleanedreference_pass' + str(thepass) + '.txt')
                else:
                    cleaned_referencetc = 1.0 * referencetc
                    cleaned_nonosreferencetc = 1.0 * nonosreferencetc
            else:
                print('no sidelobes found in range')
                cleaned_referencetc = 1.0 * referencetc
                cleaned_nonosreferencetc = 1.0 * nonosreferencetc
        else:
            cleaned_referencetc = 1.0 * referencetc
            cleaned_nonosreferencetc = 1.0 * nonosreferencetc

        # Step 0 - estimate significance
        if numestreps > 0:
            timings.append(['Significance estimation start, pass {0}'.format(thepass),
                            time.time(), None, None])
            print('\n\nSignificance estimation, pass {0}'.format(thepass))
            if verbose:
                print('calling getNullDistributionData with args:', oversampfreq, fmritr, corrorigin, lagmininpts,
                      lagmaxinpts)
            if memprofile:
                getNullDistributionData_func = profile(tide_nullcorr.getNullDistributionData, precision=2)
            else:
                tide_util.logmem('before getnulldistristributiondata', file=memfile)
                getNullDistributionData_func = tide_nullcorr.getNullDistributionData
            corrdistdata = getNullDistributionData_func(
                cleaned_referencetc,
                corrscale,
                theprefilter,
                oversampfreq,
                corrorigin,
                lagmininpts,
                lagmaxinpts,
                temp_dict,
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype
                )
            tide_io.writenpvecs(corrdistdata,
                                '{0}_corrdistdata_pass{1}.txt'.format(prefix, thepass))

            # calculate percentiles for the crosscorrelation from the distribution data
            sighistlen = 1000
            thepercentiles = np.array([0.95, 0.99, 0.995, 0.999])
            thepvalnames = []
            for thispercentile in thepercentiles:
                thepvalnames.append("{:.3f}".format(1.0 - thispercentile).replace('.', 'p'))

            pcts, pcts_fit, sigfit = tide_stats.sigFromDistributionData(
                corrdistdata,
                sighistlen,
                thepercentiles,
                twotail=bipolar,
                displayplots=displayplots,
                nozero=nohistzero,
                dosighistfit=dosighistfit)
            if ampthreshfromsig:
                print('setting ampthresh to the p<{0:.3f} threshold'.format(1.0 - thepercentiles[0]))
                ampthresh = pcts[2]
            tide_stats.printthresholds(pcts, thepercentiles,
                                       ('Crosscorrelation significance '
                                        'thresholds from data:'))
            if dosighistfit:
                tide_stats.printthresholds(pcts_fit, thepercentiles,
                                           ('Crosscorrelation significance '
                                            'thresholds from fit:'))
                tide_stats.makeandsavehistogram(corrdistdata, sighistlen, 0,
                                                prefix + '_nullcorrelationhist_pass' + str(thepass),
                                                displaytitle='Null correlation histogram, pass{0}'.format(thepass),
                                                displayplots=displayplots, refine=False)
            del corrdistdata
            timings.append(['Significance estimation end, pass {0}'.format(thepass),
                            time.time(), numestreps, 'repetitions'])

        # Step 1 - Correlation step
        print('\n\nCorrelation calculation, pass ' + str(thepass))
        timings.append(['Correlation calculation start, pass {0}'.format(thepass),
                        time.time(), None, None])
        if memprofile:
            correlationpass_func = profile(tide_corrpass.correlationpass, precision=2)
        else:
            tide_util.logmem('before correlationpass', file=memfile)
            correlationpass_func = tide_corrpass.correlationpass
        voxelsprocessed_cp, theglobalmaxlist = correlationpass_func(
            fmri_data_valid[:, addedskip:],
            fft_fmri_data,
            cleaned_referencetc,
            initial_fmri_x,
            os_fmri_x,
            fmritr,
            corrorigin,
            lagmininpts,
            lagmaxinpts,
            corrout,
            meanval,
            theprefilter,
            temp_dict,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype
            )
        for i in range(len(theglobalmaxlist)):
            theglobalmaxlist[i] = corrscale[theglobalmaxlist[i]]
        tide_stats.makeandsavehistogram(
            np.asarray(theglobalmaxlist),
            len(corrscale),
            0,
            prefix + '_globallaghist_pass' + str(thepass),
            displaytitle='lagtime histogram',
            displayplots=displayplots,
            therange=(corrscale[0], corrscale[-1]),
            refine=False)
        timings.append(['Correlation calculation end, pass {0}'.format(thepass),
                        time.time(), voxelsprocessed_cp, 'voxels'])

        # Step 2 - correlation fitting and time lag estimation
        print('\n\nTime lag estimation pass {0}'.format(thepass))
        timings.append(['Time lag estimation start, pass {0}'.format(thepass),
                        time.time(), None, None])

        if memprofile:
            fitcorr_func = profile(tide_corrfit.fitcorr, precision=2)
        else:
            tide_util.logmem('before fitcorr', file=memfile)
            fitcorr_func = tide_corrfit.fitcorr
        voxelsprocessed_fc = fitcorr_func(
            genlagtc,
            initial_fmri_x,
            lagtc,
            slicesize,
            corrscale[corrorigin - lagmininpts:corrorigin + lagmaxinpts],
            lagmask,
            lagtimes,
            lagstrengths,
            lagsigma,
            corrout,
            meanval,
            gaussout,
            R2,
            temp_dict,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype
            )
        timings.append(['Time lag estimation end, pass {0}'.format(thepass),
                        time.time(), voxelsprocessed_fc, 'voxels'])

        # Step 2b - Correlation time despeckle
        if despeckle_passes > 0:
            print('\n\nCorrelation despeckling pass {0}'.format(thepass))
            print('\tUsing despeckle_thresh = {0}'.format(despeckle_thresh))
            timings.append(['Correlation despeckle start, pass {0}'.format(thepass),
                            time.time(), None, None])

            # find lags that are very different from their neighbors, and refi
            # starting at the median lag for the point
            voxelsprocessed_fc_ds = 0
            for despecklepass in range(despeckle_passes):
                print('\n\nCorrelation despeckling subpass ' + str(despecklepass + 1))
                outmaparray *= 0.0
                outmaparray[validvoxels] = eval('lagtimes')[:]
                medianlags = ndimage.median_filter(outmaparray.reshape(nativespaceshape), 3).reshape(numspatiallocs)
                initlags = \
                    np.where(np.abs(outmaparray - medianlags) > despeckle_thresh,
                             medianlags, -1000000.0)[validvoxels]
                if len(initlags) > 0:
                    voxelsprocessed_fc_ds += fitcorr_func(
                        genlagtc,
                        initial_fmri_x,
                        lagtc,
                        slicesize,
                        corrscale[corrorigin - lagmininpts:corrorigin + lagmaxinpts],
                        lagmask,
                        lagtimes,
                        lagstrengths,
                        lagsigma,
                        corrout,
                        meanval,
                        gaussout,
                        R2,
                        temp_dict,
                        initiallags=initlags,
                        rt_floatset=rt_floatset,
                        rt_floattype=rt_floattype)
                else:
                    print('Nothing left to do! Terminating despeckling')
                    break

            print('\n\n{0}voxels despeckled in passes'.format(voxelsprocessed_fc_ds, despeckle_passes))
            timings.append(['Correlation despeckle end, pass {0}'.format(thepass),
                            time.time(), voxelsprocessed_fc_ds, 'voxels'])

        # Step 3 - regressor refinement for next pass
        if thepass < passes:
            print('\n\nRegressor refinement, pass' + str(thepass))
            timings.append(['Regressor refinement start, pass {0}'.format(thepass),
                            time.time(), None, None])
            if refineoffset:
                peaklag, peakheight, peakwidth = tide_stats.gethistprops(lagtimes[np.where(lagmask > 0)],
                                                                         histlen)
                offsettime = peaklag
                offsettime_total += peaklag
                print('offset time set to {0}, total is '
                      '{1}'.format(offsettime, offsettime_total))

            # regenerate regressor for next pass
            if memprofile:
                refineregressor_func = profile(tide_refine.refineregressor,
                                               precision=2)
            else:
                tide_util.logmem('before refineregressor', file=memfile)
                refineregressor_func = tide_refine.refineregressor
            voxelsprocessed_rr, outputdata, refinemask = refineregressor_func(
                fmri_data_valid[:, :],
                fmritr,
                shiftedtcs,
                weights,
                thepass,
                lagstrengths,
                lagtimes,
                lagsigma,
                R2,
                theprefilter,
                temp_dict,
                padtrs=numpadtrs,
                includemask=internalincludemask_valid,
                excludemask=internalexcludemask_valid,
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype)
            normoutputdata = tide_math.stdnormalize(theprefilter.apply(fmrifreq, outputdata))
            tide_io.writenpvecs(normoutputdata,
                                '{0}_refinedregressor_pass{1}.txt'.format(prefix, thepass))

            if dodetrend:
                resampnonosref_y = tide_fit.detrend(
                    tide_resample.doresample(initial_fmri_x, normoutputdata, initial_fmri_x,
                                             method=interptype),
                    demean=dodemean)
                resampref_y = tide_fit.detrend(tide_resample.doresample(initial_fmri_x, normoutputdata, os_fmri_x,
                                                                        method=interptype),
                                               demean=dodemean)
            else:
                resampnonosref_y = tide_resample.doresample(initial_fmri_x, normoutputdata, initial_fmri_x,
                                                            method=interptype)
                resampref_y = tide_resample.doresample(initial_fmri_x, normoutputdata, os_fmri_x,
                                                       method=interptype)
            if bool(tmaskname):
                resampnonosref_y *= tmask_y
                thefit, R = tide_fit.mlregress(tmask_y, resampnonosref_y)
                resampnonosref_y -= thefit[0, 1] * tmask_y
                resampref_y *= tmaskos_y
                thefit, R = tide_fit.mlregress(tmaskos_y, resampref_y)
                resampref_y -= thefit[0, 1] * tmaskos_y

            # reinitialize lagtc for resampling
            genlagtc = tide_resample.fastresampler(initial_fmri_x, normoutputdata, padvalue=padvalue)
            nonosrefname = '_reference_fmrires_pass{0}.txt'.format(thepass+1)
            osrefname = '_reference_resampres_pass{0}.txt'.format(thepass+1)
            tide_io.writenpvecs(tide_math.stdnormalize(resampnonosref_y), prefix + nonosrefname)
            tide_io.writenpvecs(tide_math.stdnormalize(resampref_y), prefix + osrefname)
            timings.append(['Regressor refinement end, pass {0}'.format(thepass),
                            time.time(), voxelsprocessed_rr, 'voxels'])

    # Post refinement step 0 - Wiener deconvolution
    if dodeconv:
        timings.append(['Wiener deconvolution start', time.time(), None, None])
        print('\n\nWiener deconvolution')
        reportstep = 1000

        # now allocate the arrays needed for Wiener deconvolution
        wienerdeconv = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        wpeak = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)

        if memprofile:
            wienerpass_func = profile(tide_wiener.wienerpass, precision=2)
        else:
            tide_util.logmem('before wienerpass', file=memfile)
            wienerpass_func = tide_wiener.wienerpass
        voxelsprocessed_wiener = wienerpass_func(
            numspatiallocs,
            reportstep,
            fmri_data_valid,
            threshval,
            temp_dict,
            wienerdeconv,
            wpeak,
            resampref_y,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype
            )
        timings.append(['Wiener deconvolution end', time.time(),
                        voxelsprocessed_wiener, 'voxels'])

    # Post refinement step 1 - GLM fitting to remove moving signal
    if doglmfilt or doprewhiten:
        timings.append(['GLM filtering start', time.time(), None, None])
        if doglmfilt:
            print('\n\nGLM filtering')
        if doprewhiten:
            print('\n\nPrewhitening')
        reportstep = 1000
        if (gausssigma > 0) or (glmsourcefile is not None):
            if glmsourcefile is not None:
                print('reading in {0} for GLM filter, please wait'.format(glmsourcefile))
                if datatype == 'text':
                    nim_data = tide_io.readvecs(glmsourcefile)
                else:
                    nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(glmsourcefile)
            else:
                print('rereading {0} for GLM filter, please wait'.format(in_file))
                if datatype == 'text':
                    nim_data = tide_io.readvecs(in_file)
                else:
                    nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(in_file)
            fmri_data_valid = (nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1])[validvoxels, :] + 0.0

            # move fmri_data_valid into shared memory
            if sharedmem:
                print('moving fmri data to shared memory')
                timings.append(['Start moving fmri_data to shared memory',
                                time.time(), None, None])
                if memprofile:
                    numpy2shared_func = profile(numpy2shared, precision=2)
                else:
                    tide_util.logmem('before movetoshared (glm)', file=memfile)
                    numpy2shared_func = numpy2shared
                (fmri_data_valid,
                 fmri_data_valid_shared,
                 fmri_data_valid_shared_shape) = numpy2shared_func(fmri_data_valid,
                                                                   rt_floatset)
                timings.append(['End moving fmri_data to shared memory',
                                time.time(), None, None])
            del nim_data

        # now allocate the arrays needed for GLM filtering
        meanvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        rvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        r2value = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitNorm = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitcoff = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        if sharedmem:
            datatoremove, _, _ = allocshared(internalvalidfmrishape, rt_outfloatset)
            filtereddata, _, _ = allocshared(internalvalidfmrishape, rt_outfloatset)
        else:
            datatoremove = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
            filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)

        if doprewhiten:
            prewhiteneddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
            arcoffs = np.zeros(internalvalidarmodelshape, dtype=rt_outfloattype)

        if memprofile:
            memcheckpoint('about to start glm noise removal...')
        else:
            tide_util.logmem('before glm', file=memfile)

        if preservefiltering:
            for i in range(len(validvoxels)):
                fmri_data_valid[i] = theprefilter.apply(fmrifreq, fmri_data_valid[i])

        if memprofile:
            glmpass_func = profile(tide_glmpass.glmpass, precision=2)
        else:
            tide_util.logmem('before glmpass', file=memfile)
            glmpass_func = tide_glmpass.glmpass

        voxelsprocessed_glm = glmpass_func(
            numvalidspatiallocs,
            fmri_data_valid,
            threshval,
            lagtc,
            meanvalue,
            rvalue,
            r2value,
            fitcoff,
            fitNorm,
            datatoremove,
            filtereddata,
            reportstep=reportstep,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            addedskip=addedskip,
            mp_chunksize=mp_chunksize,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype
            )
        del fmri_data_valid

        timings.append(['GLM filtering end, pass {0}'.format(thepass),
                        time.time(), voxelsprocessed_glm, 'voxels'])
        if memprofile:
            memcheckpoint('...done')
        else:
            tide_util.logmem('after glm filter', file=memfile)

        if doprewhiten:
            arcoff_ref = pacf_yw(resampref_y, nlags=armodelorder)[1:]
            print('\nAR coefficient(s) for reference waveform: {0}'.format(arcoff_ref))
            resampref_y_pw = rt_floatset(prewhiten(resampref_y, arcoff_ref))
        else:
            resampref_y_pw = rt_floatset(resampref_y)

        if windowfunc:
            referencetc_pw = tide_math.stdnormalize(
                tide_filt.windowfunction(np.shape(resampref_y_pw)[0],
                                         type=windowfunc) * tide_fit.detrend(
                    tide_math.stdnormalize(resampref_y_pw))) / np.shape(resampref_y_pw)[0]
        else:
            referencetc_pw = tide_math.stdnormalize(tide_fit.detrend(
                tide_math.stdnormalize(resampref_y_pw))) / np.shape(resampref_y_pw)[0]

        print('')
        if displayplots:
            fig = figure()
            ax = fig.add_subplot(111)
            ax.set_title('initial and prewhitened reference')
            plot(os_fmri_x, referencetc, os_fmri_x, referencetc_pw)
    else:
        # get the original data to calculate the mean
        print('rereading {0} for GLM filter, please wait'.format(in_file))
        if datatype == 'text':
            nim_data = tide_io.readvecs(in_file)
        else:
            nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(in_file)
        fmri_data = nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1]
        meanvalue = np.mean(fmri_data, axis=1)

    # Post refinement step 2 - prewhitening
    if doprewhiten:
        print('Step 3 - reprocessing prewhitened data')
        timings.append(['Step 3 start', time.time(), None, None])
        _, _ = tide_corrpass.correlationpass(
            prewhiteneddata,
            fft_fmri_data,
            referencetc_pw,
            initial_fmri_x,
            os_fmri_x,
            fmritr,
            corrorigin,
            lagmininpts,
            lagmaxinpts,
            corrout,
            meanval,
            theprefilter,
            temp_dict,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype
            )

    # Post refinement step 3 - make and save interesting histograms
    timings.append(['Start saving histograms', time.time(), None, None])
    tide_stats.makeandsavehistogram(lagtimes[np.where(lagmask > 0)], histlen, 0, prefix + '_laghist',
                                    displaytitle='lagtime histogram', displayplots=displayplots,
                                    refine=False)
    tide_stats.makeandsavehistogram(lagstrengths[np.where(lagmask > 0)], histlen, 0,
                                    prefix + '_strengthhist',
                                    displaytitle='lagstrength histogram', displayplots=displayplots,
                                    therange=(0.0, 1.0))
    tide_stats.makeandsavehistogram(lagsigma[np.where(lagmask > 0)], histlen, 1,
                                    prefix + '_widthhist',
                                    displaytitle='lagsigma histogram', displayplots=displayplots)
    if doglmfilt:
        tide_stats.makeandsavehistogram(r2value[np.where(lagmask > 0)], histlen, 1, prefix + '_Rhist',
                                        displaytitle='correlation R2 histogram',
                                        displayplots=displayplots)
    timings.append(['Finished saving histograms', time.time(), None, None])

    # Post refinement step 4 - save out all of the important arrays to nifti files
    # write out the options used
    tide_io.writedict(temp_dict, '{0}_options.txt'.format(prefix))

    if datatype == 'cifti':
        outsuffix3d = '.dscalar'
        outsuffix4d = '.dtseries'
    else:
        outsuffix3d = ''
        outsuffix4d = ''

    # do ones with one time point first
    timings.append(['Start saving maps', time.time(), None, None])
    if not datatype == 'text':
        theheader = nim_hdr
        if datatype == 'cifti':
            theheader['intent_code'] = 3006
        else:
            theheader['dim'][0] = 3
            theheader['dim'][4] = 1

    # first generate the MTT map
    MTT = np.square(lagsigma) - (acwidth * acwidth)
    MTT = np.where(MTT > 0.0, np.sqrt(MTT), 0.0)

    for mapname in ['lagtimes', 'lagstrengths', 'R2', 'lagsigma', 'lagmask', 'MTT']:
        if memprofile:
            memcheckpoint('about to write ' + mapname)
        else:
            tide_util.logmem('about to write ' + mapname, file=memfile)
        outmaparray[:] = 0.0
        outmaparray[validvoxels] = eval(mapname)[:]
        if datatype == 'text':
            tide_io.writenpvecs(outmaparray.reshape(nativespaceshape, 1),
                                '{0}_{1}{2}.txt'.format(prefix, mapname, outsuffix3d))
        else:
            tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, thesizes,
                                '{0}_{1}{2}'.format(prefix, mapname, outsuffix3d))

    if doglmfilt:
        for mapname, mapsuffix in [('rvalue', 'fitR'),
                                   ('r2value', 'fitR2'),
                                   ('meanvalue', 'mean'),
                                   ('fitcoff', 'fitcoff'),
                                   ('fitNorm', 'fitNorm')]:
            if memprofile:
                memcheckpoint('about to write ' + mapname)
            else:
                tide_util.logmem('about to write ' + mapname, file=memfile)
            outmaparray[:] = 0.0
            outmaparray[validvoxels] = eval(mapname)[:]
            if datatype == 'text':
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    '{0}_{1}{2}.txt'.format(prefix, mapsuffix, outsuffix3d))
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, thesizes,
                                    '{0}_{1}{2}'.format(prefix, mapsuffix, outsuffix3d))
        del rvalue, r2value, meanvalue, fitcoff, fitNorm
    else:
        for mapname, mapsuffix in [('meanvalue', 'mean')]:
            if memprofile:
                memcheckpoint('about to write {0}'.format(mapname))
            else:
                tide_util.logmem('about to write {0}'.format(mapname), file=memfile)
            outmaparray[:] = 0.0
            outmaparray = eval(mapname)[:]
            if datatype == 'text':
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    '{0}_{1}{2}.txt'.format(prefix, mapsuffix, outsuffix3d))
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, thesizes,
                                    '{0}_{1}{2}'.format(prefix, mapsuffix, outsuffix3d))
        del meanvalue

    if numestreps > 0:
        for i in range(0, len(thepercentiles)):
            pmask = np.where(np.abs(lagstrengths) > pcts[i], lagmask, 0 * lagmask)
            if dosighistfit:
                tide_io.writenpvecs(sigfit, '{0}_sigfit.txt'.format(prefix))
            tide_io.writenpvecs(np.array([pcts[i]]),
                                '{0}_p_lt_{1}_thresh.txt'.format(prefix, thepvalnames[i]))
            outmaparray[:] = 0.0
            outmaparray[validvoxels] = pmask[:]
            if datatype == 'text':
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    '{0}_p_lt_{1}_mask{2}.txt'.format(prefix, thepvalnames[i], outsuffix3d))
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape),
                                    theheader, thesizes,
                                    '{0}_p_lt_{1}_mask{2}'.format(prefix, thepvalnames[i], outsuffix3d))

    if passes > 1:
        outmaparray[:] = 0.0
        outmaparray[validvoxels] = refinemask[:]
        if datatype == 'text':
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                '{0}_lagregressor{1}.txt'.format(prefix, outsuffix4d))
        else:
            tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, thesizes,
                                '{0}_refinemask{1}'.format(prefix, outsuffix4d))
        del refinemask

    # clean up arrays that will no longer be needed
    del lagtimes, lagstrengths, lagsigma, R2, lagmask

    # now do the ones with other numbers of time points
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = gaussout[:, :]
    if datatype != 'text':
        theheader = nim_hdr
        theheader['toffset'] = corrscale[corrorigin - lagmininpts]
        theheader['pixdim'][4] = corrtr
        if datatype == 'cifti':
            theheader['intent_code'] = 3002
        else:
            theheader['dim'][4] = np.shape(corrscale)[0]
        tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                            thesizes, prefix + '_gaussout' + outsuffix4d)
    else:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                            prefix + '_gaussout' + outsuffix4d + '.txt')
    del gaussout

    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = corrout[:, :]
    if datatype == 'text':
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                            prefix + '_corrout' + outsuffix4d + '.txt')
    else:
        tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader, thesizes,
                            prefix + '_corrout' + outsuffix4d)
    del corrout

    if saveprewhiten:
        outarmodelarray[validvoxels, :] = arcoffs[:, :]
        outfmriarray[validvoxels, :] = prewhiteneddata[:, :]
        if datatype != 'text':
            theheader = nim.header
            theheader['toffset'] = 0.0
            if datatype == 'cifti':
                theheader['intent_code'] = 3002
            else:
                theheader['dim'][4] = armodelorder
            tide_io.savetonifti(outarmodelarray.reshape(nativearmodelshape), theheader, thesizes,
                                prefix + '_arN' + outsuffix4d)
            tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, thesizes,
                                prefix + '_prewhiteneddata' + outsuffix4d)
        else:
            tide_io.writenpvecs(outarmodelarray.reshape(nativearmodelshape),
                                prefix + '_arN' + outsuffix4d + '.txt')
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                prefix + '_prewhiteneddata' + outsuffix4d + '.txt')
        del arcoffs
        del prewhiteneddata

    if datatype != 'text':
        theheader = nim_hdr
        theheader['pixdim'][4] = fmritr
        theheader['toffset'] = 0.0
        if datatype == 'cifti':
            theheader['intent_code'] = 3002
        else:
            theheader['dim'][4] = np.shape(initial_fmri_x)[0]

    if savelagregressors:
        outfmriarray[validvoxels, :] = lagtc[:, :]
        if datatype == 'text':
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                prefix + '_lagregressor' + outsuffix4d + '.txt')
        else:
            tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, thesizes,
                                prefix + '_lagregressor' + outsuffix4d)
        del lagtc

    if passes > 1:
        if savelagregressors:
            outfmriarray[validvoxels, :] = shiftedtcs[:, :]
            if datatype == 'text':
                tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                    prefix + '_shiftedtcs' + outsuffix4d + '.txt')
            else:
                tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, thesizes,
                                    prefix + '_shiftedtcs' + outsuffix4d)
        del shiftedtcs

    if doglmfilt and saveglmfiltered:
        del datatoremove
        outfmriarray[validvoxels, :] = filtereddata[:, :]
        if datatype == 'text':
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                prefix + '_filtereddata' + outsuffix4d + '.txt')
        else:
            tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, thesizes,
                                prefix + '_filtereddata' + outsuffix4d)
        del filtereddata

    timings.append(['Finished saving maps', time.time(), None, None])
    print('done')

    if displayplots:
        show()
    timings.append(['Done', time.time(), None, None])

    # Post refinement step 5 - process and save timing information
    nodeline = 'Processed on {0}'.format(platform.node())
    tide_util.proctiminginfo(timings, outputfile=prefix + '_runtimings.txt',
                             extraheader=nodeline)


def _main(argv=None):
    """
    Compile arguments for rapidtide workflow.

    TODO: Move post-parsing logic into workflow.
    """
    args = vars(_get_parser().parse_args(argv))

    # Additional argument parsing not handled by argparse
    if args['arbvec'] is not None:
        if len(args['arbvec']) not in [2, 4]:
            raise ValueError("Argument '--arb' (or '-F') must be either two "
                             "or four floats.")

    args['offsettime_total'] = -1 * args['offsettime']

    if args['saveprewhiten'] is True:
        args['doprewhiten'] = True

    reg_ref_used = ((args['lagminthresh'] != 0.5) or
                    (args['lagmaxthresh'] != 5.) or
                    (args['ampthresh'] != 0.3) or
                    (args['sigmathresh'] != 100.) or
                    (args['refineoffset']))
    if reg_ref_used and args['passes'] == 1:
        args['passes'] = 2

    if args['ampthresh'] != 100.:
        args['ampthreshfromsig'] = False
    else:
        args['ampthreshfromsig'] = True

    if args['despeckle_thresh'] != 5 and args['despeckle_passes'] == 0:
        args['despeckle_passes'] = 1

    if args['zerooutbadfit']:
        args['nohistzero'] = False
    else:
        args['nohistzero'] = True

    if args['fixeddelayvalue'] is not None:
        args['fixdelay'] = True
        args['lag_extrema'] = (args['fixeddelayvalue'] - 10.0,
                               args['fixeddelayvalue'] + 10.0)
    else:
        args['fixdelay'] = False

    if args['in_file'].endswith('txt') and args['realtr'] == 'auto':
        raise ValueError('Either --datatstep or --datafreq must be provided '
                         'if data file is a text file.')

    if args['realtr'] != 'auto':
        fmri_tr = args['realtr']
    else:
        fmri_tr = nib.load(args['in_file']).header.get_zooms()[3]
        args['realtr'] = fmri_tr

    if args['inputfreq'] == 'auto':
        args['inputfreq'] = 1. / fmri_tr

    args['usebutterworthfilter'] = bool(args['butterorder'])

    if args['venousrefine']:
        print('WARNING: Using "venousrefine" macro. Overriding any affected '
              'arguments.')
        args['lagminthresh'] = 2.5
        args['lagmaxthresh'] = 6.
        args['ampthresh'] = 0.5
        args['lagmaskside'] = 'upper'

    if args['nirs']:
        print('WARNING: Using "nirs" macro. Overriding any affected '
              'arguments.')
        args['nothresh'] = False
        args['preservefiltering'] = True
        args['refineprenorm'] = 'var'
        args['ampthresh'] = 0.7
        args['lagmaskthresh'] = 0.1

    # write out the command used
    temp_args = {k: str(v) for k, v in args.items()}
    with open(args['prefix']+'_call.json', 'w') as fo:
        json.dump(temp_args, fo, indent=4, sort_keys=True)

    rapidtide_workflow(**args)


if __name__ == '__main__':
    _main()
