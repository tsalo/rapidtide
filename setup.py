#!/bin/env python
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import subprocess
import re
import sys
import versioneer


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# Write version number out to VERSION file
version = versioneer.get_version()
with open(path.join(here, 'VERSION'), "w", encoding='utf-8') as f:
    f.write(version)


addtidepool = True

modules_list = ['rapidtide/miscmath',
                'rapidtide/correlate',
                'rapidtide/filter',
                'rapidtide/fit',
                'rapidtide/io',
                'rapidtide/resample',
                'rapidtide/stats',
                'rapidtide/util',
                'rapidtide/multiproc',
                'rapidtide/calcnullsimfunc',
                'rapidtide/calcsimfunc',
                'rapidtide/calccoherence',
                'rapidtide/simfuncfit',
                'rapidtide/peakeval',
                'rapidtide/helper_classes',
                'rapidtide/glmpass',
                'rapidtide/dlfilter',
                'rapidtide/wiener',
                'rapidtide/refine',
                'rapidtide/_version',
                'rapidtide/workflows/parser_funcs']

if addtidepool:
    modules_list.append('rapidtide/tidepoolTemplate')
    modules_list.append('rapidtide/tidepoolTemplate_alt')
    modules_list.append('rapidtide/OrthoImageItem')

script_list = ['rapidtide/scripts/rapidtide2x_legacy',
               'rapidtide/scripts/rapidtide',
               'rapidtide/scripts/rapidtide2std',
               'rapidtide/scripts/happy2std',
               'rapidtide/scripts/showxcorr',
               'rapidtide/scripts/aligntcs',
               'rapidtide/scripts/plethquality',
               'rapidtide/scripts/showxcorrx',
               'rapidtide/scripts/ccorrica',
               'rapidtide/scripts/showstxcorr',
               'rapidtide/scripts/resamp1tc',
               'rapidtide/scripts/resamplenifti',
               'rapidtide/scripts/simdata',
               'rapidtide/scripts/fingerprint',
               'rapidtide/scripts/pixelcomp',
               'rapidtide/scripts/spatialdecomp',
               'rapidtide/scripts/temporaldecomp',
               'rapidtide/scripts/spectrogram',
               'rapidtide/scripts/showtc',
               'rapidtide/scripts/histtc',
               'rapidtide/scripts/filtnifti',
               'rapidtide/scripts/filttc',
               'rapidtide/scripts/showxy',
               'rapidtide/scripts/spatialfit',
               'rapidtide/scripts/glmfilt',
               'rapidtide/scripts/histnifti',
               'rapidtide/scripts/atlastool',
               'rapidtide/scripts/atlasaverage',
               'rapidtide/scripts/linfit',
               'rapidtide/scripts/happy_legacy',
               'rapidtide/scripts/happy',
               'rapidtide/scripts/happywarp',
               'rapidtide/scripts/applydlfilter',
               'rapidtide/scripts/threeD',
               'rapidtide/scripts/endtidalproc',
               'rapidtide/scripts/synthASL',
               'rapidtide/scripts/tcfrom3col',
               'rapidtide/scripts/physiofreq',
               'rapidtide/scripts/rapidtide_dispatcher',
               # 'rapidtide/scripts/endtidalproc',
               'rapidtide/scripts/showhist',
               'rapidtide/scripts/rapidtide_dispatcher']

if addtidepool:
    script_list.append('rapidtide/scripts/tidepool')


setup(
    name='rapidtide',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    description='Tools for performing correlation analysis on fMRI data.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/bbfrederick/rapidtide',

    # Author details
    author="Blaise Frederick",
    author_email="bbfrederick@mclean.harvard.edu",

    # Choose your license
    license='Apache Software License',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords=['fMRI', 'correlation', 'RIPTiDe', 'noise'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    py_modules=modules_list,

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=["numpy>=1.16",
                        'scipy',
                        'pandas',
                        'scikit-image',
                        'scikit-learn',
                        'nibabel',
                        'nilearn',
                        'matplotlib',
                        'pyfftw',
                        'pyqtgraph>=0.11.0',
                        'statsmodels'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'jit': ['numba'],
        'doc': [
            'sphinx>=1.5.3',
            'sphinx_rtd_theme',
            'sphinx-argparse',
            'sphinx-gallery',
            'm2r',
            'recommonmark',
            ],
        'tests': [
            'codecov',
            'coverage',
            'coveralls',
            'flake8',
            'pytest',
            'pytest-cov'
        ],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    include_package_data=True,
    zip_safe=False,
    package_data={
        'reference': ['reference/*.txt',
                      'reference/*.nii.gz',
                      ],
        'testdata': ['tests/testdata/*.txt',
                      ],
        'testtargets': ['tests/testtargets/*.txt',
                      'tests/testtargets/*.nii.gz',
                      ],
        'tmpdata': ['tests/tmp/.placeholder.txt',
                      ],
    },

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    scripts=script_list,
)
