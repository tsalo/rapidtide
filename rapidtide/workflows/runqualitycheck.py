#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2024-2025 Blaise Frederick
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
import argparse

import rapidtide.io as tide_io
import rapidtide.qualitycheck as tide_quality


def _get_parser():
    """
    Argument parser for runqualitycheck
    """
    parser = argparse.ArgumentParser(
        prog="runqualitycheck",
        description=("Run a quality check on a rapidtide dataset."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputfileroot",
        type=str,
        help="The root of the rapidtide dataset name (without the underscore.)",
    )

    # add optional arguments
    parser.add_argument(
        "--graymaskspec",
        metavar="MASK[:VALSPEC]",
        type=str,
        help="The name of a gray matter mask that matches the input dataset. If VALSPEC is given, only voxels "
        "with integral values listed in VALSPEC are used.  If using an aparc+aseg file, set to APARC_GRAY.",
        default=None,
    )
    parser.add_argument(
        "--whitemaskspec",
        metavar="MASK[:VALSPEC]",
        type=str,
        help="The name of a white matter mask that matches the input dataset. If VALSPEC is given, only voxels "
        "with integral values listed in VALSPEC are used.  If using an aparc+aseg file, set to APARC_WHITE.",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Output additional debugging information."),
        default=False,
    )

    return parser


def runqualitycheck(args):
    resultsdict = tide_quality.qualitycheck(
        args.inputfileroot,
        graymaskspec=args.graymaskspec,
        whitemaskspec=args.whitemaskspec,
        debug=args.debug,
    )
    tide_io.writedicttojson(
        resultsdict,
        f"{args.inputfileroot}_desc-qualitymetrics_info.json",
    )
