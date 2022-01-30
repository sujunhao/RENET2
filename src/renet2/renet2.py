#!/usr/bin/env python
import sys
from pathlib import Path
import os, sys

from os.path import dirname, abspath
from importlib import import_module
from collections import namedtuple

from renet2.raw_handler import *

REPO_NAME = "RENET2"
RENET2_FOLDER="renet2"
VERSION="v1.3"


renet2_folder = ["predict", "train", "evaluate_renet2_ft_cv"]
prep_scripts_folder = ["download_data", "parse_data", "normalize_ann", "install_geniass"]

def print_help_messages():
    from textwrap import dedent
    print(dedent("""\
        RENET2 ({0}): High-Performance Full-text Gene-Disease Relation Extraction with Iterative Training Data Expansion
        {1} submodule invocator:
            Usage: renet2 [submodule] [options of the submodule]
        Available renet2 submodules:\n{2}
        Available data preparation submodules:\n{3}
        """.format(
            VERSION,
            REPO_NAME,
            "\n".join("          - %s" % submodule_name for submodule_name in renet2_folder),
            "\n".join("          - %s" % submodule_name for submodule_name in prep_scripts_folder),
        )
    ))

def main():
    print_renet2_log()
    if len(sys.argv) <= 1 or sys.argv[1] == "-h" or sys.argv[1] == "-v" or sys.argv[1] == "--help":
        print_help_messages()
        sys.exit(0)

    submodule_name = sys.argv[1]
    if ( submodule_name not in renet2_folder and \
         submodule_name not in prep_scripts_folder):
        sys.exit("[ERROR] Submodule %s not found." % (submodule_name))

    directory = RENET2_FOLDER
    submodule = import_module("%s.%s" % (directory, submodule_name))

    sys.argv = sys.argv[1:]
    sys.argv[0] += (".py")

    # Note: need to make sure every submodule contains main() method
    submodule.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
