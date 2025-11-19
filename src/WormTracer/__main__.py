import argparse
import os
import pathlib
import subprocess
import sys

from WormTracer import wt


def get_parser():
    parser = argparse.ArgumentParser("WormTracer")

    parser.add_argument(
        "parameter_file",
        type=pathlib.Path,
        help="Path to the parameter file (*.yaml) containing configuration for wormtracer.",
    )

    parser.add_argument(
        "dataset_path",
        help="Path to a folder containing the input images for analyzing",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_directory",
        default=None,
        help="Path to a directory to save the output results. Defaults to the input folder if not specified.",
    )

    return parser


def main():
    parser = get_parser()
    args, unknown_params = parser.parse_known_args()
    wt.run(**vars(args))


def main_wrapper():
    # To run script mode with optimized flag.
    # Equivalent to uv run python -O -m WormTracer ...
    command = [sys.executable, "-O", "-m", "WormTracer"] + sys.argv[1:]
    ret = subprocess.run(
        command,
        env=os.environ.copy(),
        stdout=subprocess.STDOUT,
        stderr=subprocess.STDOUT,
    )
    return ret.returncode


if __name__ == "__main__":
    main()
