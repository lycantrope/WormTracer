import argparse
import pathlib

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
        default="",
        help="Path to a directory to save the output results. Defaults to the input folder if not specified.",
    )

    return parser


def main():
    parser = get_parser()
    args, unkown_params = parser.parse_known_args()
    wt.run(**vars(args))


if __name__ == "__main__":
    main()
