from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

import matplotlib

from WormTracer import wt


def get_parser():
    parser = argparse.ArgumentParser("WormTracer")

    parser.add_argument(
        "--parameter_file",
        type=pathlib.Path,
        help="Path to the parameter file (*.yaml) containing configuration for wormtracer.",
        required=True,
    )

    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to a folder containing the input images for analyzing",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_directory",
        default=None,
        help="Path to a directory to save the output results. Defaults to the input folder if not specified.",
    )

    parser.add_argument(
        "-g",
        "--guide_files",
        dest="guide_files",
        nargs="+",
        type=pathlib.Path,
        help="Path to guided centerline. (*_x.csv, *_y.csv or *.h5)",
    )

    parser.add_argument("--start_T", type=int, help="Start frame of WormTracer")
    parser.add_argument(
        "--end_T",
        type=int,
        help="End frame of WormTracer (set to 0 process all frame)",
    )

    parser.add_argument("--rescale", type=float, help="Rescale of input images")

    parser.add_argument("--Tscale", type=int)
    parser.add_argument("--continuity_loss_weight", type=float)
    parser.add_argument("--smoothness_loss_weight", type=float)
    parser.add_argument("--length_loss_weight", type=float)
    parser.add_argument("--center_loss_weight", type=float)
    parser.add_argument("--plot_n", type=int)
    parser.add_argument("--epoch_plus", type=int)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--body_ratio", type=float)
    parser.add_argument("--judge_head_method", choices=["amplitude", "frequency"])

    parser.add_argument(
        "--num_t",
        type=int,
    )

    parser.add_argument(
        "--ShowProgress",
        action="store_true",
        help="If True, shows progress during optimization repeats.",
        default=None,
    )
    parser.add_argument(
        "--SaveProgress",
        action="store_true",
        help="If True, saves worm images during optimization in `progress_image` folder created in datafolder.",
        default=None,
    )

    parser.add_argument(
        "--show_progress_freq",
        type=int,
        help="This value is epoch frequency of displaying tracing progress. (default: 200)",
    )
    parser.add_argument(
        "--save_progress_freq",
        type=int,
        help="This value is epoch frequency of saving tracing progress. (default: 50)",
    )
    parser.add_argument(
        "--save_progress_num",
        type=int,
        help="This value is the number of images that are included in saved progress tracing. (default: 50)",
    )

    parser.add_argument(
        "--SaveCenterlinedWormsSerial",
        action="store_true",
        help="If True, saves input images with estimated centerline as seirial numbered png files in full_line_images folder.",
        default=None,
    )
    parser.add_argument(
        "--SaveCenterlinedWormsMovie",
        action="store_true",
        help="If True, saves input images with estimated centerline as a movie full_line_images.mp4",
        default=None,
    )
    parser.add_argument(
        "--SaveCenterlinedWormsMultitiff",
        action="store_true",
        help="If True, saves input images with estimated centerline as a multipage tiff full_line_images.tif",
        default=None,
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    conf = {k: v for k, v in vars(args).items() if v is not None}
    backend = matplotlib.get_backend()
    # Set matplotlib backend to Agg while running WormTracer as script.
    if backend != "Agg":
        matplotlib.use("Agg")
    wt.run(**conf)
    try:
        matplotlib.use(backend)
    except Exception:
        pass


def main_wrapper():
    # To run script mode with optimized flag.
    # Equivalent to uv run python -O -m WormTracer ...
    command = [sys.executable, "-O", "-m", "WormTracer"] + sys.argv[1:]
    ret = subprocess.run(
        command,
        env=os.environ.copy(),
    )
    return ret.returncode


if __name__ == "__main__":
    main()
