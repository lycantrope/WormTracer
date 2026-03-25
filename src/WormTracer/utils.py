from __future__ import annotations

import collections
import datetime
import functools
import glob
import logging
import os
import shutil
import typing
from pathlib import Path

import matplotlib
import numpy as np
import roifile
import torch
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Iterator, Sequence

    import numpy.typing as npt

### read, preprocess images and get information ###
logger = logging.getLogger(__name__)


def set_output_path(
    dataset_path: str | os.PathLike,
    output_directory: str | os.PathLike | None,
) -> tuple[str, Path, str]:
    if not output_directory:
        if Path(dataset_path).is_dir():
            output_dir_p = Path(dataset_path)
        else:
            output_dir_p = Path(dataset_path).parent
    else:
        output_dir_p = Path(output_directory)

    Path(output_dir_p).mkdir(exist_ok=True, parents=True)

    dataset_prefix = Path(dataset_path).stem
    # If the output folder with the same name already exists, the series number is incremented by 1 from 001 to 999.
    for i in range(1, 1001):
        try:
            output_path = output_dir_p.joinpath(f"{dataset_prefix}_output_{i:0>3d}")
            output_path.mkdir()
            break
        except FileExistsError:
            # If output_path existed we pass
            pass
    else:
        # If the series number is incremented to 1000, it will throw an error to notify the user clearup the output folder.
        raise FileExistsError(
            "The output folder exists, please delete or move the previous output folder."
        )
    return dataset_prefix, output_path, Path(output_path).stem


def get_filenames(dataset_path: str | os.PathLike) -> Sequence[Path]:
    extensions_available = (
        ".bmp",
        ".dib",
        ".pbm",
        ".pgm",
        ".ppm",
        ".pnm",
        ".ras",
        ".png",
        ".tiff",
        ".tif",
        ".jp2",
        ".jpeg",
        ".jpg",
        ".jpe",
    )
    dataset_path = Path(dataset_path)
    if dataset_path.is_file() and dataset_path.suffix in extensions_available:
        return [dataset_path]

    ext_files_map = collections.defaultdict(list)  # type: collections.defaultdict[str, list[Path]]
    for name in dataset_path.glob("*.*"):
        if name.suffix in extensions_available:
            ext_files_map[name.suffix].append(name)

    if not ext_files_map:
        msg = "No extensions were found for openCV available. Please check if image files with the following extensions exist in the specified path"
        logger.error(msg)
        logger.error(extensions_available)
        raise FileNotFoundError(dataset_path)

    ext, files = max(ext_files_map.items(), key=lambda x: len(x[1]))

    if len(ext_files_map) > 1:
        logger.error("We found several extensions available in OpenCV.")
        logger.error(
            f"In this case, we loaded a {ext} file, but if you want to load a file with a different extension, delete the unrelated file."
        )
    return sorted(files, key=lambda x: x.name)


def clear_dir(output_path: os.PathLike, foldername: str) -> None:
    output_folder = Path(output_path) / foldername
    if output_folder.is_dir():
        shutil.rmtree(os.fspath(output_folder))
    output_folder.mkdir(parents=True)


def remove_progress(output_pathh, filename):
    remove_files = glob.glob(os.path.join(output_pathh, "progress_image", filename))
    for f in remove_files:
        os.remove(f)


def calc_cap_span(image_shape: Sequence[int], plot_n: int) -> int:
    """Calculate maximum span of trainig in terms of CUDA memory."""
    GB = float(1024**3)

    T = image_shape[0]
    # dim_size = d0 x d1 x d2 x ... x dn
    dim_size = np.prod(image_shape[1:]).item()

    # bytes used per stack under float32 and multiple by 8 for some margin case.
    mem_used_per_stack = float(8 * 4 * dim_size * (plot_n - 1)) / GB
    device = torch.accelerator.current_accelerator()
    try:
        with torch.cuda.device(device=device):
            free_memory, total_memory = torch.cuda.mem_get_info()  # bytes
            free_memory_gb = free_memory / GB  # bytes to GB
        # GB
        # Since the continuity and center loss require two consecutive frames, the cap_span must be greater than 1.
        cap_span = max(int(free_memory_gb / mem_used_per_stack), 5)
    except Exception as _:
        cap_span = T
    return cap_span


# 2. Setup your specific logger
def ensure_clearup(logger: logging.Logger) -> Callable:
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*arg, **kwargs):
            tic = datetime.datetime.now()
            backend = matplotlib.get_backend()
            original_level = logger.getEffectiveLevel()
            for h in logger.handlers[:]:
                h.close()
                logger.removeHandler(h)

            logger.handlers = []
            logger.propagate = False
            logger.setLevel(logging.DEBUG)
            stream_stderr = logging.StreamHandler()
            stream_stderr.setLevel(logging.DEBUG)
            logger.addHandler(stream_stderr)
            try:
                # Run function
                ret = fn(*arg, **kwargs)
                toc = datetime.datetime.now()
                elapsed_time = toc - tic
                logger.info(f"Elapse time: {elapsed_time.total_seconds():.1f} (sec)")
                return ret
            finally:
                for h in logger.handlers[:]:
                    h.close()
                    logger.removeHandler(h)

                logger.setLevel(original_level)
                logger.propagate = True
                try:
                    matplotlib.use(backend)
                except Exception:
                    pass

        return wrapper

    return decorator


def centerline_to_roi_iter(
    x: npt.NDArray,
    y: npt.NDArray,
    head_idx: int = 0,
) -> Iterator[roifile.ImagejRoi]:
    # (A,R,G,B)
    RED = b"\xff\xff\x00\x00"
    YELLOW = b"\xff\xff\xff\x00"
    n_digit = len(str(x.shape[0]))

    for pos, skel in enumerate(zip(x, y)):
        # (2, plot_n) => (plot_n, 2)
        centerline = np.asarray(skel).T
        name = str(pos + 1).zfill(n_digit)
        head_roi = roifile.ImagejRoi.frompoints(
            [centerline[head_idx]],
            name=name + "-Head",
            position=pos,
        )

        head_roi.roitype = roifile.ROI_TYPE.POINT
        head_roi.options |= roifile.ROI_OPTIONS.SHOW_LABELS
        head_roi.arrow_style_or_aspect_ratio = 3
        head_roi.stroke_width = 5
        head_roi.stroke_color = RED

        yield head_roi

        skel_roi = roifile.ImagejRoi.frompoints(
            centerline,
            name=name,
            position=pos,
        )
        skel_roi.roitype = roifile.ROI_TYPE.POLYLINE
        skel_roi.options |= roifile.ROI_OPTIONS.SHOW_LABELS
        skel_roi.stroke_color = YELLOW
        skel_roi.stroke_width = 2
        yield skel_roi


def save_centerline_to_roi(
    outputpath: str,
    x: npt.NDArray,
    y: npt.NDArray,
    head_idx: int = 0,
) -> None:
    roifile.roiwrite(
        outputpath,
        centerline_to_roi_iter(x, y, head_idx),
        mode="w",
    )


def save_params_into_commented_yaml(outputpath: os.PathLike, conf: dict[str, Any]):
    # Clone parameters
    conf_for_save = conf.copy()
    for key, value in conf_for_save.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            conf_for_save[key] = conf_for_save[key].item()
        if isinstance(value, Path):
            conf_for_save[key] = str(value)

    yaml = YAML(typ="rt")
    yaml.indent(mapping=4, sequence=4, offset=2)
    # Initialize our mapping object
    data = CommentedMap()

    # --- General Section ---
    data.yaml_set_start_comment(
        """This file is automatically generated by WormTracer.
# General Settings"""
    )
    data["local_time_difference"] = conf_for_save["local_time_difference"]
    data.yaml_add_eol_comment("UTC timezone", "local_time_difference")
    data.yaml_set_comment_before_after_key(
        "plot_n", before="\nNumber of segmented points placed on the centerline"
    )
    data["plot_n"] = conf_for_save["plot_n"]
    # --- Preprocess Section ---
    # Adding a blank line for organization before the next section
    data.yaml_set_comment_before_after_key("start_T", before="\nPreprocess")
    data["start_T"] = conf_for_save["start_T"]
    data.yaml_add_eol_comment("Number of start frames (default to 0)", "start_T")

    data["end_T"] = conf_for_save["end_T"]
    data.yaml_add_eol_comment("0 = process all frames", "end_T")

    data["rescale"] = conf_for_save["rescale"]
    data.yaml_add_eol_comment("Scaling ratio of original images", "rescale")

    data["Tscale"] = conf_for_save["Tscale"]
    data.yaml_add_eol_comment("Timestep of each frame", "Tscale")

    # --- Training Section ---
    data.yaml_set_comment_before_after_key(
        "continuity_loss_weight",
        before="\nLoss Weights",
    )
    data["continuity_loss_weight"] = conf_for_save["continuity_loss_weight"]
    data.yaml_add_eol_comment(
        "Ensures smooth movement between time frames",
        "continuity_loss_weight",
    )

    data["smoothness_loss_weight"] = conf_for_save["smoothness_loss_weight"]
    data.yaml_add_eol_comment(
        "Prevents sharp bends and keeps the body shape smooth",
        "smoothness_loss_weight",
    )

    data["length_loss_weight"] = conf_for_save["length_loss_weight"]
    data.yaml_add_eol_comment(
        "Prevents the worm from stretching or shrinking unnaturally",
        "length_loss_weight",
    )
    data["center_loss_weight"] = conf_for_save["center_loss_weight"]
    data.yaml_add_eol_comment(
        "Keeps the centerline inside the worm's silhouette",
        "center_loss_weight",
    )
    data["body_ratio"] = conf_for_save["body_ratio"]
    data.yaml_add_eol_comment(
        "Weight ratio between the middle body and head/tail",
        "body_ratio",
    )
    data.yaml_set_comment_before_after_key(
        "speed",
        before="\nTraining ",
    )
    data["speed"] = conf_for_save["speed"]
    data["lr"] = conf_for_save["lr"]
    data["epoch_plus"] = conf_for_save["epoch_plus"]
    data.yaml_add_eol_comment("Additional epochs after final step", "epoch_plus")

    data.yaml_set_comment_before_after_key(
        "judge_head_method",
        before="\nPostprocess",
    )

    data["judge_head_method"] = conf_for_save["judge_head_method"]
    data.yaml_add_eol_comment(
        "Judge the head or tail by `frequency` or `amplitude` (default to `frequency`)",
        "judge_head_method",
    )

    # --- Display Section ---
    data.yaml_set_comment_before_after_key("num_t", before="\nDisplay & Progress")
    data["num_t"] = conf_for_save["num_t"]
    data["ShowProgress"] = conf_for_save["ShowProgress"]
    data["SaveProgress"] = conf_for_save["SaveProgress"]
    data["show_progress_freq"] = conf_for_save["show_progress_freq"]
    data["save_progress_freq"] = conf_for_save["save_progress_freq"]
    data["save_progress_num"] = conf_for_save["save_progress_num"]

    # --- Output Section ---
    data.yaml_set_comment_before_after_key(
        "SaveCenterlinedWormsSerial", before="\nOutput Formats"
    )
    data["SaveCenterlinedWormsSerial"] = conf_for_save["SaveCenterlinedWormsSerial"]
    data["SaveCenterlinedWormsMovie"] = conf_for_save["SaveCenterlinedWormsMovie"]
    data["SaveCenterlinedWormsMultitiff"] = conf_for_save[
        "SaveCenterlinedWormsMultitiff"
    ]

    if "dataset_path" in conf_for_save:
        data.yaml_set_comment_before_after_key(
            "dataset_path", before="\nDataset and output directory"
        )
        data["dataset_path"] = conf_for_save["dataset_path"]
        data["output_path"] = conf_for_save["output_path"]

    # ---- Model Parameters ---
    if "init_alpha" in conf_for_save:
        data.yaml_set_comment_before_after_key(
            "init_alpha", before="\nInitial Model Weights"
        )
        data["init_alpha"] = conf_for_save["init_alpha"]
        data["init_gamma"] = conf_for_save["init_gamma"]
        data["init_delta"] = conf_for_save["init_delta"]

    if "alpha" in conf_for_save:
        data.yaml_set_comment_before_after_key(
            "alpha", before="\nTrained Model Weights"
        )
        data["alpha"] = conf_for_save["alpha"]
        data["gamma"] = conf_for_save["gamma"]
        data["delta"] = conf_for_save["delta"]

    # Write to a file
    with open(outputpath, "w") as f:
        yaml.dump(data, f)
