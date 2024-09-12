from __future__ import annotations

import collections
import glob
import math
import os
import shutil
from math import pi
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import distance_matrix
from scipy.special import expit as np_sigmoid
from skimage import morphology

if TYPE_CHECKING:
    from typing import Tuple, Union

    from numpy.typing import NDArray


def show_image(image, num_t=5, title="", x=0, y=0, x2=0, y2=0):
    T = image.shape[0]
    num_t = min(num_t, T)
    t_sparse = np.linspace(0, T - 1, min(num_t, T), dtype=int)
    if torch.is_tensor(image):
        image = image.clone().detach().cpu().numpy()
    if torch.is_tensor(x):
        x = x.clone().detach().cpu().numpy()
        y = y.clone().detach().cpu().numpy()
    n_rows = (t_sparse.shape[0] + 4) // 5
    fig, axes = plt.subplots(
        n_rows, 5, figsize=(12, 2 * n_rows), squeeze=False, tight_layout=True
    )
    for i in range(t_sparse.shape[0]):
        axes[i // 5, i % 5].imshow(
            image[t_sparse[i], :, :], cmap="gray", vmax=np.max(image), vmin=0
        )
        axes[i // 5, i % 5].axis([0, image.shape[2], 0, image.shape[1]])
        axes[i // 5, i % 5].set_title(title + " t = {}".format(t_sparse[i]))
        if isinstance(x, np.ndarray):
            axes[i // 5, i % 5].scatter(x[t_sparse[i]], y[t_sparse[i]], c="r", s=30)
        if isinstance(x2, np.ndarray):
            axes[i // 5, i % 5].scatter(x2[t_sparse[i]], y2[t_sparse[i]], c="y", s=30)
    # plt.show()
    plt.close(fig)


### read, preprocess images and get information ###


def set_output_path(dataset_path, output_directory):
    output_directory = Path(output_directory or Path(dataset_path).parent)
    Path(output_directory).mkdir(exist_ok=True)

    # for compatiblity of py3.6 using PurePath to get stem
    dataset_prefix = Path(dataset_path).stem

    for i in range(int(1e32)):
        output_path = output_directory.joinpath(
            "{}_output_{:d}".format(dataset_prefix, i + 1)
        )
        if not output_path.is_dir():
            break

    Path(output_path).mkdir()
    return dataset_prefix, output_path, Path(output_path).stem


def get_filenames(dataset_path: Union[str, bytes, os.PathLike]):
    extensions_available = {
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
    }
    dataset_path = Path(os.fspath(dataset_path))
    if dataset_path.is_file() and dataset_path.suffix in extensions_available:
        return [os.fspath(dataset_path)]

    ext_files_map = collections.defaultdict(list)
    for name in dataset_path.glob("*.*"):
        if name.suffix not in extensions_available:
            continue
        ext_files_map[name.suffix].append(os.fspath(name))

    if not ext_files_map:
        print(
            "No extensions were found for openCV available. Please check if image files with the following extensions exist in the specified path"
        )
        print(extensions_available)

    ext, files = max(ext_files_map.items(), key=lambda x: len(x[1]))

    if len(ext_files_map) > 1:
        print("We found several extensions available in openCV.")
        print(
            f"In this case, we loaded a {ext} file, but if you want to load a file with a different extension, delete the unrelated file."
        )
    return sorted(files)


def get_property(filenames, rescale):
    _, ims = cv2.imreadmulti(filenames[0])
    im = ims[0]
    if np.any((0 < np.asarray(im)) & (np.asarray(im) < 255)):
        print("Warning! : Input images seem not to be binary.")
    if not math.isclose(rescale, 1.0, rel_tol=1e4):
        im = cv2.resize(
            im,
            dsize=None,
            fy=rescale,
            fx=rescale,
            interpolation=cv2.INTER_NEAREST,
        )
    white_pixel = (
        np.sum(im[0, :-1]) // 255
        + np.sum(im[-1, 1:]) // 255
        + np.sum(im[1:, 0]) // 255
        + np.sum(im[:-1, -1]) // 255
    )
    # if sum of white pixel is larger than height + width
    Worm_is_black = white_pixel > sum(im.shape[:2])
    multi_flag = len(ims) > 1
    n_input_images = len(ims) if multi_flag else len(filenames)
    return im.shape, Worm_is_black, multi_flag, n_input_images


def read_serial_images(filenames, Tscaled_ind):
    return [cv2.imread(filenames[ind], cv2.IMREAD_GRAYSCALE) for ind in Tscaled_ind]


def read_image(
    filenames,
    rescale,
    Worm_is_black,
    multi_flag,
    Tscaled_ind,
) -> Tuple[NDArray, float, float]:
    """read images and get skeletonized plots"""
    if multi_flag:
        _, ims = cv2.imreadmulti(filenames[0], flags=0)  # multipage tiff file
        ims = [ims[ind] for ind in Tscaled_ind]
    else:
        ims = read_serial_images(filenames, Tscaled_ind)  # serial-numbered image files

    def preprocess(im):
        im = im.astype("uint8")
        if Worm_is_black:
            im = cv2.bitwise_not(im)
        _, labelImages, stuts, _ = cv2.connectedComponentsWithStats(im, connectivity=4)
        im[labelImages != stuts[1:, 4].argmax() + 1] = 0
        if not math.isclose(rescale, 1.0, rel_tol=1e-3):
            im = cv2.resize(
                im,
                dsize=None,
                fy=rescale,
                fx=rescale,
                interpolation=cv2.INTER_NEAREST,
            )

        return im

    ims = [preprocess(im) for im in ims]
    imagestack = np.asarray(ims)
    imagestack, y_st, x_st = cut_image(imagestack)
    return imagestack, y_st, x_st


def calc_xy_and_prewidth(
    imagestack: NDArray,
    plot_n: int,
    x_st: float,
    y_st: float,
) -> Tuple[NDArray, NDArray, NDArray, float]:
    """read images and get skeletonized plots"""
    T = imagestack.shape[0]
    assert T > 0, "Input is empty"

    # Intitial output data
    x = np.zeros((T, plot_n))
    y = np.zeros((T, plot_n))
    pre_width = np.zeros(T)

    x[0, :], y[0, :] = get_skeleton(imagestack[0], plot_n)
    pre_width[0] = get_width(imagestack[0], x[0], y[0])
    print("")
    for t in range(1, T):
        bar = "\rget_skeleton and width:[{:<100}] {}/{}".format(
            "▉" * round(t * 100 / T + 1), t + 1, T
        )
        print(bar, end="")
        im = imagestack[t]
        x1, y1 = get_skeleton(im, plot_n)

        x0, y0 = x[t - 1, :], y[t - 1, :]

        gap_headtail = ((x1 - x0) ** 2 + (y1 - y0) ** 2).sum()
        gap_headtail_rev = ((x1 - x0[::-1]) ** 2 + (y1 - y0[::-1]) ** 2).sum()

        x[t, :], y[t, :] = x1, y1
        if gap_headtail > gap_headtail_rev:
            x[t, :] = x1[::-1]
            y[t, :] = y1[::-1]

        pre_width[t] = get_width(im, x[t], y[t])

    print("")
    unitLength = np.sqrt(
        np.median(((x[:, :-1] - x[:, 1:]) ** 2 + (y[:, :-1] - y[:, 1:]) ** 2))
    )
    x += x_st
    y += y_st
    return x, y, pre_width, unitLength


def get_skeleton(im: NDArray, plot_n: int):
    """skeletonize image and get splined plots"""

    # skeletonize image
    im_filled = ndi.binary_fill_holes(im)
    im_skeleton = morphology.skeletonize(im_filled)
    point_list = np.argwhere(im_skeleton == 1)

    if len(point_list) == 1:
        x_splined = np.ones(plot_n) * point_list[0][1]
        y_splined = np.ones(plot_n) * point_list[0][0]
        return x_splined, y_splined

    # make distance matrix
    cube_len = len(point_list)
    adj_mtx = distance_matrix(
        point_list, point_list, threshold=cube_len * cube_len * 2 + 10
    )
    adj_mtx[adj_mtx > 1.5] = 0  # delete distance between isolated points
    csr = csr_matrix(adj_mtx)
    adj_sum = np.sum(adj_mtx, axis=0)

    # get tips of longest path
    d1 = shortest_path(csr, indices=np.argmax(adj_sum < 1.5))
    while np.sum(d1 == np.inf) > d1.shape[0] // 2:
        adj_sum[np.argmax(adj_sum < 1.5)] = 2
        d1 = shortest_path(csr, indices=np.argmax(adj_sum < 1.5))
    d1[d1 == np.inf] = 0
    d2, p = shortest_path(csr, indices=np.argmax(d1), return_predecessors=True)
    d2[d2 == np.inf] = 0

    # get longest path
    plots = []
    arclen = []
    point = np.argmax(d2)  # This is the start point(the end point is np.argmax(d1))
    while point != np.argmax(d1) and point >= 0:
        plots.append(point_list[point])
        arclen.append(d2[point])
        point = p[point]
    plots.append(point_list[point])
    arclen.append(d2[point])
    plots = np.array(plots)
    arclen = np.array(arclen)[::-1]

    # interpolation
    div_linespace = np.linspace(0, np.max(arclen), plot_n)
    x_splined = np.interp(div_linespace, arclen, plots[:, 1], kind="linear")
    y_splined = np.interp(div_linespace, arclen, plots[:, 0], kind="linear")

    return x_splined, y_splined


def get_width(im, x, y):
    """Get width of the object by measure distance of centerline to the object's surface."""
    im_filled = ndi.binary_fill_holes(im)
    x = x.reshape([-1, 1, 1])
    y = y.reshape([-1, 1, 1])
    y_3d = np.arange(im_filled.shape[0]).reshape([1, -1, 1])
    x_3d = np.arange(im_filled.shape[1]).reshape([1, 1, -1])
    segment_distance = np.sqrt((x - x_3d) ** 2 + (y - y_3d) ** 2)
    max_dist = im_filled.shape[0] + im_filled.shape[1]
    new_segment_distance = segment_distance + im_filled * max_dist
    wid = new_segment_distance.min(axis=(1, 2)).max()
    return wid


def flip_check(x, y):
    """Check if plots of head and tail is flipping."""
    gap_headtail = np.mean(
        (x[1:, :] - x[:-1, :]) ** 2 + (y[1:, :] - y[:-1, :]) ** 2, axis=1
    )
    gap_headtail_ex = np.mean(
        (x[1:, :] - x[:, ::-1][:-1, :]) ** 2 + (y[1:, :] - y[:, ::-1][:-1, :]) ** 2,
        axis=1,
    )
    ex_t = gap_headtail > gap_headtail_ex
    ex_r = np.zeros(ex_t.shape, dtype=np.bool)
    ex_r[np.cumsum(ex_t) % 2 == 1] = True
    x[1:, :][ex_r] = x[:, ::-1][1:, :][ex_r]
    y[1:, :][ex_r] = y[:, ::-1][1:, :][ex_r]
    return x, y


def cut_image(image):
    """Cut images to minimum size."""
    thresh = np.bitwise_or.reduce(image > 0, axis=0)

    (ys, xs) = np.nonzero(thresh)
    if ys.size == 0:
        print("[Warning] the imagestack have no signal")
        return image, 0, 0

    max_h, max_w = thresh.shape

    x1 = max(xs.min() - 5, 0)
    x2 = min(xs.max() + 5, max_w)
    y1 = max(ys.min() - 5, 0)
    y2 = min(ys.max() + 5, max_h)

    return image[:, y1:y2, x1:x2], y1, x1


def make_theta_from_xy(x: NDArray, y: NDArray) -> NDArray:
    assert x.ndim == 2, "x should be 2D ndarray"
    T, plot_n = x.shape
    n_segs = plot_n - 1
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    theta = np.arctan2(dy, dx)
    # Arrange theta if the gap is larget than pi
    # Adjust the middle theta between time point
    pi = np.pi
    mid = n_segs // 2
    t_gap = theta[1:, mid] - theta[:-1, mid]
    t_adjust = np.sign(t_gap) * 2 * pi
    t_adjust[np.abs(t_gap) < pi] = 0
    theta[1:, :] -= t_adjust.cumsum().reshape(-1, 1)

    gap = theta[:, 1:] - theta[:, :-1]
    # adjust right hand side of theta within same time points
    r_gap = gap[:, mid:]
    r_adjust = np.sign(r_gap) * 2 * pi
    r_adjust[np.abs(r_gap) < pi] = 0
    theta[:, mid + 1 :] -= r_adjust.cumsum(axis=1)

    # adjust left hand side
    l_gap = gap[:, :mid]
    l_adjust = np.sign(l_gap * -1) * 2 * pi
    l_adjust[np.abs(l_gap) < pi] = 0
    l_adjust_rev = np.flip(l_adjust, axis=1)
    theta[:, :mid] -= np.flip(np.cumsum(l_adjust_rev, axis=1), axis=1)
    return theta


### prepare for training ###


def calc_cap_span(image_info, plot_n, s_m=8000):
    """Calculate maximum span of trainig in terms of CUDA memory."""
    device = image_info["device"]
    GiB = 1e20
    try:
        free_memory = (
            torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_allocated(device)
        ) / GiB
        cap_span = int(
            s_m
            * free_memory
            / (
                image_info["image_shape"][1]
                * image_info["image_shape"][2]
                * (plot_n - 1)
            )
        )
    except Exception as _:
        cap_span = image_info["image_shape"][0]
    return cap_span


def pixel_value_from_dist_max_np(
    max_dist: NDArray,
    contrast: float = 1.2,
    sharpness: float = 2.0,
) -> NDArray:
    """Get pixel value when distance from midline is given."""
    return 255 * (contrast * (np_sigmoid(max_dist * sharpness) - 0.5) + 0.5)


def worm_width_all_np(
    *,
    plot_n: int,
    alpha: float,
    gamma: float,
    delta: float,
) -> NDArray:
    """Get all worm widths when segment number is given."""

    worm_x = np.linspace(-1.0, 1.0, plot_n - 1)

    delta_sigmoid = np_sigmoid(delta)
    gamma_e = 0.5 + np.exp(gamma)
    worm_x_abs = np.abs(worm_x)
    width = alpha * np.sqrt(
        1
        + 1e-5
        - worm_x_abs ** (2 * gamma_e)
        * (
            1
            + (2 * gamma_e) * delta_sigmoid
            - (2 * gamma_e) * delta_sigmoid * worm_x_abs
        )
    )
    return width


def make_distance_matrix_np(radius: int) -> NDArray:
    diameter = radius * 2 + 1
    delta = (np.arange(diameter) - radius) ** 2
    distance_matrix = np.sqrt(delta[None, :] + delta[:, None])

    # let distance_kernel become circular
    distance_matrix[distance_matrix > radius] = np.inf
    return distance_matrix


def make_distance_matrix(radius: int) -> NDArray:
    diameter = radius * 2 + 1
    delta = (torch.arange(diameter) - radius) ** 2
    distance_matrix = torch.sqrt(delta[None, :] + delta[:, None])

    # let distance_kernel become circular
    distance_matrix[distance_matrix > radius] = torch.inf
    return distance_matrix


def make_single_image(
    x: NDArray,
    y: NDArray,
    width: int,
    height: int,
    pixel_matrix: NDArray,
) -> NDArray:
    cent_x = ((x[:-1] + x[1:]) / 2).astype(np.int32)
    cent_y = ((y[:-1] + y[1:]) / 2).astype(np.int32)

    diameter = pixel_matrix.shape[1]
    radius = diameter // 2
    pad_image = np.full(
        (height + radius * 2, width + radius * 2),
        fill_value=-25.5,
    )
    for i, j, pix in zip(cent_x, cent_y, pixel_matrix):
        pad_image[j : j + diameter, i : i + diameter] = np.maximum(
            pad_image[j : j + diameter, i : i + diameter],
            pix,
        )
    return pad_image[radius:-radius, radius:-radius]


def make_image(x, y, x_st, y_st, params, image_info):
    """Create model image by dividing them to avoid CUDA memory error."""
    T = x.shape[0]
    worm_wid = worm_width_all_np(
        plot_n=params["plot_n"],
        alpha=params["alpha"],
        gamma=params["gamma"],
        delta=params["delta"],
    )
    max_radius = int(np.ceil(worm_wid.max())) + 2
    distance_matrix = make_distance_matrix_np(max_radius)

    distance_matrix_3d = worm_wid[:, None, None] - distance_matrix[None, :, :]
    pixel_matrix = pixel_value_from_dist_max_np(distance_matrix_3d)
    im_height = image_info["image_shape"][1]
    im_width = image_info["image_shape"][2]
    image = np.zeros((T, im_height, im_width))

    for i in range(T):
        image[i, :, :] = make_single_image(
            x[i] - x_st,
            y[i] - y_st,
            width=im_width,
            height=im_height,
            pixel_matrix=pixel_matrix,
        )

    return image


def get_image_loss_max(
    image_losses, real_image, x, y, x_st, y_st, params, image_info, cap_span
):
    """Create bad image and get bad image_loss to judge complex area."""
    small_loss_frame = np.argmin(image_losses)
    im = real_image[small_loss_frame].reshape(
        -1, real_image.shape[1], real_image.shape[2]
    )
    x0 = np.ones(params["plot_n"]) * x[small_loss_frame, 0].reshape(1, -1)
    y0 = np.ones(params["plot_n"]) * y[small_loss_frame, 0].reshape(1, -1)
    im0 = make_image(x0, y0, x_st, y_st, params, image_info)
    image_loss_max = np.mean((im - im0) ** 2)
    return image_loss_max


def get_use_points(
    image_losses, image_loss_max, cap_span, x, y, plot_n, show_plot=True
):
    """Judge flames complex or not and get span for training."""
    T = image_losses.shape[0]
    # find complex area
    borderline = 0.4 * image_loss_max + 0.6 * np.min(image_losses)
    under_borderline = 0.2 * image_loss_max + 0.8 * np.min(image_losses)
    nont_ini, nont_end, simple_area = find_nont_area(
        image_losses, borderline, under_borderline
    )

    try:
        if nont_ini[0] > nont_end[0]:
            nont_end = nont_end[1:]
            print("Warning! The initial frame of images is difficult to skeletonize.")
            print("Biginning of Results will be incorrect.")
        if nont_ini[-1] > nont_end[-1]:
            print("Warning! The end frame of images is difficult to skeletonize.")
            print("End of Results will be incorrect.")
            nont_ini = nont_ini[:-1]

        # expand complex area
        nont_span = nont_end - nont_ini
        target_area = np.full(nont_ini.shape[0], True)
        while sum(target_area) > 0:
            temp_ini = nont_ini.copy()
            temp_end = nont_end.copy()
            temp_ini[target_area] = nont_ini[target_area] - 1
            temp_end[target_area] = nont_end[target_area] + 1
            enough_expanded = check_enough_expanded(nont_span, temp_ini, temp_end)
            collision = check_collision(temp_ini, temp_end, T)
            target_area = target_area * enough_expanded * collision
            nont_ini[target_area] = nont_ini[target_area] - 1
            nont_end[target_area] = nont_end[target_area] + 1

        # set use_points
        nont_flag = []
        max_span = np.max(nont_end - nont_ini)
        nont_end = np.append(0, nont_end + 1)
        nont_ini = np.append(nont_ini, T - 1)
        num_span = (nont_ini - nont_end) // max_span
        one_span = (nont_end == nont_ini).astype(np.int32)
        use_points = np.array([0])
        for i in range(num_span.shape[0]):
            use_points = np.append(
                use_points,
                np.linspace(
                    nont_end[i], nont_ini[i], num_span[i] + 2 - one_span[i], dtype=int
                ),
            )
            nont_flag += [0] * (num_span[i] + 1 - one_span[i])
            nont_flag.append(1)
        use_points = use_points[1:]

        # check memory
        if max_span > cap_span:
            unitL = np.median(
                np.sqrt(
                    (
                        (x[simple_area, :-1] - x[simple_area, 1:]) ** 2
                        + (y[simple_area, :-1] - y[simple_area, 1:]) ** 2
                    )
                )
            )
            rescale_rec = max(np.sqrt(cap_span / max_span), 200 / unitL / plot_n)
            Tscale_rec = 1
            if int(max_span * rescale_rec**2) > cap_span:
                Tscale_rec = max(max_span // 200, 1)
                if int((max_span * rescale_rec**2) / Tscale_rec) > cap_span:
                    rescale_rec = max(
                        np.sqrt(Tscale_rec * cap_span / max_span), 120 / unitL / plot_n
                    )
                    if int((max_span * rescale_rec**2) / Tscale_rec) > cap_span:
                        Tscale_rec = max(max_span // 150, 1)
            if int((max_span * rescale_rec**2) / Tscale_rec) > cap_span:
                rescale_rec = np.sqrt(cap_span / max_span)
                print(
                    """
        Warning! This task uses large memory.
        If CUDA run out of memory, please go back to setting hyperparameters and set rescale as {:.2f}, Tscale as {}.
        The result may be not precise enough.
        """.format(rescale_rec, Tscale_rec)
                )
            else:
                print(
                    """
        Warning! This task uses large memory.
        If CUDA run out of memory, please go back to setting hyperparameters and set rescale as {:.2f}, Tscale as {}.
        """.format(rescale_rec, Tscale_rec)
                )

    except IndexError:
        print("All frames seem to be simple; easy to skeletonize.")
        use_points = np.linspace(0, T - 1, (T - 1) // (cap_span + 1) + 2, dtype=int)
        nont_flag = [0] * (use_points.shape[0])

    if show_plot:
        plt.plot(image_losses)
        plt.plot([borderline] * T)
        plt.plot([under_borderline] * T)
        plt.plot([image_loss_max] * T)
        for i in range(len(use_points)):
            plt.plot(
                [use_points[i]] * 2,
                [
                    np.min(image_losses),
                    0.1 * image_loss_max + 0.9 * np.min(image_losses),
                ],
                c="r",
            )
        plt.xlabel("frames", fontsize=20)
        plt.ylabel("image loss", fontsize=20)
        # plt.show()

    return use_points, nont_flag[:-1], simple_area


def find_nont_area(image_losses, borderline, under_borderline):
    complex_area = (image_losses > borderline).astype(np.int32)
    under_complex_area = (image_losses > under_borderline).astype(np.int32)
    complex_area_check = complex_area + under_complex_area
    checkpoint = None
    continent = 0
    for i in range(complex_area_check.shape[0]):
        if complex_area_check[i] == 1:
            if continent == 0:
                checkpoint = i
                continent = 1
            if continent == 2:
                complex_area[i] = 1
        if complex_area_check[i] == 0:
            checkpoint = None
            continent = 0
        if complex_area_check[i] == 2:
            if continent == 1:
                complex_area[checkpoint:i] = 1
            continent = 2
    nont_ini = np.where(complex_area[1:] - complex_area[:-1] == 1)[0]
    nont_end = np.where(complex_area[1:] - complex_area[:-1] == -1)[0]
    return nont_ini, nont_end, 1 - complex_area


def check_enough_expanded(nont_span, temp_ini, temp_end, enough_rate=2):
    expand_amount = temp_end - temp_ini - nont_span
    return expand_amount < nont_span // enough_rate


def check_collision(temp_ini, temp_end, T):
    nont_end = np.append(0, temp_end + 1)
    nont_ini = np.append(temp_ini, T - 1)
    gap_span_safe = (nont_ini - nont_end) >= 0
    return gap_span_safe[1:] & gap_span_safe[:-1]


def prepare_for_train(pre_width, simple_area, x, y, params):
    params["init_alpha"] = torch.tensor(pre_width[simple_area].mean())
    params["init_gamma"] = torch.tensor(0.0)
    params["init_delta"] = torch.tensor(0.0)
    unitLength = np.sqrt(
        np.median(
            (
                (x[simple_area, :-1] - x[simple_area, 1:]) ** 2
                + (y[simple_area, :-1] - y[simple_area, 1:]) ** 2
            )
        )
    )
    return unitLength


### training ###
def make_progress_image(image, num_t=20):
    """Make one large image with images laid out on it."""
    if torch.is_tensor(image):
        image = image.clone().detach().cpu().numpy()
    assert image.ndim == 3, "image must be (batch, height, width)"
    T, H, W = image.shape
    t_sparse = np.linspace(0, T - 1, min(num_t, T), dtype=int)
    subset = image[t_sparse]
    n_chunk = (subset.shape[0] + 1) // 5
    progress_image = np.zeros((H * n_chunk, W * 5))
    for i, chunk in enumerate(np.array_split(subset, n_chunk, axis=0)):
        merge = np.hstack(chunk)
        progress_image[i * H : (i + 1) * H, : merge.shape[1]] = merge
    return progress_image


def save_progress(image, output_path, output_name: str, params, txt="real"):
    if params["SaveProgress"]:
        use_area = params["use_area"]
        progress_image = make_progress_image(image, params["save_progress_num"])
        filename = os.path.join(
            output_path,
            output_name + "_progress_image",
            "{}-{}_{}.png".format(use_area[0], use_area[1], txt),
        )
        cv2.imwrite(filename, progress_image)


def remove_progress(output_pathh, filename):
    remove_files = glob.glob(os.path.join(output_pathh, "progress_image", filename))
    for f in remove_files:
        os.remove(f)


def get_center(binimg):
    """Calculate center of images."""
    if torch.is_tensor(binimg):
        binimg = binimg.clone().detach().cpu().numpy()
    ys, xs = np.where(binimg == np.max(binimg))
    x = np.average(xs)
    y = np.average(ys)
    return x, y


def set_init_xy(real_image):
    """Set init center plots for training."""
    T = real_image.shape[0]
    init_cx = torch.zeros(T)
    init_cy = torch.zeros(T)
    for t in range(T):
        init_cx[t], init_cy[t] = get_center(real_image[t, :, :])
    return init_cx, init_cy


def find_theta(theta, pretheta, plus=1):
    """Find min MSE theta by theta(t=0)"""
    i = plus
    mse_list = [np.sum((theta[0, :] - pretheta) ** 2)]
    while True:
        theta_cand = pretheta + i * 2 * pi
        mse_0T = np.sum((theta[0, :] - theta_cand) ** 2)
        if mse_list[-1] < mse_0T:
            break
        mse_list.append(mse_0T)
        i += plus
    return len(mse_list)


def make_theta_cand(theta):
    i_normal = find_theta(theta, theta[-1, :]) - find_theta(theta, theta[-1, :], -1)
    pretheta = theta[-1, :][::-1] + pi
    i_reverse = find_theta(theta, pretheta) - find_theta(theta, pretheta, -1)
    theta_pair = (theta[-1, :] + i_normal * 2 * pi, pretheta + i_reverse * 2 * pi)
    theta_cands_normal = [theta_pair[0] + 2 * pi, theta_pair[0] - 2 * pi]
    loss_normal_p = np.sum((theta[0, :] - theta_cands_normal[0]) ** 2)
    loss_normal_m = np.sum((theta[0, :] - theta_cands_normal[1]) ** 2)
    theta_cands_reverse = [theta_pair[1] + 2 * pi, theta_pair[1] - 2 * pi]
    loss_reverse_p = np.sum((theta[0, :] - theta_cands_reverse[0]) ** 2)
    loss_reverse_m = np.sum((theta[0, :] - theta_cands_reverse[1]) ** 2)
    theta_subpair = (
        theta_cands_normal[int(loss_normal_p > loss_normal_m)],
        theta_cands_reverse[int(loss_reverse_p > loss_reverse_m)],
    )
    return theta_pair, theta_subpair


def body_axis_function(body_ratio, plot_n, base=0.5):
    x = torch.arange(-0.5 * plot_n + 2, 0.5 * plot_n) - 0.5
    n = 1 / base - 1
    body_axis_weight = (
        n
        * (torch.sigmoid(x + body_ratio // 2) + torch.sigmoid(-x + body_ratio // 2) - 1)
        + 1
    ) / (n + 1)
    return body_axis_weight.reshape(1, plot_n - 2)


def annealing_function(epoch, T, speed=0.2, start=0, slope=1):
    x = torch.arange(-T / 2 + 0.5, T / 2 + 0.5)
    annealing_weight = torch.sigmoid(
        (torch.abs(x) - T / 2 + start + epoch * speed) * slope
    )
    return annealing_weight


def worm_width_all(
    plot_n: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Get all worm widths when segment number is given."""
    device = alpha.device
    worm_x = torch.linspace(-1.0, 1.0, plot_n - 1).to(device)
    delta_sigmoid = torch.sigmoid(delta)
    gamma_e = 0.5 + torch.exp(gamma)
    worm_x_abs = torch.abs(worm_x)
    width = alpha * torch.sqrt(
        1
        + 1e-5
        - worm_x_abs ** (2 * gamma_e)
        * (
            1
            + (2 * gamma_e) * delta_sigmoid
            - (2 * gamma_e) * delta_sigmoid * worm_x_abs
        )
    )
    return width


def pixel_value_from_dist_max(
    max_dist: torch.Tensor,
    contrast: float = 1.2,
    sharpness: float = 2.0,
) -> torch.Tensor:
    """Get pixel value when distance from midline is given."""
    return 255 * (contrast * (torch.sigmoid(max_dist * sharpness) - 0.5) + 0.5)


PIXEL_MINIMUM = 255 * -0.1
PIXEL_MAXIMUM = 255 * 1.1


def make_single_worm(
    x: torch.Tensor,
    y: torch.Tensor,
    width: int,
    height: int,
    pixel_matrix: torch.Tensor,
) -> torch.Tensor:
    cent_x = ((x[:-1] + x[1:]) / 2).long()
    cent_y = ((y[:-1] + y[1:]) / 2).long()
    n_pts, diameter = pixel_matrix.shape[:2]
    radius = diameter // 2
    pad_image = torch.full(
        (n_pts, height + diameter, width + diameter),
        fill_value=-25.5,
        device=x.device,
    )
    idx_x, idx_y, idx_z = torch.meshgrid(
        torch.arange(diameter, device=x.device),
        torch.arange(diameter, device=x.device),
        torch.arange(n_pts, device=x.device),
        indexing="ij",
    )

    all_idx_x = idx_x + cent_x.unsqueeze(0).unsqueeze(0)
    all_idx_y = idx_y + cent_y.unsqueeze(0).unsqueeze(0)

    pad_image_max, _ = pad_image.index_put(
        (idx_z.flatten(), all_idx_y.flatten(), all_idx_x.flatten()),
        pixel_matrix.flatten(),
        accumulate=True,
    ).max(dim=0)

    return pad_image_max[radius : radius + height, radius : radius + width]


def make_worm(
    x: torch.Tensor,
    y: torch.Tensor,
    width: int,
    height: int,
    worm_wid: torch.Tensor,
) -> torch.Tensor:
    # batchsize = x.shape[0]
    # n_pts, diameter = pixel_matrix.shape[:2]
    # radius = diameter // 2

    # cent_x = ((x[:, :-1] + x[:, 1:]) / 2).long()
    # cent_y = ((y[:, :-1] + y[:, 1:]) / 2).long()

    # pad_image = torch.full(
    #     (batchsize, n_pts, height + diameter, width + diameter),
    #     fill_value=-25.5,
    #     device=x.device,
    # )
    # idx_x, idx_y, idx_z, idx_t = torch.meshgrid(
    #     torch.arange(diameter, device=x.device),
    #     torch.arange(diameter, device=x.device),
    #     torch.arange(n_pts, device=x.device),
    #     torch.arange(batchsize, device=x.device),
    #     indexing="ij",
    # )

    # all_idx_x = idx_x + cent_x.unsqueeze(0).unsqueeze(0)
    # all_idx_y = idx_y + cent_y.unsqueeze(0).unsqueeze(0)

    # pad_image_max, _ = pad_image.index_put_(
    #     (idx_t.flatten(), idx_z.flatten(), all_idx_y.flatten(), all_idx_x.flatten()),
    #     pixel_matrix.flatten(),
    # ).max(dim=1)

    # return pad_image_max[:, radius : radius + height, radius : radius + width]

    # return torch.stack(
    #     [
    #         make_single_worm(
    #             x,
    #             y,
    #             height=height,
    #             width=width,
    #             pixel_matrix=pixel_matrix,
    #         )
    #         for x, y in zip(torch.unbind(x, dim=0), torch.unbind(y, dim=0))
    #     ],
    #     dim=0,
    # )
    H, W = height, width
    T, plot_n = x.shape
    device = x.device
    # midpoints of segments, length plot size
    cent_mid_x_3d = (x[:, :-1] + x[:, 1:]) / 2
    # midpoints of segments, length plot size
    cent_mid_y_3d = (y[:, :-1] + y[:, 1:]) / 2

    x_3d = torch.arange(W).reshape([1, 1, W]).to(device)
    cent_mid_x_3d = cent_mid_x_3d.reshape([T, plot_n - 1, 1]).to(torch.float32)
    delta_x = (cent_mid_x_3d - x_3d) ** 2

    y_3d = torch.arange(H).reshape([1, 1, H]).to(device)
    cent_mid_y_3d = cent_mid_y_3d.reshape([T, plot_n - 1, 1]).to(torch.float32)
    delta_y = (cent_mid_y_3d - y_3d) ** 2

    worm_wid_3d = worm_wid.reshape([1, plot_n - 1, 1, 1])
    segment_distance_3d = torch.sqrt(
        delta_x.reshape(T, plot_n - 1, 1, W) + delta_y.reshape(T, plot_n - 1, H, 1)
    )
    delta_max = (worm_wid_3d - segment_distance_3d).max(dim=1)
    image = pixel_value_from_dist_max(delta_max.values)
    return image


def make_model_image(cent_x, cent_y, theta, unitLength, image_info, params):
    T = image_info["image_shape"][0]
    device = image_info["device"]
    x = torch.cat(
        (
            torch.zeros((T, 1)).to(device),
            torch.cumsum(
                unitLength.reshape((T, 1)).to(device) * torch.cos(theta), dim=1
            ),
        ),
        dim=1,
    )
    x = (
        x - torch.mean(x, dim=1).reshape((T, 1)) + cent_x.reshape((T, 1))
    )  # length plot size +1
    y = torch.cat(
        (
            torch.zeros((T, 1)).to(device),
            torch.cumsum(
                unitLength.reshape((T, 1)).to(device) * torch.sin(theta), dim=1
            ),
        ),
        dim=1,
    )
    y = (
        y - torch.mean(y, dim=1).reshape((T, 1)) + cent_y.reshape((T, 1))
    )  # length plot size +1
    image = make_worm(x, y, image_info, params)
    return image


class Model(torch.nn.Module):
    def __init__(
        self,
        init_cx,
        init_cy,
        init_theta,
        init_unitLength,
        image_info,
        params,
    ):
        super().__init__()
        self.cx = nn.parameter.Parameter(init_cx)
        self.cy = nn.parameter.Parameter(init_cy)
        self.theta = nn.parameter.Parameter(init_theta)
        self.unitLength = nn.parameter.Parameter(init_unitLength)
        self.alpha = nn.parameter.Parameter(params["init_alpha"])
        self.gamma = nn.parameter.Parameter(params["init_gamma"])
        self.delta = nn.parameter.Parameter(params["init_delta"])
        self.image_info = image_info
        params["alpha"] = self.alpha
        params["gamma"] = self.gamma
        params["delta"] = self.delta
        self.params = params

    def forward(self):
        device = self.alpha.device
        T, im_height, im_width = self.image_info["image_shape"]
        plot_n = self.params["plot_n"]
        worm_wid = worm_width_all(
            plot_n,
            self.alpha,
            self.gamma,
            self.delta,
        )
        # worm_wid_max = worm_wid.max().long().item() + 15
        # distance_matrix = make_distance_matrix(worm_wid_max).to(device)

        # distance_matrix_3d = worm_wid.unsqueeze(-1).unsqueeze(
        #     -1
        # ) - distance_matrix.unsqueeze(0)
        # pixel_matrix = pixel_value_from_dist_max(distance_matrix_3d)

        x = torch.cat(
            (
                torch.zeros((T, 1)).to(device),
                torch.cumsum(
                    self.unitLength.reshape((T, 1)).to(device) * torch.cos(self.theta),
                    dim=1,
                ),
            ),
            dim=1,
        )
        x = (
            x - torch.mean(x, dim=1).reshape((T, 1)) + self.cx.reshape((T, 1))
        )  # length plot size +1
        y = torch.cat(
            (
                torch.zeros((T, 1)).to(device),
                torch.cumsum(
                    self.unitLength.reshape((T, 1)).to(device) * torch.sin(self.theta),
                    dim=1,
                ),
            ),
            dim=1,
        )
        y = (
            y - torch.mean(y, dim=1).reshape((T, 1)) + self.cy.reshape((T, 1))
        )  # length plot size +1
        image = make_worm(x, y, im_width, im_height, worm_wid)
        return image


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=30, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.delta = delta

    def __call__(self, loss, model):
        if loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0


def train3(
    model,
    real_image,
    optimizer,
    params,
    device,
    init_data,
    output_path,
    output_name,
    is_nont=True,
):
    T = real_image.shape[0]
    speed = params["speed"]
    epochs = int(T / (2 * speed) + params["epoch_plus"])
    continuity_loss_weight = params["continuity_loss_weight"]
    smoothness_loss_weight = params["smoothness_loss_weight"]
    length_loss_weight = params["length_loss_weight"]
    center_loss_weight = params["center_loss_weight"]
    save_progress_freq = params["save_progress_freq"]
    init_cx = init_data[0].to(device)
    init_cy = init_data[1].to(device)
    unitL = init_data[2]
    annealing_weight = torch.ones(T).to(device)
    body_axis_weight = body_axis_function(params["body_ratio"], params["plot_n"]).to(
        device
    )
    model.alpha.requires_grad = False
    model.gamma.requires_grad = False
    model.delta.requires_grad = False
    early_stopping = EarlyStopping()
    if not torch.is_tensor(real_image):
        real_image = torch.tensor(real_image).to(device)

    # main optimization
    for e in range(epochs):
        model_image = model().to(device)
        optimizer.zero_grad()

        if is_nont:
            annealing_weight = annealing_function(e, T, speed).to(device)
        image_loss = torch.mean(
            ((model_image - real_image) ** 2) * annealing_weight.reshape([T, 1, 1])
        )
        continuity_loss = continuity_loss_weight * torch.mean(
            (model.theta[:-1, :] - model.theta[1:, :]) ** 2
        )
        smoothness_loss = smoothness_loss_weight * torch.mean(
            (
                (model.theta[:, :-1] - model.theta[:, 1:])
                * body_axis_weight
                * annealing_weight.reshape([T, 1])
            )
            ** 2
        )
        length_loss = (
            10000
            * length_loss_weight
            * torch.mean((model.unitLength[:-1] - model.unitLength[1:]) ** 2)
        )
        center_loss = (
            center_loss_weight
            / unitL
            * torch.mean((model.cx - init_cx) ** 2 + (model.cy - init_cy) ** 2)
        )
        loss = (
            image_loss + continuity_loss + smoothness_loss + length_loss + center_loss
        )
        loss.backward()
        if torch.min(annealing_weight) > 0.99:
            early_stopping(loss.item(), model)
        del loss
        if early_stopping.early_stop:
            if params["ShowProgress"]:
                print("Early stopping at epoch +{}.".format(e - int(T / (2 * speed))))
            break
        optimizer.step()

        if e % save_progress_freq > 0:
            continue

        # Save Progres
        save_progress(
            model_image,
            output_path,
            output_name,
            params,
            txt="id{}_{}".format(params["id"], e),
        )
        if not params["ShowProgress"]:
            continue

        # Show Progress
        print(
            "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                image_loss.item(),
                continuity_loss.item(),
                smoothness_loss.item(),
                length_loss.item(),
                center_loss.item(),
            )
        )
        show_image(model_image, params["num_t"], title=f"epoch {e}")

    model.alpha.requires_grad = True
    model.gamma.requires_grad = True
    model.delta.requires_grad = True
    body_axis_weight = body_axis_function(
        params["body_ratio"],
        params["plot_n"],
        base=0.3,
    ).to(device)
    early_stopping = EarlyStopping()

    # minor adjustment
    for e in range(params["epoch_plus"]):
        model_image = model().to(device)
        optimizer.zero_grad()

        image_loss = torch.mean((model_image - real_image) ** 2)
        continuity_loss = continuity_loss_weight * torch.mean(
            (model.theta[:-1, :] - model.theta[1:, :]) ** 2
        )
        smoothness_loss = smoothness_loss_weight * torch.mean(
            ((model.theta[:, :-1] - model.theta[:, 1:]) * body_axis_weight) ** 2
        )
        length_loss = (
            10000
            * length_loss_weight
            * torch.mean((model.unitLength[:-1] - model.unitLength[1:]) ** 2)
        )
        loss = image_loss + continuity_loss + smoothness_loss + length_loss
        loss.backward()
        early_stopping(loss.item(), model)
        del loss
        if early_stopping.early_stop:
            if params["ShowProgress"]:
                print("Minor adjustment done.")
            break
        optimizer.step()

    if not params["ShowProgress"]:  # Show Progress
        print(
            "{:.2f} {:.2f} {:.2f} {:.2f}".format(
                image_loss.item(),
                continuity_loss.item(),
                smoothness_loss.item(),
                length_loss.item(),
            )
        )
        show_image(model_image, params["num_t"], title="final")

    save_progress(
        model_image,
        output_path,
        output_name,
        params,
        txt="id{}_{}".format(params["id"], "final"),
    )

    losses = [
        torch.mean((model_image - real_image) ** 2, axis=(1, 2)),
        continuity_loss_weight
        * torch.mean((model.theta[:-1, :] - model.theta[1:, :]) ** 2, axis=1),
        smoothness_loss_weight
        * torch.mean((model.theta[:, :-1] - model.theta[:, 1:]) ** 2, axis=1),
        length_loss_weight * ((model.unitLength[:-1] - model.unitLength[1:]) ** 2),
        center_loss_weight * ((model.cx - init_cx) ** 2 + (model.cy - init_cy) ** 2),
    ]
    for i in range(len(losses)):
        losses[i] = losses[i].clone().detach().cpu().numpy()
    return losses


def make_plot(theta, unitLength, x_cent, y_cent, x_st=0, y_st=0):
    T = theta.shape[0]
    x = np.hstack((np.zeros((T, 1)), np.cumsum(unitLength * np.cos(theta), axis=1)))
    y = np.hstack((np.zeros((T, 1)), np.cumsum(unitLength * np.sin(theta), axis=1)))
    x = x - np.mean(x, axis=1).reshape((T, 1)) + x_cent.reshape((T, 1)) + x_st
    y = y - np.mean(y, axis=1).reshape((T, 1)) + y_cent.reshape((T, 1)) + y_st
    return x, y


def get_shape_params(shape_params, params):
    T_sum = 0
    params["init_alpha"] = 0
    params["init_gamma"] = 0
    params["init_delta"] = 0
    for para in shape_params:
        T_sum += para[0]
        params["init_alpha"] += para[0] * para[1]
        params["init_gamma"] += para[0] * para[2]
        params["init_delta"] += para[0] * para[3]
    return (
        params["init_alpha"] / T_sum,
        params["init_gamma"] / T_sum,
        params["init_delta"] / T_sum,
    )


def loss_compare(loss_pair):
    im_select = int(max(loss_pair[0][0]) > max(loss_pair[1][0]))
    con_select = int(max(loss_pair[0][1]) > max(loss_pair[1][1]))
    smo_select = int(max(loss_pair[0][2]) > max(loss_pair[1][2]))
    if im_select + con_select + smo_select == 3:
        return 1
    if im_select + con_select + smo_select == 0:
        return 0
    q75, q50, q25 = np.percentile(loss_pair[im_select][0], [75, 50, 25])
    im_exrate = (max(loss_pair[1 - im_select][0]) - q50) / (q75 - q25)
    q75, q50, q25 = np.percentile(loss_pair[im_select][1], [75, 50, 25])
    con_exrate = (max(loss_pair[1 - con_select][1]) - q50) / (q75 - q25)
    q75, q50, q25 = np.percentile(loss_pair[im_select][2], [75, 50, 25])
    smo_exrate = (max(loss_pair[1 - smo_select][2]) - q50) / (q75 - q25)
    exrate_loss = np.argmax(np.array([im_exrate, con_exrate, smo_exrate]))
    return [im_select, con_select, smo_select][exrate_loss]


def show_loss_plot(losses, title=""):
    _ = losses[0].shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses[0], label="im")
    ax.plot(losses[1], label="con")
    ax.plot(losses[2], label="smo")
    ax.plot(losses[3], label="len")
    ax.plot(losses[4], label="cen")
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("frames", fontsize=20)
    ax.set_xlabel("loss", fontsize=20)
    ax.legend()
    # plt.show()


def find_losslarge_area(losses_all):
    losslarge_area = np.zeros(len(losses_all))
    for i in range(3):
        lossi = []
        for j in range(len(losses_all)):
            lossi = lossi + list(losses_all[j][i])
        q75, q50, q25 = np.percentile(lossi, [75, 50, 25])
        for j in range(len(losses_all)):
            if np.max(losses_all[j][i]) - q50 > (q75 - q25) * 4:
                losslarge_area[j] += 1
    return losslarge_area


### arrange and save data ###


def judge_head_amplitude(x, y):
    """Judge which tip is head based on variance of body curve rate."""
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    theta = np.arctan2(dy, dx)

    curve_rate_var = ((theta[:, 1:] - theta[:, :-1] + np.pi) % (2 * np.pi) - np.pi).var(
        axis=0
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(curve_rate_var)
    ax.set_xlabel("body segment", fontsize=20)
    ax.set_ylabel("curve rate var", fontsize=20)
    # plt.show()
    idx15per = int(np.round(x.shape[1] * 0.15))
    idx20per = int(np.round(x.shape[1] * 0.20))
    curve_mean1 = curve_rate_var[idx15per : idx20per + 1].mean()
    curve_mean2 = curve_rate_var[-idx20per - 1 : -idx15per].mean()

    x_rev, y_rev = x[:, ::-1], y[:, ::-1]
    # Reversed
    if curve_mean1 < curve_mean2:
        x, x_rev = x_rev, x
        y, y_rev = y_rev, y

    return x, y, x_rev, y_rev


def judge_head_frequency(x, y):
    """Judge which tip is head based on frequency of body curve rate."""

    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    theta = np.arctan2(dy, dx)

    curve_rate = (theta[:, 1:] - theta[:, :-1] + np.pi) % (2 * np.pi) - np.pi
    T = curve_rate.shape[0]

    # fast fourier transform
    spa = np.abs(np.fft.fft(curve_rate, axis=0))

    # the latter half of fourier power spectrum is the same as the first half
    T2 = int(np.ceil((T - 1) / 2))
    cut = int(np.round(x.shape[1] / 20))  # cut end 5% of worm
    spat = spa[1 : (T2 + 1), cut : x.shape[1] - cut]

    # cutoff high-freq area with values < peak/10
    sp_sum = np.sum(spat, axis=1)
    freq_cut = np.max(np.where(sp_sum > np.max(sp_sum) / 10)[0])
    spat = spat[: freq_cut + 1, :]
    # print('freq_cut =', freq_cut)

    # calculate correlation
    xmean = np.sum(spat.sum(axis=0) / spat.sum() * np.arange(spat.shape[1]))
    ymean = np.sum(spat.sum(axis=1) / spat.sum() * np.arange(spat.shape[0]))
    xcoord = (np.arange(spat.shape[1]) - xmean).reshape((1, -1))
    ycoord = (np.arange(spat.shape[0]) - ymean).reshape((-1, 1))
    cor = (
        np.sum(spat * xcoord * ycoord)
        / np.sqrt(np.sum(spat * xcoord * xcoord))
        / np.sqrt(np.sum(spat * ycoord * ycoord))
    )
    # print('correlation =', cor)

    # show power spectrum plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(spat)
    ax.set_aspect(0.1)
    ax.set_xlabel("body segment", fontsize=20)
    ax.set_ylabel("peak curve freq", fontsize=20)
    ax.set_title(f"Correlation = {cor:.3g}")
    # plt.show()

    x_rev, y_rev = x[:, ::-1], y[:, ::-1]
    # Reversed
    if cor > 0:
        x, x_rev = x_rev, x
        y, y_rev = y_rev, y
    return x, y, x_rev, y_rev


def clear_dir(output_path, foldername):
    if os.path.isdir(os.path.join(output_path, foldername)):
        shutil.rmtree(os.path.join(output_path, foldername))
    os.mkdir(os.path.join(output_path, foldername))


def cancel_reduction(x, y, n_input_images, start_T, end_T, Tscaled_ind, plot_n):
    if end_T == 0:
        end_T = n_input_images - 1
    if len(Tscaled_ind) == end_T - start_T + 1:
        return x, y

    x_splined = np.zeros((end_T - start_T + 1, plot_n))
    y_splined = np.zeros((end_T - start_T + 1, plot_n))

    # interpolation
    div_linespace = np.arange(end_T - start_T + 1)
    Tscaled_dif_ind = [ind - start_T for ind in Tscaled_ind]
    for i in range(plot_n):
        x_splined[:, i] = np.interp(div_linespace, Tscaled_dif_ind, x[:, i])
        y_splined[:, i] = np.interp(div_linespace, Tscaled_dif_ind, y[:, i])
    return x_splined, y_splined


def straigthen_multi(
    src: NDArray,
    x: NDArray,
    y: NDArray,
    width: int,
    height: int,
):
    """
    Straightens an image based on given x and y coordinates using affine transformation and interpolation.

    Args:
        src: Input image as a NumPy array [N, H, W].
        x: x-coordinates of points to be straightened [N, x].
        y: y-coordinates of points to be straightened [N, y].
        width: Desired width of the straightened image.
        height: Desired height of the straightened image.

    Returns:
        The straightened image as a NumPy array [N, height, width].
    """

    assert src.ndim == 3, "The shape of source images is not (number, height, width)"
    assert x.shape == y.shape, "The coordinates of x and y have different shape."
    N, H, W = src.shape
    assert (
        x.shape[0] == N
    ), "The number of frames to be straightened is different from given coordinates."

    dist = np.zeros_like(x)
    dist[:, 1:] = np.sqrt((x[:, 1:] - x[:, :-1]) ** 2 + (y[:, 1:] - y[:, :-1]) ** 2)

    acc_dist = np.cumsum(dist, axis=1)
    src_xy = np.zeros((N, width, 2))
    xy = np.stack([x, y], axis=-1)
    out_xcoords = np.arange(width)

    # Interpolate x and y coordinates (T, width) based on accumulated distances
    for i in range(N):
        f_xy = CubicSpline(acc_dist[i], xy[i])
        src_xy[i] = f_xy(out_xcoords)

    # Calculate vectors (T, width-1, 2) between consecutive x and y coordinates
    dxy = np.diff(src_xy, axis=1)

    # Padding to each end with same values (T, width+1, 2)
    dxy = np.pad(
        dxy,
        pad_width=((0, 0), (1, 1), (0, 0)),
        mode="edge",
    )

    # Compute average vectors for each point (including boundary points)
    dxya = (dxy[:, 1:] + dxy[:, :-1]) / 2.0  # (T, width)

    # Tangential vectors to the centerlines
    xt_vec = -dxya[:, :, 1]
    yt_vec = dxya[:, :, 0]

    # Calculate normalized tangential vectors to the centerlines
    vec_norm = np.sqrt((dxya**2).sum(axis=-1))
    xt_norm = xt_vec / vec_norm
    yt_norm = yt_vec / vec_norm

    # Create a grid of y-coordinates for interpolation
    y_grid = np.arange(height) - (height - 1) / 2  # (height,)

    src_x = src_xy[:, :, 0]
    src_y = src_xy[:, :, 1]

    # Calculate new x and y coordinates based on tangential vectors and y-grid
    # (T, 1, width) * (1, height, 1) + (T, 1, width)
    gx = xt_norm[:, None, :] * y_grid[None, :, None] + src_x[:, None, :]
    gy = yt_norm[:, None, :] * y_grid[None, :, None] + src_y[:, None, :]

    # Let gx and gy normalize within [-1., 1.]
    gx = 2 * gx / W - 1.0
    gy = 2 * gy / H - 1.0
    gxy = np.stack((gx, gy), axis=-1).reshape((-1, height, width, 2))

    # Create a 2D grid for interpolation
    src_t = torch.from_numpy(src).reshape((N, -1, H, W)).float()
    grid = torch.from_numpy(gxy).float()

    straigthen_dst = F.grid_sample(src_t, grid, align_corners=True)
    straigthen_dst = (
        torch.clamp(straigthen_dst, src.min(), src.max())
        .detach()
        .numpy()
        .astype(src.dtype)
        .reshape(N, height, width)
    )

    return straigthen_dst


def straigthen(
    src: NDArray,
    x: NDArray,
    y: NDArray,
    width: int,
    height: int,
):
    """
    Straightens an image based on given x and y coordinates using affine transformation and interpolation.

    Args:
        src: Input image as a NumPy array [H, W].
        x: x-coordinates of points to be straightened.
        y: y-coordinates of points to be straightened.
        width: Desired width of the straightened image.
        height: Desired height of the straightened image.

    Returns:
        The straightened image as a NumPy array [N, height, width].
    """

    assert src.ndim == 2, "The shape of source images is not (height, width)"
    assert x.shape == y.shape, "The coordinates of x and y have different shape."
    H, W = src.shape

    dist = np.zeros_like(x)
    dist[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)

    acc_dist = np.cumsum(dist)
    src_xy = np.zeros((width, 2))
    xy = np.stack([x, y], axis=-1)
    out_xcoords = np.arange(width)

    # Interpolate x and y coordinates based on accumulated distances
    f_xy = CubicSpline(acc_dist, xy)
    src_xy = f_xy(out_xcoords)

    # Calculate vectors (width-1, 2) between consecutive x and y coordinates
    dxy = src_xy[1:] - src_xy[:-1]

    # Padding to each end with same values (width+1, 2)
    dxy = np.pad(
        dxy,
        pad_width=((1, 1), (0, 0)),
        mode="edge",
    )

    # Compute average vectors for each point (including boundary points)
    dxya = (dxy[1:] + dxy[:-1]) / 2.0  # (T, width)

    # Tangential vectors to the centerlines
    xt_vec = -dxya[:, 1]
    yt_vec = dxya[:, 0]

    # Calculate normalized tangential vectors to the centerlines
    vec_norm = np.sqrt((dxya**2).sum(axis=-1))
    xt_norm = xt_vec / vec_norm
    yt_norm = yt_vec / vec_norm

    # Create a grid of y-coordinates for interpolation
    y_grid = np.arange(height) - (height - 1) / 2  # (height,)

    src_x = src_xy[:, 0]
    src_y = src_xy[:, 1]

    # Calculate new x and y coordinates based on tangential vectors and y-grid
    # (1, width) * (height, 1) + (1, width)
    gx = xt_norm[None, :] * y_grid[:, None] + src_x[None, :]
    gy = yt_norm[None, :] * y_grid[:, None] + src_y[None, :]

    # Let gx and gy normalize within [-1., 1.]
    gx = 2 * gx / W - 1.0
    gy = 2 * gy / H - 1.0
    gxy = np.stack((gx, gy), axis=-1).reshape((-1, height, width, 2))

    # Create a 2D grid for interpolation
    src_t = torch.from_numpy(src).reshape((1, -1, H, W)).float()
    grid = torch.from_numpy(gxy).float()

    straigthen_dst = F.grid_sample(src_t, grid, mode="bicubic", align_corners=False)
    straigthen_dst = (
        torch.clamp(straigthen_dst, src.min(), src.max())
        .detach()
        .numpy()
        .astype(src.dtype)
        .reshape(height, width)
    )

    return straigthen_dst
