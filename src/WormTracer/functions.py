from __future__ import annotations

import collections
import itertools
import logging
import math
import os
import typing
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import distance_matrix
from scipy.special import expit as np_sigmoid
from skimage import morphology

if typing.TYPE_CHECKING:
    from typing import (
        Any,
        Generator,
        Optional,
        Sequence,
        Set,
    )

    import numpy.typing as npt


logger = logging.getLogger(__name__)


def show_image(
    image: npt.NDArray | torch.Tensor,
    num_t: int = 5,
    title: str = "",
    x: None | npt.NDArray = None,
    y: None | npt.NDArray = None,
    x2: None | npt.NDArray = None,
    y2: None | npt.NDArray = None,
):
    if not __debug__:
        return

    T = image.shape[0]
    num_t = min(num_t, T)
    t_sparse = np.linspace(0, T - 1, min(num_t, T), dtype=int)
    if torch.is_tensor(image):
        image = image.clone().detach().cpu().numpy()
    if torch.is_tensor(x) and torch.is_tensor(y):
        x = x.clone().detach().cpu().numpy()
        y = y.clone().detach().cpu().numpy()

    if torch.is_tensor(x2) and torch.is_tensor(y2):
        x2 = x2.clone().detach().cpu().numpy()
        y2 = y2.clone().detach().cpu().numpy()

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
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            axes[i // 5, i % 5].scatter(x[t_sparse[i]], y[t_sparse[i]], c="r", s=30)
        if isinstance(x2, np.ndarray) and isinstance(y2, np.ndarray):
            axes[i // 5, i % 5].scatter(x2[t_sparse[i]], y2[t_sparse[i]], c="y", s=30)
    # plt.show()
    plt.close(fig)


def get_guide_points(
    guide_files: Sequence[str | os.PathLike],
    TScale_ind: Sequence[int],
    plot_n: int,
    n_frame: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    guide_file_ext = (".csv", ".h5")
    guide_file_map = collections.defaultdict(list)

    for file in guide_files:
        file = Path(file)
        if file.suffix not in guide_file_ext:
            raise ValueError("guide_files must be two csv files or one hdf file")
        guide_file_map[file.suffix].append(file)

    if len(guide_file_map[".csv"]) == 2:
        fx, fy = guide_file_map[".csv"]
        if "_y" in fx.name:
            fx, fy = fy, fx
        logger.info(f"guide_x = {os.fspath(fx)}\nguide_y = {os.fspath(fy)}")
        guide_x = np.loadtxt(fx, delimiter=",")
        guide_y = np.loadtxt(fy, delimiter=",")
    elif len(guide_file_map[".h5"]) == 1:
        f = guide_file_map[".h5"][0]
        logger.info(f"guide_hdf = {os.fspath(f)}")
        with h5py.File(f, "r") as handler:
            guide_x = np.asarray(handler["x"])
            guide_y = np.asarray(handler["y"])
    else:
        raise ValueError("guide_files must be two csv files or one hdf file")

    assert guide_x.shape == guide_y.shape, "guide_x and guide_y have different shape."
    assert guide_x.shape[0] == n_frame, (
        "guide_x and guide_y have different length from input images"
    )
    guide_x = guide_x[TScale_ind]
    guide_y = guide_y[TScale_ind]
    guide_idx = np.where(np.all(np.isfinite(guide_x), axis=1))[0]
    if guide_idx.size == 0:
        raise ValueError(f"guide_files is empty at TScale_ind: {TScale_ind}")

    guide_x = np.nan_to_num(guide_x, copy=False)
    guide_y = np.nan_to_num(guide_y, copy=False)
    # Interpolate the guide points if required.
    if guide_x.shape[1] != plot_n:
        arc = np.linspace(0, 1, guide_x.shape[1])
        xy_func = CubicSpline(arc, (guide_x, guide_y), axis=2)
        guide_x, guide_y = xy_func(np.linspace(0.0, 1.0, plot_n))

    return guide_x, guide_y, guide_idx


def get_property(
    filenames: Sequence[Path], rescale: float
) -> tuple[Sequence[int], bool, bool, int]:
    if filenames[0].name.lower().endswith((".tif", ".tiff")):
        try:
            ims = tifffile.imread(filenames[0])
        except ValueError as e:
            err_msg = "This file is not a valid ImageJ format. Please save your Tiff file using ImageJ: {}"
            raise ValueError(err_msg.format(e))
    else:
        _, ims = cv2.imreadmulti(filename=os.fspath(filenames[0]), mats=[], flags=0)
        ims = np.asarray(ims)

    im = np.asarray(ims[0])
    if not np.all((0 == im) | (im == 255)):
        logger.warning("Warning! : Input images seem not to be binary.")
    if not math.isclose(rescale, 1.0, rel_tol=1e4):
        im = cv2.resize(
            im,
            dsize=None,
            fy=rescale,
            fx=rescale,
            interpolation=cv2.INTER_NEAREST,
        )
    white_pixel = int(
        np.sum(im[0, :-1]) // 255
        + np.sum(im[-1, 1:]) // 255
        + np.sum(im[1:, 0]) // 255
        + np.sum(im[:-1, -1]) // 255
    )
    # if sum of white pixel is larger than height + width
    Worm_is_black = white_pixel > sum(im.shape[:2])
    multi_flag = ims.shape[0] > 1
    n_input_images = len(ims) if multi_flag else len(filenames)
    return im.shape, Worm_is_black, multi_flag, n_input_images


def read_serial_images(
    filenames: Sequence[Path],
    Tscaled_ind: Sequence[int],
):
    for ind in Tscaled_ind:
        yield cv2.imread(os.fspath(filenames[ind]), cv2.IMREAD_GRAYSCALE)


def load_image(
    filenames: Sequence[Path],
    rescale: float,
    Worm_is_black: bool,
    multi_flag: bool,
    Tscaled_ind: Sequence[int],
) -> tuple[npt.NDArray, int, int]:
    """read images and get skeletonized plots"""
    if multi_flag:
        # multipage tiff file
        if filenames[0].name.lower().endswith((".tif", ".tiff")):
            try:
                ims = tifffile.imread(filenames[0])
            except ValueError as e:
                err_msg = "This file is not a valid ImageJ format. Please save your Tiff file using ImageJ: {}"
                raise ValueError(err_msg.format(e))
        else:
            # Unknown Data Type
            _, ims = cv2.imreadmulti(os.fspath(filenames[0]), flags=0)
        # Use generator instead of read image
        ims_gen = (ims[ind] for ind in Tscaled_ind)
    else:
        ims_gen = read_serial_images(
            filenames, Tscaled_ind
        )  # serial-numbered image files

    def preprocess(im: npt.NDArray) -> npt.NDArray:
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

    ims_subset = [preprocess(im) for im in ims_gen]
    imagestack = np.asarray(ims_subset)
    imagestack, y_st, x_st = trim_image(imagestack)
    return imagestack, y_st, x_st


def calc_xy_and_prewidth(
    imagestack: npt.NDArray,
    plot_n: int,
    x_st: float,
    y_st: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, float]:
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
        if __debug__:
            bar = "\rget_skeleton and width:[{:<100}] {}/{}".format(
                "▉" * round((t + 1) * 100 / T), t + 1, T
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


def get_skeleton(im: npt.NDArray, plot_n: int) -> tuple[npt.NDArray, npt.NDArray]:
    """skeletonize image and get splined plots"""

    # skeletonize image
    im_filled = ndi.binary_fill_holes(im)
    im_skeleton = morphology.skeletonize(im_filled)
    point_list = np.argwhere(im_skeleton == 1)

    if len(point_list) == 0:
        raise ValueError("Original image is empty")
    elif len(point_list) == 1:
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
    x_splined = np.interp(div_linespace, arclen, plots[:, 1])
    y_splined = np.interp(div_linespace, arclen, plots[:, 0])

    return x_splined, y_splined


def get_width(im: npt.NDArray, x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Get width of the object by measure distance of centerline to the object's surface."""
    im_filled = ndi.binary_fill_holes(im)
    assert im_filled is not None, "Err after binary_fill_holes"
    im_dist = ndi.distance_transform_edt(im_filled)
    coordinates = [y, x]  # Shape (2, plot_n)
    #
    dist_to_zero = ndi.map_coordinates(im_dist, coordinates, order=1)
    wid = dist_to_zero.max()
    return wid


def flip_check(x: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Check if plots of head and tail is flipping."""
    assert x.shape == y.shape, "The coordinates of x and y have different shape."
    gap_headtail = np.mean(
        (x[1:, :] - x[:-1, :]) ** 2 + (y[1:, :] - y[:-1, :]) ** 2,
        axis=1,
    )
    gap_headtail_ex = np.mean(
        (x[1:, :] - x[:-1, ::-1]) ** 2 + (y[1:, :] - y[:-1, ::-1]) ** 2,
        axis=1,
    )
    ex_t = gap_headtail > gap_headtail_ex
    ex_r = np.bitwise_xor.accumulate(ex_t)
    idx = np.where(ex_r)[0] + 1
    x[idx, :] = x[idx, ::-1]
    y[idx, :] = y[idx, ::-1]
    return x, y


def trim_image(image: npt.NDArray, *, padding: int = 5) -> tuple[npt.NDArray, int, int]:
    """Cut images to minimum size."""
    assert image.ndim in (2, 3), "Only support 2D (Y, X) or 3D (Z, Y, X)"
    thresh = image > 0
    if image.ndim > 2:
        thresh = np.bitwise_or.reduce(thresh, axis=0)

    assert thresh.sum() > 0, (
        "Image has no signal, please confirm your image is properly loaded"
    )
    (ys, xs) = np.nonzero(thresh)
    max_h, max_w = thresh.shape

    x1 = max(xs.min() - padding, 0)
    x2 = min(xs.max() + padding, max_w)
    y1 = max(ys.min() - padding, 0)
    y2 = min(ys.max() + padding, max_h)

    if image.ndim == 2:
        return image[y1:y2, x1:x2], y1, x1
    else:
        return image[:, y1:y2, x1:x2], y1, x1


def make_theta_from_xy(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    assert x.ndim == 2 and (x.shape == y.shape), "x, y should be 2D ndarray"

    T, plot_n = x.shape
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    length = np.sqrt(dx**2 + dy**2)
    # polar coordinates
    theta = (dx + 1j * dy) / (length + 1e-8)

    if T > 1:
        dot_products = theta[1:, :] * np.conj(theta[:-1, :])
        global_alignment = np.sum(np.real(dot_products), axis=1)
        flip_mask = np.bitwise_xor.accumulate(global_alignment < 0.0)
        where_to_flip = np.where(flip_mask)[0] + 1
        theta[where_to_flip, :] *= -1.0

    # theta = np.arctan2(dy, dx)
    # # Arrange theta if the gap is largest than pi
    # # Adjust the middle theta between time point
    # mid = n_seqs // 2
    # theta /= theta[0, mid]
    # t_gap = theta[1:, mid] - theta[:-1, mid]
    # t_adjust = np.sign(t_gap) * 2 * np.pi
    # t_adjust[np.abs(t_gap) < np.pi] = 0
    # theta[1:, :] -= t_adjust.cumsum().reshape(-1, 1)
    #
    # gap = theta[:, 1:] - theta[:, :-1]
    # # adjust right-hand side of theta within same time points
    # r_gap = gap[:, mid:]
    # r_adjust = np.sign(r_gap) * 2 * np.pi
    # r_adjust[np.abs(r_gap) < np.pi] = 0
    # theta[:, mid + 1 :] -= r_adjust.cumsum(axis=1)
    #
    # # adjust left-hand side
    # l_gap = gap[:, :mid]
    # l_adjust = np.sign(l_gap * -1) * 2 * np.pi
    # l_adjust[np.abs(l_gap) < np.pi] = 0
    # l_adjust_rev = np.flip(l_adjust, axis=1)
    # theta[:, :mid] -= np.flip(np.cumsum(l_adjust_rev, axis=1), axis=1)
    return theta


### prepare for training ###


def pixel_value_from_dist_max_np(
    max_dist: npt.NDArray,
    contrast: float = 1.2,
    sharpness: float = 2.0,
) -> npt.NDArray:
    """Get pixel value when distance from midline is given."""
    return 255 * (contrast * (np_sigmoid(max_dist * sharpness) - 0.5) + 0.5)


def worm_width_all_np(
    *,
    plot_n: int,
    alpha: float,
    gamma: float,
    delta: float,
) -> npt.NDArray:
    """Get all worm widths when segment number is given."""
    # w_i  = α√(1-|h|^2γ (1+2γδ-2γδ|h|))
    worm_x = np.linspace(-1.0, 1.0, plot_n)  # h
    worm_x_abs = np.abs(worm_x)  # |h|
    delta_sig = np_sigmoid(delta)  # δ
    # 2γ
    gamma_e = 1 + 2 * np.exp(gamma)  # 2 * γ
    eps = 1e-5  # To avoid some floating points below zeros.
    width = alpha * np.sqrt(
        1 - worm_x_abs ** (gamma_e) * (1 + gamma_e * delta_sig * (1 - worm_x_abs)) + eps
    )
    return width


def make_distance_matrix_np(radius: int) -> npt.NDArray:
    diameter = radius * 2 + 1
    delta = (np.arange(diameter) - radius) ** 2
    distance_matrix = np.sqrt(delta[None, :] + delta[:, None])
    # let distance_kernel become circular
    distance_matrix[distance_matrix > radius] = np.inf
    return distance_matrix


def make_distance_matrix(radius: int) -> torch.Tensor:
    diameter = radius * 2 + 1
    delta = (torch.arange(diameter) - radius) ** 2
    distance_matrix = torch.sqrt(delta[None, :] + delta[:, None])

    # let distance_kernel become circular
    distance_matrix[distance_matrix > radius] = torch.inf
    return distance_matrix


def make_single_image(
    x: npt.NDArray,
    y: npt.NDArray,
    width: int,
    height: int,
    pixel_matrix: npt.NDArray,
) -> npt.NDArray:

    cent_x = (x[:-1] + x[1:]) / 2
    cent_x = cent_x.astype(np.int32)
    cent_y = (y[:-1] + y[1:]) / 2
    cent_y = cent_y.astype(np.int32)

    diameter = pixel_matrix.shape[1]
    radius = diameter // 2

    min_val = pixel_matrix.min()

    pad_image = np.full(
        (height + radius * 2, width + radius * 2),
        fill_value=min_val,
    )
    for i, j, pix in zip(cent_x, cent_y, pixel_matrix):
        pad_image[j : j + diameter, i : i + diameter] = np.maximum(
            pad_image[j : j + diameter, i : i + diameter],
            pix,
        )
    return pad_image[radius : radius + height, radius : radius + width]


def get_pixel_matrix(
    *,
    plot_n: int,
    alpha: float,
    gamma: float,
    delta: float,
):

    worm_wid = worm_width_all_np(
        plot_n=plot_n,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
    )
    max_radius = int(np.ceil(worm_wid.max())) + 2
    distance_matrix = make_distance_matrix_np(max_radius)

    distance_matrix_3d = worm_wid[:, None, None] - distance_matrix[None, :, :]
    return pixel_value_from_dist_max_np(distance_matrix_3d)


def make_image(
    x: npt.NDArray,
    y: npt.NDArray,
    width: int,
    height: int,
    pixel_matrix: npt.NDArray,
) -> npt.NDArray:
    """Create Model imaging using precalculated mask"""
    T = x.shape[0]

    image = np.zeros((T, height, width))

    for i in range(T):
        image[i, :, :] = make_single_image(
            x[i],
            y[i],
            width=width,
            height=height,
            pixel_matrix=pixel_matrix,
        )

    return image


def get_image_loss_max(
    best_fit_image: npt.NDArray,
    cx: float,
    cy: float,
    pixel_matrix: npt.NDArray,
) -> float:
    """Create bad image and get bad image_loss to judge complex area."""
    # Create a straigthen line to make a bad image that maximize the loss.
    plot_n = pixel_matrix.shape[0]
    x0 = np.ones((plot_n)) * cx
    y0 = np.ones((plot_n)) * cy
    height, width = best_fit_image.shape
    im0 = make_single_image(x0, y0, width, height, pixel_matrix)
    image_loss_max = float(np.mean((best_fit_image - im0) ** 2))
    return image_loss_max


class TrainingBlocks:
    class Block(typing.NamedTuple):
        start: int
        end: int
        idx: int
        is_complex: bool

        @property
        def size(self) -> int:
            return self.end - self.start + 1

        def __repr__(self) -> str:
            return f"({self.start:d}, {self.end:d}, {'complex' if self.is_complex else 'simple'})"

    def __init__(self, losses: npt.NDArray, relaxed: float, rigid: float):
        assert rigid > relaxed, "rigid margin must be greater than relaxed margin"

        # Use relaxed criteria to separate the blocks
        complex_area = losses > relaxed
        distinct_from_prev = np.zeros_like(complex_area).astype(bool)
        distinct_from_prev[1:] = complex_area[:-1] ^ complex_area[1:]
        # labeling all blocks in 0-index
        blocks = distinct_from_prev.astype(int).cumsum()
        # If the block contains a complex block, label it as complex block.
        complex_block_count = np.bincount(blocks, weights=(losses > rigid))
        complex_block = np.where(complex_block_count > 0)[0]

        # Get mask of complex_area that is segmented by relaxed criteria that fulfilled rigid criteria
        self.complex_area = np.isin(blocks, complex_block)
        self.simple_area = np.bitwise_not(self.complex_area)

        # Merge remaining non-complex blocks
        distinct_from_prev = np.zeros_like(complex_area).astype(bool)
        distinct_from_prev[1:] = self.complex_area[:-1] ^ self.complex_area[1:]
        merged_blocks = distinct_from_prev.astype(int).cumsum()
        self.blocks = merged_blocks
        complex_block_count = np.bincount(merged_blocks, weights=self.complex_area)
        complex_block = np.where(complex_block_count > 0)[0]
        self.complex_block = complex_block

        label = np.unique(self.blocks)
        self.nblock = len(label)

    @property
    def nframe(self) -> int:
        return len(self.blocks)

    def batch_iter(
        self, batchsize: Optional[int] = None
    ) -> Generator[TrainingBlocks.Block, None, None]:
        """Return an iterator that yields Block(idx, is_complex, start, end) within the batchsize"""
        block_sizes = np.bincount(self.blocks)
        # it will return the index of first occurence.
        label, onset = np.unique(self.blocks, return_index=True)
        offset = onset + block_sizes - 1

        mask = self.complex_area[onset]
        if batchsize is None:
            # We set the batchsize greater than the maximum block.
            batchsize = int(block_sizes.max()) + 1

        counter = itertools.count()
        for is_complex, start, end in zip(mask, onset, offset):
            # Since end is inclusive, we should add one to include the end index.
            for st in range(start, end + 1, batchsize):
                yield TrainingBlocks.Block(
                    start=st,
                    end=min(st + batchsize - 1, end),
                    idx=next(counter),
                    is_complex=is_complex,
                )


def get_use_blocks(
    image_losses: npt.NDArray,
    image_loss_max: float,
) -> TrainingBlocks:
    """
    Judge frames complex or not and get span for training.
    """
    # the criteria to filter complex area
    image_losses_min = np.min(image_losses)
    rigid = 0.4 * image_loss_max + 0.6 * image_losses_min
    relaxed = 0.2 * image_loss_max + 0.8 * image_losses_min
    return TrainingBlocks(losses=image_losses, relaxed=relaxed, rigid=rigid)


### training ###
def make_progress_image(
    image: np.ndarray | torch.Tensor,
    num_t: int = 20,
) -> np.ndarray:
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
        merge = np.hstack(list(chunk))
        progress_image[i * H : (i + 1) * H, : merge.shape[1]] = merge
    return progress_image


def save_progress(
    image: torch.Tensor | npt.NDArray,
    output_path: os.PathLike,
    output_name: str,
    start: int,
    end: int,
    num_t: int,
    txt="real",
) -> None:
    progress_image = make_progress_image(image, num_t)
    filename = os.path.join(
        output_path,
        output_name + "_progress_image",
        "{}-{}_{}.png".format(start, end, txt),
    )
    cv2.imwrite(filename, progress_image)


def get_center(binimg: torch.Tensor | npt.NDArray):
    """Calculate center of images."""
    if torch.is_tensor(binimg):
        binimg = binimg.clone().detach().cpu().numpy()
    ys, xs = np.where(binimg == np.max(binimg))
    x = np.average(xs)
    y = np.average(ys)
    return x, y


def set_init_xy(
    imstack: torch.Tensor | npt.NDArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set init center plots for training."""
    assert imstack.ndim == 3, "real_image is not 3-d array (T, H, W)"
    if torch.is_tensor(imstack):
        imstack = imstack.clone().detach().cpu().numpy()
    imstack_max = np.max(imstack, axis=(1, 2), keepdims=True)
    (zs, ys, xs) = np.where(imstack == imstack_max)
    count_per_frame = np.bincount(zs)
    init_cx = np.bincount(zs, weights=xs) / count_per_frame
    init_cy = np.bincount(zs, weights=ys) / count_per_frame

    # This parts is to normalize the center coordinates to (-1, 1)
    # _, H, W = imstack.shape
    # scale = max(H, W)

    # init_cx = (init_cx * 2).astype("f4") / scale - 1.0
    # init_cy = (init_cy * 2).astype("f4") / scale - 1.0

    return torch.from_numpy(init_cx.astype("f4")), torch.from_numpy(
        init_cy.astype("f4")
    )


def find_minimal_winding_number(theta1: npt.NDArray, theta2: npt.NDArray) -> int:
    # 1. Find the average numerical difference across all elements.
    avg_diff = np.mean(theta1 - theta2)
    # 2. Convert the average difference into a fractional number of 2*pi cycles.
    frac_shift = avg_diff / (2 * np.pi)
    # 3. Round to the nearest integer to find the optimal winding number (k).
    return int(np.round(frac_shift))


def make_polar_cand(
    theta_begin: npt.NDArray, theta_end: npt.NDArray
) -> tuple[np.NDArray, np.NDArray]:
    loss_fwd = np.sum(np.abs(theta_begin - theta_end))
    theta_end_rev = theta_end[::-1] * -1
    loss_rev = np.sum(np.abs(theta_begin - theta_end_rev))

    if loss_fwd < loss_rev:
        return theta_end, theta_end_rev
    else:
        return theta_end_rev, theta_end


def make_theta_cand(
    theta_begin: npt.NDArray, theta_end: npt.NDArray
) -> tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]:
    k_normal = find_minimal_winding_number(theta_begin, theta_end)
    shift_fw = (np.array([0, 1, -1]) + k_normal) * 2 * np.pi

    theta_cands_normal = shift_fw[:, None] + theta_end
    loss_normal = np.sum((theta_cands_normal - theta_begin) ** 2, axis=1)
    sort_indices = np.argsort(loss_normal)
    theta_cands_normal = theta_cands_normal[sort_indices]

    theta_rev = theta_end[::-1] + np.pi
    k_reverse = find_minimal_winding_number(theta_begin, theta_rev)
    shift_rv = (np.array([0, 1, -1]) + k_reverse) * 2 * np.pi

    theta_cands_reversal = shift_rv[:, None] + theta_rev
    loss_reversal = np.sum((theta_cands_reversal - theta_begin) ** 2, axis=1)
    sort_indices = np.argsort(loss_reversal)
    theta_cands_reversal = theta_cands_reversal[sort_indices]

    top_candidates = (theta_cands_normal[0], theta_cands_reversal[0])
    next_candidates = (theta_cands_normal[1], theta_cands_reversal[1])
    return top_candidates, next_candidates


def body_axis_function(body_ratio, plot_n, base=0.5):
    x = torch.arange(2, plot_n) - (plot_n + 1) * 0.5
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
    plot_n: int,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Get all worm widths when segment number is given."""
    device = alpha.device
    worm_x = torch.linspace(-1.0, 1.0, plot_n - 1, requires_grad=False).to(device)
    worm_x_abs = torch.abs(worm_x)

    delta_sigmoid = torch.sigmoid(delta)
    gamma_e = 1 + 2 * torch.exp(gamma)
    eps = torch.tensor((1e-5,), requires_grad=False).to(device)
    width = alpha * torch.sqrt(
        1 - worm_x_abs**gamma_e * (1 + gamma_e * delta_sigmoid * (1 - worm_x_abs)) + eps
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
    H, W = height, width
    T, plot_n = x.shape
    device = x.device
    # midpoints of segments, length plot size
    cent_mid_x_3d = (x[:, 1:] + x[:, :-1]) / 2
    # midpoints of segments, length plot size
    cent_mid_y_3d = (y[:, 1:] + y[:, :-1]) / 2
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

    plot_n = params["plot_n"]
    worm_wid = worm_width_all(
        plot_n,
        params["alpha"],
        params["gamma"],
        params["delta"],
    )

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
    image = make_worm(x, y, image_info, params, worm_wid)
    return image


def to_param(data: Any, dtype=torch.float32) -> nn.Parameter:
    # Ensure it's a tensor first, then wrap as Parameter
    t = data if torch.is_tensor(data) else torch.tensor(data, dtype=dtype)
    return nn.Parameter(t.detach().clone())


class Model(torch.nn.Module):
    def __init__(
        self,
        init_cx: torch.Tensor,
        init_cy: torch.Tensor,
        init_theta: torch.Tensor,
        init_unitLength: torch.Tensor,
        params: dict[str, Any],
    ):
        super().__init__()
        self.cx = to_param(init_cx)
        self.cy = to_param(init_cy)
        self.theta = to_param(init_theta, dtype=torch.complex64)
        self.unitLength = to_param(init_unitLength)

        self.alpha = to_param(params["init_alpha"])
        self.gamma = to_param(params["init_gamma"])
        self.delta = to_param(params["init_delta"])
        params["alpha"] = self.alpha
        params["gamma"] = self.gamma
        params["delta"] = self.delta
        self.params = params

    def forward(
        self,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plot_n = self.params["plot_n"]
        worm_wid = worm_width_all(plot_n, self.alpha, self.gamma, self.delta)

        theta_unit = self.theta / (torch.abs(self.theta) + 1e-8)
        # (batch, ) => (batch, 1)
        unitLength = self.unitLength.unsqueeze(1)
        cx = self.cx.unsqueeze(1)
        cy = self.cy.unsqueeze(1)
        xy = F.pad(torch.cumsum(theta_unit, dim=1), pad=(1, 0))
        xy = (xy - xy.mean(axis=1, keepdim=True)) * unitLength

        x = torch.real(xy) + cx
        y = torch.imag(xy) + cy

        image = make_worm(x, y, width=width, height=height, worm_wid=worm_wid)
        return x, y, image

    def zero_masked_gradients(self, mask: torch.Tensor):
        assert mask.ndim == 1, "Input mask must be a 1D tensor."
        assert mask.shape[0] == self.cx.shape[0], (
            "The length of mask is not equal to the first dimension of cx."
        )

        # Ensure the mask is a float tensor for multiplication
        if mask.dtype != self.cx.dtype:
            mask = mask.to(dtype=self.cx.dtype)

        if not Model._is_binary_mask(mask):
            raise ValueError(
                "Mask must contain only 0s and 1s for selective gradient zeroing."
            )

        for param in (self.cx, self.cy, self.theta, self.unitLength):
            if param.grad is not None:
                num_dims = param.ndim
                reshaped_mask = mask
                # Add necessary dimensions of size 1 for broadcasting
                while reshaped_mask.ndim < num_dims:
                    reshaped_mask = reshaped_mask.unsqueeze(-1)

                # Apply the adjusted mask
                param.grad.mul_(reshaped_mask)

    @staticmethod
    def _is_binary_mask(mask: torch.Tensor) -> bool:
        """Checks if the mask tensor contains only 0s and 1s."""
        with torch.no_grad():
            float_mask = mask.to(torch.float32)
            # Check: x * (1 - x) == 0 only when x is 0 or 1.
            check_tensor = float_mask * (1.0 - float_mask)
            zero_tensor = torch.zeros(
                1, device=float_mask.device, dtype=float_mask.dtype
            )
            return torch.allclose(check_tensor.sum(), zero_tensor)


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

    def __call__(self, loss):
        if loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0


def train3(
    model: Model,
    real_image,
    optimizer,
    params,
    output_path,
    output_name,
    is_nont=True,
    gradient_mask: Optional[torch.Tensor] = None,
):
    device = next(model.parameters()).device
    T, H, W = real_image.shape
    speed = params["speed"]
    # Make sure at least 1 epochs will be executed
    epochs = max(int(T / (2 * speed) + params["epoch_plus"]), 1)
    block = params["use_area"]
    # Loss Weight
    continuity_loss_weight = float(params["continuity_loss_weight"])
    smoothness_loss_weight = float(params["smoothness_loss_weight"])
    length_loss_weight = float(params["length_loss_weight"])
    center_loss_weight = float(params["center_loss_weight"])
    # Progress setup
    save_progress_freq = params["save_progress_freq"]
    save_flag = params.get("SaveProgress", False)
    show_flag = params.get("ShowProgress", False)

    init_cx = model.cx.clone().detach()
    init_cy = model.cy.clone().detach()

    annealing_weight = torch.ones(T, device=device)
    body_axis_weight = body_axis_function(params["body_ratio"], params["plot_n"]).to(
        device
    )
    model.alpha.requires_grad = False
    model.gamma.requires_grad = False
    model.delta.requires_grad = False
    early_stopping = EarlyStopping()
    if not torch.is_tensor(real_image):
        real_image = torch.tensor(real_image, device=device)

    if gradient_mask is None:
        mask = torch.ones(T, dtype=model.cx.dtype, device=device)

    elif isinstance(gradient_mask, torch.Tensor):
        mask = gradient_mask.to(device)
    else:
        raise TypeError("loss_mask must be a torch.Tensor or None.")

    if torch.allclose(
        mask.sum(), torch.tensor(0.0, device=mask.device, dtype=mask.dtype)
    ):
        # Skip the main optimization if all parts are masked (fully frozen).
        epochs = 0
        logger.info("Mask is all zeros. Setting epochs to 0 to skip main optimization.")

    # main optimization
    for e in range(epochs):
        optimizer.zero_grad()
        _, _, model_image = model(width=W, height=H)
        model_image = model_image.to(device)

        if is_nont:
            annealing_weight = annealing_function(e, T, speed).to(device)
        image_loss = (
            torch.mean(
                ((model_image - real_image) ** 2),
                dim=(1, 2),
            )
            * annealing_weight
        )  # (T, W, H) => (T, )

        image_loss = torch.mean(image_loss)
        theta_normalized = model.theta / (torch.abs(model.theta) + 1e-8)
        rotation_at_t = theta_normalized[:, :-1] * torch.conj(theta_normalized[:, 1:])
        smoothness_loss = (
            torch.mean(
                torch.abs(rotation_at_t - 1.0) ** 2 * body_axis_weight,
                dim=1,
            )
            * annealing_weight
        )

        smoothness_loss = smoothness_loss_weight * torch.mean(smoothness_loss)

        unitL = torch.mean(model.unitLength)
        center_loss = (
            center_loss_weight
            / unitL
            * torch.mean(((model.cx - init_cx) ** 2 + (model.cy - init_cy) ** 2))
        )

        if T < 2:
            # If training block contains only single frame. Then, we assigned continuity_loss and length_loss to zeros.
            continuity_loss = torch.zeros(1, device=device)
            length_loss = torch.zeros(1, device=device)
        else:
            rotation_along_t = theta_normalized[:-1, :] * torch.conj(
                theta_normalized[1:, :]
            )

            continuity_loss = torch.mean(
                torch.abs(rotation_along_t - 1.0) ** 2,
                dim=1,
            )
            continuity_loss = continuity_loss_weight * torch.mean(continuity_loss)

            length_loss = (
                10000
                * length_loss_weight
                * torch.mean((model.unitLength[:-1] - model.unitLength[1:]) ** 2)
            )

        loss = (
            image_loss + continuity_loss + smoothness_loss + length_loss + center_loss
        )

        model.zero_masked_gradients(mask)
        loss.backward()
        if torch.min(annealing_weight) > 0.99:
            early_stopping(loss.item())
        del loss
        if early_stopping.early_stop:
            if params["ShowProgress"]:
                logger.info(
                    "Early stopping at epoch +{}.".format(e - int(T / (2 * speed)))
                )
            break
        optimizer.step()

        if not (show_flag or save_flag) or e % save_progress_freq > 0:
            continue

        # Save Progress
        if save_flag:
            save_progress(
                real_image,
                output_path,
                output_name,
                block.start,
                block.end,
                params["save_progress_num"],
                txt="id{}_{}".format(params["id"], e),
            )

        # Show Progress
        if show_flag:
            logger.info(
                "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                    image_loss.item(),
                    continuity_loss.item(),
                    smoothness_loss.item(),
                    length_loss.item(),
                    center_loss.item(),
                )
            )
            if __debug__:
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
        optimizer.zero_grad()
        _, _, model_image = model(width=W, height=H)
        model_image = model_image.to(device)

        image_loss = torch.mean((model_image - real_image) ** 2)
        theta_normalized = model.theta / (torch.abs(model.theta) + 1e-8)

        rotation_at_t = theta_normalized[:, :-1] * torch.conj(theta_normalized[:, 1:])

        smoothness_loss = smoothness_loss_weight * torch.mean(
            torch.abs(rotation_at_t - 1.0) ** 2 * body_axis_weight
        )

        if T < 2:
            # If training block contains only single frame. Then, we assigned continuity_loss and length_loss to zeros.
            continuity_loss = torch.zeros(1, device=device)
            length_loss = torch.zeros(1, device=device)
        else:
            rotation_along_t = theta_normalized[:-1, :] * torch.conj(
                theta_normalized[1:, :]
            )

            continuity_loss = continuity_loss_weight * torch.mean(
                torch.abs(rotation_along_t - 1.0) ** 2
            )

            length_loss = (
                10000
                * length_loss_weight
                * torch.mean((model.unitLength[:-1] - model.unitLength[1:]) ** 2)
            )

        loss = image_loss + continuity_loss + smoothness_loss + length_loss

        model.zero_masked_gradients(mask)
        loss.backward()
        early_stopping(loss.item())
        del loss
        if early_stopping.early_stop:
            if params["ShowProgress"]:
                logger.info("Minor adjustment done.")
            break
        optimizer.step()

    with torch.no_grad():
        _, _, model_image = model(width=W, height=H)
        # Calculate the loss for display, this part does not require grad.
        image_loss = torch.mean((model_image - real_image) ** 2, dim=(1, 2))
        theta_normalized = model.theta / (torch.abs(model.theta) + 1e-8)
        rotation_at_t = theta_normalized[:, :-1] * torch.conj(theta_normalized[:, 1:])

        smoothness_loss = smoothness_loss_weight * torch.mean(
            torch.abs(rotation_at_t - 1.0) ** 2,
            dim=1,
        )
        center_loss = center_loss_weight * (
            (model.cx - init_cx) ** 2 + (model.cy - init_cy) ** 2
        )

        if T < 2:
            # If training block contains only single frame. Then, we ignore continuity_loss and length_loss by filled it to zeros
            continuity_loss = torch.zeros(1, device=model.theta.device)
            length_loss = torch.zeros(1, device=model.unitLength.device)
        else:
            rotation_along_t = theta_normalized[:-1, :] * torch.conj(
                theta_normalized[1:, :]
            )

            continuity_loss = continuity_loss_weight * torch.mean(
                torch.abs(rotation_along_t - 1.0) ** 2,
                dim=1,
            )
            length_loss = length_loss_weight * (
                (model.unitLength[:-1] - model.unitLength[1:]) ** 2
            )
            # We padding the loss to make it has same length as other loss
            continuity_loss = F.pad(continuity_loss, (1, 0), mode="constant", value=0.0)
            length_loss = F.pad(length_loss, (1, 0), mode="constant", value=0.0)

        losses = (
            image_loss,
            continuity_loss,
            smoothness_loss,
            length_loss,
            center_loss,
        )

        # (5, T)
        losses = np.asarray([loss.clone().detach().cpu().numpy() for loss in losses])
    if show_flag:
        mean_loss = losses.mean(axis=1)
        logger.info(
            f"avg:{mean_loss[0]:.2f} {mean_loss[1]:.2f} {mean_loss[2]:.2f} {mean_loss[3]:.2f} {mean_loss[4]:.2f}"
        )
        if __debug__:
            show_image(model_image, params["num_t"], title="final")

    if save_flag:
        save_progress(
            real_image,
            output_path,
            output_name,
            block.start,
            block.end,
            params["save_progress_num"],
            txt="id{}_{}".format(params["id"], "final"),
        )
    torch.cuda.empty_cache()
    return losses


def make_plot(theta, unitLength, x_cent, y_cent):
    T = theta.shape[0]
    x = np.hstack((np.zeros((T, 1)), np.cumsum(unitLength * np.cos(theta), axis=1)))
    y = np.hstack((np.zeros((T, 1)), np.cumsum(unitLength * np.sin(theta), axis=1)))
    x = x - np.mean(x, axis=1).reshape((T, 1)) + x_cent.reshape((T, 1))
    y = y - np.mean(y, axis=1).reshape((T, 1)) + y_cent.reshape((T, 1))
    return x, y


def loss_compare(loss_pair) -> bool:
    im_select = int(max(loss_pair[0][0]) > max(loss_pair[1][0]))
    con_select = int(max(loss_pair[0][1]) > max(loss_pair[1][1]))
    smo_select = int(max(loss_pair[0][2]) > max(loss_pair[1][2]))
    if im_select + con_select + smo_select == 3:
        return True
    if im_select + con_select + smo_select == 0:
        return False

    if len(loss_pair[im_select][0]) > 2:
        q75, q50, q25 = np.percentile(loss_pair[im_select][0], [75, 50, 25])
        im_exrate = (max(loss_pair[1 - im_select][0]) - q50) / (q75 - q25 + 1e-8)
    else:
        im_exrate = max(loss_pair[1 - im_select][0]) / max(
            max(loss_pair[im_select][0]), 1e-8
        )

    if len(loss_pair[con_select][1]) > 2:
        q75, q50, q25 = np.percentile(loss_pair[con_select][1], [75, 50, 25])
        con_exrate = (max(loss_pair[1 - con_select][1]) - q50) / (q75 - q25 + 1e-8)
    else:
        con_exrate = max(loss_pair[1 - con_select][1]) / max(
            max(loss_pair[con_select][1]), 1e-8
        )

    if len(loss_pair[smo_select][2]) > 2:
        q75, q50, q25 = np.percentile(loss_pair[smo_select][2], [75, 50, 25])
        smo_exrate = (max(loss_pair[1 - smo_select][2]) - q50) / (q75 - q25 + 1e-8)
    else:
        smo_exrate = max(loss_pair[1 - smo_select][2]) / max(
            max(loss_pair[smo_select][2]), 1e-8
        )

    # Choossing most significant loss to do comparison
    # (boolean, exrate)
    return bool(
        max(
            [
                (im_select, im_exrate),
                (con_select, con_exrate),
                (smo_select, smo_exrate),
            ],
            key=lambda x: x[1],
        )
    )


def show_loss_plot(losses, title=""):
    if not __debug__:
        return
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
    plt.close(fig)


def find_losslarge_area(losses_all) -> list[tuple[int, int]]:
    losslarge_area = dict()
    for i in range(3):
        lossi = np.concatenate([loss[i] for loss in losses_all.values()], axis=None)
        q75, q50, q25 = np.percentile(lossi, [75, 50, 25])
        for (step, idx), loss in losses_all.items():
            if np.max(loss[i]) - q50 > (q75 - q25) * 4:
                # Can be step 1 or 2
                losslarge_area[idx] = step
            elif idx in losslarge_area:
                # Assume that loss in step2 is lower than step1
                # If the loss in the step2 passed the criteria
                # removing the idx from losslarge_area
                losslarge_area.pop(idx)

    return [(step, idx) for idx, step in losslarge_area.items()]


### arrange and save data ###


def judge_head_amplitude(x, y) -> bool:
    """Judge which tip is head based on variance of body curve rate."""
    assert x.shape == y.shape, "The coordinates of x and y have different shape."
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    theta = np.arctan2(dy, dx)

    curve_rate_var = ((theta[:, 1:] - theta[:, :-1] + np.pi) % (2 * np.pi) - np.pi).var(
        axis=0
    )
    if __debug__:
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

    return curve_mean1 < curve_mean2


def judge_head_frequency(x, y) -> bool:
    """Judge which tip is head based on frequency of body curve rate."""
    assert x.shape == y.shape, "The coordinates of x and y have different shape."
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    theta = np.arctan2(dy, dx)

    curve_rate = np.unwrap(theta[:, 1:] - theta[:, :-1] + np.pi) - np.pi
    T = curve_rate.shape[0]

    # fast fourier transform
    spa = np.abs(np.fft.fft(curve_rate, axis=0))

    # the latter half of fourier power spectrum is the same as the first half
    T2 = int(np.ceil((T - 1) / 2))
    cut = int(np.round(x.shape[1] / 20))  # cut end 5% of worm
    spat = spa[1 : (T2 + 1), cut : x.shape[1] - cut]

    try:
        # cutoff high-freq area with values < peak/10
        sp_sum = np.sum(spat, axis=1)
        freq_cut = np.max(np.where(sp_sum > np.max(sp_sum) / 10)[0]) + 1
    except ValueError as _:
        # no high-freq area with values < peak/10
        freq_cut = spat.shape[0]

    spat = spat[:freq_cut, :]

    # logger.info('freq_cut =', freq_cut)

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
    # logger.info('correlation =', cor)

    # show power spectrum plot
    if __debug__:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(spat)
        ax.set_aspect(0.1)
        ax.set_xlabel("body segment", fontsize=20)
        ax.set_ylabel("peak curve freq", fontsize=20)
        ax.set_title(f"Correlation = {cor:.3g}")
        # plt.show()

    return cor > 0


def cancel_reduction(
    x: npt.NDArray,
    y: npt.NDArray,
    n_input_images: int,
    start_T: int,
    end_T: int,
    Tscaled_ind: Sequence[int],
    plot_n: int,
) -> tuple[npt.NDArray, npt.NDArray]:
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
