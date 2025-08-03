# from __future__ import annotations

import os
from typing import Tuple, Union

import numpy as np
import torch

def show_image(image, num_t=5, title="", x=0, y=0, x2=0, y2=0): ...

### read, preprocess images and get information ###

def set_output_path(dataset_path, output_directory): ...
def get_filenames(dataset_path: Union[str, bytes, os.PathLike]): ...
def get_property(filenames, rescale): ...
def read_serial_images(filenames, Tscaled_ind): ...
def read_image(
    filenames,
    rescale,
    Worm_is_black,
    multi_flag,
    Tscaled_ind,
) -> Tuple[np.ndarray, float, float]:
    """read images and get skeletonized plots"""

def calc_xy_and_prewidth(
    imagestack: np.ndarray,
    plot_n: int,
    x_st: float,
    y_st: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """read images and get skeletonized plots"""

def get_skeleton(im: np.ndarray, plot_n: int):
    """skeletonize image and get splined plots"""

def get_skeleton_networkx(im: np.ndarray, plot_n: int):
    """skeletonize image and get splined plots
    2024/10/01 Speed is same as previous implemenetation
    """

def get_width(im, x, y):
    """Get width of the object by measure distance of centerline to the object's surface."""

def flip_check(x, y):
    """Check if plots of head and tail is flipping."""

def cut_image(image):
    """Cut images to minimum size."""

def make_theta_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

### prepare for training ###

def calc_cap_span(image_info, plot_n, s_m=8000):
    """Calculate maximum span of trainig in terms of CUDA memory."""

def pixel_value_from_dist_max_np(
    max_dist: np.ndarray,
    contrast: float = 1.2,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Get pixel value when distance from midline is given."""

def worm_width_all_np(
    *,
    plot_n: int,
    alpha: float,
    gamma: float,
    delta: float,
) -> np.ndarray:
    """Get all worm widths when segment number is given."""

def make_distance_matrix_np(radius: int) -> np.ndarray: ...
def make_distance_matrix(radius: int) -> np.ndarray: ...
def make_single_image(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
    pixel_matrix: np.ndarray,
) -> np.ndarray: ...
def make_image(x, y, x_st, y_st, params, image_info):
    """Create model image by dividing them to avoid CUDA memory error."""

def get_image_loss_max(
    image_losses, real_image, x, y, x_st, y_st, params, image_info, cap_span
):
    """Create bad image and get bad image_loss to judge complex area."""

def get_use_points(
    image_losses, image_loss_max, cap_span, x, y, plot_n, show_plot=True
):
    """Judge flames complex or not and get span for training."""

def find_nont_area(image_losses, borderline, under_borderline): ...
def check_enough_expanded(nont_span, temp_ini, temp_end, enough_rate=2): ...
def check_collision(temp_ini, temp_end, T): ...
def prepare_for_train(pre_width, simple_area, x, y, params): ...

### training ###
def make_progress_image(image, num_t=20):
    """Make one large image with images laid out on it."""

def save_progress(image, output_path, output_name: str, params, txt="real"): ...
def remove_progress(output_pathh, filename): ...
def get_center(binimg):
    """Calculate center of images."""

def set_init_xy(real_image):
    """Set init center plots for training."""

def find_theta(theta, pretheta, plus=1):
    """Find min MSE theta by theta(t=0)"""

def make_theta_cand(theta): ...
def body_axis_function(body_ratio, plot_n, base=0.5): ...
def annealing_function(epoch, T, speed=0.2, start=0, slope=1): ...
def worm_width_all(
    plot_n: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Get all worm widths when segment number is given."""

def pixel_value_from_dist_max(
    max_dist: torch.Tensor,
    contrast: float = 1.2,
    sharpness: float = 2.0,
) -> torch.Tensor:
    """Get pixel value when distance from midline is given."""

PIXEL_MINIMUM: float
PIXEL_MAXIMUM: float

def make_single_worm(
    x: torch.Tensor,
    y: torch.Tensor,
    width: int,
    height: int,
    pixel_matrix: torch.Tensor,
) -> torch.Tensor: ...
def make_worm(
    x: torch.Tensor,
    y: torch.Tensor,
    width: int,
    height: int,
    worm_wid: torch.Tensor,
) -> torch.Tensor: ...
def make_model_image(cent_x, cent_y, theta, unitLength, image_info, params): ...
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
): ...
def make_plot(theta, unitLength, x_cent, y_cent, x_st=0, y_st=0): ...
def get_shape_params(shape_params, params): ...
def loss_compare(loss_pair): ...
def show_loss_plot(losses, title=""): ...
def find_losslarge_area(losses_all): ...

### arrange and save data ###

def judge_head_amplitude(x, y): ...
def judge_head_frequency(x, y):
    """Judge which tip is head based on frequency of body curve rate."""

def clear_dir(output_path, foldername): ...
def cancel_reduction(x, y, n_input_images, start_T, end_T, Tscaled_ind, plot_n): ...
def straigthen_multi(
    src: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
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

def straigthen(
    src: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
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

class Model(torch.nn.Module):
    def __init__(
        self, init_cx, init_cy, init_theta, init_unitLength, image_info, params
    ): ...
    def forward(self): ...

class EarlyStopping:
    def __init__(self, patience=30, delta=0): ...
    def __call__(self, loss, model): ...
