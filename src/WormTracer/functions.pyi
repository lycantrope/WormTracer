# from __future__ import annotations

import os
from typing import (
    Generator,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import roifile
import torch

def show_image(image, num_t=5, title="", x=0, y=0, x2=0, y2=0): ...

### read, preprocess images and get information ###

def set_output_path(dataset_path, output_directory): ...
def get_filenames(dataset_path: Union[str, bytes, os.PathLike]) -> List[str]: ...
def get_property(filenames, rescale) -> Tuple[Sequence[int], bool, bool, int]: ...
def read_serial_images(filenames, Tscaled_ind): ...
def load_image(
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

def get_skeleton(im: np.ndarray, plot_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """skeletonize image and get splined plots"""

def get_skeleton_networkx(im: np.ndarray, plot_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """skeletonize image and get splined plots
    2024/10/01 Speed is same as previous implemenetation
    """

def get_width(im: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Get width of the object by measure distance of centerline to the object's surface."""

def flip_check(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Check if plots of head and tail is flipping."""

def trim_image(image, *, padding: int = 5) -> np.ndarray:
    """Crop images to minimum size."""

def make_theta_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

### prepare for training ###

def calc_cap_span(image_shape, plot_n) -> int:
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

def get_image_loss_max(best_fit_image, cx, cy, x_st, y_st, params, image_info) -> float:
    """Create bad image and get bad image_loss to judge complex area."""

def get_use_points(
    image_losses, image_loss_max, cap_span, x, y, plot_n, show_plot=True
):
    """Judge flames complex or not and get span for training."""

def find_nont_area(image_losses, borderline, under_borderline): ...
def check_enough_expanded(nont_span, temp_ini, temp_end, enough_rate=2): ...
def check_collision(temp_ini, temp_end, T): ...

### training ###
def make_progress_image(image, num_t=20):
    """Make one large image with images laid out on it."""

def save_progress(
    image: np.ndarray,
    output_path: str,
    output_name: str,
    start: int,
    end: int,
    num_t: int,
    txt="real",
) -> None: ...
def remove_progress(output_pathh, filename): ...
def get_center(binimg):
    """Calculate center of images."""

def set_init_xy(real_image):
    """Set init center plots for training."""

def find_theta(theta, pretheta, plus=1) -> int:
    """Find min MSE theta by theta(t=0)"""

def find_minimal_winding_number(theta1: np.ndarray, theta2: np.ndarray) -> int: ...
def make_theta_cand(
    theta_begin: np.ndarray, theta_end: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: ...
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
    model: Model,
    real_image,
    optimizer,
    params,
    device,
    init_data,
    output_path,
    output_name,
    is_nont=True,
    gradient_mask: Optional[torch.Tensor] = None,
): ...
def make_plot(theta, unitLength, x_cent, y_cent): ...
def loss_compare(loss_pair): ...
def show_loss_plot(losses, title=""): ...
def find_losslarge_area(losses_all) -> Set[int]: ...

### arrange and save data ###

def judge_head_amplitude(x, y) -> bool: ...
def judge_head_frequency(x, y) -> bool:
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
    def __init__(self, init_cx, init_cy, init_theta, init_unitLength, params): ...
    def forward(self, batch, width, height): ...
    def zero_masked_gradients(self, mask: torch.Tensor): ...
    @staticmethod
    def _is_binary_mask(mask: torch.Tensor) -> bool: ...

class EarlyStopping:
    def __init__(self, patience=30, delta=0): ...
    def __call__(self, loss, model): ...

class TrainingBlocks:
    blocks: np.ndarray
    complex_block: np.ndarray

    complex_area: np.ndarray
    simple_area: np.ndarray

    nblock: int

    class Block(NamedTuple):
        start: int
        end: int
        idx: int
        is_complex: bool

        @property
        def size(self) -> int: ...
        def __repr__(self) -> str: ...

    def __init__(self, losses, relaxed, rigid) -> None: ...
    @property
    def nframe(self) -> int: ...
    def batch_iter(
        self, batchsize: Optional[int] = None
    ) -> Generator[TrainingBlocks.Block, None, None]:
        """Return an iterator that yields Block(idx, is_complex, start, end) within the batchsize"""

def get_use_blocks(
    image_losses: np.ndarray,
    image_loss_max: float,
) -> TrainingBlocks:
    """Judge frames complex or not and get span for training."""

def centerline_to_roi_iter(x, y, head_idx=0) -> Iterator[roifile.ImagejRoi]: ...
def save_centerline_to_roi(
    outputpath: str, x: np.ndarray, y: np.ndarray, head_idx: int = 0
) -> None: ...
