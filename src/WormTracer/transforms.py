from __future__ import annotations

import typing

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def straigthen_multi(
    src: npt.NDArray,
    x: npt.NDArray,
    y: npt.NDArray,
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
    assert x.shape[0] == N, (
        "The number of frames to be straightened is different from given coordinates."
    )

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

    straigthen_dst = F.grid_sample(src_t, grid, mode="bicubic", align_corners=True)
    straigthen_dst = (
        torch.clamp(straigthen_dst, src.min(), src.max())
        .detach()
        .numpy()
        .astype(src.dtype)
        .reshape(N, height, width)
    )

    return straigthen_dst


def straigthen(
    src: npt.NDArray,
    x: npt.NDArray,
    y: npt.NDArray,
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
