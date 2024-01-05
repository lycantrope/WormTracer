import torch as _torch
import torch.nn as _nn
import numpy as _np

from wormtracer.types import _NP_T


def pixel_value(
    d: _torch.Tensor,
    width: _torch.Tensor,
    contrast: float = 1.2,
    sharpness: float = 2.0,
) -> _torch.Tensor:
    """Get pixel value when distance from midline and worm width is given."""
    return 255 * contrast * _torch.sigmoid((width - d) * sharpness) + 127.5 * (
        1.0 - contrast
    )


def calc_worm_width(
    n_pts: int,
    alpha: _torch.Tensor,
    gamma: _torch.Tensor,
    delta: _torch.Tensor,
) -> _torch.Tensor:
    """
    Get worm width when segment number is given.
    Return:
        worm_width: size of n_segs-1
    """

    unit_i = _torch.arange(n_pts - 1)
    worm_x = _torch.abs(2 * (unit_i / (n_pts - 2) - 0.5))

    delta_sigmoid = _torch.sigmoid(delta)
    gamma_e = 0.5 + _torch.exp(gamma)
    gamma_2e = 2 * gamma_e

    worm_wid = alpha * _torch.sqrt(
        1.00001 - worm_x**gamma_2e * (1 + gamma_2e * delta_sigmoid * (1 - worm_x))
    )
    return worm_wid


def create_skel_from_theta_and_pos(ct_pos, theta, segment_unit) -> _torch.Tensor:
    # theta: shape [T, n_segs]
    T = theta.size(dim=0)
    n_segs = theta.size(dim=1)
    # txy [T, 2, n_segs]
    txy = _torch.zeros((T, 2, n_segs + 1))
    txy[:, 0, 1:] = _torch.cumsum(_torch.cos(theta), dim=1)
    txy[:, 1, 1:] = _torch.cumsum(_torch.sin(theta), dim=1)
    t_mean = txy.mean(dim=2).reshape(T, 2, 1)
    return (txy - t_mean) * segment_unit.reshape(T, 1, 1) + ct_pos.reshape(T, 2, 1)


def make_worm(txy: _torch.Tensor, worm_width: _torch.Tensor, im_width, im_height):
    # txy: [T, 2, n_pts]
    T = txy.size(dim=0)
    n_pts = txy.size(dim=2)
    cent_mid = (txy[:, :, :-1] + txy[:, :, 1:]) * 0.5
    # cent_mid_3d: [T, 2, n_pts-1, 1, 1]
    cent_mid_3d = cent_mid.reshape(T, 2, n_pts - 1, 1, 1)

    cent_mid_x_3d = cent_mid_3d[:, 0]
    cent_mid_y_3d = cent_mid_3d[:, 1]
    # worm_wid_3d: [1, n_pts-1, 1, 1]
    worm_wid_3d = worm_width.reshape(1, n_pts - 1, 1, 1)
    y_3d = _torch.arange(im_height).reshape(1, 1, im_height, 1)
    x_3d = _torch.arange(im_width).reshape(1, 1, 1, im_width)

    # segment_distance_3d: [T, n_pts-1, im_height, im_width]
    # worm_wid_3d: [1, n_pts-1, 1, 1]
    segment_distance_3d = _torch.sqrt(
        _torch.pow(cent_mid_x_3d - x_3d, 2) + _torch.pow(cent_mid_y_3d - y_3d, 2)
    )
    # the max_span will be used in here (n_segs * im_height*im_width)

    # image_3d = [T, n_pts-1, im_height, im_width]
    image_3d = pixel_value(segment_distance_3d, worm_wid_3d)
    image, _ = _torch.max(image_3d, dim=1)
    return image


def make_worm_batch(
    txy: _NP_T,
    pre_width: _NP_T,
    im_width: int,
    im_height: int,
    batchsize: int,
    device: str,
) -> _NP_T:
    """Create model image by dividing them to avoid CUDA memory error."""
    from numpy import split

    T, _, n_pts = txy.shape
    image = _np.zeros((T, im_height, im_width), dtype=_np.uint8)
    cut = _np.arange(0, T, step=batchsize)
    chunks = zip(
        split(image, cut),
        split(txy, cut),
        split(pre_width, cut),
    )
    with _torch.no_grad():
        for dst, t, w in chunks:
            dst[:, :, :] = (
                make_worm(
                    txy=_torch.from_numpy(t).to(device),
                    worm_width=_torch.from_numpy(w).to(device),
                    im_width=im_width,
                    im_height=im_height,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(_np.uint8)
            )

    return image


def get_image_loss_max(
    nearest_idx,
    txy: _NP_T,
    imagestack: _NP_T,
    worm_width: _NP_T,
):
    """Create bad image and get bad image_loss to judge complex area."""
    im = imagestack[nearest_idx]
    height, width = im.shape
    xy = txy[nearest_idx]
    _, n_pts = xy.shape
    txy0 = _np.ones((2, n_pts)) * xy[:, 0][:, None]
    wid0 = worm_width[nearest_idx]
    with _torch.no_grad():
        txy0_t = _torch.from_numpy(txy0).reshape(1, 2, -1)
        wid0_t = _torch.from_numpy(wid0).reshape(1, -1)
        im0 = make_worm(txy0_t, wid0_t, width, height).reshape(height, width)
    image_loss_max = _np.mean((im.astype("i4") - im0.astype("i4")) ** 2)
    return image_loss_max
