from __future__ import annotations

import itertools
import logging
import typing

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.spatial import distance_matrix
from skimage import morphology

if typing.TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger(__name__)


def find_theta(theta: npt.NDArray, pretheta: npt.NDArray, plus: int = 1) -> int:
    """Find min MSE theta by theta(t=0)"""
    i = plus
    mse_list = [float(np.sum((theta[0, :] - pretheta) ** 2))]
    while True:
        theta_cand = pretheta + i * 2 * np.pi
        mse_0T = float(np.sum((theta[0, :] - theta_cand) ** 2))
        if mse_list[-1] < mse_0T:
            break
        mse_list.append(mse_0T)
        i += plus
    return len(mse_list)


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


def get_skeleton_networkx(
    im: npt.NDArray, plot_n: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """skeletonize image and get splined plots
    2024/10/01 Speed is same as previous implemenetation
    """
    # skeletonize image
    im_filled = ndi.binary_fill_holes(im)
    assert im_filled is not None, "Err after binary_fill_holes"
    im_skeleton = morphology.skeletonize(im_filled)
    point_list = np.argwhere(im_skeleton == 1)

    if len(point_list) == 1:
        x_splined = np.ones(plot_n) * point_list[0][1]
        y_splined = np.ones(plot_n) * point_list[0][0]
        return x_splined, y_splined

    # make distance matrix
    adj_mtx = distance_matrix(
        point_list,
        point_list,
    )
    adj_mtx[adj_mtx > 1.5] = 0  # delete distance between isolated points

    G = nx.from_numpy_array(adj_mtx)
    assert not isinstance(G.degree, int), "G.degree must be a list here"
    # Obtain end from 1 degree node.
    ends = [node for node, deg in G.degree if deg == 1]
    # Calculate the shortest path of all ends pairing
    pairs = itertools.combinations(ends, 2)
    paths = [nx.dijkstra_path(G, st, end, weight="weight") for st, end in pairs]
    # Obtain the maximum distance
    skel_idx = max(paths, key=lambda x: adj_mtx[x[1:], x[:-1]].sum())
    arclen = np.zeros_like(skel_idx, dtype="f8")
    arclen[1:] = np.cumsum(adj_mtx[skel_idx[1:], skel_idx[:-1]])
    plots = point_list[skel_idx]

    # Interpolation
    div_linespace = np.linspace(0, np.max(arclen), plot_n)
    x_splined = np.interp(div_linespace, arclen, plots[:, 1])
    y_splined = np.interp(div_linespace, arclen, plots[:, 0])

    return x_splined, y_splined


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
            logger.warning(
                "Warning! The initial frame of images is difficult to skeletonize."
            )
            logger.warning("Beginning of Results will be incorrect.")
        if nont_ini[-1] > nont_end[-1]:
            logger.warning(
                "Warning! The end frame of images is difficult to skeletonize."
            )
            logger.warning("End of Results will be incorrect.")
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
                logger.warning(
                    """
        Warning! This task uses large memory.
        If CUDA run out of memory, please go back to setting hyperparameters and set rescale as {:.2f}, Tscale as {}.
        The result may be not precise enough.
        """.format(rescale_rec, Tscale_rec)
                )
            else:
                logger.warning(
                    """
        Warning! This task uses large memory.
        If CUDA run out of memory, please go back to setting hyperparameters and set rescale as {:.2f}, Tscale as {}.
        """.format(rescale_rec, Tscale_rec)
                )

    except IndexError:
        logger.warning("All frames seem to be simple; easy to skeletonize.")
        use_points = np.linspace(0, T - 1, (T - 1) // (cap_span + 1) + 2, dtype=int)
        nont_flag = [0] * (use_points.shape[0])

    if __debug__ and show_plot:
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
