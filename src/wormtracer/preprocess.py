import typing as _T
from functools import partial as _partial
from functools import reduce as _reduce
from pathlib import Path

import attrs as _attrs
import cv2 as _cv
import numpy as np
import scipy.ndimage as _ndi
import skimage.morphology as _morphology
from scipy.interpolate import interp1d as _interp1d
from scipy.sparse import csgraph as _csgraph
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.spatial import distance_matrix as _distance_matrix
from tqdm import tqdm as _tqdm

from wormtracer.utils import eprint

_NP_T: _T.TypeAlias = np.ndarray
_OPTIONAL_NP: _T.TypeAlias = _T.Optional[_NP_T]
_PATH_T: _T.TypeAlias = _T.Union[str, Path]
_IMREAD_T: _T.TypeAlias = _T.Callable[[_PATH_T], _NP_T]
_TRANSFORM_T: _T.TypeAlias = _T.Callable[[_NP_T], _NP_T]


class Offset(_T.NamedTuple):
    x: int = 0
    y: int = 0


__all__ = ["get_skeleton", "imread", "ImageReader"]


def imread(path: _PATH_T) -> _NP_T:
    """A grayscale imread that accept str or Path

    Args:
        path (_T.Union[str,_Path]): An image path

    Returns:
        np.ndarray: image loaded by cv2.imread
    """
    return _cv.imread(str(path), flags=_cv.IMREAD_GRAYSCALE)


def _find_max_contour(im: _NP_T) -> _NP_T:
    _, label_im, stats, _ = _cv.connectedComponentsWithStatsWithAlgorithm(
        im,
        connectivity=8,
        ltype=_cv.CV_16U,
        ccltype=_cv.CCL_BBDT,
    )
    im[label_im != stats[1:, 4].argmax() + 1] = 0
    return im


def _otsu(im: _NP_T) -> _NP_T:
    _, im = _cv.threshold(
        im,
        thresh=0,
        maxval=255,
        type=_cv.THRESH_BINARY + _cv.THRESH_OTSU,
        dst=None,
    )
    return im


@_attrs.define
class ImageReader:
    rescale: float = _attrs.field(kw_only=True, default=1.0)
    worm_is_black: bool = _attrs.field(kw_only=True, default=False)
    is_binarized: bool = _attrs.field(kw_only=True, default=True)
    use_max_contour: bool = _attrs.field(kw_only=True, default=True)
    binarize_fn: _TRANSFORM_T = _attrs.field(
        kw_only=True,
        default=_otsu,
    )
    # hidden property
    src_width: int = _attrs.field(init=False, default=0)
    src_height: int = _attrs.field(init=False, default=0)

    _imread: _T.Callable[[str], _NP_T] = _attrs.field(
        init=False,
        repr=False,
        default=imread,
    )

    def __attrs_post_init__(self):
        steps: _T.List[_TRANSFORM_T] = []
        if not self.is_binarized:
            steps.append(self.binarize_fn)

        if self.worm_is_black:
            steps.append(_cv.bitwise_not)

        if self.use_max_contour:
            steps.append(_find_max_contour)

        if not np.isclose(self.rescale, 1.0, atol=1e2):
            # scaling
            resize = _partial(
                _cv.resize,
                dsize=None,
                fy=self.rescale,
                fx=self.rescale,
                interpolation=_cv.INTER_NEAREST,
            )
            steps.append(resize)

        self._imread = ImageReader._compose_sequential_fn(imread, steps)

    @staticmethod
    def _compose_sequential_fn(
        imread: _IMREAD_T,
        operators: _T.List[_TRANSFORM_T],
    ) -> _IMREAD_T:
        def chain_func(im_path: _PATH_T):
            im = imread(im_path)
            return _reduce(lambda x, f: f(x), operators, im)

        return chain_func

    def imread(self, im_path: _PATH_T) -> _NP_T:
        try:
            im = self._imread(str(im_path))
        except Exception as e:
            eprint(e)
            raise e

        self.src_height, self.src_width = im.shape[:2]
        return im

    @classmethod
    def build_reader_from_image(
        cls,
        im_path: _PATH_T,
        rescale: float = 1.0,
        binarize_fn: _T.Optional[_TRANSFORM_T] = None,
    ) -> "ImageReader":
        im = imread(im_path)
        is_binarized = np.unique(im).size == 2
        if not is_binarized:
            if binarize_fn is None:
                im = _otsu(im)
                eprint(
                    "[Warning] Input images seem not to be binary.",
                    "Automatically threshold by otsu or use `ImageReader.build_reader_from_image_with_binarized` to setup thesholding method",
                    sep="\n",
                )
            else:
                im = binarize_fn(im)

        # use 0.01 as a cut off.
        if not np.isclose(rescale, 1.0, atol=1e-2):
            im: _NP_T = _cv.resize(
                im,
                dsize=None,
                fy=rescale,
                fx=rescale,
                interpolation=_cv.INTER_NEAREST,
            )

        h, w = im.shape[:2]
        # Checking the nonzero count of boarder pixel.
        while_pixel = np.count_nonzero(im[(0, h - 1), :]) + np.count_nonzero(
            im[1 : h - 1, (0, w - 1)]
        )
        side_length = (h + w) * 2 - 4
        is_white_background = while_pixel > (side_length * 0.5)

        return cls(
            rescale=rescale,
            worm_is_black=is_white_background,
            is_binarized=is_binarized,
            binarize_fn=binarize_fn,
        )


def trim_imagestack(imagestack: _NP_T) -> _T.Tuple[_NP_T, Offset]:
    """trim imagestack to minimum size."""
    assert (
        imagestack.ndim == 3
    ), "Only accept 3D-image stack in (time, width, height) order"

    thresh = np.bitwise_or.reduce(imagestack > 0, axis=0)
    max_h, max_w = thresh.shape

    (ys, xs) = np.nonzero(thresh)
    if ys.size == 0:
        eprint("[Warning] the imagestack have no signal")
        return imagestack, Offset()

    # padding the border with 5 pixel width if possible
    x1 = max(xs.min() - 5, 0)
    x2 = min(xs.max() + 5, max_w)
    y1 = max(ys.min() - 5, 0)
    y2 = min(ys.max() + 5, max_h)
    imagestack = imagestack[:, y1:y2, x1:x2]
    # because we knew the size of trimmed image
    return imagestack, Offset(x1, y1)


def get_width(
    im: _NP_T,
    splines: _NP_T,
):
    """Get width of the object by measure distance of centerline and object's surface."""
    im_filled = _ndi.binary_fill_holes(im)
    x = splines[0, :].reshape((-1, 1, 1))
    y = splines[1, :].reshape((-1, 1, 1))
    y_3d = np.arange(im_filled.shape[0]).reshape([1, -1, 1])
    x_3d = np.arange(im_filled.shape[1]).reshape([1, 1, -1])
    segment_distance = np.sqrt(np.power(x - x_3d, 2) + np.power(y - y_3d, 2))
    max_dist = im_filled.shape[0] + im_filled.shape[1]
    new_segment_distance = (segment_distance / max_dist + im_filled) * max_dist
    wid = new_segment_distance.min(axis=(1, 2)).max()
    return wid


def get_width_by_distance(im: _NP_T, splines: _NP_T) -> float:
    im_filled = _ndi.binary_fill_holes(im)
    x = splines[0].astype(int)
    y = splines[1].astype(int)
    dist = _cv.distanceTransform(im_filled.astype("u1"), _cv.DIST_L2, 3)
    wid = dist[y, x]
    return wid


def flip_check(x: _NP_T, y: _NP_T):
    """Check if plots of head and tail is flipping."""
    gap_headtail = np.mean(
        np.power(np.diff(x, n=1, axis=0), 2) + np.power(np.diff(y, n=1, axis=0), 2),
        axis=1,
    )
    gap_headtail_ex = np.mean(
        np.power(x[1:, :] - x[:-1, ::-1], 2) + np.power(y[1:, :] - y[:-1, ::-1], 2),
        axis=1,
    )
    ex_t = gap_headtail > gap_headtail_ex
    ex_r = np.bitwise_xor.accumulate(ex_t)
    x[1:, :][ex_r] = x[1:, ::-1][ex_r]
    y[1:, :][ex_r] = y[1:, ::-1][ex_r]
    return x, y


def read_imagestack(
    filenames: _T.List[_PATH_T],
    reader: ImageReader,
) -> _T.Tuple[_NP_T, Offset]:
    assert len(filenames), "Input is empty"
    T = len(filenames)

    items = iter(_tqdm(enumerate(filenames), total=T, desc="loading: "))
    # read the first item to get size of
    _, f0 = next(items)
    im = reader.imread(f0)

    imagestack = np.zeros((T, im.shape[0], im.shape[1]), dtype="uint8")
    imagestack[0, :, :] = im

    for t, f in items:
        imagestack[t, :, :] = reader.imread(f)
    imagestack, offset = trim_imagestack(imagestack)
    return (imagestack > 0).astype("f8"), offset


def get_skeleton(
    im: _NP_T, n_seg: int = 100, *, fill_hole: bool = True
) -> _OPTIONAL_NP:
    """skeletonize image and get splined plots"""

    # skeletonize image
    im_filled = im.copy()
    if fill_hole:
        im_filled = _ndi.binary_fill_holes(im_filled)
    im_skeleton = _morphology.skeletonize(im_filled)

    if im_skeleton.sum() == 0:
        return None

    pts = np.argwhere(im_skeleton)

    splines = np.ones((2, n_seg + 1)) * pts[0][1]
    splines[0, :] = splines[0, :] * pts[0][1]
    splines[1, :] = splines[1, :] * pts[0][1]

    if len(pts) == 1:
        return splines

    # make distance matrix
    adj_mtx = _distance_matrix(pts, pts, threshold=len(pts) * len(pts) * 2 + 10)

    adj_mtx[adj_mtx >= 1.5] = 0  # delete distance between isolated points
    csr = _csr_matrix(adj_mtx)
    adj_sum = np.sum(adj_mtx, axis=0)

    # get tips of longest path
    d1 = _csgraph.shortest_path(csr, indices=np.argmax(adj_sum < 1.5))
    while np.sum(d1 == np.inf) > (d1.shape[0] >> 1):
        adj_sum[np.argmax(adj_sum < 1.5)] = 2
        d1 = _csgraph.shortest_path(csr, indices=np.argmax(adj_sum < 1.5))
    d1[d1 == np.inf] = 0
    d2, p = _csgraph.shortest_path(
        csr,
        indices=np.argmax(d1),
        return_predecessors=True,
    )
    d2[d2 == np.inf] = 0

    # get longest path
    plots = []
    arclen = []
    point = np.argmax(d2)  # This is the start point(the end point is np.argmax(d1))
    while point != np.argmax(d1) and point >= 0:
        plots.append(pts[point])
        arclen.append(d2[point])
        point = p[point]
    plots.append(pts[point])
    arclen.append(d2[point])
    plots = np.array(plots)
    arclen = np.array(arclen)[::-1]

    # interpolation
    div_linespace = np.linspace(0, np.max(arclen), n_seg + 1)
    f_x = _interp1d(arclen, plots[:, 1], kind="linear")
    f_y = _interp1d(arclen, plots[:, 0], kind="linear")
    splines[0] = f_x(div_linespace).round().astype(int)
    splines[1] = f_y(div_linespace).round().astype(int)

    return splines


def calc_all_skeleton_and_width(
    imagestack: _NP_T,
    n_segs: int,
) -> _T.Tuple[_NP_T, _NP_T, float]:
    """read images and get skeletonized plots"""
    assert imagestack.ndim == 3, "Input is empty"
    T = imagestack.shape[0]
    txy = np.zeros((T, 2, n_segs + 1))
    pre_width = np.zeros((T, n_segs + 1))

    # first item does not require the flipping check
    items = iter(
        _tqdm(enumerate(imagestack), total=T, desc="skeletonize & get_width: ")
    )
    _, im = next(items)
    txy[0, :, :] = get_skeleton(im, n_segs)
    pre_width[0] = get_width_by_distance(imagestack[0], txy[0])

    for t, im in items:
        xy1 = get_skeleton(im, n_segs)
        xy0 = txy[t - 1, :, :]
        gap_headtail = np.power(xy1 - xy0, 2).sum()
        gap_headtail_rev = np.power(xy1[:, ::-1] - xy0, 2).sum()

        # Revert the orientation if the flipping is smaller than original
        if gap_headtail > gap_headtail_rev:
            txy[t] = xy1[:, ::-1]
        else:
            txy[t] = xy1

        pre_width[t] = get_width_by_distance(imagestack[t], txy[t])

    delta = np.power(np.diff(txy, n=1, axis=2), 2).sum(axis=1)
    unit_per_seg = np.sqrt(np.median(delta))
    return txy, pre_width, unit_per_seg


def calc_theta_from_xy(txy: _NP_T) -> _NP_T:
    assert txy.ndim == 3, "txy is not a three dimension np.ndarray (T, xy, n_pts)"
    T, _, n_pts = txy.shape
    n_segs = n_pts - 1
    dxy = np.diff(txy, n=1, axis=2)
    theta = np.arctan2(dxy[:, 1], dxy[:, 0])
    # Arrange theta if the gap is larget than pi
    # Adjust the middle theta between time point
    pi = np.pi
    mid = n_segs // 2
    t_gap = np.diff(theta[:, mid], n=1)
    t_adjust = np.sign(t_gap) * 2 * pi
    t_adjust[np.abs(t_gap) < pi] = 0
    theta[1:, :] -= t_adjust.cumsum().reshape(-1, 1)

    # adjust right hand side of theta within same time points
    r_gap = np.diff(theta[:, mid:], n=1, axis=1)
    r_adjust = np.sign(r_gap) * 2 * pi
    r_adjust[np.abs(r_gap) < pi] = 0
    theta[:, mid + 1 :] -= r_adjust.cumsum(axis=1)

    # adjust left hand side
    l_gap = np.diff(theta[:, : mid + 1], n=1, axis=1)
    l_adjust = np.sign(l_gap) * (-2) * pi
    l_adjust[np.abs(l_gap) < pi] = 0
    l_adjust_rev = np.flip(l_adjust, axis=1)
    theta[:, :mid] += np.flip(np.cumsum(l_adjust_rev, axis=1), axis=1)

    return theta


def estimate_batchsize(
    device: str,
    time_span: int,
    im_width: int,
    im_height: int,
    n_segs: int,
    s_m: int = 8000,
):
    """Estimate maximum time span used for training section in terms of CUDA memory."""
    MiB = 1 << 20
    try:
        from torch.cuda import get_device_properties, memory_allocated

        tot_mem = get_device_properties(device).total_memory
        allocated_mem = memory_allocated(device)
        free_mem = (tot_mem - allocated_mem) / MiB
        batchsize = int(s_m * free_mem // (im_width * im_height * n_segs))
    except (ImportError, RuntimeError) as e:
        print(e)
        batchsize = time_span

    return batchsize


def calc_init_center(read_image: _NP_T) -> _NP_T:
    assert read_image.ndim == 3, "read_image is not 3-d np.ndarray (T, Y, X)"
    mask = read_image >= read_image.max(axis=(1, 2))[:, None, None]
    ts, ys, xs = np.where(mask)
    # calculate split points, and omit end slice
    ts_count = np.bincount(ts).cumsum()[:-1]
    cy = [y.mean() if y.size else -1 for y in np.split(ys, ts_count)]
    cx = [x.mean() if x.size else -1 for x in np.split(xs, ts_count)]
    return np.array([cx, cy]).T


def _find_theta_gen(first_theta: _NP_T, pretheta: _NP_T, step: int) -> int:
    """
    Find minimun MSE theta by theta(t=0)

    Args:
        theta (np.ndarray): The time series of worm theta data. [Time, theta]
        pretheta (np.ndarray): the pretheta [theta]
        step (int, optional): step size. Defaults to 1.

    Returns:
        int: _description_
    """

    import itertools as its

    steps = its.count(step=step * np.pi * 2)
    pre_mse = np.inf
    for delta in steps:
        mse = np.sum((first_theta - pretheta + delta) ** 2)
        if pre_mse < mse:
            break
        # yield mse count
        yield 1
        pre_mse = mse


def find_theta(first_theta: _NP_T, pretheta: _NP_T, step: int = 1) -> int:
    """
    Find minimun MSE theta by theta(t=0)

    Args:
        theta (np.ndarray): The time series of worm theta data. [Time, theta]
        pretheta (np.ndarray): the pretheta [theta]
        step (int, optional): step size. Defaults to 1.

    Returns:
        int: _description_
    """
    return sum(_find_theta_gen(first_theta, pretheta, step))


def find_theta_candidate(theta) -> _T.Tuple[_NP_T, _NP_T]:
    PI = np.pi
    first, last = theta[(0, -1), :]
    i_normal = find_theta(first, last) - find_theta(first, last, -1)
    last_rev = last[::-1] + PI
    i_reverse = find_theta(first, last_rev) - find_theta(first, last_rev, -1)

    theta_pair = np.array((last + i_normal * 2 * PI, last_rev + i_reverse * 2 * PI))

    theta_cands = theta_pair[:, None, :] + np.array((2 * PI, -2 * PI))[None, :, None]

    first_3d = first[None, None, :]
    # theta_losses = [[normal_p, normal_m],[reverse_p,reverse_m]]
    theta_losses = np.sum((theta_cands - first_3d) ** 2, axis=2)
    # return the index of minimum loss.
    cand_idx = theta_losses.argmin(axis=1)

    theta_subpair = theta_cands[(0, 1), cand_idx, :]
    return theta_pair, theta_subpair


# def make_worm_numpy(skel, pre_width, im_width, im_height):

#     assert skel.ndim == 2, "Dimension must be (2, n_pts)"
#     _, n_pts = skel.shape

#     cent_mid_3d = ((skel[:, :-1] + skel[:, 1:]) / 2.0).reshape(2, n_pts - 1, 1)

#     worm_wid = torch.zeros(plot_n - 1).to(device)
#     for i in range(plot_n - 1):
#         worm_wid[i] = worm_width(
#             i, plot_n, params["alpha"], params["gamma"], params["delta"]
#         )

#     cent_mid_x_3d = cent_mid_x_3d.reshape([-1, plot_n - 1, 1, 1]).to(torch.float32)
#     cent_mid_y_3d = cent_mid_y_3d.reshape([-1, plot_n - 1, 1, 1]).to(torch.float32)
#     worm_wid_3d = worm_wid.reshape([1, plot_n - 1, 1, 1])
#     y_3d = torch.arange(image_shape[1]).reshape([1, 1, -1, 1]).to(device)
#     x_3d = torch.arange(image_shape[2]).reshape([1, 1, 1, -1]).to(device)
#     segment_distance_3d = torch.sqrt(
#         (cent_mid_x_3d - x_3d) ** 2 + (cent_mid_y_3d - y_3d) ** 2
#     )
#     image_3d = pixel_value(segment_distance_3d, worm_wid_3d)
#     image, _ = torch.max(image_3d, dim=1)
#     return image
