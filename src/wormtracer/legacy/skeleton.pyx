import cython as _cython
import numpy as np
cimport numpy as np
import scipy.ndimage as _ndi
from scipy.interpolate import interp1d as _interp1d
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import csgraph as _csgraph
from scipy.spatial import distance_matrix as _distance_matrix
import skimage.morphology as _morphology


@_cython.boundscheck(False)
@_cython.wraparound(False)
def get_skeleton(np.ndarray[unsigned char, ndim=2, mode="c"] im, size_t n_seg):
    """
        [Note] Current implementation is slower than pure python
        skeletonize image and get splined plots
    """
    
    cdef:
         long point, n_pts
         np.ndarray[long, ndim=1, mode="c"] x_splined, y_splined

         np.ndarray[unsigned char, ndim=2, mode="c"] im_filled, im_skeleton
         np.ndarray[long, ndim=2, mode="c"] pts, plots_np
         np.ndarray[double, ndim=1, mode="c"] adj_sum, div_linespace, d1, d2, arclen_np
         np.ndarray[double, ndim=2, mode="c"] adj_mtx

    # skeletonize image
    im_filled = _ndi.binary_fill_holes(im)
    im_skeleton = _morphology.skeletonize(im_filled)
    if np.sum(im_skeleton) == 0:
        return None,None

    pts =np.ascontiguousarray(np.argwhere(im_skeleton == 1))
    n_pts = len(pts)
    if n_pts == 1:
        x_splined = np.ones(n_seg) * pts[0][1]
        y_splined = np.ones(n_seg) * pts[0][0]
        return x_splined, y_splined

    # make distance matrix
    adj_mtx = _distance_matrix(pts, pts, threshold= n_pts*n_pts+100)

    adj_mtx[adj_mtx >= 1.5] = 0  # delete distance between isolated points
    csr = _csr_matrix(adj_mtx)
    adj_sum = np.sum(adj_mtx, axis=0)

    # get tips of longest path
    d1 = _csgraph.shortest_path(csr, indices=np.argmax(adj_sum < 1.5))
    while np.sum(d1 == np.inf) > d1.shape[0] // 2:
        adj_sum[np.argmax(adj_sum < 1.5)] = 2
        d1 = _csgraph.shortest_path(csr, indices=np.argmax(adj_sum < 1.5))
    d1[d1 == np.inf] = 0
    d2, p = _csgraph.shortest_path(csr, indices=np.argmax(d1), return_predecessors=True)
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
    plots_np = np.array(plots)
    arclen_np = np.array(arclen[::-1])
    
    # interpolation
    div_linespace = np.linspace(0, np.max(arclen_np), n_seg)
    f_x = _interp1d(arclen_np, plots_np[:, 1], kind="linear")
    f_y = _interp1d(arclen_np, plots_np[:, 0], kind="linear")
    x_splined = f_x(div_linespace).round().astype("long")
    y_splined = f_y(div_linespace).round().astype("long")

    return x_splined, y_splined
