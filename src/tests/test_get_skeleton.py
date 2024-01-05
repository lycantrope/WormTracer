import pytest
from wormtracer import preprocess, skeleton
import numpy as np


# content of test_sample.py

np.random.seed(1953)
IMAGE_W = 100
IMAGE_H = 100


@pytest.fixture
def img():
    return (
        np.random.randint(0, 2, IMAGE_W * IMAGE_H)
        .reshape(IMAGE_H, IMAGE_W)
        .astype("uint8")
        * 255
    )


@pytest.fixture
def im_empty():
    return (
        np.random.randint(0, 2, IMAGE_W * IMAGE_H)
        .reshape(IMAGE_H, IMAGE_W)
        .astype("uint8")
        * 0
    )


# def test_skeleton_eq(img):
#     skel_old_x, skel_old_y = preprocess._old_get_skeleton(img, 101)
#     skel_x, skel_y = preprocess.get_skeleton(img, 101)
#     c_skel_x, c_skel_y = skeleton.get_skeleton(img, 101)

#     np.testing.assert_equal(skel_x, skel_old_x.round().astype(int))
#     np.testing.assert_equal(skel_y, skel_old_y.round().astype(int))

#     np.testing.assert_equal(skel_x, c_skel_x)
#     np.testing.assert_equal(skel_y, c_skel_y)


def test_empty_im_skeleton(im_empty):
    skel_x, skel_y = preprocess.get_skeleton(im_empty, 101)
    c_skel_x, c_skel_y = skeleton.get_skeleton(im_empty, 101)

    np.testing.assert_equal(skel_x, c_skel_x)
    np.testing.assert_equal(skel_y, c_skel_y)


@pytest.mark.benchmark(
    disable_gc=True,
    warmup=False,
)
def test_py_skeleton(benchmark, img):
    skel_x, skel_y = benchmark(preprocess.get_skeleton, img, 101)


@pytest.mark.benchmark(
    disable_gc=True,
    warmup=False,
)
def test_cython_skeleton(benchmark, img):
    skel_x, skel_y = benchmark(skeleton.get_skeleton, img, 101)


# @pytest.mark.benchmark(
#     max_time=5.0,
#     disable_gc=True,
#     warmup=False,
# )
# def _test_old_py_skeleton(benchmark, img):
#     skel_x, skel_y = benchmark(preprocess._old_get_skeleton, img, 101)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
