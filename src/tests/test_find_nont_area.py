# from wormtracer.preprocess import find_nont_area as new_find_nont_area


import numpy as np


def old_find_nont_area(image_losses, borderline, under_borderline):
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
    if len(nont_end) > len(nont_ini):
        nont_end = nont_end[1:]

    if len(nont_end) < len(nont_ini):
        nont_ini = nont_ini[:-1]
    nont = np.array([nont_ini, nont_end])
    return nont


def new_find_nont_area(image_losses, q1, q2):
    complex_area = image_losses > q2
    area_diff = np.ones_like(complex_area).astype(bool)
    area_diff[1:] = complex_area[1:] ^ complex_area[:-1]
    nont_ini = np.where(area_diff & complex_area)[0]
    nont_end = np.where(area_diff & (~complex_area))[0]
    if len(nont_end) > len(nont_ini):
        nont_end = nont_end[1:]

    if len(nont_end) < len(nont_ini):
        nont_ini = nont_ini[:-1]

    nont = np.array([nont_ini, nont_end])
    return nont


# content of test_sample.py

np.random.seed(1953)


def test_find_nont_area():
    data = np.random.randint(0, 255, 100 * 10).reshape(10, 100)
    maxi = data.max(axis=1)
    mini = data.min(axis=1)
    q1_all = maxi * 0.4 + 0.6 * mini
    q2_all = maxi * 0.2 + 0.8 * mini

    for d, q1, q2 in zip(data, q1_all, q2_all):
        v1 = old_find_nont_area(d, q1, q2)
        v2 = new_find_nont_area(d, q1, q2)
        np.testing.assert_equal(v1, v2)

        # np.testing.assert_equal(v1[2], v2[2])
