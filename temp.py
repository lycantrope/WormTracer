# %%

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from wormtracer.dataset import TrainingBlocks
from wormtracer.loss import find_outliner
from wormtracer.preprocess import (
    calc_all_skeleton_and_width,
    trim_imagestack,
)

# %%
data_p = Path("data/worm040_0632_0/0040N2AssayOffLeft00.tif")


def stack_from_mutli_tiff(data_p):

    ret, images = cv2.imreadmulti(str(data_p))
    if not ret or len(images) == 0:
        raise IOError(f"Cannot read: {data_p}")

    H, W = images[0].shape
    T = len(images)

    imagestack = np.zeros((T, H, W), dtype="u1")

    for i, im in enumerate(images):
        contours = cv2.findContours(
            im,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )[-2]
        if not contours:
            imagestack[i] = im
            continue
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        imagestack[i, y : y + h, x : x + w] = im[y : y + h, x : x + w]

    return trim_imagestack(imagestack)


imagestack, offset = stack_from_mutli_tiff(data_p)

# %%

txy, pre_width, tot_unit_length = calc_all_skeleton_and_width(
    imagestack.astype("u1"),
    100,
)

# %%

imagestack_view = np.lib.stride_tricks.sliding_window_view(imagestack, 16, axis=0)
DECAY = (-1.0 + np.sqrt(5)) / 2.0
prev = np.zeros_like(imagestack[0])
weight = np.power(DECAY, np.arange(1, 17)[::-1])
fig, _ = plt.subplots(3, 5)
for i, ax in enumerate(fig.get_axes()):
    idx = i + 2410 - 20

    im = (imagestack_view[idx] * weight.reshape(1, 1, -1)).sum(axis=-1)
    det = im.max() - im.min()
    im = (im - im.min()) / det * 255
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"T:{idx:d}")


fig.tight_layout()

plt.show()

# %%


def sigmoid(x):
    return np.reciprocal(1 + np.exp(x * -1.0))


def gaussian(): ...


# %%

fig, _ = plt.subplots(3, 5, figsize=(8, 5.5))
for i, ax in enumerate(fig.get_axes()):
    idx = 2415 + i
    im = imagestack[idx]
    ax.imshow(im)
    # ax.imshow(im_skeleton)
    ax.plot(txy[idx, 0], txy[idx, 1], color="red")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"T:{idx:d}, W:{pre_width[idx].max():.1f}")


plt.show()

# %%


# def find_outliner(seq):
#     p25, p50, p75 = np.percentile(seq, [25, 50, 75])
#     IQR = p75 - p25
#     upper = p50 + 3.5 * IQR
#     lower = p50 - 3.5 * IQR
#     return (seq < lower) | (seq > upper)


def find_outliner(seq):
    mean = np.nanmean(seq)
    sd = np.nanstd(seq)
    upper = mean + 3.5 * sd
    lower = mean - 3.5 * sd

    return (seq < lower) | (seq > upper)


pre_width_max = pre_width.max(axis=1)

tot_delta = np.sqrt(np.power(np.diff(txy, n=1, axis=2), 2).sum(axis=1)).sum(axis=1)

complex_area = find_outliner(pre_width_max) | find_outliner(tot_delta)


complex_area


# %%

p = np.argwhere(complex_area)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pre_width_max, color="blue")
ax.scatter(p, pre_width_max[p], color="red", marker=".")


# %%

p = np.argwhere(complex_area)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tot_delta, color="blue")
ax.scatter(p, tot_delta[p], color="red", marker=".")


# %%

blocks = TrainingBlocks.from_complex_area(complex_area=complex_area)


for block in blocks.batch_iter():
    if not block.is_complex:
        continue
    _, _, start, end = block
    size = end - start + 1
    idx = slice(start, end + 1)
    fig, _ = plt.subplots(max(size // 5, 1), 5, figsize=(8, 1.76 + (size / 5 * 0.9)))
    im_block = imagestack[idx]
    txy_block = txy[idx]
    [ax.set_axis_off() for ax in fig.get_axes()]
    for i, (im, xy, ax) in enumerate(zip(im_block, txy_block, fig.get_axes())):
        # skel = get_skeleton(im, fill_hole=False)
        ax.set_title(f"T:{i+start:d} len:{tot_delta[i+start]:.1f}")
        ax.imshow(im)
        # ax.imshow(im_skeleton)
        ax.plot(xy[0], xy[1], color="red")
        # ax.plot(skel[0], skel[1], color="blue")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_on()

    fig.tight_layout()


# %%


for block in blocks.batch_iter():
    if block.is_complex:
        continue
    _, _, start, end = block
    size = min(end - start + 1, 15)
    idx = slice(start, end + 1)
    fig, _ = plt.subplots(max(size // 5, 1), 5, figsize=(8, 1.76 + (size / 5 * 0.9)))
    im_block = imagestack[idx]
    txy_block = txy[idx]
    [ax.set_axis_off() for ax in fig.get_axes()]
    for im, xy, ax in zip(im_block, txy_block, fig.get_axes()):
        # skel = get_skeleton(im, fill_hole=False)

        ax.imshow(im)
        # ax.imshow(im_skeleton)
        ax.plot(xy[0], xy[1], color="red")
        # ax.plot(skel[0], skel[1], color="blue")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_on()

    fig.tight_layout()

# %%
