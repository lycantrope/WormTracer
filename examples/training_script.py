# %% import dependency
import numpy as np
import matplotlib.pyplot as plt
import torch
import wormtracer as wt
from wormtracer.dataset import get_use_points
from wormtracer.loss import find_outliner
from wormtracer.train import train3
from wormtracer.preprocess import (
    ImageReader,
    calc_init_center,
    calc_theta_from_xy,
    estimate_batchsize,
    find_theta_candidate,
    read_imagestack,
    calc_all_skeleton_and_width,
)
from wormtracer.model import (
    get_image_loss_max,
    make_worm_batch,
    WormModel,
    WormSkeletonLayer,
    WormShapeLayer,
    WormImageLayer,
)
from pathlib import Path

from wormtracer.utils import calc_avg_shape_params

#  %%
home = Path(r".")

n_segs = 100
ext = "jpg"


params = wt.parameter.HyperParameters(
    continuity_loss_weight=1e5,
    smoothness_loss_weight=1e6,
    length_loss_weight=50,
    center_loss_weight=50,
    epoch_plus=1500,
    speed=0.05,
    lr=0.05,
    body_ratio=90,
    n_segments=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
# %%
filenames = sorted(home.glob("*.{}".format(ext)), key=lambda x: x.stem)

# %%
import cv2 as cv


def binary_function_builder(filenames, size: int = 50):
    cands = np.random.choice(filenames, size)
    imgs = np.array([cv.imread(str(p), flags=cv.IMREAD_GRAYSCALE) for p in cands])
    background = np.median(imgs, axis=0).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    def threshold(im: np.ndarray) -> np.ndarray:
        im = (im <= background).astype(np.uint8) * 255
        # _, im = cv.threshold(im, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY)

        im = cv.morphologyEx(im, cv.MORPH_OPEN, kernel=kernel, iterations=2)
        return im

    return threshold


test_im = cv.imread(str(filenames[0]), cv.IMREAD_GRAYSCALE)
binarize_fn = binary_function_builder(filenames)
binarize_fn(test_im)
# %%
# reader = ImageReader.build_reader_from_image(filenames[0], 1.0, binarize_fn= binary_function_builder(filenames),)
# custom reader

reader = ImageReader(worm_is_black=True, is_binarized=False, binarize_fn=binarize_fn)

# %%
imagestack, all_offset = read_imagestack(
    filenames[:100],
    reader,
)
imagestack_all = imagestack
imagestack = imagestack[:, 100:600, 65:370] == 0
imagestack = imagestack.astype("f8")

# %%


# %%
T, im_height, im_width = imagestack.shape
txy, pre_width, tot_unit_length = calc_all_skeleton_and_width(
    imagestack.astype("u1"),
    params.n_segments,
)

assert txy.shape == (
    T,
    2,
    params.n_segments + 1,
), "The skeleton and imagesize got wrong"

theta = calc_theta_from_xy(txy)

alpha = pre_width.min()

cap_span = estimate_batchsize(params.device, T, im_width, im_height, params.n_segments)
cap_span = 20
# %% screen for complex and normal block
im_model = make_worm_batch(
    txy,
    pre_width,
    im_width,
    im_height,
    cap_span,
    params.device,
)
image_losses = np.mean(
    (im_model - imagestack) ** 2,
    axis=(1, 2),
)

q25, q75 = np.percentile(image_losses, q=[25, 75])
min_idx = np.argmin(image_losses)
image_loss_max = get_image_loss_max(imagestack[min_idx], txy[min_idx], pre_width)

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(image_losses)
ax.axhline(image_loss_max, 0.0, 1.0)
plt.show()

# %%
training_blocks = get_use_points(image_losses, image_loss_max)

# %%
for xy, im, im_m in zip(txy, imagestack, im_model):
    fig, ax = plt.subplots(1, 2)
    cax = ax[0].imshow(im, cmap="grey")
    ax[1].imshow((im_m * 255), cmap="gray")
    ax[0].plot(xy[0], xy[1], color="red")
    plt.show()


# %%
# main loop 1
simple_mask = training_blocks.get_block_mask(get_complex=False)
unit_length = np.sqrt(np.median((np.diff(txy[simple_mask], axis=2) ** 2).sum(axis=1)))
shape_layer = WormShapeLayer(alpha=pre_width[simple_mask].mean(), delta=0.0, gamma=0.0)

losses_all = []
shape_params = []
for idx, is_complex, start, end in training_blocks.batch_iter(cap_span):
    if is_complex:
        continue

    file_block = filenames[start : end + 1]
    T = len(file_block)
    ims_block, block_offset = read_imagestack(
        file_block,
        reader,
    )

    theta_block = theta[start : end + 1, :].copy()

    _, H, W = ims_block.shape
    # read and preprocess images
    target_theta, _ = find_theta_candidate(theta_block)
    theta_block[-1, :] = target_theta[0]

    init_ct = calc_init_center(ims_block)
    skel_layer = WormSkeletonLayer(
        ct=init_ct.copy(),
        theta=theta_block,
        unit_length=unit_length,
    )
    model = WormModel(
        shape_layer=shape_layer,
        skel_layer=skel_layer,
        image_layer=WormImageLayer(width=W, height=H),
    )
    # make model instance and training
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    model, im_model, txy_model, losses = train3(
        model, ims_block, optimizer, params, is_complex
    )

    # get trace information
    theta_model = skel_layer.theta.detach().cpu().numpy()
    unit_model = skel_layer.unit_length.detach().cpu().numpy().reshape(-1, 1)
    ct_model = skel_layer.ct.detach().cpu().numpy()

    losses_all.append((idx, losses))
    shape_params.append((T, model.shape_layer.get_shape_params()))

    txy[start:end, :, :] = (
        txy_model + ct_model.reshape(T, 2, 1) - np.array(all_offset).reshape(1, 2, 1)
    )

# %%

avg_shape_params = calc_avg_shape_params(shape_params)
# main loop 2
shape_layer = WormShapeLayer.from_shape_params(avg_shape_params)

# main loop 2
for idx, is_complex, start, end in training_blocks.batch_iter(cap_span):
    if not is_complex:
        continue

    file_block = filenames[start : end + 1]
    T = len(file_block)
    ims_block, block_offset = read_imagestack(
        file_block,
        reader,
    )

    theta_block = theta[start : end + 1, :].copy()

    _, H, W = ims_block.shape
    # read and preprocess images
    init_ct = calc_init_center(ims_block)

    target_theta, _ = find_theta_candidate(theta_block)

    init_theta = np.linspace(theta_block[0, :], target_theta[0], T)
    model = WormModel(
        skel_layer=WormSkeletonLayer(
            ct=init_ct.copy(),
            theta=init_theta,
            unit_length=unit_length,
        ),
        shape_layer=shape_layer,
        image_layer=WormImageLayer(width=W, height=H),
    )
    # make model instance and training
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    model, im_model, txy_model, losses = train3(
        model,
        ims_block,
        optimizer,
        params,
        is_complex,
    )

    # flip final theta to trace again
    init_theta_filp = np.linspace(theta_block[0, :], target_theta[1], T)

    model2 = WormModel(
        skel_layer=WormSkeletonLayer(
            ct=init_ct.copy(),
            theta=init_theta_filp,
            unit_length=unit_length,
        ),
        shape_layer=shape_layer,
        image_layer=WormImageLayer(width=W, height=H),
    )

    optimizer = torch.optim.Adam(model2.parameters(), lr=params.lr)
    model2, im_rev_model, txy_model_rev, losses_rev = train3(
        model2,
        ims_block,
        optimizer,
        params,
        is_complex,
    )
    skel_layer = model.skel_layer
    # get trace information if loss is smaller
    if losses > losses_rev:
        skel_layer = model2.skel_layer
        im_model = im_rev_model
        losses = losses_rev

    # get trace information
    theta_model = skel_layer.theta.detach().cpu().numpy()
    unit_model = skel_layer.unit_length.detach().cpu().numpy().reshape(-1, 1)
    ct_model = skel_layer.ct.detach().cpu().numpy()

    losses_all.append((idx, losses))
    # reconstruct plots from model results
    txy[start:end, :, :] = (
        txy_model + ct_model.reshape(T, 2, 1) - np.array(all_offset).reshape(1, 2, 1)
    )

# %%
# revise areas which have too large loss
losses_all = sorted(losses_all, key=lambda x: x[0])
losses_all = [l for (_, l) in losses_all]
outliner_mask = find_outliner(losses_all)
for idx, is_complex, start, end in training_blocks.batch_iter(cap_span):
    if not is_complex and not is_complex:
        continue

    file_block = filenames[start : end + 1]
    T = len(file_block)
    ims_block, block_offset = read_imagestack(
        file_block,
        reader,
    )

    theta_block = theta[start : end + 1, :].copy()

    _, H, W = ims_block.shape

    init_ct = calc_init_center(ims_block)

    ## use sub theta candidate as target
    _, target_theta = find_theta_candidate(theta_block)

    init_theta_filp = np.linspace(theta_block[0, :], target_theta[0], T)
    model = WormModel(
        shape_layer=WormSkeletonLayer(
            ct=init_ct.copy(),
            theta=init_theta_filp,
            unit_length=unit_length,
        ),
        skel_layer=skel_layer,
        image_layer=WormImageLayer(width=W, height=H),
    )
    # make model instance and training
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    model, im_model, txy_model, losses = train3(
        model,
        ims_block,
        optimizer,
        params,
        is_complex,
    )

    # get trace information

    # flip final theta to trace again

    init_theta_flip = np.linspace(theta_block[0, :], target_theta[1], T)

    model2 = WormModel(
        skel_layer=WormSkeletonLayer(
            ct=init_ct.copy(),
            theta=init_theta_flip,
            unit_length=unit_length,
        ),
        shape_layer=shape_layer,
        image_layer=WormImageLayer(width=W, height=H),
    )

    optimizer = torch.optim.Adam(model2.parameters(), lr=params.lr)
    model2, im_rev_model, txy_model_rev, losses_rev = train3(
        model2,
        ims_block,
        optimizer,
        params,
        is_complex,
    )

    skel_layer = model.skel_layer
    # get trace information if loss is smaller
    if losses > losses_rev:
        skel_layer = model2.skel_layer
        im_model = im_rev_model
        losses = losses_rev

    theta_model = skel_layer.theta.detach().cpu().numpy()
    unit_model = skel_layer.unit_length.detach().cpu().numpy().reshape(-1, 1)
    ct_model = skel_layer.ct.detach().cpu().numpy()

    if losses > losses_all[idx]:
        continue

    # reconstruct plots from model results
    txy[start:end, :, :] = (
        txy_model + ct_model.reshape(T, 2, 1) - np.array(all_offset).reshape(1, 2, 1)
    )
# %%


fig, _ = plt.subplots(4, 4)


for i, ax in enumerate(fig.get_axes()):
    ax.imshow(imagestack[i + 50])

fig.tight_layout()

# %%
