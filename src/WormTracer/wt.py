### WormTracer main package wt.py ###

import datetime
import json
import logging
import os
import sys
from pathlib import Path, PurePath

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import roifile
import tifffile
import torch
import yaml
from matplotlib import animation, rc

from . import __version__
from .functions import (
    Model,
    calc_cap_span,
    calc_xy_and_prewidth,
    cancel_reduction,
    clear_dir,
    find_losslarge_area,
    flip_check,
    get_filenames,
    get_image_loss_max,
    get_property,
    get_use_blocks,
    judge_head_amplitude,
    judge_head_frequency,
    load_image,
    loss_compare,
    make_image,
    make_plot,
    make_theta_cand,
    make_theta_from_xy,
    remove_progress,
    save_centerline_to_roi,
    save_progress,
    set_init_xy,
    set_output_path,
    show_image,
    show_loss_plot,
    train3,
)

### input information and params ###
"""
dataset_path (mandatory):
Path to a folder including input images.
Images are either as a single multipage tiff file or serial numbered image files, with either of the following format.
".bmp", ".dib", ".pbm", ".pgm", ".ppm", ".pnm", ".ras", ".png", ".tiff", ".tif", ".jp2", ".jpeg", ".jpg", ".jpe"
ALL RESULTS ARE SAVED in dataset_path.

output_directory (can be omitted):
Path to a directory in which output of WormTracer will be saved in a folder named xxxx_output_n, where xxxx comes from dataset_path name, n is a serial number.
If output_directory is not given at all or is an empty string, the folder xxxx_output_n is created at the same level as dataset_path.
If the directory output_directory does not exist, a directory is created.

functions_path (mandatory):
Path to functions.py file, which is essential.

local_time_difference:
Time difference relative to UTC (hours). Affects time stamps used in result file names.

start_T, end_T(int, > 0):
You can set frames which are applied to WormTracer.
If you want to use all frames, set both start_T and end_T as 0 (assuming the image number starts from 0).

rescale(float, > 0, <= 1):
You can change the scale of image to use for tracing by this value.
If MEMORY ERROR occurs, set this value lower.
For example if you set it 0.5, the size of images will be half of the original.
Default value is 1.

Tscale(int, > 0):
You can reduce frames by thinning out the movie by this value.
If MEMORY ERROR occurs, set this value higher.
For example, if you set it to 2, even-numbered frames will be picked up.
This parameter is useful in case frame rate is too high.
Default value is 1.

continuity_loss_weight(float, > 0):
This value is the weight of the continuity constraint.
Around 10000 is recommended, but if the object moves fast, set it lower.

smoothness_loss_weight(float, > 0):
This value is the weight of the smoothness constraint.
Around 50000 is recommended, but if the object bends sharply, set it lower.

length_loss_weight(float, > 0):
This value is the weight of the length continuity constraint.
Around 50 is recommended, but if length of the object changes drastically, set it lower.

center_loss_weight(float, > 0):
This value is the weight of the center position constraint.
Around 50 is recommended.

plot_n(int, > 1):
This value is plot number of center line.
Around 100 is recommended.

epoch_plus(int, > 0):
This value is additional training epoch number.
After annealing is finished, training will be performed for at most epoch_plus times.
Over 1000 is recommended.

speed(float, > 0):
This value is speed of annealing progress.
The larger this value, the faster the learning is completed.
0.1 is efficient, 0.05 is cautious.

lr(float, > 0):
This value is learning rate of training.
Around 0.05 is recommended.

body_ratio(float, > 0):
This value is body (rigid part of the object) ratio of the object.
If the object is a typical worm, set it around 90.

judge_head_method (string, 'amplitude' or 'frequency'):
Discriminate head and tail by eigher of the following criteria,
Variance of body curvature is larger near the head ('amplitude')
Frequency of body curvature change is larger near the head ('frequency')

num_t(int, > 0):
This value means the number of images which are displayed
when show_image function is called.
Default value is 5.
If you want to see all frames, set it to "np.inf".

ShowProgress (True or False):
If True, shows progress during optimization repeats.

SaveProgress (True or False):
If True, saves worm images during optimization in "progress_image" folder created in datafolder.

show_progress_freq(int, > 0):
This value is epoch frequency of displaying tracing progress.

save_progress_freq(int, > 0):
This value is epoch frequency of saving tracing progress.

save_progress_num(int, > 0):
This value is the number of images that are included in saved progress tracing.

SaveCenterlinedWormsSerial (True or False):
If True, saves input images with estimated centerline as seirial numbered png files in full_line_images folder.

SaveCenterlinedWormsMovie (True or False):
If True, saves input images with estimated centerline as a movie full_line_images.mp4

SaveCenterlinedWormsMultitiff (True or False):
If True, saves input images with estimated centerline as a multipage tiff full_line_images.tif

"""

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def run(
    parameter_file, dataset_path, output_directory=None, **kwargs
):  # execute the whole WormTracer process, kwargs are optional parameter=value pairs
    matplotlib.use("Agg")

    with open(parameter_file, "r") as yml:
        params = yaml.safe_load(yml)

    params.update(kwargs)

    # Check filenames
    filenames_all = get_filenames(dataset_path)

    # After check dataset set_output_path
    # output_path is created in output_directory
    dataset_name, output_path, output_name = set_output_path(
        dataset_path, output_directory
    )
    # Clean the previous progress if existed.
    if params["SaveProgress"]:
        clear_dir(output_path, output_name + "_progress_image")

    # setup logger
    logging.basicConfig(
        filename=output_path.joinpath(f"{output_name}.log"),
        format="%(message)s",
        level=logging.INFO,
    )
    # log
    time_now = datetime.datetime.now()
    logger.info(f"Code executed at {time_now}")
    logger.info(f"Python: {sys.version_info}")
    logger.info("WormTracer:" + __version__)
    logger.info(f"Params : {params}")
    logger.info("dataset_path = " + os.fspath(PurePath(dataset_path)))
    logger.info("output_path = " + os.fspath(PurePath(output_path)))

    #### make use of GPU ####
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Running using GPU.")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Running using Apple silicon (mps).")
    else:
        device = "cpu"
        logger.info("Running using CPU. GPU is recommended")

    # basic informatin to save
    params["dataset_path"] = dataset_path
    params["output_path"] = output_path

    imshape, Worm_is_black, multi_flag, n_input_images = get_property(
        filenames_all, params["rescale"]
    )
    Tscaled_ind = list(range(n_input_images))
    Tscaled_ind = (
        Tscaled_ind[params["start_T"] : params["end_T"] + 1]
        if params["end_T"]
        else Tscaled_ind[params["start_T"] :]
    )
    Tscaled_ind = Tscaled_ind[:: params["Tscale"]]

    # read images and get information: load_image
    real_image, y_st, x_st = load_image(
        filenames_all,
        params["rescale"],
        Worm_is_black,
        multi_flag,
        Tscaled_ind,
    )
    # getting xy plots by thinning in function ; calc_xy_and_prewidth()
    x, y, pre_width, unitLength = calc_xy_and_prewidth(
        real_image,
        params["plot_n"],
        x_st,
        y_st,
    )
    theta = make_theta_from_xy(x, y)

    # log
    time_now = datetime.datetime.now()
    logger.info(f"Reading images finished at {time_now}")
    logger.info(f"Original shape = {imshape} \n")
    logger.info(
        f"frame = {len(Tscaled_ind)} cropped_shape = {real_image.shape} unitLength = {unitLength}\n"
    )

    # make worm model image from plots
    params["alpha"] = pre_width.min()
    params["gamma"] = 0.0
    params["delta"] = 0.0
    image_info = {"image_shape": real_image.shape, "device": device}
    cap_span = calc_cap_span(image_info["image_shape"], params["plot_n"])
    model_image = make_image(x, y, x_st, y_st, params, image_info)

    # get points for trace blocks
    image_losses = np.mean((model_image - real_image) ** 2, axis=(1, 2))

    # Retrieve the best frame which has the lowest loss.
    best_frame_idx = np.argmin(image_losses)

    image_loss_max = get_image_loss_max(
        best_fit_image=real_image[best_frame_idx],
        cx=x[best_frame_idx, 0],
        cy=y[best_frame_idx, 0],
        x_st=x_st,
        y_st=y_st,
        params=params,
        image_info=image_info,
    )

    if __debug__:
        show_image(real_image, params["num_t"], title="real image")
        show_image(model_image, params["num_t"], title="model image")

    # Since dataset will be loaded during each training block, the entire dataset can be drop here.
    del real_image
    del model_image

    training_block = get_use_blocks(image_losses, image_loss_max)

    # log 3
    time_now = datetime.datetime.now()
    logger.info(f"Determining time blocks finished at {time_now}")
    logger.info(f"Total blocks: {training_block.nblock}")
    logger.info(f"Complex blocks: {len(training_block.complex_block)}")
    logger.info(f"Capspan: {cap_span}")

    all_blocks = list(training_block.batch_iter(cap_span))
    logger.info(f"{all_blocks}")

    assert all_blocks, "The training block is empty. Something goes wrong."
    if all_blocks[0].is_complex:
        logger.warning(
            "Warning! The initial frame of images is difficult to skeletonize."
        )
        logger.warning("Beginning of Results will be incorrect.")

    if all_blocks[-1].is_complex:
        logger.warning("Warning! The last frame of images is difficult to skeletonize.")
        logger.warning("Last of Results will be incorrect.")

    losses_all = {}
    shape_params = []

    # Prepare for training.
    simple_area = training_block.simple_area
    params["init_alpha"] = torch.tensor(pre_width[simple_area].mean())
    params["init_gamma"] = torch.tensor(0.0)
    params["init_delta"] = torch.tensor(0.0)

    unitLength = np.sqrt(
        np.median(np.sum(np.diff((x[simple_area], y[simple_area])) ** 2, axis=0))
    )

    logger.info("STEP1 : optimization for simple posture blocks\n")

    # main loop 1
    for block in all_blocks:
        if block.is_complex:
            continue

        logger.info(str(block))
        params["use_area"] = block
        # filenames_ = filenames[use_area[0]:use_area[1]+1]
        theta_ = theta[block.start : block.end + 1, :].copy()

        # read and preprocess images
        real_image, y_st, x_st = load_image(
            filenames_all,
            params["rescale"],
            Worm_is_black,
            multi_flag,
            Tscaled_ind[block.start : block.end + 1],
        )

        T, H, W = real_image.shape

        logger.info(f"im_shape: {real_image.shape}")

        if params.get("SaveProgress"):
            save_progress(
                real_image,
                output_path,
                output_name,
                block.start,
                block.end,
                params["save_progress_num"],
                txt="real",
            )

        # set init value
        theta_cand, _ = make_theta_cand(theta_)
        theta_[-1, :] = theta_cand[0]
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.tensor(theta_)
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength
        init_data = [init_cx, init_cy, unitLength]

        # make model instance and training
        model = (
            Model(init_cx, init_cy, init_theta, init_unitLength, params)
            .to(torch.float32)
            .to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        params["id"] = 0
        losses = train3(
            model,
            real_image,
            optimizer,
            params,
            device,
            init_data,
            output_path,
            output_name,
            is_nont=False,
        )

        # get trace information
        losses_all[block.idx] = losses
        theta_model = model.theta.detach().cpu().numpy()
        unitL_model = model.unitLength.detach().cpu().numpy().reshape(-1, 1)
        x_cent, y_cent = (
            model.cx.detach().cpu().numpy(),
            model.cy.detach().cpu().numpy(),
        )
        shape_params.append(
            (
                block.size,
                model.alpha.detach().cpu(),
                model.gamma.detach().cpu(),
                model.delta.detach().cpu(),
            )
        )
        # Add x_st, y_st to restore original position before reconstruction.
        x_cent += x_st
        y_cent += y_st
        x_model, y_model = make_plot(theta_model, unitL_model, x_cent, y_cent)

        x[block.start : block.end + 1, :] = x_model
        y[block.start : block.end + 1, :] = y_model

        # log
        logger.info(
            f"""{str(block)}
image loss : {np.mean(losses[0])}
continuity loss : {np.mean(losses[1])}
smoothing loss : {np.mean(losses[2])}
length loss : {np.mean(losses[3])}
center loss : {np.mean(losses[4])}
"""
        )
        if __debug__:
            # Only compute the model_image, if we want to show the result.
            model_image = model(batch=T, width=W, height=H)
            show_image(real_image, params["num_t"], title="real image")
            show_image(model_image, params["num_t"], title="model image")
            show_loss_plot(losses_all[block.idx], title="losses of model")

    time_now = datetime.datetime.now()
    logger.info(f"STEP1 finished at {time_now}\n")
    print(shape_params)
    shape_params = np.array(shape_params)
    # Calculating weighted average parameters
    weighted_params = np.average(
        shape_params[:, 1:],
        weights=shape_params[:, 0],
        axis=0,
    )
    params["init_alpha"] = torch.tensor(weighted_params[0])
    params["init_gamma"] = torch.tensor(weighted_params[1])
    params["init_delta"] = torch.tensor(weighted_params[2])

    logger.info("STEP2 : optimization for complex posture blocks\n")

    padding = 3
    # main loop 2
    for block in all_blocks:
        if not block.is_complex:
            continue

        # Inclusive both end [Start-3, end+3]
        start, end = (
            max(block.start - padding, 0),
            min(block.end + padding, training_block.nframe - 1),
        )

        params["use_area"] = block
        # filenames_ = filenames[use_area[0]:use_area[1]+1]
        theta_ = theta[start : end + 1, :].copy()

        # read and preprocess images
        real_image, y_st, x_st = load_image(
            filenames_all,
            params["rescale"],
            Worm_is_black,
            multi_flag,
            Tscaled_ind[start : end + 1],
        )

        T, H, W = real_image.shape
        # make flipping theta candidate
        theta_cand, _ = make_theta_cand(theta_)

        # set init value
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[0], T))
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength
        init_data = [init_cx, init_cy, unitLength]

        # make model instance and training
        model = (
            Model(init_cx, init_cy, init_theta, init_unitLength, params)
            .to(torch.float32)
            .to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        params["id"] = 0
        losses_all[block.idx] = train3(
            model,
            real_image,
            optimizer,
            params,
            device,
            init_data,
            output_path,
            output_name,
        )

        # get trace information
        theta_model = model.theta.detach().cpu().numpy()
        unitL_model = model.unitLength.detach().cpu().numpy().reshape(-1, 1)
        x_cent, y_cent = (
            model.cx.detach().cpu().numpy(),
            model.cy.detach().cpu().numpy(),
        )

        model_image = model(batch=T, width=W, height=H)

        # flip final theta to trace again
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[1], T))

        # make model instance and training
        model = (
            Model(init_cx, init_cy, init_theta, init_unitLength, params)
            .to(torch.float32)
            .to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        params["id"] = 1
        losses = train3(
            model,
            real_image,
            optimizer,
            params,
            device,
            init_data,
            output_path,
            output_name,
        )

        # get trace information if loss is smaller
        select_ind = loss_compare([losses_all[block.idx], losses])
        if select_ind:
            theta_model = model.theta.detach().cpu().numpy()
            unitL_model = model.unitLength.detach().cpu().numpy().reshape(-1, 1)
            x_cent, y_cent = (
                model.cx.detach().cpu().numpy(),
                model.cy.detach().cpu().numpy(),
            )
            model_image = model(batch=T, width=W, height=H)
            losses_all[block.idx] = losses

        # Add x_st, y_st to restore original position before reconstruction.
        x_cent += x_st
        y_cent += y_st
        x_model, y_model = make_plot(theta_model, unitL_model, x_cent, y_cent)

        l_pad = block.start - start
        r_pad = l_pad + block.size
        x_model = x_model[l_pad:r_pad]
        y_model = y_model[l_pad:r_pad]

        x[block.start : block.end + 1, :] = x_model
        y[block.start : block.end + 1, :] = y_model

        # (scale, y_t, x_t)
        if params.get("SaveProgress"):
            remove_progress(
                output_path,
                "{}-{}_id{}*.png".format(start, end, 1 - select_ind),
            )
            save_progress(
                real_image,
                output_path,
                output_name,
                start,
                end,
                params["save_progress_num"],
                txt="real",
            )

        if __debug__:
            show_image(real_image, params["num_t"], title="real image")
            show_image(model_image, params["num_t"], title="model image")
            show_loss_plot(
                losses_all[block.idx], title="losses of model{}".format(select_ind)
            )

        # log
        logger.info(
            f"""{str((start, end))}
image loss : {np.mean(losses[0])}
continuity loss : {np.mean(losses[1])}
smoothing loss : {np.mean(losses[2])}
length loss : {np.mean(losses[3])}
center loss : {np.mean(losses[4])}

"""
        )

    time_now = datetime.datetime.now()
    logger.info(f"STEP2 finished at {time_now}\n")

    # revise areas which have too large loss
    losslarge_area = find_losslarge_area(losses_all)
    logger.info(
        "STEP3 : re-optimization for unsuccessful blocks with complex postures\n"
    )

    padding = 3
    for i in losslarge_area:
        block = all_blocks[i]
        if not block.is_complex:
            continue

        start, end = (
            max(block.start - padding, 0),
            min(block.end + padding, training_block.nframe - 1),
        )
        print(start, end)

        print(start, ":", end, " too large loss! ")
        params["use_area"] = block
        # filenames_ = filenames[use_area[0]:use_area[1]+1]

        theta_ = theta[start : end + 1, :].copy()

        # read and preprocess images
        # real_image, y_st, x_st = read_image(imshape, filenames_, params['rescale'], Worm_is_black)
        real_image, y_st, x_st = load_image(
            filenames_all,
            params["rescale"],
            Worm_is_black,
            multi_flag,
            Tscaled_ind[start : end + 1],
        )
        T, H, W = real_image.shape

        # make flipping candidate
        _, theta_cand = make_theta_cand(theta_)

        # set init value
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[0], T))
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength
        init_data = [init_cx, init_cy, unitLength]

        # make model instance and training
        update = 0
        model = (
            Model(init_cx, init_cy, init_theta, init_unitLength, params)
            .to(torch.float32)
            .to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        params["id"] = 2
        losses = train3(
            model,
            real_image,
            optimizer,
            params,
            device,
            init_data,
            output_path,
            output_name,
        )

        # get trace information if loss is smaller
        if loss_compare([losses_all[i], losses]):
            print("update")
            update = 2
            theta_model = model.theta.detach().cpu().numpy()
            unitL_model = model.unitLength.detach().cpu().numpy().reshape(-1, 1)
            x_cent, y_cent = (
                model.cx.detach().cpu().numpy(),
                model.cy.detach().cpu().numpy(),
            )
            model_image = model(batch=T, width=W, height=H)
            losses_all[i] = losses
            remove_progress(output_path, "{}-{}_id[0-1]*.png".format(start, end))
        else:
            print("no update")
            remove_progress(output_path, "{}-{}_id2*.png".format(start, end))

        # flip final theta and trace again
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[1], T))

        # make model instance and training
        model = (
            Model(init_cx, init_cy, init_theta, init_unitLength, params)
            .to(torch.float32)
            .to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        params["id"] = 3
        losses = train3(
            model,
            real_image,
            optimizer,
            params,
            device,
            init_data,
            output_path,
            output_name,
        )

        # get trace information if loss is smaller
        if loss_compare([losses_all[i], losses]):
            print("update")
            update = 3
            theta_model = model.theta.detach().cpu().numpy()
            unitL_model = model.unitLength.detach().cpu().numpy().reshape(-1, 1)
            x_cent, y_cent = (
                model.cx.detach().cpu().numpy(),
                model.cy.detach().cpu().numpy(),
            )
            model_image = model(batch=T, width=W, height=H)
            losses_all[i] = losses
            remove_progress(output_path, "{}-{}_id[0-2]*.png".format(start, end))
        else:
            print("no update")
            remove_progress(output_path, "{}-{}_id3*.png".format(start, end))

        if update:
            # Add x_st, y_st to restore original position before reconstruction.
            x_cent += x_st
            y_cent += y_st
            x_model, y_model = make_plot(theta_model, unitL_model, x_cent, y_cent)
            if __debug__:
                show_image(real_image, params["num_t"], title="real image")
                show_image(model_image, params["num_t"], title="model image")
                show_loss_plot(losses_all[i], title="losses of new model")

            # reconstruct plots from model results
            l_pad = block.start - start
            r_pad = l_pad + block.size
            x_model = x_model[l_pad:r_pad]
            y_model = y_model[l_pad:r_pad]

            x[block.start : block.end + 1, :] = x_model
            y[block.start : block.end + 1, :] = y_model

            # log
            logger.info(
                f"""{str((start, end))} updated
image loss : {np.mean(losses_all[i][0])}
continuity loss : {np.mean(losses_all[i][1])}
smoothing loss : {np.mean(losses_all[i][2])}
length loss : {np.mean(losses_all[i][3])}
center loss : {np.mean(losses_all[i][4])}

"""
            )

    time_now = datetime.datetime.now()
    logger.info(f"STEP3 finished at {time_now}\n")

    # save params and plots
    params_for_save = params.copy()
    for key, value in params_for_save.items():
        if torch.is_tensor(value):
            params_for_save[key] = params_for_save[key].item()
        if isinstance(value, Path):
            params_for_save[key] = os.fspath(value)
    del params_for_save["use_area"]

    # check flipping
    x, y = flip_check(x, y)

    # check which side is head or tail
    if (
        "judge_head_method" not in params.keys()
        or params["judge_head_method"] == "amplitude"
    ):
        x, y, x_rev, y_rev = judge_head_amplitude(x, y)
    elif params["judge_head_method"] == "frequency":
        x, y, x_rev, y_rev = judge_head_frequency(x, y)

    # cancel reduction
    # T_read_all = params['end_T'] - params['start_T'] if params['end_T'] else len(filenames_all) - params['start_T']
    # x, y = cancel_reduction(x, y, T_read_all, len(filenames), params['plot_n'])
    # x, y = cancel_reduction(x, y, n_input_images, len(Tscaled_ind), params['plot_n'])
    x, y = cancel_reduction(
        x,
        y,
        n_input_images,
        params["start_T"],
        params["end_T"],
        Tscaled_ind,
        params["plot_n"],
    )

    # x_rev, y_rev = cancel_reduction(x_rev, y_rev, T_read_all, len(filenames), params['plot_n'])
    x_rev, y_rev = cancel_reduction(
        x_rev,
        y_rev,
        n_input_images,
        params["start_T"],
        params["end_T"],
        Tscaled_ind,
        params["plot_n"],
    )

    tz = datetime.timezone(datetime.timedelta(hours=params["local_time_difference"]))
    time_now = datetime.datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S.%f")
    # if not os.path.isdir(os.path.join(output_path, 'results')):
    #  os.mkdir(os.path.join(output_path, 'results'))
    with open(os.path.join(output_path, output_name + "_params.json"), "w") as f:
        json.dump(params_for_save, f)
    with open(os.path.join(output_path, output_name + "_params.yaml"), "w") as f:
        yaml.safe_dump(params_for_save, f, sort_keys=False)

    np.savetxt(
        os.path.join(output_path, output_name + "_x.csv"),
        x / params["rescale"],
        delimiter=",",
    )
    np.savetxt(
        os.path.join(output_path, output_name + "_y.csv"),
        y / params["rescale"],
        delimiter=",",
    )
    np.savetxt(
        os.path.join(output_path, output_name + "_x_rev.csv"),
        x_rev / params["rescale"],
        delimiter=",",
    )
    np.savetxt(
        os.path.join(output_path, output_name + "_y_rev.csv"),
        y_rev / params["rescale"],
        delimiter=",",
    )

    with h5py.File(
        os.path.join(output_path, output_name + "_skel.h5"),
        "w",
    ) as handler:
        handler.create_dataset("x", data=x)
        handler.create_dataset("y", data=y)
        handler.create_dataset("x_rev", data=x_rev)
        handler.create_dataset("y_rev", data=y_rev)

    save_centerline_to_roi(
        outputpath=os.path.join(output_path, output_name + "_RoiSet.zip"),
        x=x,
        y=y,
    )
    save_centerline_to_roi(
        outputpath=os.path.join(output_path, output_name + "_RoiSet_rev.zip"),
        x=x_rev,
        y=y_rev,
    )

    logger.info("Params and plots are successfully saved.\n")

    if not (
        params["SaveCenterlinedWormsSerial"]
        | params["SaveCenterlinedWormsMovie"]
        | params["SaveCenterlinedWormsMultitiff"]
    ):
        return

    # save full of real_image and centerline as png images
    # real_image, y_st, x_st = read_image(imshape, filenames_full, params['rescale'], Worm_is_black)
    real_image, org_y_st, org_x_st = load_image(
        filenames_all,
        params["rescale"],
        Worm_is_black,
        multi_flag,
        list(range(n_input_images)),
    )

    if params["SaveCenterlinedWormsSerial"]:
        clear_dir(output_path, output_name + "_png")
        # for t in range(len(filenames_full)):
        end_T = n_input_images - 1 if params["end_T"] == 0 else params["end_T"]
        fig, ax = plt.subplots()
        for i, t in enumerate(range(params["start_T"], end_T + 1)):
            filename = os.path.join(
                output_path,
                output_name + "_png",
                "image" + str(t).zfill(len(str(n_input_images))) + ".png",
            )
            ax.imshow(real_image[t], cmap="gray")
            ax.plot(x[i] - org_x_st, y[i] - org_y_st, c="r", lw=3)
            plt.savefig(filename)
            plt.cla()
        plt.close()
        print("\npng images saved to " + filename + " etc.")

    # save full of real_image and centerline as mp4 movie
    if params["SaveCenterlinedWormsMovie"]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ims = []
        # for t in range(n_input_images):
        end_T = n_input_images - 1 if params["end_T"] == 0 else params["end_T"]
        for i, t in enumerate(range(params["start_T"], end_T + 1)):
            if i % 100 == 0:
                print(t, end=" ")
            lines = []
            lines.extend(ax.plot(x[i] - org_x_st, y[i] - org_y_st, c="r", lw=3))
            lines.extend([ax.imshow(real_image[t], cmap="gray")])
            title = ax.text(
                0.5,
                1.01,
                "index: " + str(t),
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize="large",
                color="black",
            )
            ims.append(lines + [title])
        ani = animation.ArtistAnimation(fig, ims, interval=50)
        rc("animation", html="jshtml")
        plt.close()
        ################# ani
        filename = os.path.join(output_path, output_name + ".mp4")
        ani.save(filename)
        print("\nMovie saved to " + filename)

    # save full of real_image and centerline as multipage tiff
    if params["SaveCenterlinedWormsMultitiff"]:
        filename = os.path.join(output_path, output_name + ".tif")

        end_T = n_input_images - 1 if params["end_T"] == 0 else params["end_T"]
        T, Y, X = real_image.shape
        stack = tifffile.memmap(
            filename,
            shape=(T, 3, Y, X),
            dtype="u1",
            imagej=True,
            metadata={
                "axes": "TCYX",
                "labels": [
                    f"index: {i:d}" for i in range(params["start_T"], end_T + 1)
                ],
            },
        )
        pts = np.stack((x - org_x_st, y - org_y_st), axis=-1)

        # OpenCV only accept np.int32
        pts = np.clip(pts, 0, None).astype("i4")

        for i, (pt, im) in enumerate(zip(pts, real_image)):
            if i % 100 == 0:
                print(i + 1, end=" ")
            im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            # pt is an [N, 2] array, OpenCV only use (1, N, 2) for plotting.
            im_lines = cv2.polylines(
                im_rgb,
                [pt],
                isClosed=False,
                color=(0, 0, 255),
                thickness=3,
            )  # (Y, X, C)
            # (Y, X, C) => (C, Y, X)
            stack[i] = np.transpose(im_lines, (2, 0, 1)).astype("u1")
            stack.flush()
        logger.info("Multipage Tiff saved to " + filename)
