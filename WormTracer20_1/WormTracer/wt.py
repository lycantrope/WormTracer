### WormTracer main package wt.py ###

import datetime
import json
import logging
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import yaml
from matplotlib import animation, rc

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
    get_shape_params,
    get_use_points,
    judge_head_amplitude,
    judge_head_frequency,
    loss_compare,
    make_image,
    make_plot,
    make_theta_cand,
    make_theta_from_xy,
    prepare_for_train,
    read_image,
    remove_progress,
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


def run(
    parameter_file, dataset_path, output_directory="", **kwargs
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

    print("dataset_path =", dataset_path)
    print("output_path =", output_path)

    # setup logger
    logging.basicConfig(
        filename=output_path.joinpath(f"{output_name}.log"),
        format="%(message)s",
        level=logging.INFO,
        encoding="utf-8",
    )

    # log
    time_now = datetime.datetime.now()
    logger.info(f"Code executed at {time_now}\nParams : {params}\n")

    #### make use of GPU ####
    if torch.cuda.is_available():
        device = "cuda"
        print("Running using GPU.")
        logger.info("Running using GPU.\n")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Running using GPU.")
        logger.info("Running using GPU.\n")
    else:
        device = "cpu"
        print("Running using CPU. GPU is recommended")
        logger.info("Running using CPU. GPU is recommended\n")

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

    # read images and get information: read_image
    real_image, y_st, x_st = read_image(
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

    print(
        "\rframe = ",
        len(Tscaled_ind),
        " shape = ",
        real_image.shape,
        " unitLength = ",
        unitLength,
    )

    # log
    time_now = datetime.datetime.now()
    logger.info(f"Reading images finished at {time_now}")
    logger.info(
        f"frame = {len(Tscaled_ind)} shape = {real_image.shape} unitLength = {unitLength}\n"
    )

    # make worm model image from plots
    params["alpha"] = pre_width.min()
    params["gamma"] = 0.0
    params["delta"] = 0.0
    image_info = {"image_shape": real_image.shape, "device": device}
    cap_span = calc_cap_span(image_info, params["plot_n"], s_m=8000)
    model_image = make_image(x, y, x_st, y_st, params, image_info)

    # get points for trace blocks
    image_losses = np.mean((model_image - real_image) ** 2, axis=(1, 2))
    image_loss_max = get_image_loss_max(
        image_losses, real_image, x, y, x_st, y_st, params, image_info, cap_span
    )
    use_points, nont_flag, simple_area = get_use_points(
        image_losses, image_loss_max, cap_span, x, y, params["plot_n"], show_plot=True
    )

    show_image(real_image, params["num_t"], title="real image")
    show_image(model_image, params["num_t"], title="model image")
    print("use_points \n", use_points)

    # log 3
    time_now = datetime.datetime.now()
    logger.info(f"Determining time blocks finished at {time_now}")
    logger.info(f"use_points : {use_points}\n")

    losses_all = []
    shape_params = []
    unitLength = prepare_for_train(pre_width, simple_area, x, y, params)
    if params["SaveProgress"]:
        clear_dir(output_path, output_name + "_progress_image")
    logger.info("STEP1 : optimization for simple posture blocks\n")

    # main loop 1
    for i in range(len(use_points) - 1):
        if nont_flag[i]:
            losses_all.append(0)
            continue
        use_area = (use_points[i], use_points[i + 1])
        print(use_area)
        params["use_area"] = use_area
        # filenames_ = filenames[use_area[0]:use_area[1]+1]
        T = use_area[1] - use_area[0] + 1
        theta_ = theta[use_area[0] : use_area[1] + 1, :].copy()

        # read and preprocess images
        real_image, y_st, x_st = read_image(
            filenames_all,
            params["rescale"],
            Worm_is_black,
            multi_flag,
            Tscaled_ind[use_area[0] : use_area[1] + 1],
        )
        show_image(real_image, params["num_t"], title="real image")

        save_progress(real_image, output_path, output_name, params, txt="real")
        image_info["image_shape"] = real_image.shape

        # set init value
        theta_cand, _ = make_theta_cand(theta_)
        theta_[-1, :] = theta_cand[0]
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.tensor(theta_)
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength
        init_data = [init_cx, init_cy, unitLength]

        print(real_image.shape)
        print(init_cx.shape)
        print(init_theta.shape)
        print(init_unitLength.shape)
        print(image_info["image_shape"])

        # make model instance and training
        model = (
            Model(init_cx, init_cy, init_theta, init_unitLength, image_info, params)
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
        losses_all.append(losses)
        theta_model = model.theta.detach().cpu().numpy()
        unitL_model = model.unitLength.detach().cpu().numpy().reshape(-1, 1)
        x_cent, y_cent = (
            model.cx.detach().cpu().numpy(),
            model.cy.detach().cpu().numpy(),
        )
        shape_params.append(
            (
                T,
                model.alpha.detach().cpu(),
                model.gamma.detach().cpu(),
                model.delta.detach().cpu(),
            )
        )
        model_image = model()
        show_image(model_image, params["num_t"], title="model image")
        show_loss_plot(losses_all[-1], title="losses of model")

        # reconstruct plots from model results
        x_model, y_model = make_plot(theta_model, unitL_model, x_cent, y_cent)
        x[use_area[0] : use_area[1] + 1, :] = x_model + x_st
        y[use_area[0] : use_area[1] + 1, :] = y_model + y_st

        # log
        logger.info(f"""{str(use_area)}
image loss : {np.mean(losses[0])}
continuity loss : {np.mean(losses[1])}
smoothing loss : {np.mean(losses[2])}
length loss : {np.mean(losses[3])}
center loss : {np.mean(losses[4])}
""")
    time_now = datetime.datetime.now()
    logger.info(f"STEP1 finished at {time_now}\n")

    params["init_alpha"], params["init_gamma"], params["init_delta"] = get_shape_params(
        shape_params, params
    )
    logger.info("STEP2 : optimization for complex posture blocks\n")

    # main loop 2
    for i in range(len(use_points) - 1):
        if not nont_flag[i]:
            continue
        use_area = (use_points[i], use_points[i + 1])
        print(use_area)
        params["use_area"] = use_area
        # filenames_ = filenames[use_area[0]:use_area[1]+1]
        T = use_area[1] - use_area[0] + 1
        theta_ = theta[use_area[0] : use_area[1] + 1, :].copy()

        # read and preprocess images
        # real_image, y_st, x_st = read_image(imshape, filenames_, params['rescale'], Worm_is_black)
        real_image, y_st, x_st = read_image(
            filenames_all,
            params["rescale"],
            Worm_is_black,
            multi_flag,
            Tscaled_ind[use_area[0] : use_area[1] + 1],
        )
        show_image(real_image, params["num_t"], title="real image")
        save_progress(real_image, output_path, output_name, params, txt="real")
        image_info["image_shape"] = real_image.shape

        # make flipping theta candidate
        theta_cand, _ = make_theta_cand(theta_)

        # set init value
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[0], T))
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength
        init_data = [init_cx, init_cy, unitLength]

        # make model instance and training
        model = (
            Model(init_cx, init_cy, init_theta, init_unitLength, image_info, params)
            .to(torch.float32)
            .to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        params["id"] = 0
        losses_all[i] = train3(
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
        model_image = model()

        # flip final theta to trace again
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[1], T))

        # make model instance and training
        model = (
            Model(init_cx, init_cy, init_theta, init_unitLength, image_info, params)
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
        select_ind = loss_compare([losses_all[i], losses])
        if select_ind:
            theta_model = model.theta.detach().cpu().numpy()
            unitL_model = model.unitLength.detach().cpu().numpy().reshape(-1, 1)
            x_cent, y_cent = (
                model.cx.detach().cpu().numpy(),
                model.cy.detach().cpu().numpy(),
            )
            model_image = model()
            losses_all[i] = losses
        remove_progress(
            output_path,
            "{}-{}_id{}*.png".format(use_area[0], use_area[1], 1 - select_ind),
        )

        # reconstruct plots from model results
        x_model, y_model = make_plot(theta_model, unitL_model, x_cent, y_cent)
        show_image(model_image, params["num_t"], title="model image")
        show_loss_plot(losses_all[i], title="losses of model{}".format(select_ind))
        x[use_area[0] : use_area[1] + 1, :] = x_model + x_st
        y[use_area[0] : use_area[1] + 1, :] = y_model + y_st

        # log
        logger.info(f"""{str(use_area)}
image loss : {np.mean(losses[0])}
continuity loss : {np.mean(losses[1])}
smoothing loss : {np.mean(losses[2])}
length loss : {np.mean(losses[3])}
center loss : {np.mean(losses[4])}

""")

    time_now = datetime.datetime.now()
    logger.info(f"STEP2 finished at {time_now}\n")

    # revise areas which have too large loss
    losslarge_area = find_losslarge_area(losses_all)
    logger.info(
        "STEP3 :　re-optimization for unsuccessful blocks with complex postures\n"
    )

    for i in range(len(use_points) - 1):
        if losslarge_area[i] and nont_flag[i]:
            use_area = (use_points[i], use_points[i + 1])
            print(use_area[0], ":", use_area[1], " too large loss! ")
            params["use_area"] = use_area
            # filenames_ = filenames[use_area[0]:use_area[1]+1]
            T = use_area[1] - use_area[0] + 1
            theta_ = theta[use_area[0] : use_area[1] + 1, :].copy()

            # read and preprocess images
            # real_image, y_st, x_st = read_image(imshape, filenames_, params['rescale'], Worm_is_black)
            real_image, y_st, x_st = read_image(
                filenames_all,
                params["rescale"],
                Worm_is_black,
                multi_flag,
                Tscaled_ind[use_area[0] : use_area[1] + 1],
            )
            show_image(real_image, params["num_t"], title="real image")
            image_info["image_shape"] = real_image.shape

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
                Model(init_cx, init_cy, init_theta, init_unitLength, image_info, params)
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
                model_image = model()
                losses_all[i] = losses
                remove_progress(
                    output_path, "{}-{}_id[0-1]*.png".format(use_area[0], use_area[1])
                )
            else:
                print("no update")
                remove_progress(
                    output_path, "{}-{}_id2*.png".format(use_area[0], use_area[1])
                )

            # flip final theta and trace again
            init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[1], T))

            # make model instance and training
            model = (
                Model(init_cx, init_cy, init_theta, init_unitLength, image_info, params)
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
                model_image = model()
                losses_all[i] = losses
                remove_progress(
                    output_path, "{}-{}_id[0-2]*.png".format(use_area[0], use_area[1])
                )
            else:
                print("no update")
                remove_progress(
                    output_path, "{}-{}_id3*.png".format(use_area[0], use_area[1])
                )

            if update:
                x_model, y_model = make_plot(theta_model, unitL_model, x_cent, y_cent)
                show_image(model_image, params["num_t"], title="model image")
                show_loss_plot(losses_all[i], title="losses of new model")

                # reconstruct plots from model results
                x[use_area[0] : use_area[1] + 1, :] = x_model + x_st
                y[use_area[0] : use_area[1] + 1, :] = y_model + y_st

                # log
                logger.info(f"""{str(use_area)} updated
image loss : {np.mean(losses_all[i][0])}
continuity loss : {np.mean(losses_all[i][1])}
smoothing loss : {np.mean(losses_all[i][2])}
length loss : {np.mean(losses_all[i][3])}
center loss : {np.mean(losses_all[i][4])}

""")

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
    logger.info("Params and plots are successfully saved.\n")

    if not (
        params["SaveCenterlinedWormsSerial"]
        | params["SaveCenterlinedWormsMovie"]
        | params["SaveCenterlinedWormsMultitiff"]
    ):
        return

    # save full of real_image and centerline as png images
    # real_image, y_st, x_st = read_image(imshape, filenames_full, params['rescale'], Worm_is_black)
    real_image, y_st, x_st = read_image(
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
            ax.plot(x[i] - x_st, y[i] - y_st, c="r", lw=3)
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
            lines.extend(ax.plot(x[i] - x_st, y[i] - y_st, c="r", lw=3))
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
        T, Y, X = read_image.shape

        def image_gen():
            pts = np.stack((x - x_st, y - y_st), axis=-1)
            for i, (pt, im) in enumerate(zip(pts, read_image)):
                if i % 100 == 0:
                    print(i + 1, end=" ")
                im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                im_lines = cv2.polylines(
                    im_rgb,
                    pt,
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=3,
                )
                im_lines_5d = im_lines[None, None, :, :, :]
                yield np.transpose(im_lines_5d, (0, 1, 4, 2, 3))

        tifffile.imwrite(
            filename,
            data=image_gen(),
            imagej=True,
            shape=(T, 1, 3, Y, X),
            metadata={
                "axes": "TZCYX",
                "labels": [
                    "index: %d" % i for i in range(params["start_T"], end_T + 1)
                ],
            },
        )
        print("\nMultipage tiff saved to " + filename)
