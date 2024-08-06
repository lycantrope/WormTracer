### WormTracer main package wt.py ###

import datetime
import glob
import io
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import animation, rc
from PIL import Image

from .functions import *

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


def run(
    parameter_file, dataset_path, output_directory="", **kwargs
):  # execute the whole WormTracer process, kwargs are optional parameter=value pairs
    with open(parameter_file, "r") as yml:
        params = yaml.safe_load(yml)
    if len(kwargs) > 0:
        params.update(kwargs)

    # log
    time_now = datetime.datetime.now()
    logs = [f"Code executed at {time_now}\n"]
    logs.append(f"Params : {params}\n")

    #### make use of GPU ####
    if torch.cuda.is_available():
        device = "cuda"
        print("Running using GPU.")
        logs.append("Running using GPU.\n\n")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Running using GPU.")
        logs.append("Running using GPU.\n\n")
    else:
        device = "cpu"
        print("Running using CPU. GPU is recommended")
        logs.append("Running using CPU. GPU is recommended\n\n")

    # set output_path
    if "output_directory" not in locals() and "output_directory" not in globals():
        output_directory = ""
    dataset_name, output_path, output_name = set_output_path(
        dataset_path, output_directory
    )  # output_path is created in output_directory
    print("dataset_path =", dataset_path)
    print("output_path =", output_path)

    # basic informatin to save
    params["dataset_path"] = dataset_path
    params["output_path"] = output_path

    # read data property(image size, frame number)
    filenames_all = get_filenames(dataset_path)

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

    # read images and get information
    # getting xy plots by thinning in function ; read_image_and_xy()
    real_image, x, y, y_st, x_st, unitLength, pre_width = read_image_and_xy(
        imshape,
        filenames_all,
        params["rescale"],
        params["plot_n"],
        Worm_is_black,
        multi_flag,
        Tscaled_ind,
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
    logs.append(f"Reading images finished at {time_now}\n")
    logs.append(
        f"frame = {len(Tscaled_ind)} shape = {real_image.shape} unitLength = {unitLength}\n\n"
    )

    # make worm model image from plots
    params["alpha"] = pre_width.min()
    image_info = {"image_shape": real_image.shape, "device": device}
    cap_span = calc_cap_span(image_info, params["plot_n"], s_m=8000)
    model_image = make_image(x, y, x_st, y_st, params, image_info, cap_span)

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
    logs.append(f"Determining time blocks finished at {time_now}\n")
    logs.append(f"use_points : {use_points}\n\n")

    losses_all = []
    shape_params = []
    unitLength = prepare_for_train(pre_width, simple_area, x, y, params)
    if params["SaveProgress"]:
        clear_dir(output_path, output_name + "_progress_image")
    logs.append("STEP1 : optimization for simple posture blocks\n\n")

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
            imshape,
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
        theta_cand, _ = make_thetaCand(theta_)
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
        logs.append(str(use_area) + "\n")
        logs.append(f"image loss : {np.mean(losses[0])}\n")
        logs.append(f"continuity loss : {np.mean(losses[1])}\n")
        logs.append(f"smoothing loss : {np.mean(losses[2])}\n")
        logs.append(f"length loss : {np.mean(losses[3])}\n")
        logs.append(f"center loss : {np.mean(losses[4])}\n\n")
    time_now = datetime.datetime.now()
    logs.append(f"STEP1 finished at {time_now}\n\n")

    params["init_alpha"], params["init_gamma"], params["init_delta"] = get_shape_params(
        shape_params, params
    )
    logs.append("STEP2 : optimization for complex posture blocks\n\n")

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
            imshape,
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
        theta_cand, _ = make_thetaCand(theta_)

        # set init value
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[0], T))
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength
        init_data = [init_cx, init_cy, unitLength]

        # make model instance and training
        model = Model(
            init_cx, init_cy, init_theta, init_unitLength, image_info, params
        ).to(device)
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
        model = Model(
            init_cx, init_cy, init_theta, init_unitLength, image_info, params
        ).to(device)
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
        select_ind = loss_compair([losses_all[i], losses])
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
        logs.append(str(use_area) + "\n")
        logs.append(f"image loss : {np.mean(losses[0])}\n")
        logs.append(f"continuity loss : {np.mean(losses[1])}\n")
        logs.append(f"smoothing loss : {np.mean(losses[2])}\n")
        logs.append(f"length loss : {np.mean(losses[3])}\n")
        logs.append(f"center loss : {np.mean(losses[4])}\n\n")

    time_now = datetime.datetime.now()
    logs.append(f"STEP2 finished at {time_now}\n\n")

    # revise areas which have too large loss
    losslarge_area = find_losslarge_area(losses_all)
    logs.append(
        "STEP3 :　re-optimization for unsuccessful blocks with complex postures\n\n"
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
                imshape,
                filenames_all,
                params["rescale"],
                Worm_is_black,
                multi_flag,
                Tscaled_ind[use_area[0] : use_area[1] + 1],
            )
            show_image(real_image, params["num_t"], title="real image")
            image_info["image_shape"] = real_image.shape

            # make flipping candidate
            _, theta_cand = make_thetaCand(theta_)

            # set init value
            init_cx, init_cy = set_init_xy(real_image)
            init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[0], T))
            init_unitLength = torch.ones(T, dtype=torch.float) * unitLength
            init_data = [init_cx, init_cy, unitLength]

            # make model instance and training
            update = 0
            model = Model(
                init_cx, init_cy, init_theta, init_unitLength, image_info, params
            ).to(device)
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
            if loss_compair([losses_all[i], losses]):
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
            model = Model(
                init_cx, init_cy, init_theta, init_unitLength, image_info, params
            ).to(device)
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
            if loss_compair([losses_all[i], losses]):
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
                logs.append(str(use_area) + " updated\n")
                logs.append(f"image loss : {np.mean(losses_all[i][0])}\n")
                logs.append(f"continuity loss : {np.mean(losses_all[i][1])}\n")
                logs.append(f"smoothing loss : {np.mean(losses_all[i][2])}\n")
                logs.append(f"length loss : {np.mean(losses_all[i][3])}\n")
                logs.append(f"center loss : {np.mean(losses_all[i][4])}\n\n")

    time_now = datetime.datetime.now()
    logs.append(f"STEP3 finished at {time_now}\n\n")

    # save params and plots
    params_for_save = params.copy()
    for key, value in params_for_save.items():
        if torch.is_tensor(value):
            params_for_save[key] = params_for_save[key].item()
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
    logs.append("Params and plots are successfully saved.\n\n")

    # save log
    # if not os.path.isdir(os.path.join(output_path, 'logs')):
    #  os.mkdir(os.path.join(output_path, 'logs'))
    with open(os.path.join(output_path, f"{output_name}_log.txt"), mode="w") as f:
        for log in logs:
            f.write(log)

    # save full of real_image and centerline as png images
    # real_image, y_st, x_st = read_image(imshape, filenames_full, params['rescale'], Worm_is_black)
    real_image, y_st, x_st = read_image(
        imshape,
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
        stack = []
        fig, ax = plt.subplots(figsize=(3, 3))
        end_T = n_input_images - 1 if params["end_T"] == 0 else params["end_T"]
        for i, t in enumerate(range(params["start_T"], end_T + 1)):
            if i % 100 == 0:
                print(t, end=" ")
            ax.imshow(real_image[t], cmap="gray")
            ax.plot(x[i] - x_st, y[i] - y_st, c="r", lw=3)
            plt.title("index: " + str(t))
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.cla()
            buf.seek(0)
            img2 = Image.open(buf).convert("RGB")
            stack.append(img2)
        stack[0].save(
            filename, compression="tiff_deflate", save_all=True, append_images=stack[1:]
        )
        plt.close(fig)
        print("\nMultipage tiff saved to " + filename)
