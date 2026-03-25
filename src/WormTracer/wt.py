from __future__ import annotations

import datetime
import functools
import inspect
import logging
import os
import sys
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from loguru import logger
from matplotlib import animation, rc
from ruamel.yaml import YAML

from WormTracer import __version__
from WormTracer.functions import (
    Model,
    calc_cap_span,
    calc_xy_and_prewidth,
    cancel_reduction,
    clear_dir,
    find_losslarge_area,
    flip_check,
    get_filenames,
    get_guide_points,
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
    save_params_into_commented_yaml,
    save_progress,
    set_init_xy,
    set_output_path,
    show_image,
    show_loss_plot,
    train3,
)

if TYPE_CHECKING:
    from typing import Optional


class InterceptHandler(logging.Handler):
    """
    Default handler from logging to loguru.
    See: https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


# 2. Setup your specific logger
def ensure_clearup(fn):
    @functools.wraps(fn)
    def wrapper(*arg, **kwargs):
        tic = datetime.datetime.now()
        backend = matplotlib.get_backend()
        try:
            # Run function
            ret = fn(*arg, **kwargs)
            toc = datetime.datetime.now()
            elapsed_time = toc - tic
            logger.info(f"Elapse time: {elapsed_time.total_seconds():.1f} (sec)")
            return ret
        finally:
            logger.remove()
            logger.add(sys.stderr, level="INFO")
            try:
                matplotlib.use(backend)
            except Exception:
                pass

    return wrapper


@ensure_clearup
def run(
    parameter_file: os.PathLike,
    dataset_path: os.PathLike,
    output_directory: Optional[os.PathLike] = None,
    guide_files: Optional[os.PathLike] = None,
    **kwargs,
):
    # execute the whole WormTracer process, kwargs are optional parameter=value pairs
    matplotlib.use("Agg")
    yaml = YAML()
    with open(parameter_file, "r") as yml:
        params = yaml.load(yml)

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

    logger.add(
        output_path.joinpath(f"{output_name}.log"),
        format="{message}",
        encoding="utf8",
        mode="w",
    )

    # log
    logger.info(f"Code executed at {datetime.datetime.now()}")
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

    guide_idx = None
    if isinstance(guide_files, list):
        logger.info(f"Found guide_files: {guide_files}")
        guide_x, guide_y, guide_idx = get_guide_points(
            guide_files,
            Tscaled_ind,
            params["plot_n"],
            n_input_images,
        )

        # We assumed that non-nan in guide_x is served as guide, we overwrite the x, y by guide_x, guide_y
        # Since the guide points were made after restoring the resacle, we need to multiple the rescale to guide_x and guide_y
        x[guide_idx] = guide_x[guide_idx] * params["rescale"]
        y[guide_idx] = guide_y[guide_idx] * params["rescale"]

        # After assignment, we can drop the guide_x and guide_y, here.
        del guide_x, guide_y

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
    model_image = make_image(
        x,
        y,
        x_st,
        y_st,
        width=real_image.shape[2],
        height=real_image.shape[1],
        params=params,
    )

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
    )

    if __debug__:
        show_image(real_image, params["num_t"], title="real image")
        show_image(model_image, params["num_t"], title="model image")

    # Since dataset will be loaded during each training block, the entire dataset can be drop here.
    del real_image
    del model_image

    if guide_idx is not None:
        # we assigned loss in the guide_idx to ensure the frame of guide_idx will be simple (non zero ground truth)
        image_losses[guide_idx] = image_losses[best_frame_idx]
        del guide_idx

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

    unitLength = float(
        np.sqrt(np.mean(np.sum(np.diff((x[simple_area], y[simple_area])) ** 2, axis=0)))
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
        theta_cand, _ = make_theta_cand(theta_[0], theta_[-1])
        theta_[-1, :] = theta_cand[0]
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.tensor(theta_)
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength

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
            output_path,
            output_name,
            is_nont=False,
        )

        # get trace information
        losses_all[(1, block.idx)] = losses
        with torch.no_grad():
            x_model, y_model, model_image = model(batch=T, width=W, height=H)

        theta_model = model.theta.detach().cpu().numpy()

        shape_params.append(
            (
                block.size,
                model.alpha.detach().cpu(),
                model.gamma.detach().cpu(),
                model.delta.detach().cpu(),
            )
        )
        # Add x_st, y_st to restore original position before reconstruction.
        x_model = x_model.detach().cpu().numpy()
        y_model = y_model.detach().cpu().numpy()
        # Trim padding
        # Add x_st, y_st to restore original position before reconstruction.
        x_model += x_st
        y_model += y_st

        x[block.start : block.end + 1, :] = x_model
        y[block.start : block.end + 1, :] = y_model
        theta[block.start : block.end + 1, :] = theta_model

        # log
        logger.info(
            f"""image loss : {np.mean(losses[0])}
continuity loss : {np.mean(losses[1])}
smoothing loss : {np.mean(losses[2])}
length loss : {np.mean(losses[3])}
center loss : {np.mean(losses[4])}
"""
        )

        if __debug__:
            # Only compute the model_image, if we want to show the result.
            show_image(real_image, params["num_t"], title="real image")
            show_image(model_image, params["num_t"], title="model image")
            show_loss_plot(losses_all[(1, block.idx)], title="losses of model")

    time_now = datetime.datetime.now()
    logger.info(f"STEP1 finished at {time_now}\n")

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

    # main loop 2
    for i, block in enumerate(all_blocks):
        if not block.is_complex:
            continue

        logger.info(str(block))
        # padding the complex block of 1/10 length, minimal to 3
        padding = max(block.size // 10, 3)

        l_pad = 0
        if i > 0:
            l_pad = min(padding, all_blocks[i - 1].size)

        r_pad = 0
        if i + 1 < len(all_blocks):
            r_pad = min(padding, all_blocks[i + 1].size)

        # Inclusive both end [Start-l_pad, end+r_pad]
        start = block.start - l_pad
        end = block.end + r_pad

        # This is only for saving the output during training
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
        logger.info(f"im_shape: {real_image.shape}")
        # make flipping theta candidate
        theta_cand, _ = make_theta_cand(theta_[0], theta_[-1])

        # set init value
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[0], T))
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength

        # gradient_mask for simple area
        mask = training_block.complex_area[start : end + 1].astype("f4")
        gradient_mask = torch.from_numpy(mask).to(device)

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
            output_path,
            output_name,
            gradient_mask=gradient_mask,
        )

        # Trim the padding losses
        losses_all[(2, block.idx)] = losses[:, l_pad : l_pad + block.size]
        # get trace information
        with torch.no_grad():
            x_model, y_model, model_image = model(batch=T, width=W, height=H)

        x_model = x_model.detach().cpu().numpy()
        y_model = y_model.detach().cpu().numpy()
        theta_model = model.theta.detach().cpu().numpy()

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
            output_path,
            output_name,
            gradient_mask=gradient_mask,
        )
        # Trim the padding losses
        losses = losses[:, l_pad : l_pad + block.size]
        # get trace information if loss is smaller
        select_ind = loss_compare([losses_all[(2, block.idx)], losses])
        if select_ind:
            with torch.no_grad():
                x_model, y_model, model_image = model(batch=T, width=W, height=H)

            x_model = x_model.detach().cpu().numpy()
            y_model = y_model.detach().cpu().numpy()
            theta_model = model.theta.detach().cpu().numpy()

            losses_all[(2, block.idx)] = losses

        # Trim padding
        x_model = x_model[l_pad : l_pad + block.size]
        y_model = y_model[l_pad : l_pad + block.size]
        theta_model = theta_model[l_pad : l_pad + block.size, :]
        # Add x_st, y_st to restore original position before reconstruction.
        x_model += x_st
        y_model += y_st

        x[block.start : block.end + 1, :] = x_model
        y[block.start : block.end + 1, :] = y_model
        theta[block.start : block.end + 1, :] = theta_model

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
                losses_all[(2, block.idx)], title="losses of model{}".format(select_ind)
            )

        # log
        logger.info(
            f"""image loss : {np.mean(losses[0])}
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

    for i in losslarge_area:
        block = all_blocks[i]
        if not block.is_complex:
            continue

        # padding the complex block of 1/10 length, minimal to 3
        padding = max(block.size // 10, 3)

        l_pad = 0
        if i > 0:
            l_pad = min(padding, all_blocks[i - 1].size)

        r_pad = 0
        if i + 1 < len(all_blocks):
            r_pad = min(padding, all_blocks[i + 1].size)

        # Inclusive both end [Start-l_pad, end+r_pad]
        start = block.start - l_pad
        end = block.end + r_pad

        # This is only for saving the output during training
        params["use_area"] = block
        logger.info(f"{str(block)}: too large loss!")

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
        _, theta_cand = make_theta_cand(theta_[0], theta_[-1])

        # set init value
        init_cx, init_cy = set_init_xy(real_image)
        init_theta = torch.from_numpy(np.linspace(theta_[0, :], theta_cand[0], T))
        init_unitLength = torch.ones(T, dtype=torch.float) * unitLength

        # The gradient mask will be all zeros except loss large area.
        mask = np.zeros(T, dtype="f4")
        mask[l_pad : l_pad + block.size] = 1.0
        gradient_mask = torch.from_numpy(mask).to(device)

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
            output_path,
            output_name,
            gradient_mask=gradient_mask,
        )

        # Trim the padding losses
        losses = losses[:, l_pad : l_pad + block.size]

        with torch.no_grad():
            x_model, y_model, model_image = model(batch=T, width=W, height=H)

        x_model = x_model.detach().cpu().numpy()
        y_model = y_model.detach().cpu().numpy()
        theta_model = model.theta.detach().cpu().numpy()
        # get trace information if loss is smaller
        # Here, the losses was compared with step2 loss
        if loss_compare([losses_all[(2, i)], losses]):
            print("update")
            update = 2
            # We stored the losses in (step3, i)
            losses_all[(3, i)] = losses
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
            output_path,
            output_name,
            gradient_mask=gradient_mask,
        )
        # Trim the padding losses
        losses = losses[:, l_pad : l_pad + block.size]

        # get trace information if loss is smaller
        if loss_compare([losses_all[(3, i)], losses]):
            print("update")
            update = 3
            with torch.no_grad():
                x_model, y_model, model_image = model(batch=T, width=W, height=H)
            x_model = x_model.detach().cpu().numpy()
            y_model = y_model.detach().cpu().numpy()
            theta_model = model.theta.detach().cpu().numpy()
            losses_all[(3, i)] = losses
            remove_progress(output_path, "{}-{}_id[0-2]*.png".format(start, end))
        else:
            print("no update")
            remove_progress(output_path, "{}-{}_id3*.png".format(start, end))

        if update:
            # Trim padding
            x_model = x_model[l_pad : l_pad + block.size]
            y_model = y_model[l_pad : l_pad + block.size]
            theta_model = theta_model[l_pad : l_pad + block.size, :]
            # Add x_st, y_st to restore original position before reconstruction.
            x_model += x_st
            y_model += y_st

            if __debug__:
                show_image(real_image, params["num_t"], title="real image")
                show_image(model_image, params["num_t"], title="model image")
                show_loss_plot(losses_all[(3, i)], title="losses of new model")

            x[block.start : block.end + 1, :] = x_model
            y[block.start : block.end + 1, :] = y_model
            theta[block.start : block.end + 1, :] = theta_model

            # log
            logger.info(
                f"""{str(block)} updated
image loss : {np.mean(losses_all[(3, i)][0])}
continuity loss : {np.mean(losses_all[(3, i)][1])}
smoothing loss : {np.mean(losses_all[(3, i)][2])}
length loss : {np.mean(losses_all[(3, i)][3])}
center loss : {np.mean(losses_all[(3, i)][4])}

"""
            )

    time_now = datetime.datetime.now()
    logger.info(f"STEP3 finished at {time_now}\n")
    # save params and plots
    # Save all losses into csv files
    dtype = np.dtype(
        [
            ("step", "i4"),
            ("block", "i4"),
            ("index", "i4"),
            ("is_complex", "?"),
            ("image_loss", "f8"),
            ("continuity_loss", "f8"),
            ("smoothing_loss", "f8"),
            ("length_loss", "f8"),
            ("center_loss", "f8"),
        ]
    )

    losses_all_tmp = []
    for (stage, idx), loss in losses_all.items():
        T = loss.shape[1]
        arr = np.zeros(T, dtype=dtype)
        block = all_blocks[idx]

        arr["step"] = stage
        arr["block"] = block.idx
        arr["index"] = np.arange(block.start, block.end + 1)
        arr["is_complex"] = block.is_complex
        arr["image_loss"] = loss[0]
        arr["continuity_loss"] = loss[1]
        arr["smoothing_loss"] = loss[2]
        arr["length_loss"] = loss[3]
        arr["center_loss"] = loss[4]
        losses_all_tmp.append(arr)

    header = "step,block,index,is_complex,image_loss,continuity_lsos,smoothing_loss,length_loss,center_loss"
    fmt = ["%i", "%i", "%i", "%i", "%f", "%f", "%f", "%f", "%f"]
    losses_arr = np.concatenate(losses_all_tmp)
    np.savetxt(
        os.path.join(output_path, output_name + "_losses.csv"),
        losses_arr,
        fmt=fmt,
        header=header,
        delimiter=",",
    )

    # cancel reduction
    # T_read_all = params['end_T'] - params['start_T'] if params['end_T'] else len(filenames_all) - params['start_T']
    x, y = flip_check(x, y)
    x, y = cancel_reduction(
        x,
        y,
        n_input_images,
        params["start_T"],
        params["end_T"],
        Tscaled_ind,
        params["plot_n"],
    )

    # check which side is head or tail
    judge_head_method = params.get("judge_head_method", "amplitude")
    if judge_head_method == "frequency":
        is_reversed = judge_head_frequency(x, y)
    else:
        if judge_head_method != "amplitude":
            logger.warning(
                "judge_head_method only supported: frequency or amplitute (default)"
            )
        is_reversed = judge_head_amplitude(x, y)

    if is_reversed:
        x, y = x[:, ::-1], y[:, ::-1]

    tz = datetime.timezone(datetime.timedelta(hours=params["local_time_difference"]))
    time_now = datetime.datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S.%f")
    # if not os.path.isdir(os.path.join(output_path, 'results')):
    #  os.mkdir(os.path.join(output_path, 'results')

    params_for_save = params.copy()
    del params_for_save["use_area"]

    save_params_into_commented_yaml(
        Path(output_path).joinpath(output_name + "_params.yaml"),
        params_for_save,
    )

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

    with h5py.File(
        os.path.join(output_path, output_name + "_skel.h5"),
        "w",
    ) as handler:
        handler.create_dataset("x", data=x / params["rescale"])
        handler.create_dataset("y", data=y / params["rescale"])

    save_centerline_to_roi(
        outputpath=os.path.join(output_path, output_name + "_RoiSet.zip"),
        x=x / params["rescale"],
        y=y / params["rescale"],
    )

    if not (
        params["SaveCenterlinedWormsSerial"]
        | params["SaveCenterlinedWormsMovie"]
        | params["SaveCenterlinedWormsMultitiff"]
    ):
        logger.info("Params and plots are successfully saved.")
        logger.info(f"Code finished at {datetime.datetime.now()}")
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
        output_folder = output_name + "_png"
        clear_dir(output_path, output_name + "_png")
        # for t in range(len(filenames_full)):
        end_T = n_input_images - 1 if params["end_T"] == 0 else params["end_T"]
        # Draw the first figure as template
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        img = ax.imshow(real_image[0], cmap="gray")
        line = ax.plot([x[0] - org_x_st, y[0] - org_y_st], c="r", lw=3)[0]

        fig.canvas.draw()
        for i, t in enumerate(range(params["start_T"], end_T + 1)):
            # set data
            img.set_data(real_image[t])
            line.set_data(x[i] - org_x_st, y[i] - org_y_st)
            # redraw
            fig.canvas.draw()
            filename = os.path.join(
                output_path,
                output_folder,
                "image" + str(t).zfill(len(str(n_input_images))) + ".png",
            )
            fig.savefig(filename)
        plt.close(fig)
        logger.info("png images saved to " + output_folder + " etc.")

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
        try:
            # If matplotlib can not find the FFmpeg, a ValueError will be raised.
            ani.save(filename)
        except ValueError as e:
            raise ValueError(f"FFmpeg was not found: {e}")
        logger.info("Movie saved to " + filename)

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

    logger.info("Params and plots are successfully saved.")
    logger.info(f"Code finished at {datetime.datetime.now()}")
