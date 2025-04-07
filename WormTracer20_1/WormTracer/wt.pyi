import os

def run(
    parameter_file: os.PathLike,
    dataset_path: os.PathLike,
    output_directory: os.PathLike = "",
    **kwargs,
):
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
