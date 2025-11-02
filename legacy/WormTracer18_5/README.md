# WormTracer

WormTracer is an algorithm designed to accurately determine the centerline of a worm in time-lapse images. Unlike conventional methods that analyze individual images separately, WormTracer estimates worm centerlines within a sequence of images concurrently. This process enables the resolution of complex postures that are difficult to assess when treated as isolated images. The centerlines obtained through WormTracer exhibit higher accuracy compared to those acquired using conventional methods.

WormTracer codes are written in Python and provided in two forms, as an IPython Notebook (.ipynb), or a Python source code (.py). Users can choose either format depending on their own environment for running the codes. Please refer to 3. or 4. below for each environment.

- Reference
"WormTracer: A precise method for worm posture analysis using temporal continuity"
bioRxiv 2023.12.11.571048; doi: https://doi.org/10.1101/2023.12.11.571048
and a manuscript under review.

#### <<< How to use WormTracer >>>

In all cases, input images need to be time-lapse images in a folder, either as a single multipage tiff file or serial numbered image files with a format of your choice, such as tif, png, jpeg etc.
In all cases, results are saved in the input image folder.

1. **Data required**<br><br>
WormTracer only requires a "folder containing binarized worm video images".
Put a binarized worm video in a single folder, either as a single multipage tiff file or serial numbered image files with a format of your choice, such as tif, png, jpeg etc that can be read by OpenCV. Binarization threshold is sometimes critical (see Tips), so we recommend users binarize raw images manually, for example using the "Image > Adjust > Threshold" function of ImageJ. The folder name is arbitrary and you will specify the folder name when you run WormTracer. For the serial numbered format, image file names should include numbers at the end in a chronological order; you can save the images in this format for example by selecting “File > Save as > Image Sequence...” in ImageJ. Alternatively, stack images can be saved as a multipage tiff file by "File > Save as > Tiff..." in ImageJ. Put this file alone in a new folder.<br><br>
Tips: For a movie that include a high rate of bent postures, for example of a loopy mutant or a worm that shows high rate of omega turns, relatively strict threshold (smaller worm area and thin body) is recommended, even though holes are seen in the binarized worm body. On the other hand, in cases where accurate positioning of the centerline at the head and tail tips are required, a threshold at which these shapes appear clearly is recommended.<br>

2. **Results**<br>
In all cases, results are saved in the input image folder.<br>

3. **Running WormTracer from IPython Notebook**,<br> such as on Jupyter Notebook or Google Colaboratory (GUI environment)<br><br>
(preparation)<br>
Install following packages that WormTracer depends on, if they are not yet installed to your environment:
torch, numpy, opencv-python, matplotlib, Pillow, scikit-image, scipy, pyyaml<br><br>
Make input images in a folder
Place functions.py at an appropriate path
Place wt18_5.ipynb and change parameters, especially the path to functions.py and path to the input image folder.
<br><br>
(execution)<br>
Run all cells in wt18_5.ipynb<br>

4. **Running WormTracer from command line (CLI)**<br><br>
(installation)
Install WormTracer by entering the following command after activating appropriate python environment<br><br>
\$ pip install git+https://github.com/yuichiiino1/WormTracer.git#subdirectory=WormTracer18_5<br><br>
Alternatively, the codes can be downloaded from https://github.com/yuichiiino1/WormTracer/ (current version is in the WormTracer18_5 subfolder), either by pip clone or https download, and (after optional modifications) install as<br><br>
\$ pip install [local path to WormTracer18_5 folder]<br><br>
Confirm successfule installation by making sure that WormTracer appears by \$ pip list
In case you need to uninstall, just enter \$ pip uninstall WormTracer<br><br>
(preparation)<br>
Make input images in a folder
Make a parameter file xxxx.yaml according to your preferred options, or use the sample file, config.yaml, as is.<br><br>
(execution in case of python interactive mode)<br>
Enter following commands:<br>
\>> from WormTracer import wt<br>
\>> wt.run('[path to xxxx.yaml]', '[path to input image folder]', parameter1=1000)<br><br>
Note: after two mandatory arguments, optional (parameter_name, value) pairs (example: parameter1=1000) can be specified, to override the value of a specific parameter(s).<br><br>
(execution in case of batch mode)<br><br>
\< wt0000.sh > (adjust according to your environment)
#!/bin/sh
python3 wtexe.py [path to xxxx.yaml] [path to input image folder]<br><br>
< wtexe.py >
import sys
from WormTracer import wt
wt.run(sys.argv[1], sys.argv[2])<br>
Optional parameter pairs can be specfied also in this case.<br><br>
(Troubleshooting)<br><br>
Under some environment, saving the results as a mp4 movie may fail with an error message such as "unknown file extension: .mp4". In this case, try downloading the ffmpeg package from https://ffmpeg.org/download.html, and specify the path to ffmpeg executable in wtexe.py as follows.<br><br>
from matplotlib import pyplot as plt
plt.rcParams["animation.ffmpeg_path"] = "PATH_TO_FFMPEG"<br><br>
before "wt.run(sys.argv[1], sys.argv[2])"<br><br>
Or, just avoid making the movie file by setting the parameter SaveCenterlinedWormsMovie as False in xxxx.yaml.
<br><br>
5. Adjustable hyperparameters
```
dataset_path (mandatory):
Path to a folder including input images.
Images are either as a single multipage tiff file or serial numbered image files, with either of the following format.
".bmp", ".dib", ".pbm", ".pgm", ".ppm", ".pnm", ".ras", ".png", ".tiff", ".tif", ".jp2", ".jpeg", ".jpg", ".jpe"
ALL RESULTS ARE SAVED in dataset_path.

functions_path (mandatory):
Path to functions.py file, which is essential.

start_T, end_T(int, > 0):
You can set frames which are applied to WormTracer.
If you want to use all frames, set both start_T and end_T as 0.

rescale(float, > 0, < 1):
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
```
