# WormTracer
An algorithm designed to accurately determine the centerline of a worm in time-lapse images.

![Abstract](https://github.com/user-attachments/assets/ceae534f-2e23-40d4-808f-0a3abe929abb)


[WormTracer](#reference) estimates worm centerlines from a sequence of images. This process ensures the continuity of centerlines across time points and captures complex postures, which are usually difficult to assess from an isolated image. The centerlines obtained through WormTracer exhibit higher accuracy compared to those acquired using conventional methods.



## Installation

### Using pip
```sh
pip install "wormtracer @ git+https://github.com/lycantrope/WormTracer.git"
```
### Using uv
```sh
uv add "wormtracer @ git+https://github.com/lycantrope/WormTracer.git"
```
### Using git

Alternatively, you can "git clone" this repository to any directory via
```sh
git clone https://github.com/lycantrope/WormTracer.git && cd WormTracer
```
Then, pip or uv install it with pip or uv
```sh
pip install .
```

```sh
uv sync
```

## Installation with napari-wormtracer
WormTracer provide an napari-based GUI for manually inspecting and revising the centerlines.
To install the WormTracer together with napari GUI tools:
```sh
pip install "wormtracer[gui] @ git+https://github.com/lycantrope/WormTracer.git"
```

And you can launch the napari with plugin using:
```sh
napari
```



## Preprocess of the images

### Data Format  
WormTracer requires a time series of binarized worm images. WormTracer accepts a single file containing all images that can be read by `tifffile/OpenCV`, or a folder containing image sequence (tif/png/jpeg).  
If the folder is provided, WormTracer will grab all files from folder with the most abundant data format and WormTracer will try to load them by the name in lexicographic order. This should reflect the chornological order of your images.  

### Binarization  
Since the quality of binarized mask significantly affects the accuracy of WormTracer's centerline estimation, we recommend manually binarizing raw images.  
ImageJ provides several methods to binarize the raw images (`"Image > Adjust > Threshold"`). After binarizing the images as mask, you can export the masks as image sequence via `"File > Save as > Image Sequence..."` or 
as a multipage tiff file by `"File > Save as > Tiff..."`.

### Tips
For a movie that includes a high rate of bent postures, such as a loopy mutant or a worm behaved high rate of omega turns, it is recommended to relatively strict threshold (smaller worm area and thin body), even if holes are seen in the binarized worm body.  
When accurate positioning of the centerline at the head and tail tips are required, it is recommended to use a loose threshold so that the shape of two tips appear clearly.


## Configuration of WormTracer Hyperparameter
The following YAML file describes all the configurable hyperparameters used by WormTracer.
**`all_params.yaml`**:
```yaml 
# General Settings
local_time_difference: 9  # UTC timezone

# Number of segmented points placed on the centerline
# Around 100 is recommended.
plot_n: 100

# Preprocess
# You can set frames which are applied to WormTracer.
# If you want to use all frames, set both start_T and end_T as 0.
start_T: 0 # Number of start frames (default to 0)
end_T: 0 # 0 = process all frames


# You can change the scale of image to use for tracing by this value.
# If MEMORY ERROR occurs, set this value lower.
# For example if you set it 0.5, the size of images will be half of the original.
# Default value is 1.0
rescale: 1.0 # Scaling ratio of original images

# You can reduce frames by thinning out the movie by this value.
# If MEMORY ERROR occurs, set this value higher.
# For example, if you set it to 2, even-numbered frames will be picked up.
# This parameter is useful in case frame rate is too high.
# Default value is 1.
Tscale: 1 # Timestep of each frame

# Loss Weights

# This value is the weight of the continuity constraint.
# Around 10000 is recommended, but if the object moves fast, set it lower.
continuity_loss_weight: 10000 # Ensures smooth movement between time frames (float, > 0)
# This value is the weight of the smoothness constraint.
# Around 50000 is recommended, but if the object bends sharply, set it lower.
smoothness_loss_weight: 100000 # Prevents sharp bends and keeps the body shape smooth
# length_loss_weight(float, > 0):
# This value is the weight of the length continuity constraint.
# Around 50 is recommended, but if length of the object changes drastically, set it lower.
length_loss_weight: 50 # Prevents the worm from stretching or shrinking unnaturally

# center_loss_weight(float, > 0):
# This value is the weight of the center position constraint.
# Around 50 is recommended.
center_loss_weight: 50 # Keeps the centerline inside the worm's silhouette

# This value is body (rigid part of the object) ratio of the object.
# If the object is a typical worm, set it around 90.
body_ratio: 90 # Weight ratio between the middle body and head/tail

# Training 
# This value is speed of annealing progress.
# The larger this value, the faster the learning is completed.
# 0.1 is efficient, 0.05 is cautious.
speed: 0.05
# This value is learning rate of training.
# Around 0.05 is recommended.
lr: 0.05
# This value is additional training epoch number.
# After annealing is finished, training will be performed for at most epoch_plus times.
# Over 1000 is recommended.
epoch_plus: 1500  # Additional epochs after final step

# Postprocess
# Discriminate head and tail by eigher of the following criteria,
# Variance of body curvature is larger near the head ('amplitude')
# Frequency of body curvature change is larger near the head ('frequency')
judge_head_method: frequency # Judge the head or tail by `frequency` or `amplitude` (default to `frequency`)

# Display & Progress
# This value means the number of images which are displayed
# when show_image function is called.
# Default value is 5.
# If you want to see all frames, set it to "np.inf".
num_t: 5
# If True, shows progress during optimization repeats.
ShowProgress: false
# If True, saves worm images during optimization in "progress_image" folder created in datafolder.
SaveProgress: false 
# This value is epoch frequency of displaying tracing progress.
show_progress_freq: 200
# This value is epoch frequency of saving tracing progress.s
save_progress_freq: 50
# This value is the number of images that are included in saved progress tracing.
save_progress_num: 50

# Output Formats
# If True, saves input images with estimated centerline as serial numbered png files in full_line_images folder.
SaveCenterlinedWormsSerial: false
# If True, saves input images with estimated centerline as a movie full_line_images.mp4
SaveCenterlinedWormsMovie: false
# If True, saves input images with estimated centerline as a multipage tiff full_line_images.tif
SaveCenterlinedWormsMultitiff: false
```

And some parameters can be ignored. The following is the minimal `parameter_file.yaml` file required to run WormTracer using sample data.
* **`essential_params.yaml`**
```yaml 
# Number of points on centerline
plot_n: 100

# Weights for different loss functions used during optimization
continuity_loss_weight: 10000
smoothness_loss_weight: 100000
length_loss_weight: 50
center_loss_weight: 50

body_ratio: 90
```

## Centerlines Estimation

### Input arguments
* `parameter_file`: path to your params.yaml file. Detail setup please check the [YAML configuration](#configuration-of-wormtracer-hyperparameter)
* `dataset_path`: path to your time-series binarized images, either folder or file is acceptable. Details please check [here](#preprocess-of-the-images).
* `output_directory` (optional): If provided the final output will be save in `output_directory` but not the parent folder of `dataset_path`
* `guide_files` (list[os.Pathlike], optional): the path to the guide files. (see [Guide Files](#guide-files))  
  
### Using python scripts
* **Run with params.yaml**
```python
import WormTracer.wt as wt
parameter_file = "./params.yaml"
dataset_path = "./hoge_mask"
wt.run(
    parameter_file=parameter_file,
    dataset_path= dataset_path,
)
```

* **Run with params.yaml and overwrite the plot_n to 50**
```python
import WormTracer.wt as wt
parameter_file = "./params.yaml"
dataset_path = "./hoge_mask"
wt.run(
    parameter_file=parameter_file,
    dataset_path=dataset_path,
    plot_n=50,
)
```


* **with-guide-csv**
```python
import WormTracer.wt as wt
parameter_file = "./hoge_params.yaml"
dataset_path = "./hoge_mask"
guide_x_path = "./hoge_mask_guide_x.csv"
guide_y_path = "./hoge_mask_guide_y.csv"
wt.run(
    parameter_file=parameter_file,
    dataset_path=dataset_path,
    guide_files=[guide_x_path,guide_y_path], 
)
```

* **with-guide-hdf**
```python
import WormTracer.wt as wt
parameter_file = "./hoge_params.yaml"
dataset_path = "./hoge_mask"
guide_h5_path = "./hoge_mask_guide_skel.h5"
wt.run(
    parameter_file=parameter_file,
    dataset_path=dataset_path,
    guide_files=[guide_h5_path], # Must be a list or tuple
)
```


### Using command-line interface (CLI)

* Use `-h/--help` to see further details.
```sh
python -m WormTracer --help
```

* **without guide**
```sh
python -m WormTracer \
--parameter_file "./hoge_params.yaml" \
--dataset_path "./hoge_mask"
```
* **with guide csv**
```sh
python -m WormTracer \
--parameter_file "./hoge_params.yaml" \
--dataset_path "./hoge_mask" \
--guide_files "./hoge_mask_guide_x.csv" "./hoge_mask_guide_y.csv"
```
* **with guide hdf**
```sh
python -m WormTracer \
--parameter_file "./hoge_params.yaml" \
--dataset_path "./hoge_mask" \
--guide_files "./hoge_mask_guide_skel.h5"
```
---
The parameters defined in the `parameter_file.yaml` will be overridden by commandl-line arguments.
Following example shows that the `plot_n` was changed to 50.
```sh
python -m WormTracer \
--parameter_file "./hoge_params.yaml" \
--dataset_path "./hoge_mask" \
--plot_n 50
```

### Using Google Colab  
**A quick demo running WormTracer on Google Colab.**  
<a target="_blank" href="https://colab.research.google.com/github/lycantrope/WormTracer/blob/main/examples/inference_on_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### WormTracer Output Format
WormTracer organizes all results into a structured directory for each analysis run.
* **Output Directory**: The final output folder is located at the parent directory of `dataset_path`, or at the custom `output_directory` if one is provided. 
* **Folder naming**: The folder name is derived from the dataset_path and suffixed with an incrementing number (e.g., hoge_mask_001). WormTracer automatically creates this directory within the Output Directory.
Inside the result folder (e.g., `hoge_mask_001`), the following files are generated:
* `hoge_mask_001.log`: Detailed log information of the execution process.
* `hoge_mask_001_params.yaml`: A YAML file containing all initial parameters used for the run, supplemented with additional metadata such as the original dataset_path and the final optimized (trained) parameters.
* `hoge_mask_001_x.csv`:  X-coordinates from start_T to end_T. Rows represent time, and columns represent positions from head to tail (T, plot_n).
* `hoge_mask_001_y.csv`: Y-coordinates formatted identically to the X-coordinate file
* `hoge_mask_001_skel.h5`: An HDF5 file containing two datasets, x and y.
* `hoge_mask_001_RoiSet.zip`: A collection of ImageJ-compatible ROIs (Regions of Interest) that can be directly imported into the ImageJ RoiManager for visualization.
* `hoge_mask_001_losses.csv`: Loss values recorded across all training steps, containing
* * `step`: The specific training step.
* * `block`: The training block within WormTracer.
* * `index`: The exact frame index.
* * `is_complex`: Boolean indicating if the frame belongs to a complex posture block.
* * `is_guide`: Boolean indicating if the frame was used as a guide frame.
* * `image_loss`, `continuity_loss`,`smoothing_loss`,`length_loss`, `center_loss`
* 
If the corresponding flags are enabled (see [Configuration](#configuration-of-wormtracer-hyperparameter)), WormTracer can generate cropped mask images with overlaid centerlines in the following formats:
*  `hoge_mask_001/hoge_mask_001_png/`: A directory containing the image sequence in PNG format.
*  `hoge_mask_001.mp4`: A rendered MP4 video of the tracked sequence.
*  `hoge_mask_001.tif`: multi-page of ImageJ-Tiff

## Postprocessing or re-run WormTracer with guide points
### Guide Files
In the latest version of WormTracer, you can provide guide files to serve as a ground truth for specific frames. This is particularly useful when a training block is too large or contains no "simple" postures for the algorithm to anchor to.  
Frames marked in a guide file are treated as stationary points (fixed ground truth) during training. WormTracer will split the training block at these points, significantly improving optimization in long or complex sequences.  
The guide file follows the exact same format as the standard WormTracer output (_x.csv, _y.csv, or .h5). During processing, any rows with *`NaN` values* will be ignored and optimized normally by the algorithm; **`only`** rows containing *`non-NaN`* values will be automatically identified as a guide point.

### Generating Guide Files with napari-wormtracer
The [napari-wormtracer](https://github.com/lycantrope/napari-wormtracer) plugin allows you to refine your results through two main workflows:
* Direct Editing: Load a WormTracer output file to manually inspect and correct specific centerlines for difficult frames.

* Export as guide file (`with_guide`): Select high-confidence frames and export them as a guide file. The plugin saves all **revised frames** as non-NaN data, while all other rows are automatically set to `NaN`. These NaN rows will then be optimized normally by the algorithm during the next run.

For further details, see https://github.com/lycantrope/napari-wormtracer

## Reference
"WormTracer: A precise method for worm posture analysis using temporal continuity"
Koyo Kuze, Ukyo T. Tazawa, Karin Suwazono, Chung-Kuan Chen, Yu Toyoshima, Yuichi Iino,
Journal of Neuroscience Methods,
https://doi.org/10.1016/j.jneumeth.2025.110644
