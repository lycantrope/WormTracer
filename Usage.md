# WormTracer

WormTracer is an algorithm designed to accurately determine the centerline of a worm in time-lapse images. Unlike conventional methods that analyze individual images separately, WormTracer estimates worm centerlines within a sequence of images concurrently. This process enables the resolution of complex postures that are difficult to assess when treated as isolated images. The centerlines obtained through WormTracer exhibit higher accuracy compared to those acquired using conventional methods.

WormTracer codes are written in Python and provided in two forms, as an IPython Notebook (.ipynb), or a Python source code (.py). Users can choose either format depending on their own environment for running the codes.

Try a quick demo here.<br>
<a target="_blank" href="https://colab.research.google.com/github/yuichiiino1/WormTracer/blob/main/WormTracer19_5/WT19_5_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Reference
"WormTracer: A precise method for worm posture analysis using temporal continuity"
bioRxiv 2023.12.11.571048; doi: https://doi.org/10.1101/2023.12.11.571048
and a manuscript under review.<br><br>

We provide several versions of WormTracer in different subfolders. Please refer to the ReadMe file in each subfolder for further details.

We also provide sample images in the "Sample Images" subdirectory as a reference and for the convenience of checking whether your installation is successful.
WT_grayscale.avi is the original movie of a moving _C. elegans_. WT_binary.avi is the same images after manual binalization. This was converted into sequential tiff files in WT_binary_image_sequence.zip or a multipage tiff file in WT_binary_multipage_tiff.zip (WT17_5 only accepts sequential files, while higher versions also accept multipage tiff files). Use either of downloaded zip file, unzip, and run WormTracer assigning the file or file directory as an input, as instructed in ReadMe for each version.
WT_estimated_centerline.avi is the example output of WormTracer obtained using these files as an input.


