# WormTracer
Automatic worm centerline tracer, WormTracer

We present WormTracer, which extracts centerlines from worm images.
Special characterics of WormTracer is as follows.
- It extracts precise centerlines from worm images even when the worm assumes complex postures like curled up postures.
- It achieves this goal by treating sequencial images in parallel, trying to keep the continuity of centerline across time.
- It threfore requires a movie, or time series images, of a single worm.
- It receives only binalized images obtained by, for example, manual thresholding. It means that WormTracer does not rely on textures or brightness imformation in determining the centerline. It only utilizes the outline of a worm. 

The main code is provided in .ipynb and .py files. Use whichever is convenient for you. In both cases, the functions.py file needs to be accessible from the main program. Please see "How to use WormTracer".

Because WormTracer optimizes the candidate centerlines on multiple images in parallel, use of GPU is highly recommended.

Author: Koyo Kuze; 
Contributors: Ukyo T. Tazawa, Karin Suwazono, Yu Toyoshima, Yuichi Iino

