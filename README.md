# WormTracer
Automatic worm centerline tracer, WormTracer

We present WormTracer, which extracts centerlines from worm images.
Special characteristics of WormTracer is as follow:
- It extracts precise centerlines from worm images even when the worm assumes complex postures like curled up postures.
- It achieves this goal by treating sequencial images in parallel, trying to keep the continuity of centerline across time.
- It therefore requires a movie, or time series images, of a single worm.
- It receives only binalized images obtained by, for example, manual thresholding. It means that WormTracer does not rely on textures or brightness imformation in determining the centerline. It only utilizes the outline of a worm. 


Because WormTracer optimizes the candidate centerlines on multiple images in parallel, use of GPU is highly recommended.

Author: Koyo Kuze; 
Contributors: Ukyo T. Tazawa, Karin Suwazono, Yu Toyoshima, Yuichi Iino

