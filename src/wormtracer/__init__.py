"""
n_segments (int, == 100)
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
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (
    dataset,
    loss,
    model,
    parameter,
    plot,
    preprocess,
    train,
    utils,
)

__all__ = [
    "dataset",
    "loss",
    "model",
    "parameter",
    "plot",
    "preprocess",
    "train",
    "utils",
]
