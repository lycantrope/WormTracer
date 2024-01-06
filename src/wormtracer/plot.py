import matplotlib.pyplot as _plt
import numpy as _np
import typing as _T, _Figure, _OPTIONAL_NP
from math import ceil as _ceil


def show_image(
    im_stack: _np.ndarray,
    num_t: int = 5,
    title: str = "",
    n_cols: int = 5,
    xy1: _OPTIONAL_NP = None,
    xy2: _OPTIONAL_NP = None,
) -> _Figure:
    """
    Display a stack of images over time with optional point annotations.

    Args:
        im_stack (np.ndarray): A 3D array representing a stack of images over time (T, H, W).
        num_t (int, optional): Number of time points to display. Defaults to 5.
        title (str, optional): Title for the plot. Defaults to an empty string.
        n_cols (int, optional): Number of columns in the display grid. Defaults to 5.
        xy1 (Optional[np.ndarray]], optional): Optional coordinates
            (T, XY, N) to annotate on the plot for the first set of points. Defaults to None.
        xy2 (Optional[np.ndarray]], optional): Optional coordinates
            (T, XY, N) to annotate on the plot for the first set of points. Defaults to None.

    Returns:
        plt.Figure: The Matplotlib figure object.

    Raises:
        AssertionError: If the input `im_stack` is not a 3-D ndarray (T, H, W).

    Example:
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        from wormtracer.plot import show_image

        # Assuming im_stack is a 3D array (T, H, W)
        show_image(im_stack, num_t=5, title="Worm", n_cols=3)
        plt.show()
        ```

    Note:
        The function uses Matplotlib to create a grid of images, displaying a subset of
        time points specified by `num_t`. Optionally, it can annotate points on each
        image using the coordinates provided in `xy1` and `xy2`.
    """

    assert im_stack.ndim == 3, "stack is not a 3-D ndarray (T, H, W)"

    T, H, W = im_stack.shape[:3]

    step = max(T // (num_t - 1), 1)
    t_slice = [*range(0, T, step)]

    n_axes = len(t_slice)
    # binning plot into n_cols
    n_rows = _ceil(n_axes / n_cols)
    fig, _ = _plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols + 0.8, 2 * n_rows),
        squeeze=False,
        tight_layout=True,
    )
    # turn off all axes before plotting
    [ax.set_axis_off() for ax in fig.get_axes()]

    vmax = _np.max(im_stack)
    for idx, ax in zip(t_slice, fig.get_axes()):
        cax = ax.imshow(
            im_stack[idx, :, :],
            cmap="gray",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_title("{} t = {}".format(title, idx))

        if xy1 is not None:
            ax.scatter(xy1[idx, 0], xy1[idx, 1], c="r", s=30, label="xy1")

        if xy2 is not None and xy1.ndim == 3:
            ax.scatter(xy2[idx, 0], xy2[idx, 1], c="y", s=30, label="xy2")

        ax.set_axis_on()

    fig.tight_layout()
    return fig


def make_progress_image(im_stack: _np.ndarray, num_t=20, n_cols: int = 5):
    """Make one large image with images laid out on it."""
    assert im_stack.ndim == 3, "stack is not a 3-D ndarray (T, H, W)"

    T, H, W = im_stack.shape[:3]
    step = max(T // (num_t - 1), 1)
    t_slice = [*range(0, T, step)]
    n_im = len(t_slice)
    n_rows = _ceil(n_im / n_cols)
    # binning plot into n_cols
    progress_image = _np.zeros((H * n_rows, W * n_cols), dtype="uint8")
    for i, idx in enumerate(t_slice):
        y1 = (i // n_cols) * H
        y2 = y1 + H
        x1 = (i % n_cols) * W
        x2 = x1 + W
        progress_image[y1:y2, x1:x2] = im_stack[idx]
    return progress_image
