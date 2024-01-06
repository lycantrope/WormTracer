import json as _json
import sys as _sys
import numpy as _np

from .types import _PATH_LIST_T, _PATH_T, _Path, _T
from .parameter import ShapeParameters as _ShapeParams

__all__ = [
    "eprint",
    "read_json",
    "rmtree",
]


def eprint(
    *values,
    sep: str = " ",
    end: str = "\n",
    flush: bool = False,
) -> None:
    """
    eprint(value, ..., sep=' ', end='\n', flush=False)
    Prints the values to sys.stderr
    Optional keyword arguments:
    sep:   string inserted between values, default a space.
    end:   string appended after the last value, default a newline.
    flush: whether to forcibly flush the stream.
    """
    print(*values, sep=sep, file=_sys.stderr, end=end, flush=flush)


def read_json(file: _PATH_T) -> None:
    try:
        with open(file, "r") as f:
            return _json.load(f)
    except IOError as e:
        eprint(e)


def rmtree(root: _PATH_T) -> None:
    """Recursively remove file or directory.

    This function recursively removes all files and subdirectories under the
    root directory. If the root itself is a file, it will be deleted as well.

    Args:
        root (_Path): A path to the file or a root folder to be removed.

    Raises:
        AssertionError: If the input `root` is not an instance of `_Path`.

    Example:
        ```python
        # Remove a directory
        rmtree(_Path("/path/to/directory"))

        # Remove a file
        rmtree(_Path("/path/to/file.txt"))
        ```

    """

    root = _Path(root)
    if not root.is_dir():
        root.unlink(True)
        return
    [rmtree(c) for c in root.glob("*")]
    root.rmdir()


def glob(data_folder: _PATH_T, ext: str = "png") -> _PATH_LIST_T:
    # sort data in lexicographic order
    return sorted(_Path(data_folder).glob("*.{}".format(ext)), key=lambda x: x.stem)


def calc_avg_shape_params(
    history: _T.List[_T.Tuple[int, _ShapeParams]]
) -> _ShapeParams:
    data = _np.array([(t, p.alpha, p.delta, p.gamma) for (t, p) in history]).T
    data[1] *= data[0]
    data[2] *= data[0]
    data[3] *= data[0]

    data_sum = data.sum(axis=1)
    data_sum[1:] /= data_sum[0]
    return _ShapeParams(alpha=data_sum[1], delta=data_sum[2], gamma=data_sum[3])
