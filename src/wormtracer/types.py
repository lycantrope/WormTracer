import typing as _T
import numpy as _np
from pathlib import Path as _Path

_NP_T: _T.TypeAlias = _np.ndarray
_OPTIONAL_NP: _T.TypeAlias = _T.Optional[_NP_T]
_PATH_T: _T.TypeAlias = _T.Union[str, _Path]
_PATH_LIST_T: _T.TypeAlias = _T.List[_Path]
_IMREAD_T: _T.TypeAlias = _T.Callable[[_PATH_T], _NP_T]
_TRANSFORM_T: _T.TypeAlias = _T.Callable[[_NP_T], _NP_T]


class _Offset(_T.NamedTuple):
    x: int = 0
    y: int = 0


__all__ = [
    "_T",
    "_NP_T",
    "_OPTIONAL_NP",
    "_PATH_T",
    "_PATH_LIST_T",
    "_IMREAD_T",
    "_TRANSFORM_T",
    "_Path",
    "_Offset",
]
