import typing as _T
import numpy as _np
from pathlib import Path as _Path
import attrs as _attrs

_NP_T: _T.TypeAlias = _np.ndarray
_OPTIONAL_NP: _T.TypeAlias = _T.Optional[_NP_T]
_PATH_T: _T.TypeAlias = _T.Union[str, _Path]
_PATH_LIST_T: _T.TypeAlias = _T.List[_Path]
_IMREAD_T: _T.TypeAlias = _T.Callable[[_PATH_T], _NP_T]
_TRANSFORM_T: _T.TypeAlias = _T.Callable[[_NP_T], _NP_T]


class _Offset(_T.NamedTuple):
    x: int = 0
    y: int = 0


@_attrs.define(frozen=True)
class ShapeParams:
    alpha: float = _attrs.field(kw_only=True)
    delta: float = _attrs.field(kw_only=True)
    gamma: float = _attrs.field(kw_only=True)

    def astuple(self) -> _T.Tuple[float, float, float]:
        """
        Returns:
            (alpha, delta, gamma)
        """
        return _attrs.astuple(self)


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
