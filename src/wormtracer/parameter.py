import attrs as _attrs
import typing as _T


@_attrs.define(kw_only=True, frozen=True)
class HyperParameters:
    lr: float
    speed: float
    epoch_plus: int
    continuity_loss_weight: float
    smoothness_loss_weight: float
    length_loss_weight: float
    center_loss_weight: float
    body_ratio: float
    n_segments: int
    body_ratio: int
    device: str

    def as_dict(self) -> _T.Dict[str, _T.Any]:
        return _attrs.asdict(self)


@_attrs.define(kw_only=True, frozen=True)
class ShapeParameters:
    alpha: float
    delta: float
    gamma: float

    def as_tuple(self) -> _T.Tuple[float, float, float]:
        """
        Returns:
            (alpha, delta, gamma)
        """
        return _attrs.astuple(self)
