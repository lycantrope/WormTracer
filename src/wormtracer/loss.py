import attrs as _attrs
import numpy as _np
import torch as _t
import torch.nn as _nn
from wormtracer.types import _T, _NP_T

__all__ = [
    "AnnealingFunction",
    "ImageLoss",
    "SmoothnessLoss",
    "ContinuityLoss",
    "ISCLoss",
    "LengthLoss",
    "CenterLoss",
    "body_axis_function",
]


def body_axis_function(body_ratio, n_segments, base=0.5):
    x = _t.arange(-0.5 * n_segments + 2, 0.5 * n_segments) - 0.5
    n = 1 / base - 1
    body_axis_weight = (
        n * (_t.sigmoid(x + body_ratio // 2) + _t.sigmoid(-x + body_ratio // 2) - 1) + 1
    ) / (n + 1)
    return body_axis_weight.reshape(1, n_segments - 2)


@_attrs.define
class AnnealingFunction:
    T: float = _attrs.field(converter=float)
    speed: float = _attrs.field(default=0.2)
    start: float = _attrs.field(default=0.0)
    slope: float = _attrs.field(default=1.0)

    def get_weights(self, epoch: int):
        x = _t.abs(_t.arange(-self.T / 2 + 0.5, self.T / 2 + 0.5)) - self.T / 2
        return _t.sigmoid((x + self.start + epoch * self.speed) * self.slope)


class ImageLoss(_nn.Module):
    def __init__(self, weight=1.0):
        super(ImageLoss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        loss = _T.cdist(pred, target)
        loss = loss.mean(axis=tuple(range(1, loss.ndim)))
        return loss * self.weight


class SmoothnessLoss(_nn.Module):
    def __init__(self, body_axis_weight, smoothness_weight=1.0):
        super(SmoothnessLoss, self).__init__()
        self.body_axis_weight = body_axis_weight
        self.weight = smoothness_weight

    def forward(
        self,
        theta: _t.Tensor,
    ):
        loss = (theta.diff(n=1, axis=1) * self.body_axis_weight).pow(2) * self.weight
        return loss


class ContinuityLoss(_nn.Module):
    def __init__(self, weight=1.0):
        super(ContinuityLoss, self).__init__()
        self.weight = weight

    def forward(self, theta: _t.Tensor) -> _t.Tensor:
        loss = theta.diff(n=1, axis=0).float_power(2).mean()
        return loss.mean() * self.weight


class ISCLoss(_nn.Module):
    """ISCLoss: abbreviation loss of (Image + Smoothness + Continuity)"""

    def __init__(
        self,
        im_loss_fn: ImageLoss,
        smooth_loss_fn: SmoothnessLoss,
        continuity_loss_fn: ContinuityLoss,
        annealing_fn: _t.Optional[AnnealingFunction] = None,
    ):
        super(ISCLoss, self).__init__()
        self.image_loss_fn = im_loss_fn
        self.smooth_loss_fn = smooth_loss_fn
        self.continuity_loss_fn = continuity_loss_fn
        self.annealing_fn = annealing_fn

    def forward(self, pred, target, theta: _t.Tensor, epoch: int) -> _t.Tensor:
        img_loss = self.image_loss_fn(pred, target)
        smo_loss = self.smooth_loss_fn(theta)
        img_smo_loss = img_loss + smo_loss
        if self.annealing_fn is not None:
            img_smo_loss = img_smo_loss * self.annealing_fn.get_weights(epoch)

        cont_loss = self.continuity_loss_fn(theta)
        return img_smo_loss.mean() + cont_loss


class LengthLoss(_nn.Module):
    def __init__(self, weight: float = 1.0):
        super(CenterLoss, self).__init__()
        self.weight = weight

    def forward(self, unit_length: _t.Tensor) -> _t.Tensor:
        loss = unit_length.diff(n=1, axis=0).pow(2)
        return loss * self.weight


class CenterLoss(_nn.Module):
    def __init__(self, unit: float, weight: float = 1.0):
        super(CenterLoss, self).__init__()
        self.unit = unit
        self.weight = weight

    def forward(
        self,
        ct: _t.Tensor,
        init_ct: _t.Tensor,
    ):
        return _t.cdist(ct, init_ct).pow(2) / self.unit * self.weight


@_attrs.define(kw_only=True, frozen=True, order=True)
class WormLosses:
    image: _NP_T
    continuity: _NP_T
    smoothness: _NP_T
    length: float = _attrs.field(eq=False)
    center: float = _attrs.field(eq=False)

    def __ne__(self, other: "WormLosses") -> bool:
        pair = (self, other)

        im_select = int(max(self.image) < max(other.image))
        con_select = int(max(self.continuity) < max(self.other))
        smo_select = int(max(self.smoothness) < max(self.smoothness))

        if (im_select + con_select + smo_select) == 3:
            return True

        if (im_select + con_select + smo_select) == 0:
            return False

        im_min = pair[1 - im_select].image
        con_min = pair[1 - im_select].continuity
        smo_min = pair[1 - im_select].smoothness

        im_max = pair[im_select].image
        con_max = pair[im_select].continuity
        smo_max = pair[im_select].smoothness

        def _weight(min_l, max_l):
            q75, q50, q25 = _np.percentile(min_l, (75, 50, 25))
            return (max(max_l) - q50) / (q75 - q25)

        im_exrate = _weight(im_min, im_max)
        con_exrate = _weight(con_min, con_max)
        smo_exrate = _weight(smo_min, smo_max)

        exrate_loss = _np.argmax([im_exrate, con_exrate, smo_exrate])
        return bool([im_select, con_select, smo_select][exrate_loss])


def find_outliner(losses_all: _T.List[LengthLoss]):
    mask = _np.zeros(len(losses_all)).astype(bool)

    def _helper(losses: _T.List[_NP_T]):
        q75, q50, q25 = _np.percentile(_np.concatenate(losses, axis=0), (75, 50, 25))

        def pred(loss: _NP_T) -> bool:
            return _np.max(loss) - q50 > (q75 - q25) * 4

        return pred

    loss_attrs = [
        "image",
        "continuity",
        "smoothness",
    ]
    for n in loss_attrs:
        pred = _helper([getattr(l, n) for l in losses_all])
        mask = mask | _np.array([pred(getattr(l, n)) for l in losses_all]).astype(bool)
    return mask
