import typing as _T

import numpy as np
import torch
import torch.nn as nn
from attrs import define as _define
from attrs import field as _field

_NP_T: _T.TypeAlias = np.ndarray


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
    x = torch.arange(-0.5 * n_segments + 2, 0.5 * n_segments) - 0.5
    n = 1 / base - 1
    body_axis_weight = (
        n
        * (torch.sigmoid(x + body_ratio // 2) + torch.sigmoid(-x + body_ratio // 2) - 1)
        + 1
    ) / (n + 1)
    return body_axis_weight.reshape(1, n_segments - 2)


@_define
class AnnealingFunction:
    T: float = _field(converter=float)
    speed: float = _field(default=0.2)
    start: float = _field(default=0.0)
    slope: float = _field(default=1.0)

    def get_weights(self, epoch: int):
        x = torch.abs(torch.arange(-self.T / 2 + 0.5, self.T / 2 + 0.5)) - self.T / 2
        return torch.sigmoid((x + self.start + epoch * self.speed) * self.slope)


class ImageLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(ImageLoss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        loss = _T.cdist(pred, target)
        loss = loss.mean(axis=tuple(range(1, loss.ndim)))
        return loss * self.weight


class SmoothnessLoss(nn.Module):
    def __init__(self, body_axis_weight, smoothness_weight=1.0):
        super(SmoothnessLoss, self).__init__()
        self.body_axis_weight = body_axis_weight
        self.weight = smoothness_weight

    def forward(
        self,
        theta: torch.Tensor,
    ):
        loss = (theta.diff(n=1, axis=1) * self.body_axis_weight).pow(2) * self.weight
        return loss


class ContinuityLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(ContinuityLoss, self).__init__()
        self.weight = weight

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        loss = theta.diff(n=1, axis=0).float_power(2).mean()
        return loss.mean() * self.weight


class ISCLoss(nn.Module):
    """ISCLoss: abbreviation loss of (Image + Smoothness + Continuity)"""

    def __init__(
        self,
        im_loss_fn: ImageLoss,
        smooth_loss_fn: SmoothnessLoss,
        continuity_loss_fn: ContinuityLoss,
        annealing_fn: torch.Optional[AnnealingFunction] = None,
    ):
        super(ISCLoss, self).__init__()
        self.image_loss_fn = im_loss_fn
        self.smooth_loss_fn = smooth_loss_fn
        self.continuity_loss_fn = continuity_loss_fn
        self.annealing_fn = annealing_fn

    def forward(self, pred, target, theta: torch.Tensor, epoch: int) -> torch.Tensor:
        img_loss = self.image_loss_fn(pred, target)
        smo_loss = self.smooth_loss_fn(theta)
        img_smo_loss = img_loss + smo_loss
        if self.annealing_fn is not None:
            img_smo_loss = img_smo_loss * self.annealing_fn.get_weights(epoch)

        cont_loss = self.continuity_loss_fn(theta)
        return img_smo_loss.mean() + cont_loss


class LengthLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super(CenterLoss, self).__init__()
        self.weight = weight

    def forward(self, unit_length: torch.Tensor) -> torch.Tensor:
        loss = unit_length.diff(n=1, axis=0).pow(2)
        return loss * self.weight


class CenterLoss(nn.Module):
    def __init__(self, unit: float, weight: float = 1.0):
        super(CenterLoss, self).__init__()
        self.unit = unit
        self.weight = weight

    def forward(
        self,
        ct: torch.Tensor,
        init_ct: torch.Tensor,
    ):
        return torch.cdist(ct, init_ct).pow(2) / self.unit * self.weight


@_define(kw_only=True, frozen=True, order=True)
class WormLosses:
    image: _NP_T
    continuity: _NP_T
    smoothness: _NP_T
    length: float = _field(eq=False)
    center: float = _field(eq=False)

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
            q75, q50, q25 = np.percentile(min_l, (75, 50, 25))
            return (max(max_l) - q50) / (q75 - q25)

        im_exrate = _weight(im_min, im_max)
        con_exrate = _weight(con_min, con_max)
        smo_exrate = _weight(smo_min, smo_max)

        exrate_loss = np.argmax([im_exrate, con_exrate, smo_exrate])
        return bool([im_select, con_select, smo_select][exrate_loss])


def find_outliner(losses_all: _T.List[LengthLoss]):
    mask = np.zeros(len(losses_all)).astype(bool)

    def _helper(losses: _T.List[_NP_T]):
        q25, q50, q75 = np.percentile(np.concatenate(losses, axis=0), (25, 50, 75))

        def pred(loss: _NP_T) -> bool:
            return np.max(loss) - q50 > (q75 - q25) * 4

        return pred

    loss_attrs = [
        "image",
        "continuity",
        "smoothness",
    ]
    for n in loss_attrs:
        pred = _helper([getattr(loss, n) for loss in losses_all])
        mask = mask | np.array([pred(getattr(loss, n)) for loss in losses_all]).astype(
            bool
        )
    return mask
