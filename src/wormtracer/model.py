import torch as _torch
import torch.nn as _nn
from torch.nn.parameter import Parameter as _Parameter

from wormtracer.formula import pixel_value
from wormtracer.types import _NP_T, _T
from wormtracer.parameter import ShapeParameters as _ShapeParams


class WormShapeLayer(_nn.Module):
    def __init__(self, *, alpha: float, delta: float, gamma: float):
        super().__init__()
        self.alpha = _Parameter(_torch.tensor(alpha))
        self.delta = _Parameter(_torch.tensor(delta))
        self.gamma = _Parameter(_torch.tensor(gamma))

    def forward(self, n_pts: int):
        unit_i = _torch.arange(n_pts - 1)
        worm_x = _torch.abs(2 * (unit_i / (n_pts - 2) - 0.5))

        delta_sigmoid = _torch.sigmoid(self.delta)
        gamma_2e = 2.0 * _torch.exp(self.gamma) + 1

        worm_wid = self.alpha * _torch.sqrt(
            1.00001 - worm_x**gamma_2e * (1 + gamma_2e * delta_sigmoid * (1 - worm_x))
        )

        return worm_wid

    def get_shape_params(self) -> _ShapeParams:
        return _ShapeParams(
            alpha=self.alpha.item(),
            delta=self.delta.item(),
            gamma=self.gamma.item(),
        )

    @classmethod
    def from_shape_params(cls, params: _ShapeParams) -> "WormShapeLayer":
        return cls(alpha=params.alpha, delta=params.delta, gamma=params.gamma)


class WormSkeletonLayer(_nn.Module):
    def __init__(
        self,
        *,
        ct: _NP_T,
        theta: _NP_T,
        unit_length: float,
    ):
        super().__init__()
        T, _ = ct.shape
        unit_vec = _torch.ones(T) * unit_length
        self.ct = _Parameter(_torch.from_numpy(ct))
        self.unit_length = _Parameter(unit_vec)
        self.theta = _Parameter(_torch.from_numpy(theta))

    def forward(self):
        T = self.theta.size(dim=0)
        n_segs = self.theta.size(dim=1)
        # txy [T, 2, n_segs+1]
        txy = _torch.zeros((T, 2, n_segs + 1))
        txy[:, 0, 1:] = _torch.cumsum(_torch.cos(self.theta), dim=1)
        txy[:, 1, 1:] = _torch.cumsum(_torch.sin(self.theta), dim=1)
        txy_mean = txy.mean(dim=2).reshape(T, 2, 1)
        units = self.unit_length.reshape(T, 1, 1)
        cts = self.ct.reshape(T, 2, 1)
        return (txy - txy_mean) * units + cts


class WormImageModel(_nn.Module):
    def __init__(
        self,
        *,
        shape_layer: WormShapeLayer,
        skel_layer: WormSkeletonLayer,
        imshape: _T.Tuple[int, int],
    ):
        super().__init__()
        self.shape_layer = shape_layer
        self.skel_layer = skel_layer
        self.im_height, self.im_width = imshape

    def forward(self):
        # theta: shape [T, n_segs]
        txy = self.skel_layer()
        T = txy.size(dim=0)
        n_pts = txy.size(dim=2)
        n_segs = n_pts - 1
        worm_width = self.shape_layer(n_segs)

        # txy: [T, 2, n_pts]
        cent_mid = (txy[:, :, :-1] + txy[:, :, 1:]) * 0.5
        # cent_mid_3d: [T, 2, n_pts-1, 1, 1]
        cent_mid_3d = cent_mid.reshape(T, 2, n_segs, 1, 1)

        cent_mid_x_3d = cent_mid_3d[:, 0]
        cent_mid_y_3d = cent_mid_3d[:, 1]
        # worm_wid_3d: [1, n_pts-1, 1, 1]
        worm_wid_3d = worm_width.reshape(1, n_segs, 1, 1)

        y_3d = _torch.arange(self.im_height).reshape(1, 1, self.im_height, 1)
        x_3d = _torch.arange(self.im_width).reshape(1, 1, 1, self.im_width)

        # segment_distance_3d: [T, n_segs, im_height, im_width]
        # worm_wid_3d: [1, n_segs, 1, 1]
        segment_distance_3d = _torch.sqrt(
            _torch.pow(cent_mid_x_3d - x_3d, 2) + _torch.pow(cent_mid_y_3d - y_3d, 2)
        )

        # image_3d = [T, n_segs, im_height, im_width]
        image_3d = pixel_value(segment_distance_3d, worm_wid_3d)
        image, _ = _torch.max(image_3d, dim=1)
        return image
