import numpy as _np
import torch as _torch
import torch.nn as _nn
from torch.nn.parameter import Parameter as _Parameter

from wormtracer.types import _NP_T, _T
from wormtracer.parameter import ShapeParameters as _ShapeParams


class WormShapeLayer(_nn.Module):
    def __init__(self, *, alpha: float, delta: float, gamma: float):
        super(WormShapeLayer, self).__init__()
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
        super(WormSkeletonLayer, self).__init__()
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


class WormPixelLayer(_nn.Module):
    def __init__(self, *, contrast: float = 1.2, sharpness: float = 2.0):
        super(WormPixelLayer,self).__init__()
        self.contrast = _Parameter(_torch.tensor(contrast), requires_grad=False)
        self.sharpness = _Parameter(_torch.tensor(sharpness), requires_grad=False)
        self.scale = _Parameter(_torch.tensor(6.0), requires_grad=False)
        self.relu6 = _nn.ReLU6()
        self.sig = _nn.Sigmoid()

    def forward(self, im: _torch.Tensor) -> _torch.Tensor:
        im = self.sig(im * self.sharpness) * self.contrast
        return self.relu6(im) / self.scale


class WormImageLayer(_nn.Module):
    def __init__(
        self,
        *,
        width: int,
        height: int,
        pixel_layer: _T.Optional[WormPixelLayer] = None,
    ):  
        super(WormImageLayer,self).__init__()
        self.y_3d = _Parameter(_torch.arange(height).reshape(1, 1, -1, 1), requires_grad=False)
        self.x_3d = _Parameter(_torch.arange(width).reshape(1, 1, 1, -1), requires_grad=False)
        self.pixel_layer = pixel_layer or WormPixelLayer()

    def forward(
        self,
        skel: _torch.Tensor,
        shape: _torch.Tensor,
    ) -> _torch.Tensor:
        """Get pixel value when distance from midline and worm width is given."""
        # txy: [T, 2, n_pts]
        T = skel.size(dim=0)
        cent_mid = (skel[:, :, :-1] + skel[:, :, 1:]) * 0.5
        # cent_mid_3d: [T, 2, n_pts-1, 1, 1]
        cent_mid_3d = cent_mid.reshape(T, 2, -1, 1, 1)

        cent_mid_x_3d = cent_mid_3d[:, 0]
        cent_mid_y_3d = cent_mid_3d[:, 1]
        # worm_wid_3d: [1, n_pts-1, 1, 1]
        worm_wid_3d = shape.reshape(1, -1, 1, 1)
        
        # segment_distance_3d: [T, n_segs, im_height, im_width]
        # worm_wid_3d: [1, n_segs, 1, 1]
        segment_distance_3d = _torch.sqrt(
            _torch.pow(cent_mid_x_3d - self.x_3d, 2) + _torch.pow(cent_mid_y_3d - self.y_3d, 2)
        )

        # image_3d = [T, n_segs, im_height, im_width]
        delta_max, _ = _torch.max(segment_distance_3d - worm_wid_3d, dim=1)
        return self.pixel_layer(delta_max)


class WormModel(_nn.Module):
    def __init__(
        self,
        *,
        shape_layer: WormShapeLayer,
        skel_layer: WormSkeletonLayer,
        image_layer: WormImageLayer,
    ):
        super(WormModel, self).__init__()
        self.shape_layer = shape_layer
        self.skel_layer = skel_layer
        self.image_layer = image_layer

    def forward(self):
        # theta: shape [T, n_segs]
        txy = self.skel_layer()
        n_pts = txy.size(dim=2)
        n_segs = n_pts - 1
        worm_width = self.shape_layer(n_segs)
        image = self.image_layer(txy, worm_width)
        return image


def make_worm_batch(
    txy: _NP_T,
    pre_width: _NP_T,
    width: int,
    height: int,
    batchsize: int,
    device: str,
) -> _NP_T:
    """Create model image by dividing them to avoid CUDA memory error."""
    from numpy import split

    T = txy.shape[0]
    imagestack = _np.zeros((T, height, width))
    cut = _np.arange(0, T, step=batchsize)[1:]
    chunks = zip(
        split(imagestack, cut),
        split(txy, cut),
       
    )
    with _torch.no_grad():
        image_layer = WormImageLayer(width=width, height=height).to(device)
        for dst, t in chunks:
            im = image_layer(
                skel=_torch.from_numpy(t.copy()).to(device),
                shape=_torch.from_numpy(pre_width).to(device),
            )
            dst[:, :, :] = im.detach().cpu().numpy()

    return imagestack


def get_image_loss_max(
    im_ref: _NP_T,
    skel: _NP_T,
    pre_width: _NP_T,
) -> float:
    """Create bad image and get bad image_loss to judge complex area."""
    height, width = im_ref.shape
    image_layer = WormImageLayer(width=width, height=height)

    skel_bad = _np.ones_like(skel) * skel[:,0].reshape(2, 1)

    with _torch.no_grad():
        im_bad = image_layer(
            skel=_torch.from_numpy(skel_bad).reshape(1, 2, -1),
            shape=_torch.from_numpy(pre_width).reshape(1, -1),
        ).reshape(height, width).detach().numpy()
    image_loss_max = _np.mean((im_ref - im_bad) ** 2)
    return image_loss_max
