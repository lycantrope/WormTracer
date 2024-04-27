import typing as _T

import attrs as _attrs
import numpy as np
import torch

import wormtracer.loss as _loss
from wormtracer.model import WormModel as _Model
from wormtracer.parameter import HyperParameters as _HyperParams

_NP_T: _T.TypeAlias = np.ndarray


@_attrs.define
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How long to wait after last time validation loss improved.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """

    patience: int = _attrs.field(default=30)
    delta: float = _attrs.field(default=0)
    counter: int = _attrs.field(init=False, default=0)
    best_loss: float = _attrs.field(init=False, default=np.inf)

    def __call__(self, loss) -> bool:
        if loss <= self.best_loss + self.delta:
            self.best_loss = min(self.best_loss, loss)
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            return True

    def reset(self):
        self.counter = 0
        self.best_loss = np.inf


def train3(
    model: _Model,
    real_image: np.ndarray,
    optimizer,
    params: _HyperParams,
    is_complex=True,
) -> _T.Tuple[_Model, _NP_T, _NP_T, _loss.WormLosses]:
    T = real_image.shape[0]
    speed = params.speed
    epochs = int(T / (2 * params.speed) + params.epoch_plus)
    continuity_loss_weight = params.continuity_loss_weight
    smoothness_loss_weight = params.smoothness_loss_weight
    length_loss_weight = params.length_loss_weight
    center_loss_weight = params.center_loss_weight

    init_ct = model.skel_layer.ct.clone()
    unit_l = model.skel_layer.unit_length.clone()

    early_stopping = EarlyStopping()

    real_image = torch.from_numpy(real_image)

    annealing_fn = None
    if is_complex:
        annealing_fn = _loss.AnnealingFunction(T, speed)

    im_loss_fn = _loss.ImageLoss()
    smo_loss_fn = _loss.SmoothnessLoss(
        smoothness_loss_weight=smoothness_loss_weight,
        body_axis_weight=_loss.body_axis_function(
            body_ratio=params.body_ratio,
            n_segments=params.n_segments,
        ),
    )
    cont_loss_fn = _loss.ContinuityLoss(continuity_loss_weight)

    length_loss_fn = _loss.LengthLoss(10000 * length_loss_weight)
    center_loss_fn = _loss.CenterLoss(unit_l, center_loss_weight)

    model.shape_layer.requires_grad_(False)
    # main optimization
    for e in range(epochs):
        model_image = model()
        skel_layer = model.skel_layer

        img_loss = im_loss_fn(model_image, real_image)
        smo_loss = smo_loss_fn(skel_layer.theta)
        cont_loss = cont_loss_fn(skel_layer.theta)
        img_smo_loss = img_loss + smo_loss
        if annealing_fn is not None:
            img_smo_loss = img_smo_loss * annealing_fn.get_weights(e)
        img_smo_loss = img_smo_loss.mean()

        length_loss = length_loss_fn(skel_layer.unit_length)
        center_loss = center_loss_fn((skel_layer.ct), init_ct)

        loss = img_smo_loss + cont_loss + length_loss + center_loss

        optimizer.zero_grad()
        loss.backward()
        if annealing_fn.get_weights(e).min() > 0.99 and early_stopping(loss.item()):
            break
        optimizer.step()

    model.shape_layer.requires_grad_(True)

    early_stopping.reset()

    smo_loss_fn = _loss.SmoothnessLoss(
        smoothness_loss_weight=smoothness_loss_weight,
        body_axis_weight=_loss.body_axis_function(
            params.body_ratio,
            params.n_segs,
            base=0.3,
        ),
    )
    length_loss_fn = _loss.LengthLoss(1e5 * length_loss_weight)
    center_loss_fn = _loss.CenterLoss(1e-1, center_loss_weight)

    # minor adjustment
    for e in range(params.epoch_plus):
        model_image = model()
        skel_layer = model.skel_layer
        img_loss = im_loss_fn(model_image, real_image)
        smo_loss = smo_loss_fn(skel_layer.theta)
        length_loss = length_loss_fn(skel_layer.unit_length)

        loss = (img_loss + smo_loss).mean() + smo_loss + length_loss

        optimizer.zero_grad()
        loss.backward()
        if early_stopping(loss.item()):
            break
        optimizer.step()

    with torch.no_grad():
        model_image = model()
        skel_layer = model.skel_layer
        txy = skel_layer().detach().cpu().numpy()
        losses = {
            "image": _loss.ImageLoss()(model_image, real_image),
            "continuity": _loss.ContinuityLoss(continuity_loss_weight)(
                skel_layer.theta
            ),
            "smoothness": _loss.SmoothnessLoss(
                body_axis_weight=1.0, smoothness_weight=smoothness_loss_weight
            )(skel_layer.theta),
            "length": _loss.LengthLoss(length_loss_weight)(skel_layer.unit_length),
            "center": _loss.CenterLoss(1.0, center_loss_weight)(
                skel_layer.ct,
                init_ct,
            ),
        }

        losses = _loss.WormLosses(
            **{k: v.clone().detach().cpu().numpy() for k, v in losses.items()}
        )

    return model, model_image, txy, losses


if __name__ == "__main__":
    pass
