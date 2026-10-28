import copy
import sys
import types
from argparse import Namespace

import pytest
import torch
import torch.nn as nn


if "inplace_abn" not in sys.modules:
    inplace_abn_stub = types.ModuleType("inplace_abn")
    inplace_abn_stub.InPlaceABN = nn.BatchNorm2d
    inplace_abn_stub.ABN = nn.BatchNorm2d
    sys.modules["inplace_abn"] = inplace_abn_stub

from models.mlic import MLIC


class TinyBackbone(nn.Module):
    """Small real backbone; all CoCo modules after it remain unmodified."""

    def __init__(self, feature_dim=8):
        super().__init__()
        self.projection = nn.Conv2d(3, feature_dim, kernel_size=1)

    def forward(self, images):
        features = self.projection(images)
        return features.flatten(2).transpose(1, 2)


def _make_model(lamda=1.0):
    cfg = Namespace(
        num_classes=18,
        num_heads=2,
        enc_layers=1,
        dec_layers=1,
        num_steps=1,
        loss="bce",
        lamda=lamda,
        tau=0.7,
        gamma_neg=2.0,
        gamma_pos=0.0,
    )
    model = MLIC(TinyBackbone(feature_dim=8), feat_dim=8, cfg=cfg)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
    return model


def _targets(*kinds):
    targets = torch.zeros(len(kinds), 18, dtype=torch.float32)
    for index, kind in enumerate(kinds):
        if kind == "single":
            targets[index, 3] = 1.0
        elif kind == "multi":
            targets[index, [1, 7, 16]] = 1.0
        elif kind != "healthy":
            raise ValueError(kind)
    return targets


def _images(batch_size):
    generator = torch.Generator().manual_seed(1729 + batch_size)
    return torch.randn(batch_size, 3, 4, 4, generator=generator)


def _official_reference(model, images, targets):
    """Original CoCo math for batches where every sample has a positive label."""
    x = model.backbone(images)
    x_pool = torch.mean(x, dim=1)

    query_input = model.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)
    x = x.permute(1, 0, 2)
    x = model.attention(x, query_input)
    x = x.permute(1, 0, 2)

    x_pool = x_pool.unsqueeze(1).repeat(1, x.shape[1], 1)
    relation_input = torch.cat([x, x_pool], dim=-1)
    weights = torch.sigmoid(model.net(relation_input))

    ax, feat_list = model.encoder(x, weights)
    classification_input = torch.cat([x, ax], dim=-1)
    logits = model.fc(classification_input)

    masks = torch.bmm(targets.unsqueeze(-1), targets.unsqueeze(1))
    positive_weights = weights * masks
    negative_weights = weights * (1 - masks)
    with torch.no_grad():
        px, _ = model.encoder(x, positive_weights)
        nx, _ = model.encoder(x, negative_weights)

    contrastive = logits.new_zeros(())
    sample_logits = []
    for index in range(x.shape[0]):
        sample_target = targets[index]
        p_ax = ax[index][sample_target == 1, :]
        p_px = px[index][sample_target == 1, :]
        p_nx = nx[index][sample_target == 1, :]

        app_logits = torch.sum(p_ax * p_px, dim=-1) / model.cfg.tau
        ctr_logits = app_logits.unsqueeze(-1)
        anp_logits = torch.sum(p_ax * p_nx, dim=-1) / model.cfg.tau
        ctr_logits = torch.cat([ctr_logits, anp_logits.unsqueeze(-1)], dim=-1)

        aap_logits = torch.mm(p_ax, p_ax.transpose(0, 1)) / model.cfg.tau
        n = aap_logits.shape[0]
        aap_logits = (
            aap_logits.flatten()[1:]
            .view(n - 1, n + 1)[:, :-1]
            .reshape(n, n - 1)
        )
        ctr_logits = torch.cat([ctr_logits, aap_logits], dim=-1)
        sample_logits.append(ctr_logits)

        ce_targets = torch.zeros(
            ctr_logits.shape[0], dtype=torch.long, device=ctr_logits.device
        )
        contrastive = contrastive + nn.CrossEntropyLoss()(
            ctr_logits, ce_targets
        )

    contrastive = contrastive / x.shape[0]
    return {
        "ctr": contrastive,
        "ctr_logits": sample_logits,
        "logits": logits,
        "ax": ax,
        "feat_list": feat_list,
    }


def _capture_encoder_outputs(model, images, targets):
    outputs = []

    def capture(_module, _inputs, result):
        outputs.append(result[0])

    handle = model.encoder.register_forward_hook(capture)
    try:
        result = model(images, targets)
    finally:
        handle.remove()
    return result, outputs


def _contrastive_logits_from_captured(ax, px, nx, targets, tau):
    logits = []
    for index in range(targets.shape[0]):
        if targets[index].sum() == 0:
            continue
        selected = targets[index] == 1
        p_ax = ax[index][selected, :]
        p_px = px[index][selected, :]
        p_nx = nx[index][selected, :]
        first = (p_ax * p_px).sum(dim=-1).unsqueeze(-1) / tau
        second = (p_ax * p_nx).sum(dim=-1).unsqueeze(-1) / tau
        matrix = torch.mm(p_ax, p_ax.transpose(0, 1)) / tau
        count = matrix.shape[0]
        off_diagonal = (
            matrix.flatten()[1:]
            .view(count - 1, count + 1)[:, :-1]
            .reshape(count, count - 1)
        )
        logits.append(torch.cat([first, second, off_diagonal], dim=-1))
    return logits


@pytest.mark.parametrize(
    "kinds",
    [
        ("healthy", "healthy"),
        ("healthy", "single"),
        ("healthy", "multi"),
        ("single", "single"),
        ("multi", "multi"),
        ("healthy", "single", "multi"),
        ("healthy",),
        ("single",),
    ],
    ids=[
        "all-healthy",
        "healthy-single",
        "healthy-multi",
        "single-disease",
        "multi-disease",
        "healthy-single-multi",
        "batch1-healthy",
        "batch1-single",
    ],
)
def test_training_batch_contrastive_loss_is_finite_on_cpu(kinds):
    torch.manual_seed(100)
    model = _make_model().train()

    result = model(_images(len(kinds)), _targets(*kinds))

    assert result["ctr"].device.type == "cpu"
    assert result["ctr"].ndim == 0
    assert torch.isfinite(result["ctr"])
    assert result["logits"].shape == (len(kinds), 18)


def test_all_healthy_batch_keeps_classification_backward_active():
    torch.manual_seed(101)
    model = _make_model().train()
    targets = _targets("healthy", "healthy")

    result = model(_images(2), targets)
    total_loss = result["bce"] + model.cfg.lamda * result["ctr"]
    total_loss.backward()

    assert result["ctr"].item() == 0.0
    assert torch.isfinite(result["bce"])
    assert model.backbone.projection.weight.grad is not None
    assert torch.isfinite(model.backbone.projection.weight.grad).all()


def test_lamda_zero_skips_contrastive_path_and_returns_tensor_zero():
    torch.manual_seed(102)
    model = _make_model(lamda=0.0).train()

    result = model(_images(2), _targets("healthy", "multi"))

    assert isinstance(result["ctr"], torch.Tensor)
    assert result["ctr"].item() == 0.0
    assert torch.isfinite(result["bce"])


def test_eval_does_not_require_targets_and_preserves_return_interface():
    torch.manual_seed(103)
    model = _make_model().eval()

    with torch.no_grad():
        result = model(_images(2))

    assert set(result) == {"logits", "bce", "ctr", "ax", "feat_list"}
    assert result["logits"].shape == (2, 18)
    assert isinstance(result["ctr"], torch.Tensor)
    assert result["ctr"].item() == 0.0


@pytest.mark.parametrize(
    "kinds", [("single",), ("multi",), ("single", "multi")]
)
def test_all_valid_batch_matches_official_loss_logits_and_intermediates(kinds):
    torch.manual_seed(104)
    actual_model = _make_model().train()
    reference_model = copy.deepcopy(actual_model).train()
    images = _images(len(kinds))
    targets = _targets(*kinds)

    actual, encoder_outputs = _capture_encoder_outputs(
        actual_model, images, targets
    )
    reference = _official_reference(reference_model, images, targets)
    actual_ctr_logits = _contrastive_logits_from_captured(
        encoder_outputs[0],
        encoder_outputs[1],
        encoder_outputs[2],
        targets,
        actual_model.cfg.tau,
    )

    torch.testing.assert_close(actual["ctr"], reference["ctr"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        actual["logits"], reference["logits"], rtol=1e-6, atol=1e-7
    )
    assert len(actual_ctr_logits) == len(reference["ctr_logits"])
    for actual_logits, reference_logits in zip(
        actual_ctr_logits, reference["ctr_logits"]
    ):
        torch.testing.assert_close(
            actual_logits, reference_logits, rtol=1e-6, atol=1e-7
        )


def test_all_valid_batch_contrastive_gradients_match_official_reference():
    torch.manual_seed(105)
    actual_model = _make_model().train()
    reference_model = copy.deepcopy(actual_model).train()
    images = _images(3)
    targets = _targets("single", "multi", "single")

    actual_loss = actual_model(images, targets)["ctr"]
    reference_loss = _official_reference(reference_model, images, targets)["ctr"]
    actual_loss.backward()
    reference_loss.backward()

    actual_grads = dict(actual_model.named_parameters())
    reference_grads = dict(reference_model.named_parameters())
    compared = 0
    for name, actual_parameter in actual_grads.items():
        reference_parameter = reference_grads[name]
        if actual_parameter.grad is None and reference_parameter.grad is None:
            continue
        assert actual_parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda message, name=name: "{}: {}".format(name, message),
        )
        compared += 1
    assert compared > 0


@pytest.mark.parametrize(
    "kinds", [
        ("healthy", "single"),
        ("healthy", "multi"),
        ("healthy", "single", "multi"),
    ]
)
def test_mixed_batch_averages_contrastive_loss_over_valid_samples(kinds):
    torch.manual_seed(106)
    actual_model = _make_model().train()
    reference_model = copy.deepcopy(actual_model).train()
    images = _images(len(kinds))
    targets = _targets(*kinds)
    valid = targets.sum(dim=1) > 0

    actual = actual_model(images, targets)
    reference = _official_reference(
        reference_model, images[valid], targets[valid]
    )

    torch.testing.assert_close(actual["ctr"], reference["ctr"], rtol=1e-6, atol=1e-7)
