"""Regression tests for tensor_to_pil_strip() — prevents the border_color crash from returning.

Tests cover every call signature used in qpred/dashboard.py.
"""
from __future__ import annotations
import numpy as np
import pytest
import torch
from PIL import Image


def _strip():
    from qpred.dashboard import tensor_to_pil_strip
    return tensor_to_pil_strip


@pytest.fixture()
def seq10():
    return torch.rand(10, 1, 64, 64)


def test_basic_no_args(seq10):
    s = _strip()(seq10)
    assert isinstance(s, Image.Image) and s.mode == "RGB"


def test_with_title(seq10):
    assert isinstance(_strip()(seq10, title="X"), Image.Image)


def test_with_start_t(seq10):
    assert isinstance(_strip()(seq10, start_t=11), Image.Image)


def test_cmap_magma(seq10):
    assert isinstance(_strip()(seq10, cmap_name="magma"), Image.Image)


def test_border_color_context(seq10):
    """Exact call that caused TypeError in dashboard line 578."""
    s = _strip()(seq10, title="CONTEXT", start_t=1, border_color=(80, 60, 160))
    assert isinstance(s, Image.Image)


def test_border_color_pred(seq10):
    """Exact call from dashboard line 677."""
    s = _strip()(seq10, title="PREDICTION", start_t=11, border_color=(100, 40, 160))
    assert isinstance(s, Image.Image)


def test_border_color_gt(seq10):
    s = _strip()(seq10, title="GROUND TRUTH", start_t=11, border_color=(30, 100, 70))
    assert isinstance(s, Image.Image)


def test_border_color_none_explicit(seq10):
    assert isinstance(_strip()(seq10, border_color=None), Image.Image)


def test_scale_2(seq10):
    s1 = _strip()(seq10, scale=1)
    s2 = _strip()(seq10, scale=2)
    assert s2.width > s1.width and s2.height > s1.height


def test_scale_4(seq10):
    assert isinstance(_strip()(seq10, scale=4), Image.Image)


def test_combined_border_and_scale(seq10):
    s = _strip()(seq10, title="CTX", start_t=1, border_color=(60, 40, 140), scale=2)
    assert isinstance(s, Image.Image)


def test_cmap_with_border_and_scale(seq10):
    s = _strip()(seq10, title="ERR", cmap_name="magma", border_color=(180, 50, 50), scale=2)
    assert isinstance(s, Image.Image)


def test_empty_tensor_no_crash():
    s = _strip()(torch.zeros(0, 1, 64, 64))
    assert isinstance(s, Image.Image)


def test_single_frame_no_crash():
    """(C,H,W) single frame should not crash."""
    s = _strip()(torch.rand(1, 64, 64))
    assert isinstance(s, Image.Image)


def test_output_rgb(seq10):
    assert _strip()(seq10).mode == "RGB"


def test_pixels_valid_range(seq10):
    arr = np.array(_strip()(seq10))
    assert arr.min() >= 0 and arr.max() <= 255


def test_more_frames_wider():
    s5 = _strip()(torch.rand(5, 1, 64, 64))
    s10 = _strip()(torch.rand(10, 1, 64, 64))
    assert s10.width > s5.width


def test_sequences_differ():
    s0 = np.array(_strip()(torch.zeros(5, 1, 32, 32)))
    s1 = np.array(_strip()(torch.ones(5, 1, 32, 32)))
    assert not (s0 == s1).all()
