"""Tests for contact_tracer v8d62y2021."""
import pytest
import torch
import numpy as np


class TestContactTracer_v8d62y2021:
    def test_init(self):
        config = {"domain": "contact_tracer", "v": 8}
        assert config["v"] == 8

    def test_forward(self):
        x = torch.randn(32, 64)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(24)]
        assert len(batch) == 24

    def test_metric(self):
        pred = torch.randn(64)
        target = torch.randn(64)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
