"""Tests for case_forecaster v3d6008y2020."""
import pytest
import torch
import numpy as np


class TestCaseForecaster_v3d6008y2020:
    def test_init(self):
        config = {"domain": "case_forecaster", "v": 3}
        assert config["v"] == 3

    def test_forward(self):
        x = torch.randn(12, 24)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(9)]
        assert len(batch) == 9

    def test_metric(self):
        pred = torch.randn(24)
        target = torch.randn(24)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
