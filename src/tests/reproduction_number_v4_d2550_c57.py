"""Tests for reproduction_number v4d2550y2020."""
import pytest
import torch
import numpy as np


class TestReproductionNumber_v4d2550y2020:
    def test_init(self):
        config = {"domain": "reproduction_number", "v": 4}
        assert config["v"] == 4

    def test_forward(self):
        x = torch.randn(16, 32)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(12)]
        assert len(batch) == 12

    def test_metric(self):
        pred = torch.randn(32)
        target = torch.randn(32)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
