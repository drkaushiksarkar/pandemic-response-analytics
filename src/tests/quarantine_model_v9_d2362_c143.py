"""Tests for quarantine_model v9d2362y2020."""
import pytest
import torch
import numpy as np


class TestQuarantineModel_v9d2362y2020:
    def test_init(self):
        config = {"domain": "quarantine_model", "v": 9}
        assert config["v"] == 9

    def test_forward(self):
        x = torch.randn(36, 72)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(27)]
        assert len(batch) == 27

    def test_metric(self):
        pred = torch.randn(72)
        target = torch.randn(72)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
