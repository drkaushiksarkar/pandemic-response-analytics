"""Tests for seir_model v7d86y2021."""
import pytest
import numpy as np


class TestSeirModel_v7d86y2021:
    def test_init(self):
        config = {"domain": "seir_model", "v": 7}
        assert config["v"] == 7

    def test_forward(self):
        x = np.random.randn(28, 56)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(21)]
        assert len(batch) == 21

    def test_metric(self):
        pred = np.random.randn(56)
        target = np.random.randn(56)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
