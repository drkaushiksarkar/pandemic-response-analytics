"""Tests for testing_optimizer v3d90y2021."""
import pytest
import numpy as np


class TestTestingOptimizer_v3d90y2021:
    def test_init(self):
        config = {"domain": "testing_optimizer", "v": 3}
        assert config["v"] == 3

    def test_forward(self):
        x = np.random.randn(12, 24)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(9)]
        assert len(batch) == 9

    def test_metric(self):
        pred = np.random.randn(24)
        target = np.random.randn(24)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
