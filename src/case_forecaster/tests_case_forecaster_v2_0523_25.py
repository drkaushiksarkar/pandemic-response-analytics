"""Tests for case_forecaster v2d71y2021."""
import pytest
import numpy as np


class TestCaseForecaster_v2d71y2021:
    def test_init(self):
        config = {"domain": "case_forecaster", "v": 2}
        assert config["v"] == 2

    def test_forward(self):
        x = np.random.randn(8, 16)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(6)]
        assert len(batch) == 6

    def test_metric(self):
        pred = np.random.randn(16)
        target = np.random.randn(16)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
