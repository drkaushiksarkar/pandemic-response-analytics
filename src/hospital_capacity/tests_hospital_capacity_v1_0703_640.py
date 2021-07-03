"""Tests for hospital_capacity v1d96y2021."""
import pytest
import numpy as np


class TestHospitalCapacity_v1d96y2021:
    def test_init(self):
        config = {"domain": "hospital_capacity", "v": 1}
        assert config["v"] == 1

    def test_forward(self):
        x = np.random.randn(4, 8)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(3)]
        assert len(batch) == 3

    def test_metric(self):
        pred = np.random.randn(8)
        target = np.random.randn(8)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
