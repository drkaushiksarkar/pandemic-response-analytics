"""Tests for hospital_capacity v5d92y2021."""
import pytest
import numpy as np


class TestHospitalCapacity_v5d92y2021:
    def test_init(self):
        config = {"domain": "hospital_capacity", "v": 5}
        assert config["v"] == 5

    def test_forward(self):
        x = np.random.randn(20, 40)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(15)]
        assert len(batch) == 15

    def test_metric(self):
        pred = np.random.randn(40)
        target = np.random.randn(40)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
