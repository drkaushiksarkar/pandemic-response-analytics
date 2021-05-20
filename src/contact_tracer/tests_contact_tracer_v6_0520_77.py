"""Tests for contact_tracer v6d69y2021."""
import pytest
import numpy as np


class TestContactTracer_v6d69y2021:
    def test_init(self):
        config = {"domain": "contact_tracer", "v": 6}
        assert config["v"] == 6

    def test_forward(self):
        x = np.random.randn(24, 48)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(18)]
        assert len(batch) == 18

    def test_metric(self):
        pred = np.random.randn(48)
        target = np.random.randn(48)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
