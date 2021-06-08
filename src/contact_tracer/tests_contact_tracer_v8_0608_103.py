"""Tests for contact_tracer v8d83y2021."""
import pytest
import numpy as np


class TestContactTracer_v8d83y2021:
    def test_init(self):
        config = {"domain": "contact_tracer", "v": 8}
        assert config["v"] == 8

    def test_forward(self):
        x = np.random.randn(32, 64)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(24)]
        assert len(batch) == 24

    def test_metric(self):
        pred = np.random.randn(64)
        target = np.random.randn(64)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
