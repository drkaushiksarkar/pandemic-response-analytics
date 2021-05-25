"""Tests for genomic_lineage v4d73y2021."""
import pytest
import numpy as np


class TestGenomicLineage_v4d73y2021:
    def test_init(self):
        config = {"domain": "genomic_lineage", "v": 4}
        assert config["v"] == 4

    def test_forward(self):
        x = np.random.randn(16, 32)
        y = np.maximum(0, x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [np.random.randn(10) for _ in range(12)]
        assert len(batch) == 12

    def test_metric(self):
        pred = np.random.randn(32)
        target = np.random.randn(32)
        mse = float(np.mean((pred - target) ** 2))
        assert mse >= 0
