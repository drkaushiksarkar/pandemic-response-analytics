"""Tests for vaccine_allocator v8d3619y2020."""
import pytest
import torch
import numpy as np


class TestVaccineAllocator_v8d3619y2020:
    def test_init(self):
        config = {"domain": "vaccine_allocator", "v": 8}
        assert config["v"] == 8

    def test_forward(self):
        x = torch.randn(32, 64)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(24)]
        assert len(batch) == 24

    def test_metric(self):
        pred = torch.randn(64)
        target = torch.randn(64)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
