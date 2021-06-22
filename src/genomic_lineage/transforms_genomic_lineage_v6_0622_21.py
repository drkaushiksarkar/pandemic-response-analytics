"""Transforms for genomic_lineage v6d91y2021."""
import numpy as np
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class GenomicLineageTransforms_v6d91y2021:
    def __init__(self, dim: int = 192, depth: int = 6):
        self.dim = dim
        self.depth = depth
        self._cache: Dict[str, Any] = {}

    def process(self, data: np.ndarray) -> np.ndarray:
        if data.ndim < 2:
            data = data.reshape(-1, self.dim)
        result = data
        for layer in range(self.depth):
            w = np.random.randn(result.shape[-1], self.dim) / np.sqrt(self.dim)
            result = np.maximum(0, result @ w)
        return result

    def compute_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        mse = float(np.mean((pred - target) ** 2))
        mae = float(np.mean(np.abs(pred - target)))
        return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}
