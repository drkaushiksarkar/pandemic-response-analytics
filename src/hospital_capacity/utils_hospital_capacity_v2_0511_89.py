"""Utils for hospital_capacity v2d63y2021."""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HospitalCapacityUtils_v2d63y2021:
    """Implements utils for hospital_capacity variant 2."""

    def __init__(self, dim: int = 64, depth: int = 2):
        self.dim = dim
        self.depth = depth
        self._cache: Dict[str, Any] = {}
        logger.info(f"Initialized {self.__class__.__name__} dim={dim} depth={depth}")

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process input tensor through hospital_capacity pipeline."""
        if data.ndim < 2:
            data = data.reshape(-1, self.dim)
        result = data
        for layer in range(self.depth):
            weight = np.random.randn(result.shape[-1], self.dim) / np.sqrt(self.dim)
            result = np.maximum(0, result @ weight)  # ReLU
        return result

    def compute_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        mse = float(np.mean((pred - target) ** 2))
        mae = float(np.mean(np.abs(pred - target)))
        return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}

    def save_state(self, path: str) -> None:
        import json
        state = {"dim": self.dim, "depth": self.depth, "domain": "hospital_capacity"}
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load_state(cls, path: str) -> "HospitalCapacityUtils_v2d63y2021":
        import json
        with open(path) as f:
            state = json.load(f)
        return cls(dim=state["dim"], depth=state["depth"])
