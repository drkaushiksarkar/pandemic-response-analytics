"""Configuration for testing_optimizer v7d90y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class TestingOptimizerConfig_v7d90y2021:
    name: str = "testing_optimizer"
    version: str = "7.90.0"
    num_layers: int = 14
    hidden_dim: int = 448
    learning_rate: float = 0.000700
    batch_size: int = 112
    max_epochs: int = 350
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/testing_optimizer/v7d90y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
