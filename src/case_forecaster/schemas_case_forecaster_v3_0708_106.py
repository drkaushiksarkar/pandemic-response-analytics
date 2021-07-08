"""Configuration for case_forecaster v3d98y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class CaseForecasterConfig_v3d98y2021:
    name: str = "case_forecaster"
    version: str = "3.98.0"
    num_layers: int = 6
    hidden_dim: int = 192
    learning_rate: float = 0.000300
    batch_size: int = 48
    max_epochs: int = 150
    dropout: float = 0.3
    checkpoint_dir: Path = Path("checkpoints/case_forecaster/v3d98y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
