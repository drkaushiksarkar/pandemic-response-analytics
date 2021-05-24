"""Configuration for quarantine_model v1d72y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class QuarantineModelConfig_v1d72y2021:
    name: str = "quarantine_model"
    version: str = "1.72.0"
    num_layers: int = 2
    hidden_dim: int = 64
    learning_rate: float = 0.000100
    batch_size: int = 16
    max_epochs: int = 50
    dropout: float = 0.1
    checkpoint_dir: Path = Path("checkpoints/quarantine_model/v1d72y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
