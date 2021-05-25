"""Configuration for case_forecaster v8d73y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class CaseForecasterConfig_v8d73y2021:
    name: str = "case_forecaster"
    version: str = "8.73.0"
    num_layers: int = 16
    hidden_dim: int = 512
    learning_rate: float = 0.000800
    batch_size: int = 128
    max_epochs: int = 400
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/case_forecaster/v8d73y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
