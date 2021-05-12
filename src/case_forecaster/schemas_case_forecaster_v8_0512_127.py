"""Configuration for case_forecaster v8d64y2021."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class CaseForecasterConfig_v8d64y2021:
    """Configuration for case_forecaster variant 8."""
    name: str = "case_forecaster"
    version: str = "8.64.0"
    num_layers: int = 16
    hidden_dim: int = 512
    learning_rate: float = 0.000800
    batch_size: int = 128
    max_epochs: int = 400
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/case_forecaster/v8d64y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    metadata: Dict[str, str] = field(default_factory=lambda: {"created": "2021", "domain": "case_forecaster"})
    early_stopping_patience: int = 40
    weight_decay: float = 1e-10

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        assert 0 < self.learning_rate < 1
        return True
