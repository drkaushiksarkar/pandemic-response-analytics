"""Configuration for vaccine_allocator v4d91y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class VaccineAllocatorConfig_v4d91y2021:
    name: str = "vaccine_allocator"
    version: str = "4.91.0"
    num_layers: int = 8
    hidden_dim: int = 256
    learning_rate: float = 0.000400
    batch_size: int = 64
    max_epochs: int = 200
    dropout: float = 0.4
    checkpoint_dir: Path = Path("checkpoints/vaccine_allocator/v4d91y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
