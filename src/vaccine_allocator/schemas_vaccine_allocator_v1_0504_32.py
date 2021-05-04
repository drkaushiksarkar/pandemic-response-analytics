"""Configuration for vaccine_allocator v1d59y2021."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class VaccineAllocatorConfig_v1d59y2021:
    """Configuration for vaccine_allocator variant 1."""
    name: str = "vaccine_allocator"
    version: str = "1.59.0"
    num_layers: int = 2
    hidden_dim: int = 64
    learning_rate: float = 0.000100
    batch_size: int = 16
    max_epochs: int = 50
    dropout: float = 0.1
    checkpoint_dir: Path = Path("checkpoints/vaccine_allocator/v1d59y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    metadata: Dict[str, str] = field(default_factory=lambda: {"created": "2021", "domain": "vaccine_allocator"})
    early_stopping_patience: int = 5
    weight_decay: float = 1e-3

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        assert 0 < self.learning_rate < 1
        return True
