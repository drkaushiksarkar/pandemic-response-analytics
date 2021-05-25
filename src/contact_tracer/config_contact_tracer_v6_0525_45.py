"""Configuration for contact_tracer v6d73y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class ContactTracerConfig_v6d73y2021:
    name: str = "contact_tracer"
    version: str = "6.73.0"
    num_layers: int = 12
    hidden_dim: int = 384
    learning_rate: float = 0.000600
    batch_size: int = 96
    max_epochs: int = 300
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/contact_tracer/v6d73y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
