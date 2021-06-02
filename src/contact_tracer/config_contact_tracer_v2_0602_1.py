"""Configuration for contact_tracer v2d77y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class ContactTracerConfig_v2d77y2021:
    name: str = "contact_tracer"
    version: str = "2.77.0"
    num_layers: int = 4
    hidden_dim: int = 128
    learning_rate: float = 0.000200
    batch_size: int = 32
    max_epochs: int = 100
    dropout: float = 0.2
    checkpoint_dir: Path = Path("checkpoints/contact_tracer/v2d77y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
