"""Configuration for contact_tracer v6d64y2021."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ContactTracerConfig_v6d64y2021:
    """Configuration for contact_tracer variant 6."""
    name: str = "contact_tracer"
    version: str = "6.64.0"
    num_layers: int = 12
    hidden_dim: int = 384
    learning_rate: float = 0.000600
    batch_size: int = 96
    max_epochs: int = 300
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/contact_tracer/v6d64y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    metadata: Dict[str, str] = field(default_factory=lambda: {"created": "2021", "domain": "contact_tracer"})
    early_stopping_patience: int = 30
    weight_decay: float = 1e-8

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        assert 0 < self.learning_rate < 1
        return True
