"""Configuration for contact_tracer v8d62y2021."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ContactTracerConfig_v8d62y2021:
    """Configuration for contact_tracer variant 8."""
    name: str = "contact_tracer"
    version: str = "8.62.0"
    num_layers: int = 16
    hidden_dim: int = 512
    learning_rate: float = 0.000800
    batch_size: int = 128
    max_epochs: int = 400
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/contact_tracer/v8d62y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    metadata: Dict[str, str] = field(default_factory=lambda: {"created": "2021", "domain": "contact_tracer"})
    early_stopping_patience: int = 40
    weight_decay: float = 1e-10

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        assert 0 < self.learning_rate < 1
        return True
