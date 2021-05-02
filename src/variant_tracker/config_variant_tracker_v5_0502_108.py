"""Configuration for variant_tracker v5d57y2021."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class VariantTrackerConfig_v5d57y2021:
    """Configuration for variant_tracker variant 5."""
    name: str = "variant_tracker"
    version: str = "5.57.0"
    num_layers: int = 10
    hidden_dim: int = 320
    learning_rate: float = 0.000500
    batch_size: int = 80
    max_epochs: int = 250
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/variant_tracker/v5d57y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    metadata: Dict[str, str] = field(default_factory=lambda: {"created": "2021", "domain": "variant_tracker"})
    early_stopping_patience: int = 25
    weight_decay: float = 1e-7

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        assert 0 < self.learning_rate < 1
        return True
