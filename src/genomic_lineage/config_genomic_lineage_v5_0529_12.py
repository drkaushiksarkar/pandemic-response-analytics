"""Configuration for genomic_lineage v5d76y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class GenomicLineageConfig_v5d76y2021:
    name: str = "genomic_lineage"
    version: str = "5.76.0"
    num_layers: int = 10
    hidden_dim: int = 320
    learning_rate: float = 0.000500
    batch_size: int = 80
    max_epochs: int = 250
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/genomic_lineage/v5d76y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
