"""Configuration for genomic_lineage v4d69y2021."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class GenomicLineageConfig_v4d69y2021:
    name: str = "genomic_lineage"
    version: str = "4.69.0"
    num_layers: int = 8
    hidden_dim: int = 256
    learning_rate: float = 0.000400
    batch_size: int = 64
    max_epochs: int = 200
    dropout: float = 0.4
    checkpoint_dir: Path = Path("checkpoints/genomic_lineage/v4d69y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
