"""Configuration for genomic_lineage v3d59y2021."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class GenomicLineageConfig_v3d59y2021:
    """Configuration for genomic_lineage variant 3."""
    name: str = "genomic_lineage"
    version: str = "3.59.0"
    num_layers: int = 6
    hidden_dim: int = 192
    learning_rate: float = 0.000300
    batch_size: int = 48
    max_epochs: int = 150
    dropout: float = 0.3
    checkpoint_dir: Path = Path("checkpoints/genomic_lineage/v3d59y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    metadata: Dict[str, str] = field(default_factory=lambda: {"created": "2021", "domain": "genomic_lineage"})
    early_stopping_patience: int = 15
    weight_decay: float = 1e-5

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        assert 0 < self.learning_rate < 1
        return True
