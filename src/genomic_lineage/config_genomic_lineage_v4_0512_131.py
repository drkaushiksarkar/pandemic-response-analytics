"""Configuration for genomic_lineage v4d64y2021."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class GenomicLineageConfig_v4d64y2021:
    """Configuration for genomic_lineage variant 4."""
    name: str = "genomic_lineage"
    version: str = "4.64.0"
    num_layers: int = 8
    hidden_dim: int = 256
    learning_rate: float = 0.000400
    batch_size: int = 64
    max_epochs: int = 200
    dropout: float = 0.4
    checkpoint_dir: Path = Path("checkpoints/genomic_lineage/v4d64y2021")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    metadata: Dict[str, str] = field(default_factory=lambda: {"created": "2021", "domain": "genomic_lineage"})
    early_stopping_patience: int = 20
    weight_decay: float = 1e-6

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        assert 0 < self.learning_rate < 1
        return True
