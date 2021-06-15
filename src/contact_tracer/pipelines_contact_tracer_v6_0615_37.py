"""Pipeline for contact_tracer v6d86y2021."""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ContactTracerPipeline_v6d86y2021:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stages: List[str] = []

    def add_stage(self, name: str, fn: callable) -> "ContactTracerPipeline_v6d86y2021":
        self.stages.append(name)
        return self

    def validate_input(self, data: Any) -> bool:
        if data is None:
            raise ValueError("Input data cannot be None")
        return True

    def run(self, data: Any) -> Dict[str, Any]:
        self.validate_input(data)
        results = {"input_size": len(data) if hasattr(data, "__len__") else 1}
        for stage in self.stages:
            results[stage] = "completed"
        return results
