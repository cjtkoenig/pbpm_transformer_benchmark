"""
PGTNetAdapter for RemainingTimeTask (external runner integration)

This adapter is a thin validation and pass-through layer used by the task integration.
It enforces that pgtnet is only invoked for task='remaining_time' and data.attribute_mode='extended'.
It does not perform any tensor conversion.
"""
from __future__ import annotations

from typing import Dict, Any


class PGTNetAdapter:
    def __init__(self, task: str, config: Dict[str, Any]):
        if task != "remaining_time":
            raise NotImplementedError("PGTNetAdapter supports only task='remaining_time'")
        attr_mode = (config.get("data", {}) or {}).get("attribute_mode", "extended")
        if attr_mode != "extended":
            raise NotImplementedError("PGTNet requires data.attribute_mode='extended'")
        self.config = config

    def prepare(self, dataset_name: str) -> Dict[str, Any]:
        """Return a dict with paths/seeds suitable to pass into the external runner.
        This keeps the adapter contract light while enabling future extensions.
        """
        model_cfg = self.config.get("model", {}) or {}
        pgtnet_cfg = model_cfg.get("pgtnet", {}) if model_cfg.get("name") == "pgtnet" else model_cfg
        return {
            "dataset_name": dataset_name,
            "pgtnet": pgtnet_cfg,
            "seeds": model_cfg.get("seeds") or [self.config.get("seed", 42)],
        }
