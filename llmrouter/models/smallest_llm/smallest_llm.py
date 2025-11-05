from typing import Any, Dict, Optional

import torch.nn as nn
from llmrouter.models.meta_router import MetaRouter


def parse_size(size_str: str) -> float:
    """
    Convert a model size string (e.g., '7B', '13B', '512M') into a numeric value in billions.
    Supports unit suffixes:
        - K: thousands
        - M: millions
        - B: billions
        - T: trillions
    Returns 0.0 if parsing fails.
    """
    size_str = str(size_str).strip().upper()
    try:
        if size_str.endswith("K"):
            return float(size_str[:-1]) / 1e6
        elif size_str.endswith("M"):
            return float(size_str[:-1]) / 1e3
        elif size_str.endswith("B"):
            return float(size_str[:-1])
        elif size_str.endswith("T"):
            return float(size_str[:-1]) * 1e3
        else:
            return float(size_str)
    except Exception:
        return 0.0


class SmallestLLM(MetaRouter):
    """
    SmallestLLM Router
    ------------------
    A heuristic router that always selects the smallest LLM based on
    the 'size' field in `self.llm_data`, restricted to models whose
    size string ends with 'B'.

    This router does not require training and ignores input batches.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the SmallestLLM router.

        Args:
            yaml_path (str):
                Path to YAML configuration containing `llm_data`.
        """
        dummy_model = nn.Identity()  # This router doesn't use a real model
        super().__init__(model=dummy_model, yaml_path=yaml_path)
        print("✅ SmallestLLM initialized successfully")

    def route(self, batch: Optional[Any] = None) -> Dict[str, Any]:
        """
        Select the smallest LLM (by size) whose size string ends with 'B'.

        Args:
            batch (Any, optional):
                Unused input. Present for interface compatibility.

        Returns:
            dict:
                {
                    "model_name": name of the selected model,
                    "model_size": size string,
                    "model_info": full metadata entry
                }
        """
        if not hasattr(self, "llm_data") or not self.llm_data:
            raise ValueError(
                "LLM data not loaded or missing in YAML configuration. "
                "Expected `self.llm_data` to be populated by DataLoader."
            )

        # Filter only models with size ending in 'B'
        filtered_names = [
            name
            for name, info in self.llm_data.items()
            if isinstance(info.get("size", ""), str)
            and info["size"].upper().endswith("B")
        ]

        if not filtered_names:
            raise ValueError(
                "No models with size ending in 'B' found in `llm_data`."
            )

        # Find the smallest model
        smallest_model_name = min(
            filtered_names,
            key=lambda k: parse_size(self.llm_data[k].get("size", "0B")),
        )

        smallest_model = self.llm_data[smallest_model_name]
        print(
            f"🐜 Smallest model (ending with 'B') selected: "
            f"{smallest_model_name} ({smallest_model.get('size')})"
        )

        return {
            "model_name": smallest_model_name,
            "model_size": smallest_model.get("size"),
            "model_info": smallest_model,
        }
