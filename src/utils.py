"""
Utility functions for LLM Fine-Tuning and Evaluation Benchmark.
"""

import os
import random
import json
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic seeds for random, numpy, and torch across CPU and GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Returns the primary torch device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_compute_dtype(preferred: str = "bfloat16") -> torch.dtype:
    """
    Returns an appropriate compute dtype based on hardware support.
    """
    if not torch.cuda.is_available():
        return torch.float32
    
    if preferred == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif preferred in ("bfloat16", "float16"):
        return torch.float16
    return torch.float32


def print_gpu_memory(tag: str = "") -> Dict[str, float]:
    """
    Logs and returns GPU memory allocation and reservation in Megabytes.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    prefix = f"[{tag}] " if tag else ""
    print(
        f"{prefix}GPU Memory: Allocated = {allocated:.2f} MB | "
        f"Reserved = {reserved:.2f} MB | Max Allocated = {max_allocated:.2f} MB"
    )
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated
    }


def calculate_parameter_statistics(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Computes total parameters, trainable parameters, and percentage trainable.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": trainable_percent,
        "frozen_params": total_params - trainable_params
    }


def save_json(data: Any, filepath: str) -> None:
    """Safely saves serializable data to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Loads JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable duration."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    elif m > 0:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.2f}s"
