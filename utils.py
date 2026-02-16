# utils.py
from __future__ import annotations

import os
import random
from typing import Any, Dict

import joblib
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_model(model: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)


def pretty_print_dict(d: Dict[str, Any], title: str | None = None) -> None:
    if title:
        print(f"\n{title} ")
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{k:>18s}: {v:.4f}")
        else:
            print(f"{k:>18s}: {v}")
