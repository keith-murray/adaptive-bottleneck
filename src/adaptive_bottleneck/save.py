import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import jax


def _to_jsonable(obj):
    """Recursively convert JAX/NumPy scalars/arrays to Python types for JSON."""
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    # JAX/NumPy scalar
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    # dict
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # array-like
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    return str(obj)


def save_results_json(results: dict, save_path: str):
    """Save `results` (nested dict of python lists/scalars) to a JSON file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_to_jsonable(results), f, indent=2)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(obj: Any, path: Path) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _save_params_pkl(params: Any, path: Path) -> None:
    """Save a JAX pytree of arrays to disk.

    We device_get() to move to host numpy arrays and then pickle.
    """
    host_params = jax.device_get(params)
    with path.open("wb") as f:
        pickle.dump(host_params, f)
