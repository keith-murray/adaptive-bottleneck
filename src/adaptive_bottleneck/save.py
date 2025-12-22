import json
from pathlib import Path

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