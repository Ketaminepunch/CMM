from typing import Any
import json
import sys
import os


def load_json(path: str) -> Any:
    try:
        with open(path) as f:
            return json.load(f)

    except FileNotFoundError:
        sys.exit(f"Error file not found at: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error invalid JSON in {path}: {e}")


def save_json(data: Any,
              path: str = "data/output/function_calling_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
