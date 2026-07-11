"""Small helpers for reading and writing the pipeline's JSON files."""

import json
import os
import sys
from typing import Any


def load_json(path: str) -> Any:
    """Load and parse a JSON file, exiting with a message on failure.

    Args:
        path: Path to the JSON file to load.

    Returns:
        The parsed JSON content.
    """
    try:
        with open(path) as f:
            return json.load(f)

    except FileNotFoundError:
        sys.exit(f"Error file not found at: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error invalid JSON in {path}: {e}")


def save_json(
    data: Any, path: str = "data/output/function_calling_results.json"
) -> None:
    """Write ``data`` to ``path`` as JSON, creating parent dirs as needed.

    Args:
        data: JSON-serializable data to write.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
