"""Small, explicit helpers for immutable run directories and provenance."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Stream a file checksum, including large model weights."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    """Write strict JSON atomically; non-finite values must be handled explicitly."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def new_run(path: Path, config: dict) -> Path:
    """Refuse reuse, preserving both successful and failed experiments."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "config.json", config)
    write_json(
        path / "status.json",
        {
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
        },
    )
    return path
