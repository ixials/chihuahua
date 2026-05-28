"""Output directory layout helpers."""

from pathlib import Path


def intermediate_dir(run_dir: Path) -> Path:
    """Return outputs/<stem>/intermediate/, creating it if needed."""
    d = run_dir / "intermediate"
    d.mkdir(parents=True, exist_ok=True)
    return d
