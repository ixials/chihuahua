from pathlib import Path

from bark_detection.paths import intermediate_dir


def test_intermediate_dir_created(tmp_path: Path):
    run_dir = tmp_path / "dogs1"
    run_dir.mkdir()
    inter = intermediate_dir(run_dir)
    assert inter == run_dir / "intermediate"
    assert inter.is_dir()
