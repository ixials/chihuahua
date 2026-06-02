import pandas as pd

from bark_detection.barkseq_export import BARKSEQS_COLUMNS
from bark_detection.clips import CLIP_COLUMNS
from bark_detection.frames import FRAME_COLUMNS
from bark_detection.timeline import TIMELINE_COLUMNS

PANNS_WINDOWS_COLUMNS = [
    "window_id",
    "start_time_sec",
    "end_time_sec",
    "center_time_sec",
]

PANNS_SCORES_COLUMNS = [
    "window_id",
    "start_time_sec",
    "end_time_sec",
    "center_time_sec",
    "dog_score",
    "bark_score",
    "animal_score",
    "speech_score",
    "music_score",
    "top_1_label",
    "top_1_score",
    "top_2_label",
    "top_2_score",
    "top_3_label",
    "top_3_score",
]

FULL_BARKSEQS_COLUMNS = BARKSEQS_COLUMNS + FRAME_COLUMNS + CLIP_COLUMNS


def _assert_has_columns(df: pd.DataFrame, expected: list[str]) -> None:
    missing = set(expected) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"


def test_panns_windows_schema():
    df = pd.DataFrame(columns=PANNS_WINDOWS_COLUMNS)
    _assert_has_columns(df, PANNS_WINDOWS_COLUMNS)


def test_panns_scores_schema():
    df = pd.DataFrame(columns=PANNS_SCORES_COLUMNS)
    _assert_has_columns(df, PANNS_SCORES_COLUMNS)


def test_timeline_schema():
    df = pd.DataFrame(columns=TIMELINE_COLUMNS)
    _assert_has_columns(df, TIMELINE_COLUMNS)


def test_barkseqs_full_schema():
    df = pd.DataFrame(columns=FULL_BARKSEQS_COLUMNS)
    _assert_has_columns(df, FULL_BARKSEQS_COLUMNS)
