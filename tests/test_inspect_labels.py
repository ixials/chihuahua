"""Tests for bark_detection.inspect_labels (Phase 2).

All tests use synthetic DataFrames — no real PANNs inference.
"""

import pandas as pd
import pytest

from bark_detection.config import BarkConfig
from bark_detection.inspect_labels import (
    INSPECT_LABELS,
    format_window_summary,
    inspect_score_columns,
    label_to_column,
    nearest_window_row,
)


# ---------------------------------------------------------------------------
# label_to_column
# ---------------------------------------------------------------------------


def test_label_to_column_simple():
    assert label_to_column("Bark") == "bark_score"


def test_label_to_column_hyphen():
    assert label_to_column("Bow-wow") == "bow_wow_score"


def test_label_to_column_parens():
    assert label_to_column("Whimper (dog)") == "whimper_dog_score"


def test_label_to_column_dog():
    assert label_to_column("Dog") == "dog_score"


def test_label_to_column_all_inspect_labels_no_leading_trailing_underscore():
    for label in INSPECT_LABELS:
        col = label_to_column(label)
        assert col.endswith("_score"), f"{label!r} -> {col!r} should end with _score"
        slug = col[: -len("_score")]
        assert not slug.startswith("_"), f"slug starts with underscore: {slug!r}"
        assert not slug.endswith("_"), f"slug ends with underscore: {slug!r}"


def test_inspect_score_columns_length():
    cols = inspect_score_columns()
    assert len(cols) == len(INSPECT_LABELS)


def test_inspect_score_columns_all_end_with_score():
    for col in inspect_score_columns():
        assert col.endswith("_score")


# ---------------------------------------------------------------------------
# Synthetic DataFrame helper
# ---------------------------------------------------------------------------


def _make_scores_df() -> pd.DataFrame:
    """Build a small synthetic vocalization_scores DataFrame."""
    from bark_detection.inspect_labels import INSPECT_LABELS, label_to_column

    rows = []
    for i, center in enumerate([0.5, 1.0, 1.5, 2.0, 2.5]):
        row = {
            "window_id": i,
            "start_time_sec": center - 0.5,
            "end_time_sec": center + 0.5,
            "center_time_sec": center,
        }
        for j, label in enumerate(INSPECT_LABELS):
            # Give window 2 (center=1.5) high bark score, others low
            row[label_to_column(label)] = 0.6 if (i == 2 and j == 0) else 0.05
        vocalization_cols = [label_to_column(l) for l in INSPECT_LABELS[:-1]]
        row["vocalization_max_score"] = max(row[c] for c in vocalization_cols)
        row["max_bark_dog_score"] = max(
            row["vocalization_max_score"], row["dog_score"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# nearest_window_row
# ---------------------------------------------------------------------------


def test_nearest_window_row_exact():
    df = _make_scores_df()
    row = nearest_window_row(df, 1.0)
    assert float(row["center_time_sec"]) == pytest.approx(1.0)


def test_nearest_window_row_between():
    df = _make_scores_df()
    # 1.3 is between 1.0 and 1.5; nearest is 1.5
    row = nearest_window_row(df, 1.3)
    assert float(row["center_time_sec"]) == pytest.approx(1.5)


def test_nearest_window_row_before_start():
    df = _make_scores_df()
    row = nearest_window_row(df, 0.0)
    assert float(row["center_time_sec"]) == pytest.approx(0.5)


def test_nearest_window_row_after_end():
    df = _make_scores_df()
    row = nearest_window_row(df, 99.0)
    assert float(row["center_time_sec"]) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# format_window_summary
# ---------------------------------------------------------------------------


def test_format_window_summary_contains_center_time():
    df = _make_scores_df()
    cfg = BarkConfig()
    row = df.iloc[2]  # center=1.5
    summary = format_window_summary(row, cfg)
    assert "1.500" in summary


def test_format_window_summary_query_time_shown():
    df = _make_scores_df()
    cfg = BarkConfig()
    row = df.iloc[2]
    summary = format_window_summary(row, cfg, query_time_sec=1.45)
    assert "Query" in summary
    assert "1.45" in summary


def test_format_window_summary_high_score_flagged():
    df = _make_scores_df()
    cfg = BarkConfig()
    cfg.barkseq_threshold = 0.30
    row = df.iloc[2]  # bark_score=0.6 >= 0.30 -> should be flagged with *
    summary = format_window_summary(row, cfg)
    assert "*" in summary


def test_format_window_summary_low_score_not_flagged():
    df = _make_scores_df()
    cfg = BarkConfig()
    cfg.barkseq_threshold = 0.30
    row = df.iloc[0]  # all scores=0.05 < 0.30 -> no * marker
    summary = format_window_summary(row, cfg)
    assert "*" not in summary


def test_format_window_summary_contains_all_labels():
    df = _make_scores_df()
    cfg = BarkConfig()
    row = df.iloc[0]
    summary = format_window_summary(row, cfg)
    for label in INSPECT_LABELS:
        assert label in summary


# ---------------------------------------------------------------------------
# vocalization_max_score logic
# ---------------------------------------------------------------------------


def test_vocalization_max_score_correct():
    df = _make_scores_df()
    # Window 2 has bark_score=0.6; vocalization_max_score should be 0.6
    row = df.iloc[2]
    assert float(row["vocalization_max_score"]) == pytest.approx(0.6)


def test_vocalization_max_score_low_windows():
    df = _make_scores_df()
    for i in [0, 1, 3, 4]:
        assert float(df.iloc[i]["vocalization_max_score"]) == pytest.approx(0.05)


def test_max_bark_dog_score_is_max_of_vocalization_and_dog():
    df = _make_scores_df()
    for _, row in df.iterrows():
        expected = max(float(row["vocalization_max_score"]), float(row["dog_score"]))
        assert float(row["max_bark_dog_score"]) == pytest.approx(expected)
