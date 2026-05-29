"""Tests for PANNs vocalization label configuration."""

from bark_detection.config import BarkConfig
from bark_detection.panns_inference import SCORE_COLUMN_LABELS, resolve_label_mapping


EXPECTED_BARK_LABELS = [
    "Bark",
    "Yip",
    "Bow-wow",
    "Howl",
    "Growling",
    "Whimper (dog)",
]


def test_bark_score_labels_include_dog_vocalizations():
    assert SCORE_COLUMN_LABELS["bark"] == EXPECTED_BARK_LABELS


def test_resolve_label_mapping_finds_all_bark_labels():
    name_by_index = {
        0: "Speech",
        72: "Animal",
        24: "Whimper",
        74: "Dog",
        75: "Bark",
        76: "Yip",
        77: "Howl",
        78: "Bow-wow",
        79: "Growling",
        80: "Whimper (dog)",
        137: "Music",
    }
    fake_labels = [name_by_index.get(i, f"label_{i}") for i in range(527)]

    mapping = resolve_label_mapping(fake_labels)
    assert set(mapping["bark"].keys()) == set(EXPECTED_BARK_LABELS)


def test_high_recall_defaults():
    cfg = BarkConfig()
    assert cfg.barkseq_threshold == 0.4
    assert cfg.combined_bark_mode == "max_bark_dog"
    assert cfg.merge_gap_sec == 0.33
    assert cfg.short_window_size_sec == 0.5
    assert cfg.short_hop_size_sec == 0.1
    assert cfg.audio_normalize is True
    assert cfg.audio_normalize_target_peak == 0.9
