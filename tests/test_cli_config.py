import pytest

from bark_detection.cli import build_config, resolve_stages, _STAGE_ORDER
from bark_detection.config import BarkConfig


class TestBuildConfig:
    def test_defaults_match_barkconfig(self):
        cfg = build_config()
        defaults = BarkConfig()
        assert cfg.barkseq_threshold == defaults.barkseq_threshold
        assert cfg.combined_bark_mode == defaults.combined_bark_mode
        assert cfg.merge_gap_sec == defaults.merge_gap_sec

    def test_threshold_override(self):
        cfg = build_config(threshold=0.30)
        assert cfg.barkseq_threshold == 0.30

    def test_combined_mode_override(self):
        cfg = build_config(combined_mode="max_bark_dog")
        assert cfg.combined_bark_mode == "max_bark_dog"

    def test_merge_gap_override(self):
        cfg = build_config(merge_gap=1.0)
        assert cfg.merge_gap_sec == 1.0

    def test_all_overrides_together(self):
        cfg = build_config(threshold=0.25, combined_mode="max_bark_dog", merge_gap=0.75)
        assert cfg.barkseq_threshold == 0.25
        assert cfg.combined_bark_mode == "max_bark_dog"
        assert cfg.merge_gap_sec == 0.75

    def test_none_overrides_leave_defaults(self):
        defaults = BarkConfig()
        cfg = build_config(threshold=None, combined_mode=None, merge_gap=None)
        assert cfg.barkseq_threshold == defaults.barkseq_threshold
        assert cfg.combined_bark_mode == defaults.combined_bark_mode
        assert cfg.merge_gap_sec == defaults.merge_gap_sec


class TestResolveStages:
    def test_all_returns_full_stage_order(self):
        stages = resolve_stages(stage="all")
        assert stages == list(_STAGE_ORDER)

    def test_single_stage(self):
        stages = resolve_stages(stage="panns")
        assert stages == ["panns"]

    def test_single_stage_viz(self):
        stages = resolve_stages(stage="viz")
        assert stages == ["viz"]

    def test_from_stage_timeline(self):
        stages = resolve_stages(stage="all", from_stage="timeline")
        expected = _STAGE_ORDER[_STAGE_ORDER.index("timeline"):]
        assert stages == expected
        assert stages[0] == "timeline"
        assert stages[-1] == "viz"

    def test_from_stage_extract_is_full(self):
        stages = resolve_stages(stage="all", from_stage="extract")
        assert stages == list(_STAGE_ORDER)

    def test_from_stage_viz_is_just_viz(self):
        stages = resolve_stages(stage="all", from_stage="viz")
        assert stages == ["viz"]

    def test_from_stage_takes_priority_over_stage(self):
        # from_stage overrides stage value when both provided
        stages = resolve_stages(stage="all", from_stage="barkseqs")
        expected = _STAGE_ORDER[_STAGE_ORDER.index("barkseqs"):]
        assert stages == expected
