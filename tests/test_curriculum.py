"""Curriculum unit tests."""
from __future__ import annotations

from server.curriculum import AdaptiveCurriculum
from server.models import DifficultyConfig, EpisodeResult


def _result(correct: bool, diff: DifficultyConfig) -> EpisodeResult:
    return EpisodeResult(
        correct=correct, caved=False, fix_score=0.0,
        investigation_score=0.0, resistance_score=0.0,
        anti_cheat_score=0.0, total_reward=0.0,
        steps_used=4, tool_calls=4, difficulty=diff,
    )


def test_warmup_keeps_all_zero_first_30_episodes():
    cur = AdaptiveCurriculum()
    for _ in range(29):
        cur.update(_result(True, DifficultyConfig()))
    cfg = cur.get_difficulty()
    assert cfg.bug_subtlety == 0.0
    assert cfg.followup_count == 0


def test_correct_user_ratio_clamped_to_03():
    cur = AdaptiveCurriculum()
    cur.difficulty["correct_user_ratio"] = 0.25
    cur._bump_axis("correct_user_ratio", +0.1)
    assert cur.difficulty["correct_user_ratio"] == 0.3


def test_round_robin_rotates_axes_when_no_signal():
    cur = AdaptiveCurriculum()
    seen = set()
    for _ in range(8):
        seen.add(cur._find_weakest_axis())
    assert len(seen) == 4  # all four axes hit before we recycle


def test_high_drop_axis_chosen():
    cur = AdaptiveCurriculum()
    # Plant a deliberate drop on bug_subtlety: high difficulty fails, low passes.
    high_diff = DifficultyConfig(bug_subtlety=0.8)
    low_diff = DifficultyConfig(bug_subtlety=0.0)
    for _ in range(10):
        cur.update(_result(False, high_diff))
        cur.update(_result(True, low_diff))
    weakest = cur._find_weakest_axis()
    assert weakest == "bug_subtlety"
