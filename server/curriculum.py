"""Adaptive 5-axis difficulty curriculum.

The curriculum keeps difficulty in [0, 1] for continuous axes (and [0, 3]
for follow-up count) and adjusts based on the agent's recent accuracy.
Early episodes follow a fixed schedule; later episodes use the adaptive
controller.

Weakness tracking (was a known gap): for each axis we record the result
of recent episodes split by whether the axis was "high" (>= 0.5) or
"low" (< 0.5). The "weakest" axis is the one where the high-side
accuracy drops the most below the low-side — i.e. the axis we're most
sensitive to. ``correct_user_ratio`` is treated as high at >= 0.15
because its max is 0.3.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from server.models import DifficultyConfig, EpisodeResult


_AXIS_NAMES = [
    "bug_subtlety",
    "misconception_convincingness",
    "code_complexity",
    "correct_user_ratio",
]


def _is_high(axis: str, value: float) -> bool:
    if axis == "correct_user_ratio":
        return value >= 0.15
    return value >= 0.5


class AdaptiveCurriculum:
    def __init__(self, history_size: int = 20, axis_window: int = 50) -> None:
        self.difficulty: Dict[str, float] = {a: 0.0 for a in _AXIS_NAMES}
        self.followup_count: int = 0
        self.history: Deque[EpisodeResult] = deque(maxlen=history_size)
        self.steps: int = 0
        # Per-axis windowed accuracy split by high/low.
        self._axis_window = axis_window
        self._axis_high: Dict[str, Deque[bool]] = {
            a: deque(maxlen=axis_window) for a in _AXIS_NAMES
        }
        self._axis_low: Dict[str, Deque[bool]] = {
            a: deque(maxlen=axis_window) for a in _AXIS_NAMES
        }
        # Round-robin pointer used as a tiebreaker when no axis has enough
        # signal yet. Ensures every axis eventually gets bumped, including
        # correct_user_ratio.
        self._round_robin_idx: int = 0

    def _scheduled_difficulty(self) -> Optional[DifficultyConfig]:
        if self.steps < 30:
            return DifficultyConfig()
        if self.steps < 80:
            return DifficultyConfig(
                bug_subtlety=0.3,
                misconception_convincingness=0.3,
                followup_count=1,
                code_complexity=0.2,
                correct_user_ratio=0.0,
            )
        return None

    def get_difficulty(self) -> DifficultyConfig:
        scheduled = self._scheduled_difficulty()
        if scheduled is not None:
            return scheduled
        return DifficultyConfig(
            bug_subtlety=self.difficulty["bug_subtlety"],
            misconception_convincingness=self.difficulty[
                "misconception_convincingness"
            ],
            followup_count=self.followup_count,
            code_complexity=self.difficulty["code_complexity"],
            correct_user_ratio=self.difficulty["correct_user_ratio"],
        )

    def update(self, result: EpisodeResult) -> None:
        self.steps += 1
        self.history.append(result)
        # Record per-axis high/low accuracy buckets from this episode.
        diff = result.difficulty
        snap = {
            "bug_subtlety": diff.bug_subtlety,
            "misconception_convincingness": diff.misconception_convincingness,
            "code_complexity": diff.code_complexity,
            "correct_user_ratio": diff.correct_user_ratio,
        }
        for axis, val in snap.items():
            if _is_high(axis, val):
                self._axis_high[axis].append(result.correct)
            else:
                self._axis_low[axis].append(result.correct)

        if len(self.history) < self.history.maxlen:
            return
        recent = list(self.history)
        accuracy = sum(1 for e in recent if e.correct) / len(recent)

        if accuracy > 0.75:
            weakest = self._find_weakest_axis()
            self._bump_axis(weakest, +0.1)
        elif accuracy < 0.25:
            hardest = self._find_hardest_axis()
            self._bump_axis(hardest, -0.1)

    def _axis_drop(self, axis: str) -> Optional[float]:
        """Accuracy drop when this axis is high vs low. ``None`` if we
        don't have enough data on either side yet."""
        high = self._axis_high[axis]
        low = self._axis_low[axis]
        if len(high) < 5 or len(low) < 5:
            return None
        return (sum(low) / len(low)) - (sum(high) / len(high))

    def _find_weakest_axis(self) -> str:
        """Return the axis where increasing difficulty caused the largest
        accuracy drop. Falls back to round-robin if no axis has enough
        signal — this guarantees every axis (especially
        correct_user_ratio) eventually gets bumped."""
        scored: List = []
        for axis in _AXIS_NAMES:
            drop = self._axis_drop(axis)
            scored.append((drop, axis))

        # If at least one axis has a drop measurement, pick the largest.
        with_signal = [(d, a) for d, a in scored if d is not None]
        if with_signal:
            with_signal.sort(reverse=True)  # largest drop first
            return with_signal[0][1]

        # Fallback: round-robin so we don't always bump the same axis.
        axis = _AXIS_NAMES[self._round_robin_idx % len(_AXIS_NAMES)]
        self._round_robin_idx += 1
        return axis

    def _find_hardest_axis(self) -> str:
        # Highest current value — we're going to back it off.
        choices = [(-self.difficulty[a], a) for a in _AXIS_NAMES]
        choices.sort()
        return choices[0][1]

    def _bump_axis(self, axis: str, delta: float) -> None:
        if axis == "correct_user_ratio":
            new_val = max(0.0, min(0.3, self.difficulty[axis] + delta))
        else:
            new_val = max(0.0, min(1.0, self.difficulty[axis] + delta))
        self.difficulty[axis] = new_val

        # Followups scale with misconception_convincingness.
        mcc = self.difficulty["misconception_convincingness"]
        target_followups = int(round(mcc * 3))
        if delta > 0:
            self.followup_count = max(self.followup_count, target_followups)
        else:
            self.followup_count = min(self.followup_count, target_followups)
        self.followup_count = max(0, min(3, self.followup_count))

    def snapshot(self) -> Dict:
        snap = dict(self.difficulty)
        snap["followup_count"] = self.followup_count
        snap["steps"] = self.steps
        snap["recent_accuracy"] = (
            sum(1 for e in self.history if e.correct) / len(self.history)
            if self.history
            else 0.0
        )
        snap["axis_drops"] = {a: self._axis_drop(a) for a in _AXIS_NAMES}
        return snap
