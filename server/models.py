"""Pydantic data models for the Adversarial Reasoning Gym environment."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    task_description: str
    turn_number: int
    max_turns: int = 12
    available_tools: List[str] = Field(
        default_factory=lambda: [
            "read_code",
            "run_tests",
            "run_code",
            "apply_fix",
            "submit_fix",
        ]
    )
    last_tool_result: Optional[str] = None
    user_message: Optional[str] = None
    investigation_log: List[str] = Field(default_factory=list)
    code_current: Optional[str] = None


class Action(BaseModel):
    action_type: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class DifficultyConfig(BaseModel):
    bug_subtlety: float = 0.0
    misconception_convincingness: float = 0.0
    followup_count: int = 0
    code_complexity: float = 0.0
    correct_user_ratio: float = 0.0


class TestCase(BaseModel):
    input: str
    expected: str


class TestResult(BaseModel):
    input: str
    expected: str
    actual: str
    passed: bool
    error: Optional[str] = None


class BugDefinition(BaseModel):
    bug_type: str
    bug_line: int
    original_code: str
    buggy_code: str
    explanation: str


class GradingCriteria(BaseModel):
    bug_line: int
    wrong_line: int
    correct_code: str
    user_was_correct: bool = False
    failing_inputs: List[str] = Field(default_factory=list)


class Scenario(BaseModel):
    function_name: str
    correct_code: str
    buggy_code: str
    test_cases: List[TestCase]
    bug: BugDefinition
    wrong_line: int
    wrong_reason: str
    initial_message: str
    followup_messages: List[str] = Field(default_factory=list)
    followup_triggers: List[str] = Field(default_factory=list)
    grading: GradingCriteria
    difficulty: DifficultyConfig
    user_was_correct: bool = False


class EpisodeResult(BaseModel):
    correct: bool
    caved: bool
    fix_score: float
    investigation_score: float
    resistance_score: float
    anti_cheat_score: float
    total_reward: float
    steps_used: int
    tool_calls: int
    difficulty: DifficultyConfig
