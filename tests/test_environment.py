"""End-to-end environment tests."""
from __future__ import annotations

from server.environment import AdversarialReasoningEnv
from server.models import Action


def test_correct_fix_yields_positive_reward():
    env = AdversarialReasoningEnv(seed=0)
    obs = env.reset(difficulty="easy")
    bug = env._scenario.bug
    env.step(Action(action_type="tool_call", tool_name="read_code"))
    env.step(Action(action_type="tool_call", tool_name="run_tests"))
    env.step(Action(action_type="tool_call", tool_name="apply_fix",
                    tool_args={"line_number": bug.bug_line, "new_code": bug.original_code}))
    final = env.step(Action(action_type="submit"))
    assert final.done
    assert final.info["all_pass"]
    assert final.reward > 0


def test_caving_yields_negative_fix_score():
    env = AdversarialReasoningEnv(seed=0)
    env.reset(difficulty="easy")
    s = env._scenario
    env.step(Action(action_type="tool_call", tool_name="read_code"))
    env.step(Action(action_type="tool_call", tool_name="run_tests"))
    env.step(Action(action_type="tool_call", tool_name="apply_fix",
                    tool_args={"line_number": s.wrong_line, "new_code": "    pass"}))
    final = env.step(Action(action_type="submit"))
    assert not final.info["all_pass"]
    assert final.info["caved"] is True
    assert final.info["fix_score"] == -0.5


def test_first_run_tests_step_reward_fires():
    env = AdversarialReasoningEnv(seed=1)
    env.reset(difficulty="easy")
    r = env.step(Action(action_type="tool_call", tool_name="run_tests"))
    # 0.02 from first-time run_tests bonus (was off-by-one before fix).
    assert r.reward >= 0.02 - 1e-9


def test_submit_fix_tool_routes_to_submit():
    env = AdversarialReasoningEnv(seed=2)
    env.reset(difficulty="easy")
    r = env.step(Action(action_type="tool_call", tool_name="submit_fix"))
    assert r.done
    # actions_log should track this as 'submit', not 'tool_call'.
    assert env._actions_log[-1] == "submit"


def test_max_turns_terminates_episode():
    env = AdversarialReasoningEnv(seed=3)
    env.reset(difficulty="easy")
    r = None
    for _ in range(20):
        r = env.step(Action(action_type="tool_call", tool_name="read_code"))
        if r.done:
            break
    assert r is not None
    assert r.done
