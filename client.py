"""OpenEnv-compatible sync HTTP client for the Adversarial Reasoning Gym.

The server keeps per-session env state keyed by ``session_id``. The
client transparently threads the id through ``reset``/``step``/``state``
so callers don't need to manage it. Multiple clients (or parallel GRPO
rollouts) can share a single server without corrupting each other's
state.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from server.models import Action, Observation, StepResult


class AdversarialReasoningClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.session_id: Optional[str] = None

    def reset(
        self, difficulty: Optional[str] = None, seed: Optional[int] = None
    ) -> Observation:
        body: Dict[str, Any] = {}
        if difficulty is not None:
            body["difficulty"] = difficulty
        if seed is not None:
            body["seed"] = seed
        if self.session_id is not None:
            body["session_id"] = self.session_id
        r = self._client.post("/reset", json=body)
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        return Observation(**data["observation"])

    def step(self, action: Action) -> StepResult:
        body = {"session_id": self.session_id, "action": action.model_dump()}
        r = self._client.post("/step", json=body)
        r.raise_for_status()
        return StepResult(**r.json())

    def state(self) -> Dict[str, Any]:
        params = {"session_id": self.session_id} if self.session_id else {}
        r = self._client.get("/state", params=params)
        r.raise_for_status()
        return r.json()

    def end_session(self) -> bool:
        if self.session_id is None:
            return False
        r = self._client.delete(f"/session/{self.session_id}")
        r.raise_for_status()
        ended = bool(r.json().get("ended"))
        self.session_id = None
        return ended

    def health(self) -> Dict[str, Any]:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AdversarialReasoningClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.end_session()
        except Exception:
            pass
        self.close()


if __name__ == "__main__":
    # Demo: drive a single episode end-to-end.
    with AdversarialReasoningClient() as client:
        print(client.health())
        obs = client.reset(difficulty="easy", seed=42)
        print("USER:", obs.task_description[:200])
        print("CODE:")
        print(obs.code_current)

        for tool in ("read_code", "run_tests"):
            r = client.step(Action(action_type="tool_call", tool_name=tool))
            print(f"\n--- {tool} ---")
            print((r.observation.last_tool_result or "")[:300])

        final = client.step(Action(action_type="submit"))
        print("\nFINAL reward:", final.reward, "outcome:", final.info["fix"]["outcome"])
