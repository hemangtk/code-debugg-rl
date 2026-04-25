"""FastAPI server exposing the AdversarialReasoningEnv over HTTP.

Each session has its own AdversarialReasoningEnv keyed by a session_id
returned from /reset. This is critical: multiple GRPO rollouts
(parallel completions) hitting the same server must not share env
state. The server also evicts idle sessions on a TTL to keep memory
bounded.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from server.environment import AdversarialReasoningEnv
from server.models import Action, Observation, StepResult


app = FastAPI(title="Adversarial Reasoning Gym - Code Debugger")


_SESSION_TTL_S = float(os.environ.get("ARG_SESSION_TTL_S", "1800"))  # 30 min
_MAX_SESSIONS = int(os.environ.get("ARG_MAX_SESSIONS", "256"))


class _SessionStore:
    """Thread-safe map of session_id -> (env, last_used_ts)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._envs: Dict[str, AdversarialReasoningEnv] = {}
        self._last_used: Dict[str, float] = {}

    def create(self, seed: Optional[int]) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._evict_expired_locked()
            if len(self._envs) >= _MAX_SESSIONS:
                self._evict_oldest_locked()
            self._envs[sid] = AdversarialReasoningEnv(seed=seed)
            self._last_used[sid] = time.time()
        return sid

    def get(self, sid: str) -> AdversarialReasoningEnv:
        with self._lock:
            env = self._envs.get(sid)
            if env is None:
                raise HTTPException(status_code=404, detail=f"unknown session_id {sid!r}")
            self._last_used[sid] = time.time()
            return env

    def delete(self, sid: str) -> bool:
        with self._lock:
            if sid in self._envs:
                del self._envs[sid]
                del self._last_used[sid]
                return True
            return False

    def _evict_expired_locked(self) -> None:
        now = time.time()
        stale = [sid for sid, ts in self._last_used.items() if now - ts > _SESSION_TTL_S]
        for sid in stale:
            del self._envs[sid]
            del self._last_used[sid]

    def _evict_oldest_locked(self) -> None:
        if not self._last_used:
            return
        oldest = min(self._last_used.items(), key=lambda kv: kv[1])[0]
        del self._envs[oldest]
        del self._last_used[oldest]

    def count(self) -> int:
        with self._lock:
            return len(self._envs)


_sessions = _SessionStore()


# Backwards-compatible default env for clients that omit session_id.
# Lazily allocated so importing this module doesn't spawn a sandbox.
_default_env_lock = threading.Lock()
_default_env: Optional[AdversarialReasoningEnv] = None


def _get_default_env(seed: Optional[int] = None) -> AdversarialReasoningEnv:
    global _default_env
    with _default_env_lock:
        if _default_env is None or seed is not None:
            _default_env = AdversarialReasoningEnv(seed=seed)
        return _default_env


class ResetRequest(BaseModel):
    difficulty: Optional[str] = None
    seed: Optional[int] = None
    session_id: Optional[str] = None  # If omitted, a new session is created.


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation


class StepRequest(BaseModel):
    session_id: Optional[str] = None  # If omitted, uses default env (legacy).
    action: Action


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "environment": "adversarial_reasoning_gym",
        "active_sessions": _sessions.count(),
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    """Environment metadata. OpenEnv tooling uses this for discovery."""
    return {
        "name": "adversarial_reasoning_gym",
        "spec_version": 1,
        "description": (
            "Train LLMs to find real bugs instead of agreeing with the "
            "user's wrong diagnosis."
        ),
        "tags": ["openenv", "sycophancy", "debugging", "code", "reasoning",
                 "adaptive-difficulty"],
        "tools": ["read_code", "run_tests", "run_code", "apply_fix", "submit_fix"],
        "max_turns": 12,
        "difficulty_presets": ["easy", "medium", "hard"],
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    """JSON schema for action / observation / step result.

    Mirrors the contract OpenEnv's create_app() exposes; lets a remote
    client introspect the env without reading our source.
    """
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "step_result": StepResult.model_json_schema(),
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()) -> ResetResponse:
    if req.session_id is not None:
        env = _sessions.get(req.session_id)
        sid = req.session_id
    else:
        sid = _sessions.create(seed=req.seed)
        env = _sessions.get(sid)
    obs = env.reset(difficulty=req.difficulty)
    return ResetResponse(session_id=sid, observation=obs)


@app.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    if req.session_id is None:
        env = _get_default_env()
    else:
        env = _sessions.get(req.session_id)
    return env.step(req.action)


@app.delete("/session/{session_id}")
def end_session(session_id: str) -> Dict[str, bool]:
    return {"ended": _sessions.delete(session_id)}


@app.get("/state")
def state(session_id: Optional[str] = None) -> Dict[str, Any]:
    if session_id is None:
        env = _get_default_env()
    else:
        env = _sessions.get(session_id)
    return env.state()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_env = AdversarialReasoningEnv()
    try:
        while True:
            msg = await websocket.receive_json()
            cmd = msg.get("cmd")
            if cmd == "reset":
                obs = session_env.reset(difficulty=msg.get("difficulty"))
                await websocket.send_json({"type": "observation", "data": obs.model_dump()})
            elif cmd == "step":
                action = Action(**msg.get("action", {}))
                result = session_env.step(action)
                await websocket.send_json({"type": "step_result", "data": result.model_dump()})
            elif cmd == "state":
                await websocket.send_json({"type": "state", "data": session_env.state()})
            else:
                await websocket.send_json({"type": "error", "data": f"unknown cmd: {cmd}"})
    except WebSocketDisconnect:
        return
