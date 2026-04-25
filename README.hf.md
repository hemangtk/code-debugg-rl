---
title: Adversarial Reasoning Gym - Code Debugger
emoji: 🪲
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
short_description: RL env that trains LLMs to find real bugs, not user-pointed ones.
---

# Adversarial Reasoning Gym — Code Debugger

OpenEnv-compliant RL environment. See [`README.md`](./README.md) for full
documentation. The Space serves a FastAPI HTTP/WebSocket interface that
training scripts and clients hit on port 8000.

## Endpoints

- `GET /health` — liveness check + active session count
- `POST /reset` — start a new debugging episode (returns `session_id`)
- `POST /step` — advance the env by one action
- `GET /state?session_id=...` — full env state for a session
- `DELETE /session/{session_id}` — end a session
- `GET /ws` — WebSocket session API
