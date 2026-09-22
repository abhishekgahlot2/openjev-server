"""Shared bits for the demos: one call to a /v1/systemone server, timed, with the raw response kept for the log."""

from __future__ import annotations

import time

import httpx


class Player:
    """A /v1/systemone endpoint (OpenJev, Kev, hosted) or "random"."""

    def __init__(self, endpoint: str, model: str = "openjev", token: str = "", name: str = ""):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.name = name or endpoint
        self.random = endpoint == "random"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = None if self.random else httpx.Client(timeout=300.0, headers=headers)
        self.calls = 0
        self.ms = 0.0

    def ask(self, state, questions: dict) -> tuple[dict, float]:
        """Answers by question id, and the round-trip in ms."""
        t0 = time.perf_counter()
        r = self.client.post(self.endpoint + "/v1/systemone", json={"model": self.model, "state": state, "questions": questions})
        r.raise_for_status()
        dt = (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.ms += dt
        return r.json()["answers"], dt

    def avg_ms(self) -> float:
        return self.ms / self.calls if self.calls else 0.0
