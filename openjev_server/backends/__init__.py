"""Backends: how log-probabilities of the candidate labels are obtained. `make_backend` is the one place that knows the names."""

from __future__ import annotations

from ..config import Settings
from ..profile import Profile
from .base import Backend, BackendError

__all__ = ["Backend", "BackendError", "make_backend"]


def make_backend(s: Settings, profile: Profile) -> tuple[Backend, dict]:
    """The backend named by the settings, plus the settings worth reporting in /v1/version. Backend modules import lazily:
    mlx only exists on Apple silicon."""
    if s.backend == "vllm":
        from .vllm import VllmBackend, VllmOptions

        opts = VllmOptions(
            model=s.served_model_name or None,
            timeout_s=s.timeout_s,
            max_connections=s.max_connections,
            letter_prefix=profile.letter_prefix,
            assistant_prefix=profile.assistant_prefix,
        )
        return VllmBackend(s.vllm_url, s.model, opts), {"vllm_url": s.vllm_url}
    if s.backend == "mlx":
        from .mlx import MlxBackend

        backend = MlxBackend(s.model, prefix_cache=s.prefix_cache, letter_prefix=profile.letter_prefix, assistant_prefix=profile.assistant_prefix)
        return backend, {"prefix_cache": s.prefix_cache}
    raise ValueError(f"unknown backend {s.backend!r}: use vllm or mlx")
