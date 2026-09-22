"""Settings: CLI flags win over environment variables (OPENJEV_*), which win over the profile file."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .profile import Profile

PROFILES_DIR = Path(__file__).parent.parent / "profiles"
HELPER_ENV = {  # the released helper's READOUT_* variables still work; a CLI flag wins over them
    "temp": "READOUT_T",
    "noul_t": "READOUT_NOUL_T",
    "noul_bias": "READOUT_NOUL_BIAS",
    "perms": "READOUT_PERMS",
    "instr_style": "READOUT_INSTR_STYLE",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENJEV_", extra="ignore")
    backend: str = "vllm"  # vllm | mlx
    model: str = "openjev/openjev"  # tokenizer / model dir for vllm; the MLX model dir for mlx
    vllm_url: str = "http://localhost:8000/v1"
    served_model_name: str = ""  # empty: discovered from the backend's /models
    host: str = "127.0.0.1"
    port: int = 3000
    token: str = ""  # bearer token required on /v1/* when set
    profile: str = "openjev"  # a name under profiles/ or a path to a JSON profile
    max_connections: int = 64
    timeout_s: float = 120.0
    prefix_cache: bool = True  # mlx only
    log_level: str = "INFO"


def load_profile(name_or_path: str, overrides: dict | None = None) -> Profile:
    p = Path(name_or_path)
    if not p.exists():
        p = PROFILES_DIR / f"{name_or_path}.json"
    if not p.exists():
        raise FileNotFoundError(f"profile {name_or_path!r} not found (no file and nothing under {PROFILES_DIR})")
    d = json.loads(p.read_text())
    d.setdefault("name", p.stem)
    for k, v in (overrides or {}).items():
        if v is not None:
            d[k] = v
    for k, env in HELPER_ENV.items():
        if os.environ.get(env) and (overrides or {}).get(k) is None:
            d[k] = type(Profile.__dataclass_fields__[k].default)(os.environ[env])
    return Profile.from_dict(d)
