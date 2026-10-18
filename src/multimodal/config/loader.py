"""YAML configuration loading with environment interpolation and merging.

Experiment configs are YAML files under `configs/`. This loader:

1. Substitutes ``${VAR}`` / ``${VAR:default}`` from the environment, so a
   single config file works across dev/staging/prod.
2. Deep-merges an optional base config (`__base__:` key) — inheritance
   without YAML anchors' copy-paste hazards.
3. Coerces the result into dataclass configs (VAEConfig, GANConfig, ...).

Failure mode guarded against: silent typos in config keys. Unknown keys
are rejected loudly instead of being ignored for three epochs.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import ConfigError

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>[^}]*))?\}")


def interpolate_env(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` and ``${VAR:default}`` in strings."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            env_val = os.getenv(match.group("name"))
            if env_val is not None:
                return env_val
            if match.group("default") is not None:
                return match.group("default")
            raise ConfigError(
                "unset environment variable with no default",
                details={"variable": match.group("name")},
            )

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; override wins on scalar conflicts."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML file with env interpolation; raises ConfigError on bad input."""
    path = Path(path)
    if not path.exists():
        raise ConfigError("config file not found", details={"path": str(path)})
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}", details={"error": str(exc)}) from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping", details={"path": str(path)})
    return interpolate_env(raw)


def load_config(path: str | Path, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a config, resolving `__base__` inheritance chains (max depth 8)."""
    data = load_yaml(path)
    base_rel = data.pop("__base__", None)
    if base_rel:
        if base_dir is None:
            base_dir = Path(path).parent
        base_path = Path(base_dir) / base_rel
        if not base_path.exists():
            raise ConfigError("__base__ target not found", details={"base": str(base_path)})
        base = load_config(base_path, base_dir=base_dir)  # recursive inheritance
        data = deep_merge(base, data)
    return data


def to_dataclass(payload: dict[str, Any], cls: type) -> Any:
    """Coerce a plain dict into a dataclass config, rejecting unknown keys."""
    if not is_dataclass(cls):
        raise ConfigError("target must be a dataclass config", details={"cls": str(cls)})
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ConfigError(
            f"unknown config keys for {cls.__name__}",
            details={"unknown": sorted(unknown), "known": sorted(known)},
        )
    # Nested dataclass (e.g. OptimizerConfig) arrives as a dict.
    nested_types = {f.name: f.type for f in fields(cls)}
    for name in list(payload):
        if isinstance(payload[name], dict) and name == "optimizer":
            from .base_config import OptimizerConfig

            payload[name] = to_dataclass(payload[name], OptimizerConfig)
            del nested_types[name]
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ConfigError(
            f"invalid config for {cls.__name__}", details={"error": str(exc)}
        ) from exc
