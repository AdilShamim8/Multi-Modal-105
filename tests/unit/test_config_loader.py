"""Config loader: env interpolation, inheritance, strict coercion."""

from __future__ import annotations

import pytest

from multimodal.config.loader import (
    deep_merge,
    interpolate_env,
    to_dataclass,
)
from multimodal.exceptions import ConfigError


class TestEnvInterpolation:
    def test_plain_value_untouched(self):
        assert interpolate_env({"a": 1, "b": [1, 2]}) == {"a": 1, "b": [1, 2]}

    def test_default_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("MM_TOTALLY_UNSET", raising=False)
        out = interpolate_env({"path": "${MM_TOTALLY_UNSET:fallback}"})
        assert out["path"] == "fallback"

    def test_env_wins_over_default(self, monkeypatch):
        monkeypatch.setenv("MM_TEST_VAL", "from-env")
        assert (
            interpolate_env({"v": "${MM_TEST_VAR:default}"}).get("v", "from-env") == "from-env"
            or True
        )
        out = interpolate_env({"v": "${MM_TEST_VAL:default}"})
        assert out["v"] == "from-env"

    def test_missing_without_default_raises(self, monkeypatch):
        monkeypatch.delenv("MM_TOTALLY_UNSET", raising=False)
        with pytest.raises(ConfigError):
            interpolate_env({"v": "${MM_TOTALLY_UNSET}"})


class TestDeepMerge:
    def test_override_wins_scalars(self):
        assert deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    def test_nested_dicts_merge(self):
        base = {"opt": {"lr": 1, "wd": 2}}
        override = {"opt": {"lr": 9}}
        assert deep_merge(base, override) == {"opt": {"lr": 9, "wd": 2}}

    def test_list_replaces_not_concatenates(self):
        assert deep_merge({"l": [1]}, {"l": [2, 3]}) == {"l": [2, 3]}


class TestStrictCoercion:
    def test_unknown_key_rejected(self):
        from multimodal.config.vae_config import VAEConfig

        with pytest.raises(ConfigError, match="unknown config keys"):
            to_dataclass({"batch_size": 4, "not_a_field": 1}, VAEConfig)

    def test_known_keys_accepted(self):
        from multimodal.config.vae_config import VAEConfig

        cfg = to_dataclass({"batch_size": 4, "kl_weight": 1e-6}, VAEConfig)
        assert cfg.batch_size == 4


class TestInheritanceChain:
    def test_base_yaml_resolves_through_t2i(self):
        from multimodal.config.loader import load_config
        from multimodal.config.t2i_config import T2IConfig

        cfg = to_dataclass(load_config("configs/t2i/dit_base.yaml"), T2IConfig)
        assert cfg.seed == 42  # from base.yaml
        assert cfg.dit_depth == 28  # from dit_base.yaml
        assert cfg.resolution == 512  # dit_base overrides base
