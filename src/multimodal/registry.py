"""Component registry: names → factories, resolved at runtime.

Used for models, samplers, metrics, trainers, and artifact stores. The
registry is how YAML configs reference Python implementations without
imports scattered across the codebase — one indirection point, one place
to audit what is instantiable in production.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from .exceptions import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    """Thread-safe, name-keyed factory registry.

    Duplicate registration is a hard error: silent overwrites hide the exact
    class of "wrong implementation ran in prod" bugs. Register in module
    scope; resolve at runtime via `build()`.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}
        self._lock = threading.RLock()

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Class/function decorator form."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            self._register_plain(name, factory)
            return factory

        return decorator

    def _register_plain(self, name: str, factory: Callable[..., T]) -> None:
        with self._lock:
            if name in self._factories:
                raise ConfigError(
                    f"duplicate {self._kind} registration",
                    details={"name": name},
                )
            self._factories[name] = factory

    def build(self, name: str, /, **kwargs: Any) -> T:
        """Instantiate a registered component; unknown names are ConfigErrors."""
        with self._lock:
            factory = self._factories.get(name)
        if factory is None:
            raise ConfigError(
                f"unknown {self._kind}: '{name}'",
                details={"known": sorted(self._factories)},
            )
        return factory(**kwargs)

    def get(self, name: str) -> Callable[..., T]:
        with self._lock:
            if name not in self._factories:
                raise ConfigError(
                    f"unknown {self._kind}: '{name}'",
                    details={"known": sorted(self._factories)},
                )
            return self._factories[name]

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._factories
