# A small, shared registry primitive used by the plugin systems (covariate
# sources, cache backends, ...).
from __future__ import annotations

from typing import Callable
from typing import TypeVar

T = TypeVar("T")


class PluginNotFoundError(KeyError):
    """Raised when a plugin/backend is requested that is not registered."""

    def __init__(self, registry: str, key: str) -> None:
        self.registry = registry
        self.key = key
        super().__init__(f"No {registry} registered under '{key}'")


class Registry:
    """A name -> class registry with decorator-based registration.

    Keys are normalised to lowercase. ``register`` may be used with an explicit
    key or (for classes with a ``ref`` attribute) as ``@registry.register()``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, type] = {}

    def register(self, key: str | None = None) -> Callable[[type], type]:
        def deco(cls: type) -> type:
            k = key or getattr(cls, "ref", None)
            if not k:
                raise ValueError(f"Cannot register {cls.__name__}: no key or 'ref'")
            self._items[str(k).lower()] = cls
            return cls

        return deco

    def get(self, key: str) -> type:
        try:
            return self._items[key.lower()]
        except KeyError:
            raise PluginNotFoundError(self.name, key) from None

    def has(self, key: str) -> bool:
        return key.lower() in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def all(self) -> dict[str, type]:
        return dict(self._items)

    def unregister(self, key: str) -> None:
        self._items.pop(key.lower(), None)

    def clear(self) -> None:
        self._items.clear()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Registry '{self.name}': {self.names()}>"
