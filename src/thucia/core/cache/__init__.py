from thucia.core.registry import Registry

from .CacheBase import CacheBase
from .SQLiteCache import SQLiteCache

cache_registry: Registry = Registry("cache backend")

cache_registry.register("sqlite")(SQLiteCache)


class Cache:
    """Factory for cache backends, selected by name (case-insensitive)."""

    def __new__(cls, name: str, *args, **kwargs) -> CacheBase:
        return cache_registry.get(name)(*args, **kwargs)
