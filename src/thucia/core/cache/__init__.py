from .CacheBase import CacheBase
from .SQLiteCache import SQLiteCache


class Cache:
    _registry = {
        "sqlite": SQLiteCache,
    }

    def __new__(cls, name: str, *args, **kwargs) -> CacheBase:
        try:
            cache_class = cls._registry[name.lower()]
            return cache_class(*args, **kwargs)
        except KeyError:
            raise ValueError(f"Unknown cache type: '{name}'")
