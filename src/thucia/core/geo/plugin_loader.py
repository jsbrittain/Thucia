import importlib
import logging
import os
from pathlib import Path

from .plugin_base import source_registry

plugin_dir = Path(__file__).parent / "sources"
module_stem = "thucia.core.geo.sources"


def load_plugins(
    plugin_dir: str | Path = plugin_dir,
    module_stem: str = module_stem,
) -> dict[str, type]:
    """Import every source module so plugins self-register.

    A single broken module is logged and skipped rather than killing all
    plugins. Returns the current registry contents (ref -> class).
    """
    for fname in sorted(os.listdir(plugin_dir)):
        if fname.endswith(".py") and not fname.startswith("__"):
            mod_name = fname[:-3]
            try:
                importlib.import_module(f"{module_stem}.{mod_name}")
            except Exception as exc:  # keep other plugins importable
                logging.error(f"Failed to load source plugin '{mod_name}': {exc}")
    return source_registry.all()
