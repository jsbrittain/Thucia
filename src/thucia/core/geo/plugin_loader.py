import os
import importlib
import inspect

from pathlib import Path
from .plugin_base import SourceBase

plugin_dir = Path(__file__).parent / "sources"
module_stem = "thucia.core.geo.sources"


def load_plugins(plugin_dir=plugin_dir):
    plugins = {}
    for fname in os.listdir(plugin_dir):
        if fname.endswith(".py") and not fname.startswith("__"):
            mod_name = fname[:-3]
            mod = importlib.import_module(f"{module_stem}.{mod_name}")

            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, SourceBase) and obj is not SourceBase:
                    plugins[obj.ref] = obj()
    return plugins
