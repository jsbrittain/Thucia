from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("thucia")
except PackageNotFoundError:
    # package is not installed
    pass
