# Pytest bootstrap plugin, loaded via `addopts = "-p thucia_pytest_bootstrap"`.
#
# This is a top-level module (NOT under the `thucia` package) so importing it
# does not import `thucia` itself (which would import the whole library before
# coverage starts and make it unmeasured).
#
# It pre-imports duckdb before pytest-cov constructs its Coverage object.
# coverage.py's source discovery imports each `--cov=<pkg>` source package
# (via importlib find_spec) inside a sys.modules-saving block; on CPython 3.13
# the cleanup deletes duckdb's multi-module extension submodules
# (`_duckdb._sqltypes`) from sys.modules, after which a fresh `import duckdb`
# fails with "`_duckdb` is not a package". Keeping duckdb already imported (so
# the cleanup finds nothing to remove) avoids the corruption. duckdb is
# third-party and never under `--source=thucia`, so this doesn't affect
# measurement.
import duckdb  # noqa: F401
