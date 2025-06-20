# Developer documentation

## Developer installation

We recommend `uv` as your virtual environment manager ([installation](https://docs.astral.sh/uv/getting-started/installation/)). To install an editable version of this package with all options, including `dev` dependencies, run from the repository root folder:

```bash
uv sync --all-extras
```

Then, activate the virtual environment and install a local editable copy of Thucia:

```bash
. .venv/bin/activate  # Linux/macOS
.venv/Scripts/activate  # Windows
pip install -e .
```
