# Installation

Thucia requires **Python 3.11 or newer** (the CI runs 3.13).

## From source

Clone the repository and use [uv](https://docs.astral.sh/uv/) (recommended) to
create an environment with all optional dependencies:

```console
$ git clone https://github.com/jsbrittain/Thucia.git
$ cd Thucia
$ uv sync --all-extras
```

Then activate the virtual environment and install Thucia in editable mode:

```console
$ . .venv/bin/activate      # Linux / macOS
$ .venv\Scripts\activate    # Windows
$ pip install -e .
```

Verify the installation:

```console
$ thucia --help
```

## Pinning a version

Thucia is not yet published to a package index, so pin it to a git revision.
With uv:

```toml
# pyproject.toml
[tool.uv.sources]
thucia = { git = "ssh://git@github.com/kraemer-lab/Thucia", rev = "<commit-sha>" }
```

or with pip:

```console
$ pip install "thucia @ git+https://github.com/kraemer-lab/Thucia@<commit-sha>"
```

`uv.lock` records the resolved commit, so the environment is reproducible. Once
the package is published, `thucia==<version>` (with `--require-hashes`) becomes
the pin for index installs.

Whatever the install method, a fitted artefact records the implementation that
produced it (see {doc}`data-layer`): a **content digest** of the package that
works for editable checkouts, git installs and wheels alike, plus the commit
when — and only when — the running package really is a git checkout.

## Install extras

Thucia's dependency surface is large (deep-learning backends, a dashboard,
containers), so optional features are split into extras you install on demand.

| Extra      | Provides                                               | Install                        |
|------------|--------------------------------------------------------|--------------------------------|
| `vis`      | The Streamlit dashboard (`thucia dashboard`)           | `pip install -e .[vis]`        |
| `panels`   | Reading R `.rds` case panels (`read_rds`)              | `pip install -e .[panels]`     |
| `dev`      | Tooling: `pytest`, `pytest-cov`, `pre-commit`          | `pip install -e .[dev]`        |
| `all`      | Visualisation, panels, and development tooling         | `pip install -e .[all]`        |

```{note}
The foundation models (`chronos`, `timesfm25`, `timesfm3`) need no extra of their
own: `darts` ships the model implementations, and its PyTorch backend comes with
the base `darts[all]` dependency. Their checkpoints are downloaded from
HuggingFace Hub on first use (TimesFM 3.0 additionally requires
`accept_license=True` — see {doc}`models`).
```

The core forecasting workflow (the pipeline stages, the fast statistical
models, `sarima`, and the darts models) works from the base install without any
extra.

## Containers (optional)

The `inla` model runs an R container through Docker or Podman. Thucia
auto-detects Docker and falls back to the user Podman socket. No container
runtime is needed for any other feature.

## First forecast

Jump to the {doc}`quickstart` to build a forecast, or read the
{doc}`pipeline` page to understand the stages.
