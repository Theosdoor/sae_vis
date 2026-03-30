# CLAUDE.md — AI Assistant Guide for sae_vis

## Project Overview

**sae-vis** is a Python library for visualizing Sparse Autoencoders (SAEs), based on Anthropic's published visualizer. It generates interactive HTML visualizations for analyzing which tokens activate SAE features and how features respond to custom prompts.

- **PyPI package**: `sae-vis`
- **Python requirement**: 3.10+
- **Package manager**: Poetry
- **License**: MIT
- **Upstream repo**: https://github.com/callummcdougall/sae_vis
- **This repo**: A personal fork for making fixes and contributing PRs upstream

### Fork Workflow

This is a fork of the upstream `callummcdougall/sae_vis`. The intended workflow is:
1. Develop fixes/improvements on feature branches in this fork
2. Open PRs against the upstream repo at https://github.com/callummcdougall/sae_vis
3. The upstream author accepts PRs but is not actively developing the project

When making changes intended for upstream, keep PRs focused and ensure `make check-all` passes.

## Repository Structure

```
sae_vis/
├── sae_vis/                     # Main source package
│   ├── __init__.py              # Package exports and __version__
│   ├── data_config_classes.py   # Configuration dataclasses (layout/component configs)
│   ├── data_fetching_fns.py     # Core computation: runs inference, collects activations
│   ├── data_storing_fns.py      # Data containers and HTML serialization
│   ├── model_fns.py             # Model utilities, demo model/SAE loading
│   ├── utils_fns.py             # Shared utilities (TopK, histograms, correlation, etc.)
│   ├── init.js                  # Frontend JavaScript for interactive HTML vis
│   ├── style.css                # Frontend CSS styling
│   └── demos/
│       ├── demo.py              # Runnable examples (5 different visualization types)
│       └── demo_*.html          # Pre-generated HTML output files
├── pyproject.toml               # Poetry config, dependencies, ruff/isort/semver config
├── Makefile                     # Dev commands: format, lint, check-all
├── CHANGELOG.md                 # Version history
└── .github/workflows/ci.yaml   # CI: lint + format check + build (no tests)
```

## Architecture

The codebase follows a clean 3-layer pipeline:

```
Configuration  →  Data Fetching  →  Data Storage  →  HTML Output
(data_config_    (data_fetching_    (data_storing_    (JavaScript
 classes.py)      fns.py)            fns.py)           frontend)
```

### Layer 1: Configuration (`data_config_classes.py`)

Component configs (e.g., `SeqMultiGroupConfig`, `ActsHistogramConfig`) are composed into `Column` objects, which are composed into a `SaeVisLayoutConfig`. The top-level `SaeVisConfig` holds the layout plus global settings.

### Layer 2: Data Fetching (`data_fetching_fns.py`)

Key functions:
- `parse_feature_data()` — core computation; converts activation cache → `SaeVisData`
- `get_feature_data()` — high-level wrapper with progress tracking
- `get_sequences_data()` — extracts top-firing token sequences per feature
- `parse_prompt_data()` / `get_prompt_data()` — prompt-centric visualization

### Layer 3: Data Storage & Serialization (`data_storing_fns.py`)

Key classes:
- `SaeVisData` — master container; has `.create()` factory method and `.save_*()` output methods
- `SequenceData`, `SeqGroupData`, `SeqMultiGroupData` — token sequence containers
- `FeatureTablesData`, `ActsHistogramData`, `LogitsTableData` — component data

### Typical Usage Pattern

```python
# 1. Define layout
layout = SaeVisLayoutConfig([
    Column(SeqMultiGroupConfig(...)),
    Column(ActsHistogramConfig(), FeatureTablesConfig(...))
])

# 2. Compute data (main computation)
sae_vis_data = SaeVisData.create(
    sae=sae,
    model=model,
    tokens=tokens,
    cfg=SaeVisConfig(features=range(100), feature_centric_layout=layout),
    verbose=True,
)

# 3. Save HTML output
sae_vis_data.save_feature_centric_vis(filename="output.html", feature=0)
sae_vis_data.save_prompt_centric_vis(filename="prompt.html", prompt="Hello world")
```

## Development Workflow

### Setup

```bash
poetry install           # Install all dependencies including dev
```

### Linting and Formatting

```bash
make format              # Auto-format + fix lint issues (ruff format + ruff check --fix-only)
make lint                # Check formatting + lint without modifying (ruff format --check + ruff check)
make check-all           # Run format then lint (required before PRs)
```

**All PRs must pass `make check-all` before merging.**

### Running Demos

```bash
cd sae_vis/demos
python demo.py           # Runs all demo examples, generates HTML files
```

Note: demos require `sae-lens` (`pip install sae-lens`) which is not in the core dependencies.

### No Tests

Tests were removed in v0.3.6. The CI pipeline does not run any tests. Pytest is listed as a dev dependency but unused.

### CI/CD

Triggered on PRs and pushes to `main`. The pipeline:
1. Lint check: `poetry run ruff check`
2. Format check: `poetry run ruff format --check`
3. Build: `poetry build`
4. Release (main branch only): semantic-release → PyPI publish

## Code Conventions

### Type Annotations

- Use `jaxtyping` for tensor shape annotations: `Float[Tensor, "batch seq d_model"]`
- Use `|` for unions (NOT `typing.Union` or `typing.Optional`) — enforced by ruff
- Use built-in generics (`dict`, `list`) not `typing.Dict`, `typing.List` — enforced by ruff
- Always annotate return types

### Naming

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Config classes end in `Config`, data containers end in `Data`
- Private/internal helpers: prefix with `_`

### Dataclasses

- Use `@dataclass` for all config and data container classes
- Use `field(default_factory=...)` for mutable defaults
- Use `frozen=True` on configs that should be immutable

### Ruff Ignored Rules

The following rules are intentionally ignored (see `pyproject.toml`):
- `E203` — whitespace before `:`
- `E501` — line too long
- `E731` — do not assign lambda
- `F722` — syntax error in forward annotation (needed for jaxtyping)
- `E741` — ambiguous variable name
- `F821` — undefined name (jaxtyping uses forward refs)
- `F403` — star imports
- `ARG002` — unused method argument

### PyTorch Patterns

- Wrap inference in `@torch.inference_mode()` or `torch.inference_mode()` context manager
- Use `einops` for tensor reshaping over manual `.view()` / `.reshape()` where clarity helps
- Use `eindex-callum` for advanced indexing patterns

### Version Updates

When changing the version, update it in **two places**:
1. `pyproject.toml` → `[tool.poetry] version = "..."`
2. `sae_vis/__init__.py` → `__version__ = "..."`

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` | Tensor operations and neural network inference |
| `transformer-lens` | TransformerLens models and activation caching |
| `einops` | Readable tensor manipulation |
| `datasets` | Loading HuggingFace datasets for tokenized text |
| `dataclasses-json` | JSON serialization of dataclass instances |
| `jaxtyping` | Tensor shape type annotations |
| `eindex-callum` | Custom indexing utilities |
| `rich` | Terminal tables and formatted output |
| `matplotlib` | Histogram and plot utilities |

## SAE Types Supported

- **Standard SAEs**: residual stream or MLP output SAEs
- **Attention SAEs**: with Direct Feature Attribution (DFA) visualization
- **OthelloGPT SAEs**: with board state display and linear probe integration

## Visualization Modes

- **Feature-centric**: for a given SAE feature, show which tokens most activate it (top sequences, quantile groups)
- **Prompt-centric**: for a given prompt, show which SAE features activate and where

## Important Notes for AI Assistants

1. **No test suite exists** — do not attempt to run tests or add test infrastructure unless explicitly requested.
2. **Frontend is static** — `init.js` and `style.css` are embedded in generated HTML; changes there affect all outputs.
3. **`sae-lens` is not a dependency** — it is used only in demos to avoid a circular dependency; do not add it to `pyproject.toml`.
4. **`jaxtyping` shape strings in function signatures** cause `F722`/`F821` ruff errors; these are intentionally ignored.
5. **Config `.help()` methods** exist on config classes to print self-documentation — useful for users, not needed for development.
6. **Version is tracked in two files** — always update both `pyproject.toml` and `sae_vis/__init__.py`.
7. **The project is minimally maintained** — the original author accepts PRs but is not actively developing it.
