# Business Entity Resolution

Repository scaffold for the ML Challenge 2026 business entity-resolution competition.
This milestone provides immutable-input discovery, configuration-driven paths, safe TSV
ingestion, and tests. It intentionally does not perform profiling, candidate generation,
or model training.

## Environment

The project targets Python 3.11 and is also verified on Python 3.12. With Python 3.11
installed, create the environment with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On this workstation Python 3.11 was unavailable, so the checked environment was created
with `python3.12 -m venv .venv`.

## Checks

```bash
source .venv/bin/activate
pytest
ruff check .
python -m entity_resolution.io --validate-all
```

Dataset locations are defined only in `configs/data_paths.yaml`. Organizer-supplied files
under `student_resource/` are immutable and ignored by Git.

