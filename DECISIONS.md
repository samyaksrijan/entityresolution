# Decisions

## 2026-09-27 — Organizer resources are immutable

`student_resource/` is the common root of all supplied datasets and documentation. It is
ignored as a unit, and generated files live outside it.

## 2026-09-27 — Documentation overrides assumptions

The organizer README is authoritative. It explicitly requires open-set handling for France,
per-S1 macro F0.5 including singleton scoring, two final TSV outputs, no external lookup, and
an MIT/Apache-2.0 model of at most 8B parameters. No contradictory prior assumption was found.

## 2026-09-27 — Local verification interpreter

The project targets Python 3.11 and supports Python 3.12. Python 3.11 and `uv` were not
available on this machine, so verification uses a local `.venv` created with Python 3.12;
no system software was installed.

