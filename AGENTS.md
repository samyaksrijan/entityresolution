# Repository rules

- Preserve organizer-supplied raw data byte-for-byte and never write beneath its root.
- Resolve every data path through `configs/data_paths.yaml`; parse TSVs with an explicit tab separator.
- Keep validation leakage-safe and implement the exact per-S1 macro F0.5 competition score.
- Add focused tests for behavioral changes.
- Never use external business lookup or augmentation.
- Never use numeric entity-ID components as model features.

