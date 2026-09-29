# Implementation layout

Python modules remain in this directory because they import each other directly.

- `data/` — extracted CK+48 dataset.
- `models/` — required model assets.
- `experiments/` — experiment-specific reports and results.
- `artifacts/apex/` — apex benchmark caches, metrics, and figures.
- `artifacts/full/` — full-sequence benchmark caches, metrics, and figures.
- `artifacts/paper/` — paper-reproduction figures and tables.
- `artifacts/improved/` — improved-pipeline benchmark results.
- `artifacts/stages/` — stage previews, intermediate arrays, and stage metrics.
- `artifacts/transparency/` — transparency audit and dashboard.
- `docs/` — implementation guides and change notes.
- `logs/` — runtime logs.
