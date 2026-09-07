# Contributing

Contributions that improve portability, tests, documentation, or scientifically equivalent implementations are welcome.

1. Create a focused branch from `main`.
2. Do not commit participant-level internal data, third-party raw datasets, model checkpoints, local paths, credentials, or generated caches.
3. Preserve participant-disjoint evaluation boundaries and training-only preprocessing/feature-selection rules.
4. Add or update tests for behavioral changes.
5. Run `python -m unittest discover -s tests -p "test_*.py"`.
6. Explain the scientific and reproducibility impact in the pull request.

Changes to locked protocols or reported results must include regenerated evidence, explicit provenance, and a clear statement that the new outputs supersede prior results.

