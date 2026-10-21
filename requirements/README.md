# Environment pins

- `analysis.in`: the exact package versions of the environment that produced
  every released artifact (Python 3.12, macOS arm64). Install with
  `python -m pip install -r requirements/analysis.in` into a fresh virtual
  environment. The runs pin single-threaded numerics through environment
  variables set by the programs themselves.
- `ci.txt`: the subset used by the CI unit-test job (no PyTorch).
- `environment-macos-arm64-py3.12.json`: consolidated environment record; each
  canonical run directory also carries its own `environment.json`.

A hash-locked `requirements-lock.txt` is not shipped: generating one for a
different platform would present it as portable when it is not.
`scripts/validate_requirements_lock.py` remains available for anyone who
creates such a lock for their own platform.
