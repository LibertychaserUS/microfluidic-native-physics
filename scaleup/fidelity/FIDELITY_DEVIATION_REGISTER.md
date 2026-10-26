# FIDELITY deviation register

Numbered deviations from the upstream pipeline or from `FIDELITY_PROTOCOL_v1.md`, each with the
reason and the effect. Entries F-D1 to F-D5 were declared in the protocol before any run; later
entries are appended with the date and the run they first affected.

| ID | Where | Deviation | Reason | Effect on the comparison |
|---|---|---|---|---|
| F-D1 | M4/M5 constructor | `random_state=seed` added; upstream passes no seed to `train_test_split` or to `XGBRegressor` | Upstream runs are not reproducible; the protocol compares variants on identical splits | None on the fitted function: with `subsample=1` and `colsample_bytree=1` the seed does not enter the tree construction |
| F-D2 | M4/M5 constructor | `n_jobs=1` added | Determinism of the double run | None on the fitted function (`hist` is thread-count independent) |
| F-D3 | ENV-B | Only xgboost (1.3.3) matches the upstream pin; Python 3.9, numpy 1.26.4, scikit-learn 1.3.2, pandas 2.1.4 instead of 3.6.13 / 1.16.4 / 0.21.2 / 0.24.2; x86_64 binaries run under Rosetta 2; OpenMP runtime from conda-forge `llvm-openmp` 15.0.5 (osx-64) | Upstream's Python 3.6 stack cannot be installed on this machine | `StandardScaler` and `train_test_split` are version-stable; the fitted model is produced by the matched xgboost version |
| F-D4 | U1 generation-rate MAPE | Reported under π and under the upstream constant 3.143 | Upstream script uses 3.143 | Relative difference 4.4e-4 in the rate, far below the reported precision |
| F-D5 | G0 | Column names of the upstream CSVs are mapped to the OSF table names | Upstream used internal column names | None if G0 declares the tables equivalent; otherwise reported |
| F-D6 | driver `tables_spec`, worker table loading (2026-10-22, after the first double run) | Table paths in the worker task files were absolute and contained the run directory, so `artifact_hashes.json` differed between the two runs while every computed file was byte-identical; paths made relative to the task file, scripts re-frozen (new sidecars), both runs repeated | The protocol requires byte-identical manifests | None on any computed value (`fidelity_results.json` of the first attempt equals the repeated run); first attempt kept as `fidelity_run*_attempt0` in the workspace |
| F-D7 | G0 (post-hoc diagnostic) | The pre-registered first-match rule on the eight rounded features pairs each of the 57/59 duplicate input tuples with its first duplicate, i.e. with a different replicate measurement, so the max-relative-difference condition fails on identical tables; a post-hoc diagnostic matching on features + measured diameter + dispersed flow is reported alongside | Rule defect noticed after the run | Reporting only; the pre-registered verdict ("not same table") is kept in the results file and explained in the report |
