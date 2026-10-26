"""ENV-B worker for FIDELITY_PROTOCOL_v1 (runs under xgboost 1.3.3, Python 3.9, x86_64).

Reads a task file written by fidelity_driver.py, fits the upstream constructor
``XGBRegressor(tree_method="hist")`` (all other arguments at their library defaults) with the
upstream StandardScaler step (variant M4) and without it (variant M5), and writes per-row
predictions.  It also loads the deployed upstream model file and predicts under both feeding
conventions of protocol Section 6.  No metric is computed here; the driver scores everything.

Python 3.9 compatible on purpose (no PEP 604 unions at runtime).
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.preprocessing import StandardScaler


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_constructor(seed):
    # Upstream: xgb.XGBRegressor(tree_method="hist")  [XGBoost_gloval_stability.py; DAFD3_models.py]
    # Added for reproducibility only (protocol F-D1, F-D2): random_state, n_jobs.
    return xgboost.XGBRegressor(tree_method="hist", random_state=int(seed), n_jobs=1)


def fit_predict(x_train, y_train, x_test, seed, scale):
    if scale:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    model = upstream_constructor(seed)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.tasks, "r", encoding="utf-8") as stream:
        tasks = json.load(stream)

    tasks_dir = os.path.dirname(os.path.abspath(args.tasks))
    tables = {}
    for name, spec in tasks["tables"].items():
        path = spec["path"]
        if not os.path.isabs(path):
            path = os.path.join(tasks_dir, path)  # relative to the task file (F-D6)
        frame = pd.read_csv(path)
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError("table digest mismatch for %s" % name)
        tables[name] = frame

    rows = []
    for task in tasks["tasks"]:
        kind = task["kind"]
        if kind in ("split", "fit_predict"):
            table = tables[task["table"]]
            features = task["features"]
            target = task["target"]
            train_idx = np.asarray(task["train_idx"], dtype=int)
            test_idx = np.asarray(task["test_idx"], dtype=int)
            test_table = tables[task.get("test_table", task["table"])]
            x_train = table.iloc[train_idx][features].to_numpy(dtype=float)
            y_train = table.iloc[train_idx][target].to_numpy(dtype=float)
            x_test = test_table.iloc[test_idx][features].to_numpy(dtype=float)
            for variant, scale in (("M4", True), ("M5", False)):
                pred = fit_predict(x_train, y_train, x_test, task["seed"], scale)
                for row_index, value in zip(test_idx.tolist(), pred.tolist()):
                    rows.append(
                        {
                            "task_id": task["task_id"],
                            "variant": variant,
                            "row_index": int(row_index),
                            "predicted_normalized": float(value),
                        }
                    )
        elif kind == "deployed":
            if sha256_file(task["model_path"]) != task["sha256"]:
                raise RuntimeError("deployed model digest mismatch")
            booster = xgboost.Booster()
            booster.load_model(task["model_path"])
            table = tables[task["table"]]
            x_raw = table[task["features"]].to_numpy(dtype=float)
            x_std = StandardScaler().fit(x_raw).transform(x_raw)
            for convention, x_eval in (("C-raw", x_raw), ("C-std", x_std)):
                pred = booster.predict(xgboost.DMatrix(x_eval))
                for row_index, value in enumerate(pred.tolist()):
                    rows.append(
                        {
                            "task_id": task["task_id"],
                            "variant": "deployed_" + convention,
                            "row_index": int(row_index),
                            "predicted_normalized": float(value),
                        }
                    )
        else:
            raise ValueError("unknown task kind %s" % kind)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    env = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "platform": platform.platform(),
        "xgboost": xgboost.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "constructor": "xgboost.XGBRegressor(tree_method='hist', random_state=seed, n_jobs=1)",
        "default_params_as_seen": {
            key: value
            for key, value in upstream_constructor(0).get_params().items()
            if key
            in (
                "n_estimators",
                "learning_rate",
                "max_depth",
                "reg_lambda",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "base_score",
                "objective",
                "tree_method",
                "booster",
            )
        },
    }
    with open(args.out.replace(".csv", "_environment.json"), "w", encoding="utf-8") as stream:
        json.dump(env, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
