"""D3 post-reveal corrected recomputation (register entry D3-R1).

The frozen reveal program passed the *signed* relative error into the applicability
benefit block, whereas D3_BLIND_PROTOCOL.md §7 (and the V1 convention) define
retained-candidate risk on the absolute relative error. Point metrics, the two
pre-declared criteria, the interval block and the candidate-level block used the
absolute value and are unaffected. This script recomputes only the applicability
block from the already-revealed scored rows, adds the physically-invalid
(negative) prediction rate, and writes a separate file. Nothing frozen is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .d3_reveal import AD_RULES, CONTROLS, applicability_block, sanitize


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reveal-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reveal_dir = args.reveal_dir
    results_path = reveal_dir / "blind_reveal_results.json"
    frozen = json.loads(results_path.read_text(encoding="utf-8"))
    out: dict = {
        "schema_version": "d3-posthoc-applicability-correction-v1",
        "status": "post-reveal recomputation; frozen reveal artifacts unchanged",
        "defect": "applicability_block received signed relative error; protocol defines risk on |relative error|",
        "source_hashes": {
            "blind_reveal_results.json": sha256_file(results_path),
            "sealed_primary_scored_rows.csv": sha256_file(reveal_dir / "sealed_primary_scored_rows.csv"),
            "sealed_secondary_scored_rows.csv": sha256_file(reveal_dir / "sealed_secondary_scored_rows.csv"),
        },
        "sets": {},
    }
    for label, file in (("sealed_primary_laboratory_B", "sealed_primary_scored_rows.csv"),
                        ("sealed_secondary_operator_B", "sealed_secondary_scored_rows.csv")):
        rows = pd.read_csv(reveal_dir / file)
        actual = rows["actual_size_nm"].to_numpy(dtype=float)
        pred = rows["predicted_size_nm"].to_numpy(dtype=float)
        abs_rel = np.abs(pred - actual) / actual
        corrected = applicability_block(rows, abs_rel)
        frozen_block = frozen["sealed_primary" if "primary" in label else "sealed_secondary"]["applicability"]
        out["sets"][label] = {
            "n_rows": int(len(rows)),
            "physically_invalid_point_predictions": {
                "n_negative_predicted_size": int((pred <= 0).sum()),
                "fraction_negative_predicted_size": float((pred <= 0).mean()),
                "n_negative_interval_lower": int((rows["interval_lower_nm"].to_numpy(dtype=float) <= 0).sum()),
                "fraction_negative_interval_lower": float((rows["interval_lower_nm"].to_numpy(dtype=float) <= 0).mean()),
            },
            "frozen_full_pool_value_was_signed_mean": frozen_block["full_pool_relative_risk"],
            "corrected_applicability": corrected,
            "median_abs_relative_error_by_strict_minmax": {
                str(k): float(v) for k, v in pd.Series(abs_rel).groupby(rows["accepted__strict_minmax"].to_numpy()).median().items()
            },
        }
    args.output.write_text(json.dumps(sanitize(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    primary = out["sets"]["sealed_primary_laboratory_B"]
    print(json.dumps(sanitize({
        "negative_predictions": primary["physically_invalid_point_predictions"],
        "corrected_full_pool_abs_relative_risk": primary["corrected_applicability"]["full_pool_relative_risk"],
        "beneficial_flags": {k: v["beneficial_under_all_locked_criteria"] for k, v in primary["corrected_applicability"]["rules"].items()},
        "retained_abs_risk": {k: v["retained_relative_risk"] for k, v in primary["corrected_applicability"]["rules"].items()},
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
