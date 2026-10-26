"""E4 two-run identity check with the wall-clock column removed (deviation E-D10).

    python -I scaleup/w7/closed_loop_e4_identity.py --run1 W/e4_mac_run1 --run2 W/e4_mac_run2 --out W/e4_identity

`families.csv` and `table_cv.csv` carry a `fit_seconds` column (wall-clock fit time), which cannot be byte-identical
across runs.  This script reports, for every output file, whether the two runs are equal column by column, and the
sha256 of each CSV with `fit_seconds` dropped (re-serialised with the runner's float format).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

TIMING = "fit_seconds"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stripped(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    if TIMING in df.columns:
        df = df.drop(columns=[TIMING])
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.12g")
    return df, sha_bytes(buf.getvalue().encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", type=Path, required=True)
    ap.add_argument("--run2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    h1 = json.loads((a.run1 / "artifact_hashes.json").read_text()); h2 = json.loads((a.run2 / "artifact_hashes.json").read_text())
    files = {}
    for name in sorted(set(h1) | set(h2)):
        entry = {"byte_identical": h1.get(name) == h2.get(name), "sha256_run1": h1.get(name), "sha256_run2": h2.get(name)}
        if name.endswith(".csv"):
            d1, s1 = stripped(a.run1 / name); d2, s2 = stripped(a.run2 / name)
            entry.update({"rows": [int(len(d1)), int(len(d2))], "differing_columns_raw": [c for c in pd.read_csv(a.run1 / name).columns
                                                                                         if not pd.read_csv(a.run1 / name)[c].equals(pd.read_csv(a.run2 / name)[c])],
                          "sha256_without_fit_seconds_run1": s1, "sha256_without_fit_seconds_run2": s2, "identical_without_fit_seconds": s1 == s2})
        files[name] = entry
    ok = all(v.get("identical_without_fit_seconds", v["byte_identical"]) for v in files.values())
    res = {"schema_version": "closed-loop-e4-identity-v1", "deviation": "E-D10", "timing_column": TIMING, "files": files, "identical_except_timing": ok}
    (out / "e4_identity.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"identical_except_timing": ok, **{k: (v["byte_identical"], v.get("identical_without_fit_seconds")) for k, v in files.items()}}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
