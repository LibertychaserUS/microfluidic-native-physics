from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
SRC = WORKSPACE / "work" / "microfluidic_reliability" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import strict_json


def reject_nonstandard_constant(token: str):
    raise ValueError(f"non-standard JSON constant: {token}")


class StrictJsonTests(unittest.TestCase):
    def test_nan_variants_are_written_as_null_and_strictly_parseable(self) -> None:
        payload = {
            "builtin": float("nan"),
            "numpy": np.float64(np.nan),
            "pandas": pd.NA,
            "nested": [1.0, np.array([2.0, np.nan])],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            strict_json.write_strict_json(path, payload)
            text = path.read_text(encoding="utf-8")
        parsed = json.loads(text, parse_constant=reject_nonstandard_constant)
        self.assertIsNone(parsed["builtin"])
        self.assertIsNone(parsed["numpy"])
        self.assertIsNone(parsed["pandas"])
        self.assertEqual(parsed["nested"], [1.0, [2.0, None]])
        self.assertNotIn("NaN", text)

    def test_infinity_is_rejected_instead_of_hidden(self) -> None:
        for value in (float("inf"), float("-inf"), np.float64(np.inf)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "infinite value"):
                    strict_json.dumps_strict({"value": value})


if __name__ == "__main__":
    unittest.main()
