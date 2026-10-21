"""Strict JSON boundary for research artifacts.

Python's standard encoder deliberately emits non-standard ``NaN`` and
``Infinity`` tokens unless ``allow_nan=False`` is supplied.  Research tables
also use NaN legitimately for undefined statistics, so every artifact must
cross one shared normalization boundary before serialization.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sanitize(value: Any) -> Any:
    """Convert undefined missing values to null and reject infinities."""

    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): sanitize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(child) for child in value]
    if isinstance(value, np.ndarray):
        return [sanitize(child) for child in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError("infinite value cannot enter a JSON artifact")
        return number
    if isinstance(value, Path):
        return str(value)
    return value


def dumps_strict(
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> str:
    """Serialize with RFC-compatible non-finite handling."""

    return json.dumps(
        sanitize(payload),
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def write_strict_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
) -> None:
    path.write_text(
        dumps_strict(payload, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
