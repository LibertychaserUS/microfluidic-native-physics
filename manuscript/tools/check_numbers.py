#!/usr/bin/env python3
"""Ledger-driven number reconciliation for the microfluidic manuscript (N8).

Standard library only. For every entry in numbers_ledger.json the script
(1) recomputes the value from the named artifact file, (2) checks the printed
string agrees with it at the stated precision, (3) checks the printed string
occurs in main.tex, and (4) where a `cooccur` regex is given, checks that each
occurrence in the .tex sits next to the required denominator wording.

It also (5) resolves every truncated digest written as \\hashtrunc{prefix}{suffix}
against a registry of full SHA-256 values harvested from the artifact tree
(prefix AND suffix must match one registry entry), (6) checks that the two
canonical run pairs have byte-identical artifact_hashes.json manifests and
that every listed file re-hashes to its manifest entry, and (7) reports any
iCloud-evicted (dataless) file it had to skip instead of silently passing.

Usage:
  python3 tools/check_numbers.py --work /path/to/work/microfluidic_reliability \
      [--tex main.tex] [--ledger numbers_ledger.json]
Exit status is non-zero when any check fails.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
from typing import Any

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
HASHTRUNC = re.compile(r"\\hashtrunc\{([0-9a-f]+)\}\{([0-9a-f]+)\}")

SOURCES = {
    "repro": "artifacts/reproduction_metrics.json",
    "rel": "artifacts/reliability_metrics.json",
    "rec": "blind_artifacts/blind_reveal_results_recovered.json",
    "post": "blind_artifacts/posthoc_analysis_corrected/blind_posthoc_summary.json",
    "cust": "blind_artifacts/custodian_manifest.json",
    "rank": "blind_artifacts/prereveal/blind_candidate_rankings.csv",
    "pred": "blind_artifacts/prereveal/blind_predictions.csv",
    "queries": "blind_artifacts/prereveal/blind_target_queries.csv",
    "lock": "BLIND_PREREGISTRATION_LOCK.json",
    "incident": "BLIND_REVEAL_INCIDENT.md",
    "postmortem": "JSON_SERIALIZATION_POSTMORTEM.md",
    "v1proto": "BLIND_VALIDATION_PROTOCOL.md",
    "v2proto": "PHYSICS_CONSTRAINED_V2_PROTOCOL.md",
    "v2sum": "v2_artifacts_canonical_run2/model_summary_metrics.csv",
    "v2pair": "v2_artifacts_canonical_run2/paired_group_differences_v2p_vs_b1.csv",
    "v2rule": "v2_artifacts_canonical_run2/locked_decision_rule.json",
    "v2search": "v2_artifacts_canonical_run2/retrospective_search_summary.csv",
    "v2cfg": "v2_artifacts_canonical_run2/configuration.json",
    "v2hash": "v2_artifacts_canonical_run2/artifact_hashes.json",
    "anom": "v2_artifacts_canonical_run2/anomaly_register.json",
    "join": "v2_artifacts_canonical_run2/data_join_audit.json",
    "exp_hash": "expanded_artifacts_canonical_run2/artifact_hashes.json",
    "exp_split": "expanded_artifacts_canonical_run2/split_metrics.csv",
    "exp_ext": "expanded_artifacts_canonical_run2/external_model_summary.csv",
    "exp_bin": "expanded_artifacts_canonical_run2/applicability_binary_summary.csv",
    "exp_srcbin": "expanded_artifacts_canonical_run2/applicability_source_binary_summary.csv",
    "exp_aurc": "expanded_artifacts_canonical_run2/applicability_aurc_summary.csv",
    "exp_rc": "expanded_artifacts_canonical_run2/applicability_random_controls.csv",
    "exp_corr": "expanded_artifacts_canonical_run2/applicability_correlations.csv",
    "exp_int": "expanded_artifacts_canonical_run2/interval_summary.csv",
    "exp_replay": "expanded_artifacts_canonical_run2/inverse_replay_summary.csv",
    "exp_boot": "expanded_artifacts_canonical_run2/hierarchical_bootstrap_summary.csv",
    "exp_shuf": "expanded_artifacts_canonical_run2/shuffled_label_control_summary.csv",
    "exp_comp": "expanded_artifacts_canonical_run2/applicability_partition_composition.csv",
    "exp_thr": "expanded_artifacts_canonical_run2/applicability_thresholds.csv",
    "exp_reg": "expanded_artifacts_canonical_run2/experiment_registry.json",
    "exp_audit": "expanded_artifacts_canonical_run2/data_audit.json",
    "exp_igrv": "expanded_artifacts_canonical_run2/internal_group_risk_validation.csv",
    "manifest": "SOURCE_MANIFEST.md",
}

# Files whose full SHA-256 strings populate the digest registry.
REGISTRY_SOURCES = [
    "incident", "rec", "post", "cust", "lock", "v2cfg", "v2hash", "exp_hash",
    "manifest", "exp_audit", "join",
    "expanded_artifacts_canonical_run2/reproducibility_manifest.json",
    "PHYSICS_CONSTRAINED_V2_PROTOCOL.sha256",
    "BLIND_PREREGISTRATION_LOCK.sha256",
    "blind_artifacts/prereveal/blind_prediction_bundle.sha256",
    "w3_artifacts_canonical_run2/configuration.json",
    "d3_blind_artifacts/prereveal/code_and_lock_hashes.json",
    "d3_blind_artifacts/reveal/D3_REVEAL_CONSUMED.json",
    "d3_blind_artifacts/custodian_manifest.json",
    "w4_artifacts_canonical_run2/configuration.json",
    "w5_artifacts_canonical_run2/configuration.json",
]

LITERAL_FIELD = "literal"
COOCCUR_FIELD = "cooccur"

RUN_PAIRS = [
    ("v2_artifacts_canonical_run1", "v2_artifacts_canonical_run2"),
    ("expanded_artifacts_canonical_run1", "expanded_artifacts_canonical_run2"),
    ("w3_artifacts_canonical_run1", "w3_artifacts_canonical_run2"),
    ("w4_artifacts_canonical_run1", "w4_artifacts_canonical_run2"),
    ("w5_artifacts_canonical_run1", "w5_artifacts_canonical_run2"),
]


class Fail(Exception):
    pass


def is_dataless(path: str) -> bool:
    st = os.stat(path)
    return st.st_size > 0 and getattr(st, "st_blocks", 1) == 0


def read_text(path: str) -> str:
    if is_dataless(path):
        raise Fail(f"iCloud dataless placeholder (not materialised): {path}")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


_cache: dict[str, Any] = {}


def load(path: str) -> Any:
    if path in _cache:
        return _cache[path]
    if is_dataless(path):
        raise Fail(f"iCloud dataless placeholder (not materialised): {path}")
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh, parse_constant=_reject_constant)
    elif path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as fh:
            data = list(csv.DictReader(fh))
    else:
        data = read_text(path)
    _cache[path] = data
    return data


def _reject_constant(token: str) -> Any:
    raise Fail(f"non-strict JSON token encountered: {token}")


def json_path(obj: Any, path: str) -> Any:
    """Dotted path with [N] list indices and ['key'] for keys that contain dots."""
    cur = obj
    if path == "":
        return cur
    for part in re.findall(r"\['[^']*'\]|\[\d+\]|[^.\[\]]+", path):
        if part.startswith("['"):
            cur = cur[part[2:-2]]
        elif part.startswith("["):
            cur = cur[int(part[1:-1])]
        else:
            cur = cur[part]
    return cur


def to_float(x: Any) -> float:
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        raise Fail("empty numeric field")
    return float(s)


def csv_select(rows: list[dict], where: dict | None) -> list[dict]:
    if not where:
        return rows
    out = []
    for r in rows:
        ok = True
        for k, v in where.items():
            if isinstance(v, list):
                if r.get(k) not in [str(x) for x in v]:
                    ok = False
                    break
            elif r.get(k) != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def compute(entry: dict, work: str) -> float | str | bool:
    src = entry["src"]
    path = os.path.join(work, SOURCES.get(src, src))
    data = load(path)
    if "grep" in entry:
        if not isinstance(data, str):
            data = json.dumps(data) if not isinstance(data, list) else "\n".join(",".join(r.values()) for r in data)
        if entry["grep"] not in data:
            raise Fail(f"string {entry['grep']!r} not found in {path}")
        return entry["grep"]
    agg = entry.get("agg")
    if path.endswith(".csv"):  # CSV rows (a JSON file may itself be a list)
        rows = csv_select(data, entry.get("where"))
        if agg == "rowcount":
            return float(len(data))
        if agg == "count":
            return float(len(rows))
        if not rows:
            raise Fail(f"no CSV rows match {entry.get('where')} in {path}")
        col = entry["col"]
        vals = [to_float(r[col]) for r in rows]
        if agg is None:
            if len(rows) != 1:
                raise Fail(f"{len(rows)} CSV rows match {entry.get('where')} in {path}; expected 1")
            val = vals[0]
        elif agg == "mean":
            val = statistics.mean(vals)
        elif agg == "sd":
            val = statistics.pstdev(vals)
        elif agg == "min":
            val = min(vals)
        elif agg == "max":
            val = max(vals)
        elif agg == "median":
            val = statistics.median(vals)
        elif agg == "sum":
            val = sum(vals)
        else:
            raise Fail(f"unknown agg {agg}")
    else:  # JSON
        if "paths" in entry:
            a = to_float(json_path(data, entry["paths"][0]))
            b = to_float(json_path(data, entry["paths"][1]))
            op = entry["op"]
            if op == "ratio":
                val = a / b
            elif op == "diff":
                val = a - b
            elif op == "sum":
                val = a + b
            else:
                raise Fail(f"unknown op {op}")
        else:
            raw = json_path(data, entry["path"])
            if "expect" in entry:
                if raw != entry["expect"]:
                    raise Fail(f"{entry['path']} = {raw!r}, expected {entry['expect']!r}")
                return raw
            if agg == "len":
                return float(len(raw))
            val = to_float(raw)
    return val * entry.get("scale", 1)


def tex_regex_for(text: str, fmt: str | None) -> re.Pattern:
    if fmt == "sci":
        mant, exp = text.split("e")
        exp_i = int(exp)
        return re.compile(re.escape(mant) + r"\\times10\^\{" + str(exp_i) + r"\}")
    neg = text.startswith("-")
    body = re.escape(text.lstrip("-"))
    prefix = r"(?<![\d.])-" if neg else r"(?<![\d.])"
    return re.compile(prefix + body + r"(?![\d])")


def find_number(rx: re.Pattern, text: str, tex: str) -> list[re.Match]:
    """All matches, excluding a positive number that is really the magnitude of a
    negative one (preceded by a single '-' that is not part of a '--' range)."""
    hits = []
    for h in rx.finditer(tex):
        if not text.startswith("-") and h.start() >= 1 and tex[h.start() - 1] == "-":
            if not (h.start() >= 2 and tex[h.start() - 2] == "-"):
                continue
        hits.append(h)
    return hits


def check_value(entry: dict, val: float | str | bool, tex: str, failures: list[str]) -> None:
    eid = entry["id"]
    lit = entry.get(LITERAL_FIELD)
    if lit is not None:
        lits = lit if isinstance(lit, list) else [lit]
        for l in lits:
            if l not in tex:
                failures.append(f"[{eid}] literal phrase {l!r} not found in .tex")
    if "text" not in entry:
        return
    texts = entry["text"] if isinstance(entry["text"], list) else [entry["text"]]
    fmt = entry.get("fmt")
    for text in texts:
        if isinstance(val, float):
            if fmt == "sci":
                dec = entry.get("dec", 2)
                shown = f"{val:.{dec}e}"
                mant_v, exp_v = shown.split("e")
                mant_t, exp_t = text.split("e")
                if int(exp_v) != int(exp_t) or abs(float(mant_v) - float(mant_t)) > 0.5 * 10 ** (-dec) + 1e-9:
                    failures.append(f"[{eid}] printed {text} but artifact gives {shown}")
            else:
                dec = entry.get("dec", 2)
                tol = 0.5 * 10 ** (-dec) + 1e-9
                if abs(float(text) - val) > tol:
                    failures.append(f"[{eid}] printed {text} but artifact gives {val:.{dec + 3}f}")
        if entry.get("tex_optional"):
            continue
        rx = tex_regex_for(text, fmt)
        hits = find_number(rx, text, tex) if fmt != "sci" else list(rx.finditer(tex))
        if not hits:
            failures.append(f"[{eid}] printed value {text} not found in .tex")
            continue
        if COOCCUR_FIELD in entry:
            co = re.compile(entry[COOCCUR_FIELD], re.I)
            win = entry.get("window", 220)
            for h in hits:
                ctx = tex[max(0, h.start() - win): h.end() + win]
                if not co.search(ctx):
                    failures.append(f"[{eid}] {text} at offset {h.start()} lacks denominator wording /{entry[COOCCUR_FIELD]}/")


def build_registry(work: str, warnings: list[str]) -> set[str]:
    reg: set[str] = set()
    for src in REGISTRY_SOURCES:
        path = os.path.join(work, SOURCES.get(src, src))
        if not os.path.exists(path):
            warnings.append(f"registry source missing: {path}")
            continue
        try:
            reg.update(HEX64.findall(read_text(path)))
        except Fail as e:
            warnings.append(str(e))
    return reg


def check_hashes(tex: str, registry: set[str], failures: list[str]) -> int:
    n = 0
    for m in HASHTRUNC.finditer(tex):
        n += 1
        prefix, suffix = m.group(1), m.group(2)
        by_prefix = [h for h in registry if h.startswith(prefix)]
        if not by_prefix:
            failures.append(f"[hash] prefix {prefix} matches no registered digest")
            continue
        full = [h for h in by_prefix if h.endswith(suffix)]
        if not full:
            failures.append(f"[hash] {prefix}...{suffix}: prefix found but suffix mismatch (registry ends with {[h[-len(suffix):] for h in by_prefix]})")
        elif len(full) > 1:
            failures.append(f"[hash] {prefix}...{suffix} is ambiguous ({len(full)} digests)")
    for h in set(HEX64.findall(tex)):
        n += 1
        if h not in registry:
            failures.append(f"[hash] full digest {h[:12]}... in .tex is not in the artifact registry")
    return n


def check_run_pairs(work: str, failures: list[str], warnings: list[str]) -> None:
    for a, b in RUN_PAIRS:
        pa, pb = os.path.join(work, a, "artifact_hashes.json"), os.path.join(work, b, "artifact_hashes.json")
        try:
            ta, tb = read_text(pa), read_text(pb)
        except Fail as e:
            failures.append(f"[runpair] {e}")
            continue
        if ta != tb:
            failures.append(f"[runpair] {a} and {b} artifact_hashes.json differ")
        for d, text in ((a, ta), (b, tb)):
            manifest = json.loads(text)
            for name, expected in manifest.items():
                fp = os.path.join(work, d, name)
                if not os.path.exists(fp):
                    failures.append(f"[runpair] {d}/{name} missing")
                    continue
                if is_dataless(fp):
                    warnings.append(f"[runpair] {d}/{name} is an iCloud placeholder; hash check skipped (manifest-committed to {expected[:12]}...)")
                    continue
                with open(fp, "rb") as fh:
                    got = hashlib.sha256(fh.read()).hexdigest()
                if got != expected:
                    failures.append(f"[runpair] {d}/{name} sha256 {got[:12]} != manifest {expected[:12]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--work", default=os.environ.get("MICROFLUIDIC_WORK", ""), help="path to work/microfluidic_reliability")
    ap.add_argument("--tex", default=os.path.join(here, "main.tex"))
    ap.add_argument("--ledger", default=os.path.join(here, "numbers_ledger.json"))
    ap.add_argument("--skip-runpair", action="store_true")
    ap.add_argument("--markdown", action="store_true",
                    help="treat --tex as Markdown: keep '%%' and '$' literally, no comment stripping")
    ap.add_argument("--literal-field", default="literal",
                    help="ledger field holding the literal phrase to require (e.g. literal_zh for the Chinese mirror)")
    ap.add_argument("--cooccur-field", default="cooccur",
                    help="ledger field holding the denominator co-occurrence regex (e.g. cooccur_zh)")
    ap.add_argument("--extra-registry", action="append", default=[],
                    help="additional file (relative to --work) whose SHA-256 strings join the digest registry")
    args = ap.parse_args()
    global LITERAL_FIELD, COOCCUR_FIELD
    LITERAL_FIELD, COOCCUR_FIELD = args.literal_field, args.cooccur_field
    REGISTRY_SOURCES.extend(args.extra_registry)
    if not args.work or not os.path.isdir(args.work):
        print("error: --work must point to the microfluidic_reliability directory", file=sys.stderr)
        return 2

    with open(args.tex, encoding="utf-8") as fh:
        tex = fh.read()
    tex = tex.replace("\u2212", "-")
    if not args.markdown:
        tex = tex.replace("$", "")
        # drop comments so commented-out numbers never count as present
        tex = "\n".join(line.split("%")[0] if "\\%" not in line else re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())

    with open(args.ledger, encoding="utf-8") as fh:
        ledger = json.load(fh)

    failures: list[str] = []
    warnings: list[str] = []
    checked = 0
    for entry in ledger["entries"]:
        try:
            val = compute(entry, args.work)
            check_value(entry, val, tex, failures)
            checked += 1
        except Fail as e:
            failures.append(f"[{entry['id']}] {e}")
        except (KeyError, IndexError, ValueError, TypeError) as e:
            failures.append(f"[{entry['id']}] ledger/artifact error: {e!r}")

    registry = build_registry(args.work, warnings)
    n_hash = check_hashes(tex, registry, failures)
    if not args.skip_runpair:
        check_run_pairs(args.work, failures, warnings)

    print(f"ledger entries checked: {checked}/{len(ledger['entries'])}")
    print(f"digest references checked: {n_hash} against {len(registry)} registered digests")
    for w in warnings:
        print("WARN", w)
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(" -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
