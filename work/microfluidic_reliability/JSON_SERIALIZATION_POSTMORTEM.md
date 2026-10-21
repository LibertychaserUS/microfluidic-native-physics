# Strict JSON postmortem

Date: 2026-07-16  
Status: root cause confirmed; regression gate added

## Symptom

Research runs completed successfully, but some summary files contained bare
`NaN` tokens. Python's own decoder accepted those files while standards-compliant
JSON consumers such as JavaScript `JSON.parse` rejected them.

The pre-fix expanded summary contained 279 non-finite fields. A strict parser
failed at the first `"n_sources": NaN` field.

## Root cause chain

1. Statistical code legitimately uses IEEE NaN for undefined quantities, for
   example a conditional metric when a rule retains zero candidates.
2. Pandas also inserts NaN when records with different optional fields are
   combined into one table, for example `n_sources` on source-macro rows only.
3. These tables were converted to Python dictionaries without normalizing
   missing values at the serialization boundary.
4. Python's `json.dumps` defaults to `allow_nan=True` and emits the non-standard
   tokens `NaN`, `Infinity`, and `-Infinity`.
5. A `default=` serializer does not intercept built-in float values; therefore
   it cannot repair `float('nan')`.
6. Python's `json.loads` is permissive by default and accepts the same tokens.
   Tests that wrote and read with Python therefore produced a false pass.
7. Different scripts implemented JSON output independently. Some used strict
   output, some permissive output, and the original blind reveal used strict
   rejection without recursively sanitizing built-in float NaN.

The fundamental defect was the absence of one strict artifact-boundary contract
and a cross-parser regression test. NaN in the analytical layer was not itself
the bug.

## Corrective action

- Added `src/strict_json.py` as the shared non-frozen artifact writer.
- Recursively maps undefined NaN and pandas missing values to JSON `null`.
- Rejects positive or negative infinity instead of hiding numerical divergence.
- Forces `allow_nan=False` for every shared writer call.
- Added tests covering built-in float NaN, NumPy NaN, pandas NA, nested arrays,
  and infinity rejection.
- Final release validation uses a strict parser, not only Python's permissive
  defaults.
- Frozen blind-validation source files remain unchanged; their serialization-
  only recovery and hashes remain part of the audit trail.

## Verification

The focused strict-JSON tests and the complete 29-test suite passed after the
fix. All expanded outputs generated before this correction are excluded from the
VERIFIED release and must be regenerated after the remaining statistical audit
fixes.

