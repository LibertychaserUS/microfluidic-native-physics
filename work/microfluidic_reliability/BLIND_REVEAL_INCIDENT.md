# Blind-reveal serialization incident record

## Scope

This record documents a post-reveal file-writing failure in the prospective
TX100 sealed-label validation. It does not alter the frozen predictions,
rankings, protocol, model, applicability thresholds, target queries, or metric
definitions.

## Frozen state before reveal

- Protocol SHA-256: `1402752425f111a97222ae5c4cb7d13bd68ca1fdfabab978594ac79370e68b1c`
- Pre-registration lock SHA-256: `1f6901bc79de2cd189c7813d63e16ba561b7c502ce42721f4ed9e15a2357edc7`
- Prediction bundle SHA-256: `2b6eb42d709b367c563e62ca7c82ee764c06ac2b67dffc9f1127b3b732d6932d`
- Original frozen reveal implementation SHA-256: `46a289ad4369b10b56f8d1c233ba3a401769dec8fbcf392b1235a0baebd27a48`
- Blind rows: 87
- Fixed target queries: 78 across six strategies

## Incident

The one-time reveal gate was consumed at
`2026-07-16T12:30:53.694820+00:00`. The program verified all frozen hashes,
created the exclusive reveal gate, read and joined the sealed labels, and
computed the locked endpoints. Final strict-JSON serialization then failed
because unanswered target queries contained Python `NaN` values. The intended
result file was therefore never created.

- Consumed gate SHA-256: `e285c4cb4ad7db74b9a3aca4b6240759081095c821865ae76852486f6fb13344`
- Original intended output: absent
- Failure class: representation/serialization only
- Model fitting, candidate ranking, thresholds, and metric definitions: unchanged

## Recovery

A separate recovery program was added without modifying the original frozen
reveal implementation. It revalidated the original bundle, protocol, code,
table, label, and gate hashes; verified exact opaque-ID set equality; recomputed
the same locked metrics from the already-unsealed labels; and converted only
undefined `NaN` query fields to JSON `null`.

- Recovery implementation SHA-256: `06b6099e0d7c754681bcee7de8f16b8eec8c6c902d4ca996d3e9d0086e263a74`
- Recovery time: `2026-07-16T12:32:58.798242+00:00`
- Recovered result SHA-256: `008d026163af37b7ecddf220d2ffc5aea0818784561a389419e088cf0eaa1624`
- Frozen pre-reveal artifacts modified: no
- Model refit or retuning: no
- Rankings recomputed: no
- Endpoint definitions changed: no

## Interpretation boundary

The validation remains prospective with respect to all predictions, masks,
rankings, and decision rules because those artifacts were frozen and hashed
before the first label access. The write failure and recovery must nevertheless
be disclosed in the paper and release package; the run should not be described
as operationally flawless.
