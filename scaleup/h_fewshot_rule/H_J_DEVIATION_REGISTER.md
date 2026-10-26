# H-J deviation register

Numbered deviations from `H_J_FEWSHOT_RULE_PROTOCOL_v1.md` or observations outside its text, with the
run they first affected.

| ID | Where | Entry | Effect |
|---|---|---|---|
| H-J-D1 | affine method, 6 of 25,400 draws (run1 = run2) | The two-parameter log-affine map extrapolated to an infinite MAPE (`exp` overflow in `posthoc_round3.calibrate`) on 6 draws | Tier gains and regrets are medians over 100 draws, so an infinite draw changes no reported value; the count is recorded in `posthoc/descriptive.json` |
| H-J-D2 | affine method, 2,103 of 25,400 draws | The k source predictions were constant, so the pre-declared substitution by the one-parameter shift applied (`affine_fallback` = 1 in `draws.csv`) | Pre-declared in protocol Section 3; counted here |
| H-J-D3 | reporting | Overall fractions of tiers with a positive gain and overall medians per k are not in the frozen analysis output; they are computed post hoc from the frozen `tier_summary.csv` by `scaleup/w6/h_fewshot_rule_posthoc.py` into `posthoc/descriptive.json` | Descriptive only; the pre-registered decision uses `results.json` |
| H-J-D4 | analyzer, cosmetic | `numpy` warnings "All-NaN slice" are printed for the scratch method at k = 2, where scratch is undefined by protocol (k ≥ 3) | No effect on any value |
