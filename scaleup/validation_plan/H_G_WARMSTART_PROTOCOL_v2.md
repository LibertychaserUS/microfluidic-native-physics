# H-G 协议 v2:源模型残差微调——新数据集密封档位(R2-A1、R2-A2;post hoc)

状态:回顾性(在 R2-A1、R2-A2 揭盲之后运行)。本协议在两个数据集揭盲**之前**冻结(sha256 侧车 + 提交);程序 `w3/posthoc_round4_warmstart2.py` 首次运行前 sha256 提交。不含预期数字;判定先写死;结果无论正负进 v0.4。v1(T1 = D2 TX100、T2 = D3 实验室 B)的结果保持冻结。

## 1. 目标域

| 域 | 行 | 源模型 f_source(冻结) | 源行(λ > 0) |
|---|---|---|---|
| T3 R2-A1 S2 | sealed_S2 密封行 | `h_R2A1_run1` 预测包中 xgboost native 的点预测(训练于全部开发行) | R2-A1 开发集 523/557 行(以 custodian 为准),base_margin = 锁定 XGBoost 在开发集上重拟合的样本内预测;重拟合须复现预测包中的密封预测(容差 1e‑6),否则 λ = 0.1 臂记不可评估 |
| T4 R2-A1 S3 | sealed_S3 密封行 | 同上 | 同上 |
| T5 R2-A2 S2 | sealed_S2 密封行 | `h_R2A2_run1` 预测包 xgboost native(log 目标) | R2-A2 开发集,同上 |
| T6 R2-A2 S3 | sealed_S3 密封行 | 同上 | 同上 |
| T7 R2-A2 S4 | sealed_S4 密封行 | 同上 | 同上 |

目标尺度随该数据集 devdiag(raw/log);MAPE 一律在原尺度计算。

## 2. 沿用 round 3 / H-G v1 的一切

k 网格 {1, 2, 3, 5, 8, 10, 15, 20}(**只用 N − k ≥ 10 的 k**,N 为该域行数;其余 k 记"不适用");随机抽样 200 种子(0–199);k‑means(random_state 0);最远点(从开发集输入均值的标准化距离起);既有三臂(1 参数 log 平移、2 参数 log 仿射 k ≥ 2、从头 XGBoost k ≥ 3,从头模型的目标尺度随 devdiag);新臂 warm λ ∈ {0, 0.1},残差模型 `XGBRegressor(100, depth 2, lr 0.1, min_child_weight 1, subsample 1, colsample 1, reg_lambda 1, seed 42)`,`base_margin = log f_source`;评估在其余 N − k 行(MAPE / MdAPE / 带符号偏差)。

## 3. 规则

- R1(有效):随机抽样中位数与 p95 都低于未校准基线中位数。
- R2(参考线):**参考线 = 该数据集的 dev-ref MAPE**(h_run2 的 `dev_ref.mape_mean`;意为"把目标域误差拉回开发集水平");报告随机中位数首次 ≤ 参考线的最小 k。
- R4:k‑means / 最远点在随机分布中的百分位 ≤ 0.10 的 k 计数。
- R5:对 warm 臂不适用(同 v1)。
- R6:对仿射(k ≥ 2)与从头(k ≥ 3)的 200 抽样配对差中位数与 p05/p95;"warm 臂在全部可比较 k 上更优"仅当"中位数 < 0 且 p95 < 0"的可比较 k 数 ≥ 可比较 k 数 − 2;否则只报各 k。λ 分别判定。
- 跨域概括:只有当 R6 在 ≥ 3 个域(T3–T7)对同一比较对象允许"全部可比较 k 更优"时,才允许写"在新数据集的多数密封档位上更优";否则逐域报告。v1 的 T1/T2 结论与之并列报告,不合并。

## 4. 产物

`WORK/posthoc_review_round4_hg2_run{1,2}`:`warm_start_summary.json`、`warm_start_curves.csv`、`environment.json`、`artifact_hashes.json`;双跑哈希一致;仓库 `scaleup/posthoc_round4/v2/` 存字节拷贝。台账前缀 `p4.`。
