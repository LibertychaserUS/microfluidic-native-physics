# H-F 协议 v2:保形偏差对冲组合——A2 前置筛选版(R2-A1、R2-A2 密封档位)

状态:回顾性(post hoc;在 R2-A1、R2-A2 揭盲之后运行)。本协议在两个数据集揭盲**之前**冻结(sha256 侧车 + 提交);程序 `w6/h_portfolio2.py` 首次运行前 sha256 提交。不含预期数字;判定先写死;结果无论正负进 v0.4。v1(`H_F_PORTFOLIO_PROTOCOL_v1.md`)的四个单元与结果保持冻结,不重跑。

## 1. 目的(用户裁定)

v1 在四个单元中只有 A1 S2 成立,且事后观察到:成立单元是唯一同时满足引理假设 A2(|偏置| ≤ q)与偏置主导的单元。v2 把 **A2 作为预注册的前置筛选条件**:只在满足 A2 的单元上检验 Success@K 差;A2 不满足的单元不计入判定、只报告。目的是判断 A1 S2 的成立是否偶然。

## 2. 单元(全部来自 v2 冻结产物;不重训任何源模型)

| 单元 | 偏移集与源模型 | 候选池 | bootstrap 簇列 | 残差池(q、s;对应档位) |
|---|---|---|---|---|
| V1 R2-A1 S2 | `h_R2A1_run1/sealed_prediction_bundle.csv` + 揭盲标签,xgboost native,cell sealed_S2 | 全部 S2 密封行 | `disp_flow` 水平 | `h_R2A1_run1/point_predictions.csv` 单元 S2_dev(留一流体对)xgboost native |
| V2 R2-A1 S3 | 同上 cell sealed_S3 | 全部 S3 密封行 | s2_group(流体对) | 同上(S2 档代理,标注) |
| V3 R2-A2 S2 | `h_R2A2_run1/…` cell sealed_S2 | 全部 S2 密封行 | s3_group(来源论文) | `h_R2A2_run1/point_predictions.csv` 单元 S2_dev(留一可电离脂质)xgboost native |
| V4 R2-A2 S3 | 同上 cell sealed_S3 | 全部 S3 密封行 | 行(单篇论文) | 单元 S1(按来源论文 GroupShuffleSplit,15 种子合并)xgboost native——与"新实验室"档位对应 |
| V5 R2-A2 S4 | 同上 cell sealed_S4 | 全部 S4 密封行 | s3_group(来源论文) | 单元 S1 合并(同 V4;S4 无开发侧类比,标注代理) |

单元的具体行数以揭盲结果为准,本协议不写。簇数 ≥ 8 用簇 bootstrap,否则行 bootstrap(同 v1 §7)。

## 3. 与 v1 相同的定义(不改)

查询 = 偏移集每行观测值(自身排除为主分析,自身包含为次要);q = 对数残差 |r| 的有限样本 0.90 分位,s = r 的样本 SD;策略 = 贪心 top-K、组合(q)、朴素(s)、AD 筛选后贪心(kNN‑5 q95,在该数据集开发集 native 输入上按 v1 §5 规则计算)、区间包含认证、oracle;K ∈ {1, 2, 3, 5};主容差相对 10%,次要绝对 τ_abs = 该数据集 devdiag 的 `tau_absolute`;B = 2000、seed 0、配对;Success@K、Δ_K 与区间同 v1 §6–§7;log 尺度铺开(乘性偏置)。

## 4. A2 前置筛选(写死)

- 对每个单元在偏移集标签上计算 r = log y − log f 的均值 m(post hoc)与 q;**A2 成立 ⇔ |m| ≤ q**。
- 同时报告偏置主导度 b = |m| / SD(r) 与同号比例 s_sign(描述性,不作筛选)。
- 判定集合 U_A2 = 满足 A2 的单元;其余单元只报告 Δ_K 与区间,**不计入判定**。

## 5. 判定(写死)

- 单元在 K 上"成立":Δ_K ≥ 0.10 且 bootstrap 95% 区间下界 > 0(相对容差、自身排除)。单元"成立":K = 3 与 K = 5 同时成立。
- 协议级判定(只用 U_A2):
  - |U_A2| = 0 → "不可判定(无满足 A2 的单元)";
  - U_A2 中所有单元成立 → "A2 筛选下成立";
  - U_A2 中至少一个成立、且没有单元在 K = 5 的区间上界 < 0 → "部分成立(列出成立与不成立的单元)";
  - U_A2 中没有单元成立 → "A2 筛选下不成立"。
- 对照陈述(描述性):U_A2 之外的单元若出现成立,记为"引理假设之外的成立",只报告。
- 跨单元(v1 四单元 + v2 五单元,共九个)描述性报告 Spearman(Δ_5, b)、Spearman(Δ_5, |m|/q),不参与判定。
- "recovers usability under bias-dominated shift" 这句话只有在"A2 筛选下成立"且 A1 S2 的 v1 结果与之同向时才允许写;"部分成立"时只写具体单元。

## 6. 程序与产物

- `scaleup/w6/h_portfolio2.py`(只读冻结产物:两数据集的 `sealed_prediction_bundle.csv`、`point_predictions.csv`、揭盲 `sealed_scored_rows.csv`、custodian `development_labeled.csv` / `sealed_features.csv`;不训练模型)。
- 产物同 v1(`portfolio_summary.json`、`portfolio_selections.csv`(先于评分写出)、`portfolio_scored.csv`、`environment.json`、`artifact_hashes.json`);双跑哈希一致;仓库 `scaleup/h_portfolio/v2/` 存 summary/environment/hashes。台账前缀 `hf.`。
