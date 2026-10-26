# POSTHOC_ROUND3_RECALIBRATION_PROTOCOL_v1 — 小样本目标域重校准,冻结版 v1

- 日期:2026-10-21;用户批准原话:"同意在已揭盲数据上做重校准实验;冻结协议开跑"。
- 状态:**冻结**(sha256 侧车;改动须另存 v2 并登记)。性质:**回顾性/事后**——两个目标域的标签均已揭盲(TX100 由 V1 揭盲于 2026-07-16;D3 实验室 B 由 D3 揭盲于 2026-10-17);预注册只封死分析者自由度。
- 目录:工作区 `posthoc_review_round3/`;字节拷贝 `scaleup/posthoc_round3/`;台账前缀 `p3.`。脚本 `scaleup/w3/posthoc_round3.py` 在首次运行前 sha256 并提交。

## 1. 数据与源模型(冻结,不重训)

| 目标域 | 行 | 输入文件(工作区) | 源模型预测 f(x) | 复现断言 |
|---|---|---|---|---|
| T1:D2 TX100 组 | 87 | `blind_artifacts/blind_features.csv`(输入)、`blind_artifacts/sealed_blind_labels.csv`(标签)、`blind_artifacts/prereveal/blind_predictions.csv`(V1 冻结点预测) | 读取冻结预测,不重新拟合 | 读入预测对标签的 MAPE 必须等于 19.8436%(4 位) |
| T2:D3 实验室 B | 57 | `d3_blind_artifacts/development_labeled.csv`(246 行开发集)、`d3_blind_artifacts/reveal/sealed_primary_scored_rows.csv`(57 行输入 + 标签 + 冻结原尺度预测) | 对数目标模型按 `posthoc_round2.py` 的 `d3_analyses` 完全相同的配置(锁定 XGBoost 100/3/0.3 seed 42,六数值输入 + 独热介质,目标 log(size),246 行)重新拟合一次作为源模型 | 其实验室 B MAPE 必须等于 43.05%(2 位),否则中止 |

## 2. 因子(写死)

- k ∈ {1, 2, 3, 5, 8, 10, 15, 20}。
- 抽样:(a) 随机,200 次,种子 0–199(`numpy.random.default_rng(seed)`,无放回抽 k 行);(b) k-means 空间填充:目标域输入标准化(均值/方差用**源模型训练集**的输入统计),k-means(k 簇,`random_state=0`, n_init=10),取每簇最近质心行,1 次;(c) farthest-point:起点 = 到源训练集输入均值(标准化空间)欧氏距离最大的目标域行,逐步加入到已选集合最小距离最大的行,1 次。
- 校准形式(均在 log 空间最小二乘,预测与标签取 log;预测 ≤ 0 的行先以 1e−6 mm/nm 截断再取 log,并记录截断行数):(i) 1 参数缩放 log ŷ = log f + a;(ii) 2 参数仿射 log ŷ = α·log f + β(k ≥ 2);(iii) 对照:同 k 行从头训练 XGBoost(锁定配置,seed 42;T2 用 log 目标;k ≥ 3);(iv) 基线:未校准源模型。
- 评估:其余 N−k 行 MAPE(主)、MdAPE、带符号相对偏差(exp 回原尺度后计算);k ≥ 10 时另做 split-conformal 重标定:用 k 行的 |log 残差| 有限样本分位数(名义 0.90,⌈(k+1)·0.9⌉/k 规则)作半径,报告其余行覆盖率与平均相对宽度。

## 3. 判读规则 R1–R5(冻结,无预期值)

- R1 有效:对每个 k,随机抽样 200 次 MAPE 的中位数与 [5%, 95%] 分位;"有效" = 中位数 < 基线 MAPE 且 95% 分位 < 基线 MAPE。
- R2 参照线:T1 以 11.4%(V1 预注册复现阈值),T2 以 21.11%(对数模型自身开发参照的 1.25×);报告随机抽样中位数首次 ≤ 参照线的最小 k;若 k ≤ 20 内未达到,报 "not reached within k ≤ 20"。
- R3 对照:k ≥ 3,仿射校准 vs 从头训练:中位数更低且 200 次配对差的 [5%, 95%] 区间不含 0 → "校准优于从头训练"。
- R4 选点:k-means 与 farthest-point 的 MAPE 在随机 200 次分布中的分位;两者均 ≤ 0.10 分位且在 ≥ 4 个 k 上成立 → 可写 "label-free distance can guide which target rows to label, in these data";否则不写。
- R5 覆盖:k ≥ 10 的重标定覆盖率是否落在 [0.85, 0.95];全部报告。
- 全部数字进台账 `p3.*`;稿件标 "post hoc, labels already open"。

## 4. 产物

`recalibration_summary.json`(目标域 × k × 策略 × 形式:中位数、分位、R1–R5 判定)、`recalibration_curves.csv`(逐重复)、`environment.json`(输入 sha256、库版本)、`artifact_hashes.json`;两次独立运行目录逐字节一致(随机种子固定)。
