# H-G 协议 v1:源模型残差微调(warm-start residual boosting)

状态:回顾性(post hoc)。作为 posthoc round 3 小样本重校准实验的第 (iv) 臂,加到 round 4;T1/T2 标签均已揭盲。本协议在运行前冻结(sha256 侧车 + 提交);程序 `w3/posthoc_round4_warmstart.py` 首次运行前 sha256 提交。不含预期数字;判定规则先写死;结果无论正负进 v0.4。

## 1. 沿用 round 3 的一切(不改)

- 目标域:T1 = D2 TX100(87 行,冻结 V1 点预测),T2 = D3 实验室 B(57 行,log 目标 XGBoost 按 posthoc_round2 重拟合并断言复现)。加载、k 网格 {1, 2, 3, 5, 8, 10, 15, 20}、随机抽样 200 种子(0–199)、k‑means(random_state 0)、最远点、评估(其余 N − k 行的 MAPE / MdAPE / 带符号偏差)、既有三臂(log 空间 1 参数平移、2 参数仿射 k ≥ 2、从头 XGBoost k ≥ 3)与规则 R1–R5 全部通过 `from w3.posthoc_round3 import …` 复用,**不复制、不改写**。
- 复现断言:新脚本重算既有三臂,其每个 k 的随机抽样中位数必须与仓库内冻结的 `posthoc_round3/recalibration_summary.json` 逐一相等(容差 1e‑9);不等则中止,不出报告。

## 2. 新臂 (iv):warm-start residual boosting

- 残差模型:`XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, min_child_weight=1, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, objective="reg:squarederror", random_state=42, n_jobs=1)`——写死,不调参。
- 训练:`fit(X, log y, base_margin=log f_source(X), sample_weight=w)`;目标行 = 被抽中的 k 行(w = 1)。
- 源行降权 λ ∈ {0, 0.1}:λ = 0 只用 k 行;λ = 0.1 另加入全部源域开发行(T1:381 行;T2:246 行实验室 A),其 base_margin = 源模型对这些行的**样本内**预测(log),w = 0.1。
- 预测:`predict(X_eval, base_margin=log f_source(X_eval))` 后取指数;log f_source 用 round 3 的 FLOOR 下限。
- 源行的 f_source:T2 由 round 3 同一代码重拟合(构造上相同);T1 需用 `src/blind_predict.build_locked_model` 的声明(XGBoost 100/3/0.3, seed 42, 4 个 V1 输入、原尺度 mm)在 381 行开发集上重拟合,并断言其对 87 行 TX100 的预测与冻结 `blind_predictions.csv` 逐行相等(容差 1e‑6)。断言失败 → T1 的 λ = 0.1 臂记为"不可评估",只跑 λ = 0;报告中说明。
- 每个 (k, 抽样, 种子) 的抽样索引与既有三臂完全相同(同一 `rng.choice`),保证配对。

## 3. 指标与规则

- 对 warm 臂 (λ = 0, 0.1) 各自计算:随机抽样的 MAPE 中位数 / p05 / p95、MdAPE 中位数;R1(中位数与 p95 都低于未校准基线中位数);R2(随机中位数首次 ≤ round 3 参考线的最小 k;参考线沿用 round 3 文件内常量);R4(k‑means / 最远点在随机分布中的百分位 ≤ 0.10 的 k 计数)。
- R5(保形重校准)**不对 warm 臂评估**:它需要未用于拟合的校准行,而 k 行已全部用于拟合;声明为不适用。
- 新配对比较 R6:对每个可比较的 k,配对差 Δ_c(k) = MAPE_warm − MAPE_c 在 200 个随机抽样上的中位数与 p05/p95,c ∈ {affine(k ≥ 2), scratch(k ≥ 3)},λ 分别报告。

## 4. 判定(先写死)

- 对比较对象 c,"warm 臂在全部 k 上更优"这句话只在以下条件满足时允许写:在可比较的 k 中,满足"Δ_c(k) 中位数 < 0 且 p95 < 0"的 k 数 ≥ 可比较 k 数 − 2(即 round 4 指令的"≥ 6/8"映射到可比较集合:affine 7 个 k 中 ≥ 5,scratch 6 个 k 中 ≥ 4)。
- 否则只报告各 k 的实际结果,不作跨 k 概括;措辞限定 "in these two target domains, post hoc"。
- λ = 0 与 λ = 0.1 各自独立判定;不取其优者。

## 5. 产物

- 运行目录 `WORK/posthoc_review_round4_hg_run1`、`run2`:`warm_start_summary.json`、`warm_start_curves.csv`(含既有三臂与两个 warm 臂的全部抽样结果)、`environment.json`、`artifact_hashes.json`;两次运行哈希逐字节一致。
- 仓库 `scaleup/posthoc_round4/` 存放 summary / curves / environment / hashes 字节拷贝。台账前缀 `p4.`;中英双稿留待 v0.4。

```
round 3 (冻结)            本协议新增
┌──────────────┐          ┌───────────────────────────────┐
│ k 行抽样 idx │ ───────► │ base_margin = log f_source     │
│ baseline     │          │ + 浅层残差树 (100×depth2, lr .1)│
│ scale/affine │          │ λ∈{0, 0.1} 源行降权            │
│ scratch      │          └───────────────┬───────────────┘
└──────┬───────┘                          │ 同一 N−k 评估行
       └──────── 配对差 Δ_affine, Δ_scratch ◄┘ ──► R6 判定(§4)
```
