# H-F 协议 v1:保形偏差对冲组合(conformal bias-hedging portfolio)

状态:回顾性(post hoc)。四个单元的标签均已揭盲;本协议在任何组合评分运行之前冻结(sha256 侧车 + 提交),运行程序 `w6/h_portfolio.py` 同样在首次运行前 sha256 提交。协议中不含任何预期数字;判定规则先写死;结果无论正负进 v0.4。

## 1. 待验证陈述

当目标域偏移以近似恒定的系统性偏置为主,且偏置落在分组校准的保形半径 q 内时,把 K 个候选的预测值按子目标 d\*(m) = d\* + ((2m − K − 1)/K)·q 铺开(受硬边界与候选互异约束),其 Success@K 高于贪心 top-K。

本协议把"偏置"与 q 都定义在对数尺度(乘性偏置;与 A1/D3 的 log 目标及 H-A 的倍率表述一致):log d\*(m) = log d\* + ((2m − K − 1)/K)·q,q 为对数残差的保形半径。原生加性版本不另评(避免双重判定)。

## 2. 机制:target-search replay(复用既有定义)

- 查询 = 偏移集每一行的观测值 d\* = y_i(逐行;不是网格目标)。
- 候选池 = 同一偏移集(或同一上下文)中**除查询行自身以外**的行;候选的"预测值"= 冻结源模型对该候选输入的预测 f_c。自身排除是主分析(避免 oracle 恒为 1);含自身的版本只作次要报告。
- 选择只用 f_c 与 d\*;y_c 只在评分时读取(replay 的既有纪律:先固定选择、后揭示结果)。
- Success@K:被选的 ≤ K 个候选中至少一个满足 |y_c − d\*| ≤ τ。
- Oracle 上限:候选池中存在 |y_c − d\*| ≤ τ 的非自身行。

## 3. 单元(全部来自冻结产物;本协议**不重训任何源模型**)

| 单元 | 偏移集与源模型(冻结产物) | 查询数 | 候选池上下文 | bootstrap 簇列(簇数) |
|---|---|---|---|---|
| U1 D1 external | W3 单元 `D1_external`、模型 `xgboost`,`w3_artifacts_canonical_run2/point_predictions.csv`(cell = D1_external, model = xgboost) | 64 | 同一文献来源(`group` 列,4 个来源) | 来源(4;< 8 → 行 bootstrap) |
| U2 D2 TX100 | W3 单元 `D2_chemistry_loco` 折 repeat = 2(留出 CMC 3.5 = TX100)、模型 `xgboost`,同一文件 | 87 | 全部 TX100 行 | 表面活性剂浓度(10) |
| U3 A1 S2 | `h_A1_run1/point_predictions.csv`(cell = sealed_S2, model = xgboost, feature_set = native) | 40 | 全部 S2 密封行 | Qc 水平(13) |
| U4 A1 S3 | 同上 cell = sealed_S3 | 200 | 全部 S3 密封行 | 流体族(9) |

U2 采用 W3 矩阵的 LOCO TX100 折(与 U1 同一模型类与产物链),不采用 V1 预注册密封预测(两者训练行数不同);V1 版本不另评。

## 4. 保形半径 q 与朴素 SD(不读目标域标签)

对数残差池 r = log y − log f 取自**冻结的组外预测**(模型未见该组):

| 单元 | 残差池(对应档位) |
|---|---|
| U1 | W3 单元 `D1_geometry` 的 xgboost 测试行(30 次按几何依赖组划分的组外预测,合并) |
| U2 | W3 单元 `D2_chemistry_loco` 中 repeat ≠ 2 的 5 个折的 xgboost 测试行(每折模型未见该化学组) |
| U3、U4 | `h_A1_run1/point_predictions.csv` 单元 `S2_dev`(留一流体族,xgboost native)的 661 行 |

U4 的档位是器件实例,但 A1 开发集只有一个器件,无法做 S3 档位的分组校准;按协议用 S2 档位残差池作为代理,并在报告中标注为代理。

- q = |r| 的有限样本 0.90 分位:水平 = min(1, ⌈(n + 1)·0.9⌉/n),分位方法 `higher`(与 `blind_predict.split_conformal` 同)。
- 朴素 SD:s = r 的样本标准差(ddof = 1)。
- 两者都只用源域/开发侧的标签,不读偏移集标签。

## 5. 策略(每个单元 × 每个 K × 每种容差)

1. **贪心 top-K**:按 |log f_c − log d\*| 升序取前 K 个非自身候选(并列按 row_id 字典序)。
2. **组合(q)**:对 m = 1 … K 依次,子目标 log d\*(m) = log d\* + ((2m − K − 1)/K)·q;取尚未被选的、|log f_c − log d\*(m)| 最小的非自身候选(并列按 row_id)。K = 1 时退化为贪心。硬边界:正值性由指数形式自动满足;四个单元均无其他已知物理上界,故不设上界。
3. **朴素夹逼(s)**:同 2,把 q 换成 s。
4. **AD 筛选后贪心**:先剔除 kNN‑5 q95 适用域之外的候选,再贪心。U1/U2 用 W3 `ad_scores.csv` 中该单元的 `accepted__knn_5__q0_95`(冻结);U3/U4 在脚本内按同一规则计算:标准化 native 7 输入(用开发集均值/标准差),开发集留一的 5‑NN 距离的 0.95 分位为阈值,候选的 5‑NN 距离(参照 = 开发集)≤ 阈值为通过。若通过的候选少于 K,则只选通过者。
5. **区间整体包含认证(既有规则)**:贪心最优候选同时满足 |f_c − d\*| ≤ τ 且 [f_c·e^{−q}, f_c·e^{q}] ⊂ [d\* − τ, d\* + τ] 时"回答";报告回答率、回答中的 Success(精确率)与"回答且成功"的查询占比。
6. **Oracle 上限**:§2 定义。

## 6. 容差与 K

- 主分析:相对容差 τ = 0.10·d\*。
- 次要:绝对容差 τ_abs = 0.10 × 开发集目标中位数:U1 4.9115 µm(D1 综合表中位 49.115 µm);U2 15.205 µm(D2 非 TX100 行中位 152.05 µm);U3/U4 33.089 µm(A1 devdiag)。次要分析只报告,不参与判定。
- K ∈ {1, 2, 3, 5}。

## 7. 不确定度:bootstrap

- B = 2000,`numpy.random.default_rng(0)`;配对:同一次重抽样同时用于组合与贪心,统计量 = Success@K(组合) − Success@K(贪心)。
- 簇 bootstrap(按 §3 簇列整簇重抽样)用于簇数 ≥ 8 的单元(U2、U3、U4);U1 簇数 4,用行 bootstrap 并标注。95% 区间取 2.5/97.5 百分位。

## 8. 判定(先写死)

- 单元在 K 上"成立":Δ_K = Success@K(组合) − Success@K(贪心) ≥ 0.10 且 bootstrap 95% 区间不含 0(下界 > 0),相对容差。
- 单元"成立":K = 3 与 K = 5 同时成立。K = 1、2 只报告。
- 机理预注册(跨单元):每单元在偏移集上计算偏置主导度(标签已开,post hoc):s_sign = 残差同号比例的最大值 max(P(r > 0), P(r < 0)),b = |mean r| / SD(r),r = log y − log f。计算 Spearman ρ(Δ_5, b) 与 ρ(Δ_5, s_sign)(四个单元,n = 4,只报符号与数值,不做显著性)。
- 允许写 "recovers usability under bias-dominated shift" 的条件:至少一个单元成立 **且** ρ(Δ_5, b) > 0 **且** ρ(Δ_5, s_sign) > 0。否则只按单元报告实际 Δ_K 与区间;无单元成立时写 "no unit meets the pre-registered criterion"。
- 若某单元 oracle 上限 < 0.10 + Success@K(贪心),该单元在该 K 的判定记为"不可成立(oracle 受限)"并照常报告。

## 9. 覆盖性引理(带明确假设;不叫 theorem)

**引理(偏置主导偏移下的组合覆盖)。** 设查询 d\*,候选池中每个候选 c 的观测满足 log y_c = log f_c + b,b 为常数(假设 A1:常数乘性偏置),|b| ≤ q(假设 A2:偏置在半径内)。设候选预测值在 log 尺度上的最大相邻间隙为 δ,即对任意子目标 t 在 [log d\* − q, log d\* + q] 内都存在候选 c 使 |log f_c − t| ≤ δ(假设 A3:池密度)。则组合的 K 个候选中存在 c 使 |log y_c − log d\*| ≤ q/K + δ。
*证明概要*:子目标网格间距 2q/K、覆盖 [log d\* − (K−1)q/K, log d\* + (K−1)q/K];存在 m 使 |log d\*(m) + b − log d\*| ≤ q/K(因 |b| ≤ q);对该 m,被选候选满足 |log f_c − log d\*(m)| ≤ δ(A3;互异约束最多把候选推到下一个邻居,计入 δ),于是 |log y_c − log d\*| = |log f_c + b − log d\*| ≤ q/K + δ。∎
推论:若 q/K + δ ≤ log(1 + τ_rel) 则组合 Success@K = 1;贪心则需 |b| + δ ≤ log(1 + τ_rel)。两者的差距正是 |b| 落在 (log(1+τ) − δ, q] 的区间——这就是"偏置主导且在半径内"的含义。A1 不成立(偏置非常数、残差散布 SD 与 |mean| 可比)时引理不保证任何事;§8 的 b、s_sign 即对 A1 的经验检查。

## 10. 程序与产物

- 程序:`scaleup/w6/h_portfolio.py`(只读冻结产物;不训练模型;`--work`、`--output-dir`)。首次运行前 sha256 → `validation_plan/H_PORTFOLIO_SCRIPT.sha256` 并提交。
- 产物(每次运行目录):`portfolio_summary.json`(每单元:q、s、簇、各策略 × K × 容差的 Success、Δ 与区间、判定、b、s_sign;跨单元 Spearman;最终措辞判定)、`portfolio_selections.csv`(逐查询逐策略的被选 row_id,先于评分写出)、`portfolio_scored.csv`(评分后)、`environment.json`(输入 sha256、库版本、脚本 sha256)、`artifact_hashes.json`。
- 两次独立运行目录(`WORK/h_portfolio_run1`、`run2`)`artifact_hashes.json` 必须逐字节一致;仓库 `scaleup/h_portfolio/` 存放 summary/environment/hashes 的字节拷贝(不存第三方逐行数据)。
- 台账前缀 `hf.`;中英双稿留待 v0.4。

```
冻结产物 ──► 偏移集 (x_i, f_i, y_i) ──┐
   │                                   │  每查询 i:d* = y_i,池 = 非自身
   └─► 组外残差池 r ──► q, s           ▼
                            选择(只用 f_c, d*, q/s, AD) ──► selections.csv
                                                         │
                                       评分(读 y_c) ◄───┘ ──► Success@K, Δ, bootstrap, b, s_sign ──► 判定(§8)
```
