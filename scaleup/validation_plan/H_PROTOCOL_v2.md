# H_PROTOCOL_v2 — 预注册验证协议(H-A ~ H-E),冻结版 v2

- 日期:2026-10-21。v2 只改 §1 数据集名单与 §6 程序文件名;§2–§5、§7 与 v1 **逐字相同**(阈值、档位、密封规则、模型配置、判定规则一字不动)。v1 对 A1 的结果保持冻结,不重跑。
- 用户裁定(经主线会话转达):准入 R2-A1;条件准入 R2-A2(三条规则写死,见 §1);R2-B1、R2-B2 不准入;"如果是实验直接跑就可以"。
- 状态:**冻结**。本文件的 SHA-256 写入同名 `.sha256` 侧车并提交;任何改动都必须以新版本号另存并登记到 `H_DEVIATION_REGISTER.md`。
- 冻结时点:在下载 R2-A1、R2-A2 的任何数据文件到协议工作目录**之前**。候选阶段(`DATASET_CANDIDATES_round2.md` §0)已在隔离临时目录只读过两者的表头、分组列级别计数与目标列的非空计数/单元格格式,未读任何目标值;R2-A2 的 `paper_doi` 列已读(用于独立性核对)。

## 1. 数据集名单(写死)

| 代号 | 数据集 | 下载来源(只从这些地址取;sha256 记录) | 准入条件与绑定规则(不满足则退出并登记,**不替换**) |
|---|---|---|---|
| A1 | Hormozinezhad et al., Lab Chip 25:1681 (2025) | (v1 已完成:揭盲 2026-10-21T19:40:07Z) | 不重跑;结果按 v1 冻结,参与 §5 的多数据集判定 |
| R2-A1 | Wang J.-X. et al., *Chem. Eng. J.* 485:149467 (2024),共流微流控液滴;数据 Zenodo 10.5281/zenodo.10526441(CC BY 4.0) | `https://zenodo.org/api/records/10526441/files/Data%20for%20FrameworkA.xlsx/content`,md5 须为 `d58b4cbb8f4df47fb3eb2b44a2dc4fdb`(Zenodo 自报);FrameworkB/C 不用 | 无条件准入。输入 = 7 个原生列(`disp_flow, cont_flow, interfacial_tension, disp_viscosity, cont_viscosity, disp_density, cont_density`)+ 2 个几何列(`inner, outer`;`channel` 与器件一一对应,作为器件键的一部分);目标 = `diameter`;`flowtype` 为观测流型,**不进入输入**,只作描述性分层。S2 = 流体对(5 个物性列的组合);S3 = 器件实例(`inner, outer, channel` 组合);S1 = S2 × S3 组合(dev-ref 分组);S4 无。无量纲列由绑定给出公式(Ca_c、Ca_d、流速比、黏度比、Re_c、Re_d、几何比),流速单位未在数据中注明,仅影响常数因子,对树模型与 C2 的相等判断无影响,绑定中说明。硬边界 positivity_only;无锚点(B2 不可评估)。若 `diameter` 列出现非正或缺失,按 A1 的 v2 规则在任何分组决定之前剔除并计数 |
| R2-A2 | Song S., Baek J., Seo S., *Sci. Data* 12 (2025),"LNP Atlas";数据 Zenodo 10.5281/zenodo.17243733(CC BY 4.0) | `https://zenodo.org/api/records/17243733/files/LNP_Atlas_DB_202509_v1.csv/content`,md5 须为 `f77ec01fe88e16f558c09c41bcc10610` | **条件准入,三条规则写死(用户裁定):** ① 粒径单元格解析:纯数字与 "a ± b"(含 "a±b"、"a +/- b")取首数;区间("a–b"、"a-b")与其他文字剔除并计数;空值剔除并计数。② 输入编码:脂质身份(`ionizable_lipid, helper_lipid, peg_lipid, sterol_lipid`)为类别变量(one-hot,在开发集与密封输入的并集上定义,空字符串为一级);`lipid_molar_ratio` 解析为 4 个分数(冒号分隔的前 4 个数,归一化到和为 1;少于 4 个数的行按缺失输入剔除并计数);`target_type` one-hot;制备方法关键词标志(herringbone / t-junction / nanoassemblr 或 ignite / pipette 或 hand / ethanol injection,各一列 0/1)。③ 主分析用微流控子集(`synthesis_info` 含 microfluidic / nanoassemblr / herringbone / t-junction / ignite 之一的行),S2 = `ionizable_lipid`,S3 = `paper_doi`(来源论文 = 实验室),S1 = `paper_doi`(dev-ref 分组);全表的非微流控行只用于 S4 制备方法档位(按 §3 S4 规则:两类中的少数类整体密封,开发集为微流控子集的开发行)。目标 = 粒径(nm);无量纲列无(H-C 不可评估);硬边界 positivity_only;无锚点。稿件中标明其为**同行评审的文献汇编数据集**(63 篇来源论文),而非单一实验室原始测量 |
| R2-B1 | Tang, Chagot, Angeli, Lab Chip (2026) | — | 不准入(与 D2 同实验室) |
| R2-B2 | Eugster … Luciani, J. Control. Release (2024) | — | 不准入(逐行表位置未核实;Buttitta 同为 D3 作者) |

任何准入后发现与 D1/D2/D3/A1 共享来源行的数据集,登记并退出。

## 2. 下载与绑定(每个数据集,D3 模式)

1. 下载到隔离目录 `WORK/../data/h_<代号>/`,逐文件 sha256 写入 `h_<代号>_download_manifest.json`。
2. 只读:文件清单、工作表名、列名、每表行数、单位说明、README/PDF 文字。**不打开任何目标值列的内容,不画图,不统计。**
3. 写 `H_BINDING_<代号>.json`:文件与工作表、输入列、目标列(及单位)、分组列(S1 几何/器件 id;S2 流体/配方族;S3 器件实例/批次/实验室;S4 器件类型——不存在的档位写 `null`)、器件常数、硬边界定义(有物理上限时写上限列或常数;否则 `"positivity_only"`)、锚点定义(有文献关联式时写公式与系数来源;否则 `null`)、密封组选择结果(按 §3 规则算出的级别名)。绑定文件 sha256 后提交,再进入 custodian。
4. 绑定一经提交不得修改;修正只能以 `H_BINDING_<代号>_v2.json` 另存并登记。

## 3. 档位与密封规则(写死)

- S0 行随机参考:开发集内 80/20,15 个种子(0–14)。
- S1 同设备跨几何/跨器件 id:开发集内按 S1 分组列做 GroupShuffleSplit 80/20,15 个种子;若 S1 列不存在则不定义。
- S2 未见流体/配方族:分组列的每个级别为一个组。**密封组** = 行数 ≥ 10 且非最大级别的各级别中,行数最接近这些级别行数中位数者(并列取字典序最小)。其余级别为开发集,在开发集内做留一组(dev-LOCO,标签公开,记为回顾性)。
- S3 未见器件实例/批次/实验室:同 S2 规则选一个密封级别;若 S2 与 S3 的密封级别行重叠,S3 的密封级别改取次优候选;仍重叠则 S3 不密封、只做 dev-LOCO。
- S4 器件类型:若存在两类,密封少数类全部行(类比 D1 SE→DE),开发集为多数类;若只有一类则不定义。
- 密封集的标签由 custodian 分离(owner-only 0600,不提交);输入与开发集(含标签)可见。开发集内所有分析在揭盲前完成并冻结为预测包(sha256);单次 O_EXCL 揭盲闸门。
- 开发参照 dev-ref:开发集上 5 折分组交叉验证(组 = 最细依赖组 = S1 列,若无则 S2 列,若都无则按行),XGBoost 锁定配置(100 trees, depth 3, lr 0.3, seed 42)。
- 目标尺度诊断(只用开发集标签):若 max/min > 10 或偏度 > 2,该数据集全部模型用 log 目标;否则原尺度。诊断结果写入 `h_<代号>_devdiag.json` 并在揭盲前提交。
- 容差 τ:相对 10%;绝对 τ_abs = 开发集目标中位数的 10%(由 devdiag 计算、写死)。

## 4. 模型族与区间(与 W3/W4 一致,不调参)

XGBoost(锁定配置)、随机森林(400 trees, min leaf 2, seed 42)、MLP(64–64 ReLU, Adam 1e-3, 早停 15%/30, ≤2000 iter, seed 42, 标准化输入与(log 或原)目标)、QRF(400 trees, min leaf 5)、GP(Const×RBF-ARD+White, 2 restarts)、深度集成(5×MLP 64–64, member seeds 0–4)。区间:split conformal(名义 0.90,校准分区 20% 按组)、CQR(GBR 0.05/0.95, 300 trees, depth 3);原生区间(QRF 分位数、GP ±1.645σ、集成 ±1.645 sd)。硬边界模型(H-B):与 V2 相同的 (0,D) 参数化,D 由绑定给出;若绑定为 positivity_only,则用 log 目标 MLP 作"边界"版本,无界原尺度 MLP 作对照。

## 5. 判定规则(与草案一致,数字写死)

- **H-A**:对每个数据集每个可定义的 S2/S3 档位,r = MAPE_sealed(或 dev-LOCO 宏平均)/ dev-ref;bootstrap 2000(seed 0)。成立 = 所有 S2/S3 的 r 点估计 ∈ [2.2, 3.0] 且 S2/S3 间离散(max/min)≤ 同数据集 S1 的离散 且 S4(若有)的 r 下界 > S2/S3 的 r 上界;证伪 = 任一 S2/S3 的 r 点估计在带外且 95% 区间与带不相交,或 S2/S3 离散 > S1;不可判定 = 无数据集能定义 S2 或 S3。主判定用密封组;dev-LOCO 作次要证据并标回顾性。
- **H-B**:B1:在 ≥ 2 个数据集上,边界版 MLP 相对无界 MLP 最差组 MAPE 更低且点无效率 0 → 成立;任一数据集不满足 → 证伪。B2(需锚点):域内/域外按"锚点在该组 MAPE ≤ 开发集锚点 MAPE 中位数"划分;成立 = 域内 V2-P 型优于 V2-H 型(配对差 < 0,bootstrap 95% 不含 0)且域外不优;证伪 = 域内无改善或域外有改善;无锚点数据集 → B2 不可判定。
- **H-C**:C1:配对差 Δ = MAPE_native − MAPE_dimless(按 S2 组配对),成立 = 中位数 < 0 且 Wilcoxon p < 0.05(BH)于 ≥ 2 个数据集;证伪 = 中位数 ≥ 0 于多数数据集。C2:无量纲群相同(容差 5%)但绝对尺度不同的行对 ≥ 20 对时才判;成立 = 这些行对的 |Δd|/d 中位数显著大于(Mann–Whitney p < 0.05)完全重复行对的重复性;否则不可判定。
- **H-D**:A = 区间包含式检索回答率(点在带内且区间整体在带内),W = 宽度 < 2τ 的候选比例;成立 = 所有 S2/S3 档位 |A − W| ≤ 0.10 且 A 与点 MAPE 的跨档位 Spearman 不显著;证伪 = 存在档位 A − W > 0.10。
- **H-E**:E1:六模型 S0 排名与 S2/S3 排名的 Spearman ρ;成立 = ≥ 2 个数据集 ρ ≤ 0.5;证伪 = 多数数据集 ρ ≥ 0.7;D1/D3 既有单元一并报告。E2:原生宽度比(偏移/内部)> 1 且逐行 |误差| 与原生宽度的 Spearman ρ > 0.3(BH)于 ≥ 2 个数据集 → 成立;否则证伪。
- 多重比较:同一假设内 BH 校正。所有判定在揭盲程序中自动计算并写入结果 JSON;人工不得改判。

## 6. 程序与产物

- v1 程序(`h_download.py`、`h_custodian.py`、`h_run.py`、`h_reveal.py`)对 A1 保持冻结。v2 新数据集用绑定驱动的通用版本:`scaleup/w6/h_custodian2.py`(统一表:`row_id, s1_group, s2_group, s3_group, s4_group, <原生列>, <无量纲列>, target`;同一密封规则 + S4 规则;0600 标签)、`h_run2.py`(与 `h_run.py` 同一单元/模型/区间定义,列名由绑定给出;有 S1 列时增加 S1 单元;有 S4 行时增加 sealed_S4 单元;无无量纲列时 H-C 记不可评估)、`h_reveal2.py`(同一闸门与评分;S4 的 r 进 H-A 条件)、`h_combine.py`(A1 + 新数据集的揭盲结果 → §5 的多数据集判定,只读各数据集的 `h_reveal_results.json`)。程序在首次运行前 sha256 并提交;修改即登记。
- 产物目录:`WORK/h_<代号>_artifacts`、`WORK/h_<代号>_run{1,2}`、`WORK/h_<代号>_reveal`;字节拷贝入 `scaleup/h_<代号>/`(不含逐行第三方数据);台账前缀 `h.`。
- 偏差登记册:`scaleup/validation_plan/H_DEVIATION_REGISTER.md`。
- 结果无论正负进 v0.4 §5.10 与 §6 五假设段;RESPONSE_v0.4.md 逐条记录。

## 7. 不做的事

不调参;不按结果改档位、τ、种子或密封组;不用未批准数据集替换退出的数据集;不在揭盲前读取密封标签;不写未经上述判定支持的机理句。
