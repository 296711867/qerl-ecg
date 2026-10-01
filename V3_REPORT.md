# V3 状态报告（2026-09-29）

## 交付物

| 文件 | 内容 |
|---|---|
| `paper/v3/manuscript.md` / `.docx` | 投稿正文（摘要约 265 词，正文约 5,300 词，5 表 3 图，19 篇参考文献） |
| `paper/v3/supplement.md` / `.docx` | 补充材料 S1–S7（含 TRIPOD+AI 对照表） |
| `paper/v3/figures/` | Fig 1–3、Fig S1–S2（PNG 300 dpi + SVG） |
| `outputs/tables/v3_analysis.md` / `.json` | 重复运行、时间匹配对照、oracle 上界、操作点扫描、可靠性 |
| `outputs/tables/v3_latency.json` | 参数量与单步推理耗时 |
| `outputs/tables/event_cv_v3_{cs1,cs2,p1,p2}_causal.md` | 各次重复运行的五折明细 |

## v2 → v3 做了什么

1. **重复实验（冻结协议，零重调）**：主划分上再训 2 个分类器种子（`v3_cs1`/`v3_cs2`），另外重抽 2 套患者划分（`v3_p1`/`v3_p2`，与原划分的同测对重合率约 20%，即随机水平）。每次都重训分类器、重算轨迹、重训 DQN，共 20 个折 × 完整流水线。
2. **时间匹配对照**：DQN 总比阈值多看约 0.5 s，单比 F1 不公平。新增“在验证集上观察时间不超过同折 DQN 的最优时变阈值”作为对照。
3. **事后 oracle 上界**：oracle-any（首个正确前缀即停，0.672 / 2.57 s）、oracle-stable（10 s 决策在 2.86 s 已定型）。
4. **操作点扫描**（描述性，用测试标签）：DQN 位于所有阈值设置包络之上约 0.002–0.005。
5. **可靠性图、推理耗时与参数量**（Table 5）：DQN 单步 0.12 ms，分类器 44 ms，触发器推理成本可忽略。
6. **代码修正**：`--split` 与 `--seed-offset` 参数；`make_splits.py --seed`；补充材料更正卷积描述（v2 称“右裁剪”，实际为对称卷积 + 输入级因果）；测试覆盖所有划分文件的患者不相交与快速 macro-F1。

## 结论变化（重要）

v3 初稿曾假设“DQN 无可靠优势”。重复实验不支持这一说法，已改写为：

- **RQ1**：时变阈值 ≈ 10 s 固定窗，观察时间少 4.3–4.8 s；5 次运行中 F1 差距 ≤0.015（仅重抽划分 1 的 −0.0145 CI 不含 0）。
- **RQ2**：DQN 在 5 次运行 F1 均高于奖励选出的阈值（平均 +0.010 / +0.011，CI 不含 0），但每次多看 0.44–0.86 s；与时间匹配阈值比，平均 +0.007（CI 触及 0），2/5 次阈值占优。增益与运行间波动（10 s 固定窗 F1 0.531–0.556）同量级。
- **主张定位**：不是“RL 没用”，而是“RL 有约 0.01 的小增益，约三分之一来自多等；调好的两参数阈值拿到大部分收益，应作为学习型触发器必须超越的基线”。

## 仍需作者完成

1. **伦理声明**：`⟪AUTHOR⟫` 处确认所在单位是否需要豁免声明（公开去标识数据，PhysioNet DUA）。
2. **代码存档 DOI**：上传 Zenodo 后填入 Code availability。
3. **作者、单位、基金、利益冲突、CRediT 贡献**。
4. **期刊格式**：目标 *Biomedical Signal Processing and Control*（备选 *Computers in Biology and Medicine*、*Physiological Measurement*）。确认摘要字数上限（当前约 265 词）、Highlights（3–5 条，每条 ≤85 字符）、图 TIFF/EPS 格式要求（SVG 可转）。
5. **参考文献核对**：[1][9] 为 2026 年新文献，投稿前核对卷期页码；按期刊格式转换。
6. **声明使用了 AI 辅助写作**（Elsevier 要求在稿件中声明）。

## 建议 Highlights（草稿）

- Forward-growing annotated ECG episodes with patient-disjoint evaluation
- Time-varying confidence threshold halves observation with little F1 loss
- Deep Q-learning trigger adds about 0.01 macro-F1, partly by waiting longer
- Gains shrink against a threshold matched on observation time
- Results repeated across classifier seeds and re-drawn patient partitions

## Cover letter 要点

- 在同一分类器轨迹上、患者不相交、仅用验证集选择的条件下，公平比较固定窗、置信阈值与 DQN 停止规则。
- 显式区分 oracle 标注量（段起止）与因果输入；不声称连续监护报警延迟。
- 冻结协议后重复 5 次（种子 × 划分），并以时间匹配对照拆分 RL 增益来源——这是现有早期分类 RL 文献（SPN、ALERT）通常缺少的。
- 代码、划分文件与全部表图可一键复现（补充 S6）。

## 已知局限（已写入正文）

段起止来自回顾性标注；单中心；设计在同一数据上开发；MAT 几乎从未识别、AVB 仅 10 例；无实际弃权、质量感知无法评估；时间匹配与扫描为事后分析。

## 复现命令

见 `README.md` 与补充材料 S6。`scripts/analyze_v3.py` 约 10 分钟（CPU），`scripts/run_v3.ps1` 约 2 小时（共享 GPU）。
