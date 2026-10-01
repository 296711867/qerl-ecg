# V4 状态报告（2026-09-29）

v4 是完全重写的版本：新的叙事框架，所有图重新设计（包括架构图），正文从头撰写。实验数字沿用 v3 已验证的结果（没有重新训练），另外补充了逐 episode 的决策分析。

## 交付物

| 文件 | 内容 |
|---|---|
| `paper/v4/manuscript.md` / `.docx` | 正文：摘要、引言、方法、结果、讨论、结论，3 表 5 图，19 篇参考文献 |
| `paper/v4/supplement.md` / `.docx` | S1 校准、S2 混淆矩阵、S3 选中阈值、S4 补充数字与耗时、S5 复现 |
| `paper/v4/figures/` | Fig 1–5、Fig S1–S2（PNG 600 dpi + SVG + PDF，Arial 7 pt，全宽 183 mm）。**例外：Fig 2 已于 2026-09-29 替换为手写 SVG 复刻版**，源文件 `C:/Awork/01-Web-project/01-项目架构图绘制/ecg-paper/fig-prefix-classifier.svg`（导出 4325×2416 @600 dpi，内容同旧图仅版式优化），`make_v4_figures.py fig2` 会覆盖它，重跑时需跳过 |
| `scripts/make_v4_figures.py` | 生成全部图：`python scripts/make_v4_figures.py [fig1 ...]` |
| `scripts/v4_collect.py` | 导出主运行测试集逐 episode 的后验、TT/DQN 停止时间、DQN 贪心动作与 Q 值 → `outputs/tables/v4_decisions.npz` |

## 新叙事

核心问题：**识别一段术中节律需要多少秒 ECG？学习型停止规则是否比置信阈值用得更好？**

1. 大多数节律 3–4 s 就够：TT 0.529 / 3.52 s，与 10 s 固定窗 0.531 / 8.15 s 相当；hindsight 上界显示 85% 的 episode 在某个前缀已被判对（平均 2.6 s）。
2. DQN 5 次运行均比 TT 高约 0.01，但每次多看 0.4–0.9 s；与时间匹配阈值比，增益降到 +0.007，约三分之一来自多等。
3. 新增的机制分析（Fig 5）：
   - DQN 学到的停止边界形状与 TT 一致；
   - 多等的时间主要花在 AF/AFL（准确率 0.73 → 0.81）；
   - 对 PVC、PAC、SVTA，多等反而更差；
   - 逐 episode 配对，DQN 净多判对 24 个。

## 图

| 图 | 内容 |
|---|---|
| Fig 1 | 两个真实测试 episode：ECG、后验轨迹、TT 阈值线、两种策略的停止点（AF 同时停对；PVC 例 TT 2 s 误判，DQN 7 s 判对） |
| Fig 2 | 架构图：A 分类器（多尺度卷积 → 膨胀残差 TCN → 1×1 → GRU → 节律/质量双头），B 停止回路（状态 → TT 或 DQN → 动作，等待回到分类器） |
| Fig 3 | 各类时长分布、召回随前缀增长、按时间累计判对比例（含 hindsight 上界） |
| Fig 4 | 时间–F1 平面与局部放大；5 次运行 + 2 组平均的 ΔF1 / Δt 森林图 |
| Fig 5 | DQN 停止区域热图叠加 TT 阈值线、各类停止时间哑铃图、配对结局堆叠柱 |

## 仍需作者完成

1. 作者、单位、基金、利益冲突、伦理豁免、AI 使用声明、代码 DOI（正文中以 ⟪AUTHOR⟫ 标出）。
2. 参考文献按期刊格式调整编号顺序（目前 [12,14,15] 在 [10][11] 之前被引用；若期刊要求按引用顺序编号，需要重排）。核对 2026 年文献 [1][9] 的卷期页码。
3. 目标期刊：*Biomedical Signal Processing and Control*（备选 *Computers in Biology and Medicine*、*Physiological Measurement*）。按投稿要求转换图片格式（PDF/SVG 可转 TIFF/EPS）。

## 与 v3 的关系

v3 的稿件和图都已弃用，不再引用，仅作历史记录保留在 `paper/v3/`。所有数值来源：`outputs/tables/v3_analysis.json`、`v3_latency.json`、`v4_decisions.npz`。
