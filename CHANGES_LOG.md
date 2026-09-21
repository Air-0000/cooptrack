# 模块修复记录 · CoopTrack 自适应融合

> 记录日期：2026-09-22
> 涉及文件：`projects/mmdet3d_plugin/cooptrack/modules/` 下 3 个模块
> 本仓库已接入 git。此文件保留为改动流水与决策记录。

---

## 一、流程（怎么走到这里的）

1. **起点**：之前对三个模块做了功能增强（核心是把"高度自适应"真正做进融合链路，修掉 altitude-blind 问题），但一直以为本机没有 torch，只能等服务器验证。
2. **提示**：用户指出"torch 在别的环境找" → 定位到 conda 环境 `aic-2` 装了 **torch 2.7.1+cu128**，且 `torch.cuda.is_available() = True`（本机有可用 GPU）。
3. **可行性判断**：检查三个模块的 import，发现 `height_adaptive_fusion.py` / `adaptive_fusion.py` 只依赖 `torch` + `numpy`，**不依赖 mmdet3d**，可以直接在真机上 import 并跑前向。
4. **冒烟测试**：用 `importlib` 加载模块 + 伪造 `Instances` stub + 构造 `veh2inf_rt`（使 `inv(rt.T)[:3,3]` 等于目标飞行高度），在 GPU 上跑前向，逐个暴露运行时错误。
5. **迭代修 bug**：每修一处就重跑，直到 `forward` 通过。
6. **结论**：共修 **16 处**，全部是"第一次 forward 就会崩"的级别 → 这三个模块此前**从未被真正执行过**。

---

## 二、改动清单（16 处 / 7 类根因）

### `height_adaptive_fusion.py`（6 处）

| 行 | 改动 |
|----|------|
| 186 | `direction_encoder`：`Linear(3, …)` → `Linear(2, …)` |
| 230 | `velocity = pred_boxes[..., 8:11]` → `8:10` |
| 583 | `sigma_exp`：去掉多余的 `.unsqueeze(1)` |
| 587 | `uncertainty_exp`：去掉多余的 `.unsqueeze(1)` |
| 588 | `geo_conf_exp`：去掉多余的 `.unsqueeze(1)` |
| 621 | `h_diff.expand(-1, self.embed_dims)` → `expand(-1, 3)` |

### `cross_agent_interaction.py`（1 处）

| 行 | 改动 |
|----|------|
| 59 | **新增** `self.mahal_thresh = 2.0`（此前该属性全项目只有 1 处使用、0 处定义 → AttributeError） |

### `adaptive_fusion.py`（9 处）

| 行 | 改动 |
|----|------|
| 216 | `h_diff.expand(-1, self.embed_dims)` → `expand(-1, 3)` |
| 231 | AAF 改为返回**对齐后的特征**，不再越权做融合 |
| 265 | CAA `direction_encoder`：`Linear(3, …)` → `Linear(2, …)` |
| 280 | `matching_net` 第一层：`Linear(embed_dims, …)` → `Linear(embed_dims // 2, …)` |
| 313 | `velocity = pred_boxes[..., 8:10]`（原 `8:11`） |
| 357-358 | `inf_geo` / `veh_geo` 广播修复（去掉 unsqueeze，`veh_geo` 需转置） |
| 362 | 去掉 `geometry_score.squeeze(-1)` |
| 671 | `sigma_exp`：去掉多余的 `.unsqueeze(1)` |
| 675-676 | `confidence_exp` / `geo_conf_exp`：去掉多余的 `.unsqueeze(1)` |

### 7 类根因

- **A. 属性未定义** —— `mahal_thresh` 用了却从没赋值。
- **B. 速度维度错** —— 切片 `8:11` 与 `Linear(3)` 都错了，速度是 **2 维**。
- **C. `[N,1]` 多升一维** —— 6 处 `.unsqueeze()` 把 `[N,1]` 变成 `[N,1,1]`，`expand(N,M)` 直接报维数不匹配。
- **D. `h_diff` 广播错** —— 应占 3 个 slot，写成 `embed_dims`(256)，导致 concat 512 维撞 259 维的第一层。
- **E. `matching_net` 输入维数错** —— 实际输入是 `64+64=128`，声明成 256。
- **F. AAF 越权融合** —— 产出 `[N,M,D]`，下游 `aaf_feats[matched_idx]` 需要 `[M,D]`；且 veh_n==1 时 2D/1D concat 直接崩。
- **G. `squeeze(-1)` 压掉 2D 矩阵** —— 把 `[N,M]` 的 score 矩阵压成 `[N]`，静默破坏逐元素相乘。

### B 类修正的权威依据（来自项目自身，非猜测）

- `detectors/cooptrack.py:589`：`pred_boxes = torch.zeros((len, 10))` → box 是 **10 维**
- `spatial_temporal_reason.py:640`：`velos = track_instances.pred_boxes[..., 8:10]` → 速度是 **8:10，2 维**

故布局为：`中心xyz(0:3) + 尺寸wlh(3:6) + yaw的sin/cos(6:8) + 速度vx,vy(8:10)`。

---

## 三、验证结果（本机 GPU，torch 2.7.1+cu128）

| 项 | 结果 |
|----|------|
| 模块导入 | 通过 |
| **altitude 敏感性（核心）** | **通过**：h=25m → σ=0.95；h=55m → σ=2.43；Δσ=1.29；Δcorrected=0.85 ≠ 0 |
| `HeightAdaptiveFusion.forward` | 通过：matched=1，输出 `(1,256)`，无 NaN，fused 特征确实 ≠ 原始 veh 特征 |
| `AdaptiveFusion.forward` | 跑通不崩（matched=0，属未训练随机初始化，见下） |
| `CAA.forward`（adaptive_fusion） | 通过：输出 `(8,12)`，无 NaN |

**核心结论**：σ 随高度增长、校正量随高度变化 → 模块确实"看得见"飞行高度，altitude-blind 问题已解决。

---

## 四、遗留问题

1. **CAA（cross_agent_interaction.py）的 Mahalanobis 分支本机测不了**
   它经 `mmcv.cnn.bias_init_with_prob` 依赖 mmdet3d 生态，而 `aic-2` 装的是 **mmcv 2.1**（mmseg 用），该符号已被移除 → `ImportError`。
   `mahal_thresh` 那处是 **静态确认**的（grep 全项目仅 1 处使用、0 处定义），未经运行验证。

2. **`AdaptiveFusion` 的 σ 没有独立监督（真实风险，未修）**
   σ 只有在**匹配成功后**才有梯度；实测随机初始化下 `mahal=1165` 远超门限 2.0 → matched 恒为 0 → 无梯度 → σ 永远学不到（死循环）。
   `HeightAdaptiveFusion` 有 `compute_sigma_nll_loss` 能打破它，而实际管道（`cross_agent_interaction.py:123`）用的正是 HAF，所以**当前安全**；但 `AdaptiveFusion` 若被启用会踩坑。这属于训练策略决策，未擅自改。

3. **`adaptive_fusion.py:790` 的同名别名陷阱（已加警告注释，未删）**
   `HeightAdaptiveFusion = AdaptiveFusion` 会遮蔽 `height_adaptive_fusion.py` 里的同名类。
   已确认 `modules/__init__.py` 第 25-32 行是显式导入 `AdaptiveFusion`，**没有**覆盖 `HeightAdaptiveFusion`，实际管道安全。

---

## 五、之后要干什么（按优先级）

1. **配好真机环境**：`mmdet3d 0.17.1` + `mmcv-full 1.x`（现 aic-2 的 mmcv 2.1 不行），补齐 scipy/matplotlib。
2. **补测 CAA 的 Mahalanobis 分支**（遗留 1），确认 `mahal_thresh=2.0` 的 σ 门在真实数据下的行为。
3. **下载 Griffin 数据集，跑 CoopTrack baseline**，拿到 25/40/55m × 各误差档的 AP/AMOTA 基线数字。
4. **接入并训练**：确认这些模块在 `CrossAgentSparseInteraction` 中的调用路径能通，跑通一轮训练（重点观察训练初期 matched 是否恒为 0 —— 若恒为 0，需给 σ 加独立监督，即遗留 2）。
5. **跑消融**：`baseline → + Height-Adaptive Fusion → + Geometry-Aware Mechanisms`，每步单独验证。
6. **回填论文**：把 `PROJECT_SUMMARY.md` 里的 TBD 值与消融表填实；补框架图。

> 提示：重新验证时可用同样的冒烟套路（`importlib` 加载 + `Instances` stub + 构造 `veh2inf_rt` 使 `inv(rt.T)[:3,3]` = 目标高度），不需要完整数据集即可跑通前向。

---

## 六、仓库合并（2026-09-22）：代码 + 论文 → 单一论文仓库

此前工作分散在两个仓库，本次合并为**一个论文仓库**：

| 仓库 | 原角色 | 文件数 | 归属 |
|---|---|---|---|
| `Aerial-Ground-Cooperative-Perception`（私有） | 论文专属：tex/pdf、插件模块、文档 | 23 | 论文侧 |
| `cooptrack`（公开） | CoopTrack 官方完整代码库 + 插件 | 145 | 代码侧 |

### 合并动作

1. **并入 AGC 独有的 2 个文件**（本地此前没有）：
   - `CLAUDE.md` → 重写为「模块 ↔ 论文命名映射」（AAF/UAF/GAF → DGC/UGIM），并写明三条不可破坏的约束。
   - `docs/PROJECT_SUMMARY.md` → 中文项目纪要，命名对齐 DGC/UGIM，时间线更新。
2. **README 二合一**：
   - AGC 版（338 行，论文视角：Overview / 两大贡献 / 三个致命修复 / 死区防御 / 性能表 / 架构 / 消融 / 图表）
   - cooptrack 版（122 行，基座视角：CoopTrack 署名 / 三阶段配置 / 实验编排 / Related Works）
   - 合并后以**论文为主、代码为附**，同时保留 CoopTrack 的署名、News、Contact、Citation 与 Related Works，明确「基座来自 CoopTrack，本仓库贡献是 DGC+UGIM 插件」。
3. 论文（`adaptive_fusion.tex` 857 行）与代码（含 16 处修复 + σ 预测头）均取最新状态。

### 合并后的仓库定位

- `adaptive_fusion.tex` / `.pdf` / `references.bib` —— 未发表稿件
- `projects/mmdet3d_plugin/cooptrack/modules/` —— DGC / UGIM / 辅助模块
- `projects/configs_spd_{veh,inf,coop}/` —— 三阶段训练配置
- `tools/` —— 实验编排、绘图、profiling
- `docs/` —— INSTALL / DATA_PREP / TRAIN_EVAL / PROJECT_SUMMARY
- `CLAUDE.md`、`CHANGES_LOG.md` —— 命名映射与改动流水

### ✅ 已完成（2026-09-22）

- 完整的「论文 + 代码」已推送到**私有**仓库 `Aerial-Ground-Cooperative-Perception`
  （`6abaa02..cc9aba0`，136 个文件变更，125 新增 / 0 删除）。
- 公开仓库 `cooptrack` 已移除 `adaptive_fusion.tex`、`adaptive_fusion.pdf`、`references.bib`
  以及 `PROJECT_SUMMARY.md`、`docs/PROJECT_SUMMARY.md`、`fix_paper.py`（`bbe30d4..f1acca0`），
  并恢复为代码仓库版 README —— 仅保留 CoopTrack 基座 + DGC/UGIM 插件代码。
- 论文稿件现在**只存在于私有仓库**中。

> 后续论文修改请在私有仓库 `Aerial-Ground-Cooperative-Perception` 的工作副本中进行；
> 本仓库（公开）只维护插件代码。
