# ==============================================================================
# End-to-End Adaptive Fusion for Robust Aerial-Ground Cooperative 3D Detection
# ==============================================================================

## 项目背景

**论文标题**: End-to-End Adaptive Fusion for Robust Aerial-Ground Cooperative 3D Detection

**目标会议**: CVPR / ICCV

**Baseline**: CoopTrack (ICCV 2025)

---

## 核心贡献

### 两个模块

| 模块 | 对应 | 方法 |
|------|------|------|
| Height-Adaptive Fusion | 高度变化 (25m-55m) | End-to-end高度自适应融合 |
| Geometry-Aware Mechanisms | 定位/角度误差 | 几何感知匹配 |

### 核心Novelty

**End-to-end联合优化 vs OpenCOOD-Air静态CDSC**

---

## Geometric Uncertainties

| 类型 | 范围 | 对应模块 |
|------|------|---------|
| Altitude | 25m-55m | Height-Adaptive Fusion |
| Pose (Translation) | 0-7m | Geometry-Aware Mechanisms |
| Pose (Angle) | 0-5° | Geometry-Aware Mechanisms |

---

## Griffin基准数据 (CoopTrack)

| 条件 | AP | AMOTA |
|------|-----|-------|
| 25m + 0误差 | 47.9% | 48.8% |
| 25m + 2m误差 | 29.1% | - |
| 25m + 5°误差 | 39.0% | - |
| 40m + 0误差 | 39.6% | 44.6% |
| 55m + 0误差 | 36.4% | 40.2% |

---

## 目标

**Safe标准:**
- 大误差(2m+/5°+)：37%+ AP
- 高高度(55m+)：44%+ AP
- 完美条件：不掉点
- 消融：每个模块都有正贡献

---

## 文件结构

```
CoopTrack-main/projects/
├── adaptive_fusion.tex      # 论文主文件
├── references.bib          # 参考文献
├── experiment_runner.sh   # 实验脚本
├── configs/               # 配置文件
└── mmdet3d_plugin/         # 模块代码
    └── cooptrack/modules/
        └── height_adaptive_fusion.py
```

---

## 实验设计

### 消融实验

```
CoopTrack baseline
├─ + Height-Adaptive Fusion
│   └─ + Geometry-Aware Mechanisms  ← Full
```

### 测试条件

- 高度: 25m, 40m, 55m
- 定位偏移: 0m, 2m, 5m, 7m
- 角度偏移: 0°, 2°, 5°

---

## 时间线

| 日期 | 内容 |
|------|------|
| 2026-05-22 | 论文框架完成 |
| 2026-05-24 | 论文v3重建，命名更新 |
| 待完成 | 跑实验填TBD值 |
| 待完成 | 生成框架图 |