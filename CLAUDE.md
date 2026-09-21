# CLAUDE.md

## Project Overview

This repository is the **paper + code** home for
*End-to-End Adaptive Fusion for Robust Aerial-Ground Cooperative 3D Detection*
(target venue: CVPR / ICCV), built on top of
[CoopTrack](https://github.com/zhongjiaru/CoopTrack) (ICCV 2025, Highlight).

| Paper name | Code entry point | File |
|---|---|---|
| **DGC** — Dynamic Geometric Correction | `HeightAdaptiveFusion` (+ `UncertaintyPredictor`, `GeometryConstraintLayer`, `LocalizationSigmaHead`) | `projects/mmdet3d_plugin/cooptrack/modules/height_adaptive_fusion.py` |
| **UGIM** — Uncertainty-Gated Instance Matching | `ContextAwareAssociation` (CAA) + `ACM` | `projects/mmdet3d_plugin/cooptrack/modules/cross_agent_interaction.py`, `adaptive_fusion.py` |

### Legacy module names (older docs and config flags)

Early iterations used the names AAF / UAF / GAF. The mapping is:

- **AAF** (Altitude-Adaptive Fusion) → now **DGC**
- **UAF** (Uncertainty-Aware Fusion) → folded into **UGIM**
- **GAF** (Geometry-Aware Fusion) → folded into **UGIM** (CAA geometric matching)
- **CAA** / **ACM** → kept; both are components of **UGIM**

The config flags still carry the historical `use_aaf` / `use_caa` names for
backwards compatibility with the three-stage configs.

### Auxiliary modules (ablation only, default OFF)

- **Cross-View Embedding** — InfoNCE contrastive learning (`cross_view_embedding.py`)
- **Dual GRU** — `VehicleGRU` + `InfrastructureGRU` with separate parameters (`adaptive_fusion.py`)
- **Communication Uncertainty** — latency compensation + packet loss recovery (`communication_uncertainty.py`)

## Module Flags

All configurable in the config file:
- `use_aaf=True` — DGC / Height-Adaptive Fusion
- `use_caa=True` — UGIM / Context-Aware Association
- `use_emb=False` — Cross-View Embedding (ablation)
- `use_comm=False` — Communication Uncertainty (ablation)
- `use_gru=False` — Dual GRU (ablation, gated by `training_stage==3`)
- `h_ref=25.0` — reference drone altitude (m)
- `training_stage=3` — `1`=vehicle-only, `2`=infrastructure-only, `3`=cooperative

## Key Config

`projects/configs_spd_coop/cooptrack/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py`

## Training

Standard 3-stage CoopTrack pipeline:
1. Vehicle-side perception (`training_stage=1`)
2. Infrastructure-side perception with `save_track_query=True` (`training_stage=2`)
3. Cooperative fusion with DGC + UGIM (`training_stage=3`)

## Paper

The manuscript (`adaptive_fusion.tex` / `.pdf` / `references.bib`) is **not**
kept in this public checkout — it lives in the private paper repository.
This repository contains the plugin code only.

- `CHANGES_LOG.md` — record of code changes

## Invariants to preserve when editing

1. DGC coordinate correction must run **before** bipartite matching — matching
   indices are computed on corrected coordinates, not raw ones.
2. `ACM` takes the **AAF displacement Δp** as input, never fused features — this
   keeps CAA (discriminative) and ACM (geometric) in orthogonal gradient spaces.
3. Dual GRU parameters are **not shared** between vehicle and infrastructure.
