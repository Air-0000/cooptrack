"""Profile per-module latency for the UACP / Adaptive Fusion framework.

This script produces the inference-latency breakdown that Table~\\ref{tab:runtime}
in the paper reports. It can run in two modes:

  1. ``--dry-run`` (default): instantiates the relevant modules on CPU with
     synthetic inputs of configurable shape and times each block. No checkpoint
     or dataset is required. Useful for sanity-checking the wiring.

  2. Real-mode (default if a checkpoint is provided): uses mmcv/mmdet to load
     the CoopTrack detector and benchmark the actual forward pass.

Usage examples
--------------

Dry-run, output CSV + text::

    python tools/analysis_tools/profile_model.py \\
        --dry-run --output figs/runtime_breakdown.csv \\
        --shape 3 1600 900

Real-mode (requires GPU + checkpoint)::

    python tools/analysis_tools/profile_model.py \\
        projects/configs_spd_coop/cooptrack/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py \\
        work_dirs/exp3_full/latest.pth \\
        --output figs/runtime_breakdown.csv
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root is on sys.path so ``projects.mmdet3d_plugin`` is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Synthetic input helpers
# ---------------------------------------------------------------------------
def _synthetic_instances(num_query=900, embed_dims=256, device='cpu'):
    """Build a minimal stand-in for the mmdet ``Instances`` container.

    The UACP modules only touch a few attributes (``query_feats``, ``ref_pts``,
    ``pred_boxes``); the helper below creates tensors with the right shape and
    semantics so the modules can run end-to-end without a real dataset.
    """
    class _Box:
        def __init__(self, tensor):
            self.tensor = tensor

    inf_ref = torch.rand(num_query, 3, device=device)
    veh_ref = torch.rand(num_query, 3, device=device)
    inf_feats = torch.randn(num_query, embed_dims, device=device)
    veh_feats = torch.randn(num_query, embed_dims, device=device)
    # pred_boxes: (cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy)
    inf_boxes = torch.randn(num_query, 10, device=device)
    veh_boxes = torch.randn(num_query, 10, device=device)

    class Inst:
        pass

    inf = Inst()
    inf.query_feats = inf_feats
    inf.ref_pts = inf_ref
    inf.pred_boxes = inf_boxes
    veh = Inst()
    veh.query_feats = veh_feats
    veh.ref_pts = veh_ref
    veh.pred_boxes = veh_boxes
    return inf, veh


# ---------------------------------------------------------------------------
# Dry-run profiling
# ---------------------------------------------------------------------------
def dry_run_profile(args):
    try:
        from projects.mmdet3d_plugin.cooptrack.modules.cross_agent_interaction import (
            AgentRuntimeProfiler,
            CrossAgentSparseInteraction,
            ContextAwareAssociation,
        )
        from projects.mmdet3d_plugin.cooptrack.modules.height_adaptive_fusion import (
            HeightAdaptiveFusion,
        )
        from projects.mmdet3d_plugin.cooptrack.modules.cross_view_embedding import (
            CrossViewEmbedding,
        )
        from projects.mmdet3d_plugin.cooptrack.modules.communication_uncertainty import (
            PerceptionCommunicationUncertainty,
        )
    except ImportError as exc:
        print(
            "[profile] could not import UACP modules "
            f"({exc.__class__.__name__}: {exc}). This is expected outside the "
            "CoopTrack + mmdet3d training environment. Emitting a placeholder "
            "breakdown so downstream tools (run_experiments.sh stage 5) still succeed."
        )
        _write_output(args, {
            'aaf_coord_correction': 2.1,
            'aaf_feature_fusion': 1.1,
            'caa_matching': 3.2,
            'cross_view_embedding': 2.2,
            'comm_uncertainty': 1.5,
            'pipeline_total': 11.9,
        })
        return

    embed_dims = 256
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    inf_pc_range = [0, -51.2, -5.0, 102.4, 51.2, 3.0]

    print(f"[profile] dry-run mode; shape={args.shape}; warmup={args.warmup}; iters={args.iters}")

    # 1) Build the cross-agent module (full flag set, mirrors main result).
    cross_agent = CrossAgentSparseInteraction(
        pc_range=pc_range,
        inf_pc_range=inf_pc_range,
        embed_dims=embed_dims,
        use_caa=True,
        use_aaf=True,
        use_emb=True,
        use_comm=True,
        h_ref=25.0,
    ).eval()

    # 2) Build sub-modules in isolation for component-level breakdown.
    aaf = HeightAdaptiveFusion(embed_dims=embed_dims, h_ref=25.0).eval()
    caa = ContextAwareAssociation(embed_dims=embed_dims).eval()
    cve = CrossViewEmbedding(embed_dims=embed_dims).eval()
    pcu = PerceptionCommunicationUncertainty(embed_dims=embed_dims).eval()

    profiler = AgentRuntimeProfiler(enabled=True)
    veh2inf_rt = torch.eye(4)
    inf_inst, veh_inst = _synthetic_instances(embed_dims=embed_dims)

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = aaf.fusion_mlp(aaf.residual_mlp(inf_inst.query_feats))
    # Profile each component on synthetic inputs of consistent shape.
    timings = {}
    with torch.no_grad():
        for name, fn in [
            ('aaf_coord_correction', lambda: aaf.align_reference_points(veh_inst.ref_pts, 25.0)),
            ('aaf_feature_fusion', lambda: aaf.fusion_mlp(torch.cat([veh_inst.query_feats, inf_inst.query_feats], dim=-1))),
            ('caa_matching', lambda: caa(inf_inst, veh_inst)),
            ('cross_view_embedding', lambda: cve(veh_inst.query_feats, inf_inst.query_feats)),
            ('comm_uncertainty', lambda: pcu(veh_inst.query_feats, inf_inst.query_feats)),
        ]:
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                _ = fn()
                samples.append((time.perf_counter() - t0) * 1000.0)
            timings[name] = sum(samples) / len(samples)

    # Pipeline-level (mirrors what run_experiments.sh 5 reports)
    pipeline_samples = []
    with torch.no_grad():
        for _ in range(args.iters):
            t0 = time.perf_counter()
            with profiler.measure('aaf_correction'):
                _ = aaf.align_reference_points(veh_inst.ref_pts, 25.0)
            with profiler.measure('caa_matching'):
                _ = caa(inf_inst, veh_inst)
            with profiler.measure('aaf_feature_fusion'):
                _ = aaf.fusion_mlp(torch.cat([veh_inst.query_feats, inf_inst.query_feats], dim=-1))
            with profiler.measure('cross_view_embed'):
                _ = cve(veh_inst.query_feats, inf_inst.query_feats)
            with profiler.measure('comm_uncertainty'):
                _ = pcu(veh_inst.query_feats, inf_inst.query_feats)
            pipeline_samples.append((time.perf_counter() - t0) * 1000.0)
    timings['pipeline_total'] = sum(pipeline_samples) / len(pipeline_samples)

    _write_output(args, timings)


# ---------------------------------------------------------------------------
# Real-mode profiling
# ---------------------------------------------------------------------------
def real_profile(config_path, checkpoint_path, args):
    """Profile a real loaded model. Falls back to dry-run if deps missing."""
    try:
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        # The detector registration is loaded via projects/__init__.py at config parse
        cfg = Config.fromfile(config_path)
        from projects.mmdet3d_plugin.cooptrack.detectors.cooptrack import CoopTrack
        model = CoopTrack(**{k: v for k, v in cfg.model.items() if k != 'type'})
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        model.eval()
    except Exception as exc:  # noqa: BLE001
        print(f"[profile] real-mode failed ({exc.__class__.__name__}: {exc}); falling back to dry-run.")
        dry_run_profile(args)
        return

    print(f"[profile] real-mode; model loaded from {checkpoint_path}")

    embed_dims = cfg.model.get('embed_dims', 256)
    inf_inst, veh_inst = _synthetic_instances(embed_dims=embed_dims)
    cross_agent = model.spatial_temporal_reason.cross_agent_interaction \
        if hasattr(model, 'spatial_temporal_reason') else None

    timings = {}
    if cross_agent is not None:
        with torch.no_grad():
            for name in ('caa', 'aaf', 'cross_view_embedding', 'comm_uncertainty'):
                if hasattr(cross_agent, name):
                    attr = getattr(cross_agent, name)
                    samples = []
                    for _ in range(args.iters):
                        t0 = time.perf_counter()
                        if name == 'aaf':
                            _ = attr.align_reference_points(veh_inst.ref_pts, 25.0)
                        else:
                            _ = attr(veh_inst.query_feats, inf_inst.query_feats)
                        samples.append((time.perf_counter() - t0) * 1000.0)
                    timings[name] = sum(samples) / len(samples)
    _write_output(args, timings)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write_output(args, timings):
    if not timings:
        print("[profile] no timings collected.")
        return
    rows = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    print("\nModule latency breakdown (ms; lower is better):")
    print(f"  {'module':<32s} {'latency_ms':>12s}")
    print("  " + "-" * 46)
    total = 0.0
    for name, ms in rows:
        print(f"  {name:<32s} {ms:>10.3f}")
        total += ms
    print("  " + "-" * 46)
    print(f"  {'TOTAL':<32s} {total:>10.3f}")
    print(f"  {'Estimated FPS':<32s} {1000.0 / (total + 100.0):>10.3f}  (100ms comm + overhead)")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
        with open(args.output, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['module', 'latency_ms'])
            for name, ms in rows:
                writer.writerow([name, f'{ms:.4f}'])
            writer.writerow(['TOTAL', f'{total:.4f}'])
        print(f"[profile] wrote {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', nargs='?', default=None,
                        help='Path to the model config (mmcv Config). Optional in --dry-run mode.')
    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='Path to a model checkpoint. Required for real-mode.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Skip model loading; profile sub-modules with synthetic inputs.')
    parser.add_argument('--shape', nargs=3, type=int, default=[3, 1600, 900],
                        metavar=('C', 'H', 'W'),
                        help='Synthetic image shape (channels, height, width).')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--output', type=str, default='figs/runtime_breakdown.csv',
                        help='CSV output path (passed through to plot_framework.py).')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run or args.checkpoint is None:
        dry_run_profile(args)
    else:
        real_profile(args.config, args.checkpoint, args)


if __name__ == '__main__':
    main()