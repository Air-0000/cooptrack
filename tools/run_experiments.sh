#!/bin/bash
# ============================================================================
# Experiment Runner for Aerial-Ground Cooperative Perception
# Usage: bash tools/run_experiments.sh [stage] [gpu_ids]
#   stage: 0=all, 1=baseline, 2=dgc, 3=full, 4=ablation
#   gpu_ids: e.g. "0,1" (default: 0,1)
# ============================================================================

set -e
STAGE=${1:-0}
GPUS=${2:-"0,1"}
CONFIG_DIR="projects/configs_spd_coop/cooptrack"
CKPT_DIR="ckpts"

mkdir -p "$CKPT_DIR" logs

# ============================================================
# Shared config overrides:
#   Point these at your actual data paths
# ============================================================
DATA_OVERRIDES="data_root=./datasets/V2X-Seq-SPD-Batch-65-10-10761/cooperative/ \
                info_root=./data/infos/V2X-Seq-SPD-Batch-65-10-10761-forecasting/cooperative/"

# ============================================================
# Stage 0: Environment validation — run 1 epoch on a tiny subset
#          to catch Config/import bugs before full training.
# ============================================================
run_validate_env() {
    echo "━━━ [Stage 0] Environment Validation ━━━"
    CUDA_VISIBLE_DEVICES="$GPUS" python tools/train.py \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        --cfg-options \
        max_epochs=1 \
        samples_per_gpu=1 \
        workers_per_gpu=0 \
        load_from=ckpts/cooptrack_r50_veh.pth \
        runner.max_iters=100 \
        evaluation.interval=9999 \
        ${DATA_OVERRIDES} \
        2>&1 | tee logs/00_validate.log
    echo "✓ Env validation done. Check logs/00_validate.log"
}

# ============================================================
# Experiment 1: CoopTrack Baseline (no DGC, no UGIM)
#   Config: use_aaf=False, use_caa=False, use_emb=False,
#           use_comm=False, use_gru=False
#   Purpose: Reproduce original CoopTrack on Griffin,
#            verify your env matches published numbers
#            (target: 25m+0err → AP 47.9)
# ============================================================
run_baseline() {
    echo "━━━ [Exp 1] CoopTrack Baseline ━━━"
    CUDA_VISIBLE_DEVICES="$GPUS" ./tools/dist_train.sh \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        2 \
        --cfg-options \
        use_aaf=False use_caa=False use_emb=False \
        use_comm=False use_gru=False \
        training_stage=3 \
        load_from=ckpts/cooptrack_r50_veh.pth \
        ${DATA_OVERRIDES} \
        work_dir=work_dirs/exp1_baseline \
        2>&1 | tee logs/exp1_baseline.log
    echo "✓ Baseline done. Check logs/exp1_baseline.log"
}

# ============================================================
# Experiment 2: DGC Only (AAF + Coordinate Correction)
#   Config: use_aaf=True, use_caa=False, all else False
#   Purpose: Isolate the effect of Dynamic Geometric Correction
#            Target: +5-8% AP over baseline at 40-55m
# ============================================================
run_dgc_only() {
    echo "━━━ [Exp 2] DGC Only ━━━"
    CUDA_VISIBLE_DEVICES="$GPUS" ./tools/dist_train.sh \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        2 \
        --cfg-options \
        use_aaf=True use_caa=False use_emb=False \
        use_comm=False use_gru=False \
        training_stage=3 \
        h_ref=25.0 \
        load_from=ckpts/cooptrack_r50_veh.pth \
        ${DATA_OVERRIDES} \
        work_dir=work_dirs/exp2_dgc \
        2>&1 | tee logs/exp2_dgc.log
    echo "✓ DGC only done. Check logs/exp2_dgc.log"
}

# ============================================================
# Experiment 3: Full Framework (DGC + UGIM)
#   Config: use_aaf=True, use_caa=True, all else False
#   Purpose: Main result — our full lightweight framework
#            Target: Close gap to target APs across all conditions
# ============================================================
run_full_framework() {
    echo "━━━ [Exp 3] Full Framework (DGC + UGIM) ━━━"
    CUDA_VISIBLE_DEVICES="$GPUS" ./tools/dist_train.sh \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        2 \
        --cfg-options \
        use_aaf=True use_caa=True use_emb=False \
        use_comm=False use_gru=False \
        training_stage=3 \
        h_ref=25.0 \
        load_from=ckpts/cooptrack_r50_veh.pth \
        ${DATA_OVERRIDES} \
        work_dir=work_dirs/exp3_full \
        2>&1 | tee logs/exp3_full.log
    echo "✓ Full framework done. Check logs/exp3_full.log"
}

# ============================================================
# Experiment 4a: + Cross-View Embedding (ablation)
# ============================================================
run_abl_embed() {
    echo "━━━ [Exp 4a] + Embedding ━━━"
    CUDA_VISIBLE_DEVICES="$GPUS" ./tools/dist_train.sh \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        2 \
        --cfg-options \
        use_aaf=True use_caa=True use_emb=True \
        use_comm=False use_gru=False \
        training_stage=3 \
        load_from=ckpts/cooptrack_r50_veh.pth \
        ${DATA_OVERRIDES} \
        work_dir=work_dirs/exp4a_embed \
        2>&1 | tee logs/exp4a_embed.log
}

# ============================================================
# Experiment 4b: + Dual GRU (ablation)
# ============================================================
run_abl_gru() {
    echo "━━━ [Exp 4b] + Dual GRU ━━━"
    CUDA_VISIBLE_DEVICES="$GPUS" ./tools/dist_train.sh \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        2 \
        --cfg-options \
        use_aaf=True use_caa=True use_emb=False \
        use_comm=False use_gru=True \
        training_stage=3 \
        load_from=ckpts/cooptrack_r50_veh.pth \
        ${DATA_OVERRIDES} \
        work_dir=work_dirs/exp4b_gru \
        2>&1 | tee logs/exp4b_gru.log
}

# ============================================================
# Experiment 4c: + Communication (ablation)
# ============================================================
run_abl_comm() {
    echo "━━━ [Exp 4c] + Communication ━━━"
    CUDA_VISIBLE_DEVICES="$GPUS" ./tools/dist_train.sh \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        2 \
        --cfg-options \
        use_aaf=True use_caa=True use_emb=False \
        use_comm=True use_gru=False \
        training_stage=3 \
        load_from=ckpts/cooptrack_r50_veh.pth \
        ${DATA_OVERRIDES} \
        work_dir=work_dirs/exp4c_comm \
        2>&1 | tee logs/exp4c_comm.log
}

# ============================================================
# Experiment 4d: Everything ON (full stack — ablation)
# ============================================================
run_abl_all() {
    echo "━━━ [Exp 4d] Full Stack (ALL ON) ━━━"
    CUDA_VISIBLE_DEVICES="$GPUS" ./tools/dist_train.sh \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        2 \
        --cfg-options \
        use_aaf=True use_caa=True use_emb=True \
        use_comm=True use_gru=True \
        training_stage=3 \
        load_from=ckpts/cooptrack_r50_veh.pth \
        ${DATA_OVERRIDES} \
        work_dir=work_dirs/exp4d_all \
        2>&1 | tee logs/exp4d_all.log
}

# ============================================================
# Evaluation helper — runs all saved checkpoints on validation
# ============================================================
run_eval() {
    local work_dir=$1
    local ckpt=$(ls -t ${work_dir}/latest.pth 2>/dev/null || echo "")
    if [ -z "$ckpt" ]; then
        echo "No checkpoint found in ${work_dir}, skipping eval."
        return
    fi
    echo "Evaluating ${ckpt}..."
    CUDA_VISIBLE_DEVICES="$GPUS" python tools/test.py \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        "$ckpt" \
        --eval det track \
        2>&1 | tee "${work_dir}/eval.log"
}

# ============================================================
# Extrapolation Test (for paper Figure 3)
#   Runs inference at altitudes 25-70m on a frozen model
#   Plots coordinate correction error vs altitude
# ============================================================
run_extrapolation_test() {
    echo "━━━ [Analysis] Extrapolation Curve ━━━"
    python tools/analysis_tools/plot_extrapolation.py \
        --config "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        --checkpoint work_dirs/exp3_full/latest.pth \
        --altitudes 25 30 35 40 45 50 55 60 65 70 \
        --output figs/extrapolation_curve.pdf \
        --pc_range "-51.2 -51.2 -5.0 51.2 51.2 3.0" \
        2>&1 | tee logs/extrapolation.log
    echo "✓ Extrapolation curve saved to figs/extrapolation_curve.pdf"
}

# ============================================================
# Runtime Profile (with AgentRuntimeProfiler)
# ============================================================
run_profile() {
    echo "━━━ [Analysis] Runtime Profile ━━━"
    CUDA_VISIBLE_DEVICES="0" python tools/analysis_tools/profile_model.py \
        "${CONFIG_DIR}/uaf_gaf_track_r50_stream_bs8_48epoch_3cls.py" \
        work_dirs/exp3_full/latest.pth \
        --shape 3 1600 900 \
        --profile-memory \
        --profile-forward \
        2>&1 | tee logs/runtime_profile.log
    echo "✓ Profile saved to logs/runtime_profile.log"
}

# ============================================================
# Main dispatch
# ============================================================
case $STAGE in
    0) run_validate_env ;;
    1) run_baseline ;;
    2) run_dgc_only ;;
    3) run_full_framework ;;
    4)
        run_abl_embed
        run_abl_gru
        run_abl_comm
        run_abl_all
        ;;
    5)
        run_extrapolation_test
        run_profile
        ;;
    all|*)
        run_validate_env
        run_baseline
        run_dgc_only
        run_full_framework
        run_abl_embed
        run_abl_gru
        run_abl_comm
        run_abl_all
        run_extrapolation_test
        run_profile
        ;;
esac

echo ""
echo "=============================================="
echo "  All experiments submitted. Monitor logs in:"
echo "    logs/"
echo "    work_dirs/"
echo "=============================================="
