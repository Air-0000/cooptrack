#!/bin/bash
# ==============================================================================
# Adaptive Fusion Experiment Runner
# End-to-End Adaptive Fusion for Robust Aerial-Ground Cooperative 3D Detection
# ==============================================================================

# Configuration
DATASET_ROOT="/path/to/Griffin"
OUTPUT_DIR="./outputs/adaptive_fusion"

# ==============================================================================
# 1. Baseline (CoopTrack)
# ==============================================================================
echo "=========================================="
echo "1. Running CoopTrack Baseline"
echo "=========================================="

python -m mmdet3d eval \
    configs_spd_coop/cooptrack_25m.py \
    --work-dir ${OUTPUT_DIR}/baseline_25m \
    --checkpoint ./checkpoints/cooptrack_25m.pth

python -m mmdet3d eval \
    configs_spd_coop/cooptrack_40m.py \
    --work-dir ${OUTPUT_DIR}/baseline_40m \
    --checkpoint ./checkpoints/cooptrack_40m.pth

python -m mmdet3d eval \
    configs_spd_coop/cooptrack_55m.py \
    --work-dir ${OUTPUT_DIR}/baseline_55m \
    --checkpoint ./checkpoints/cooptrack_55m.pth

# ==============================================================================
# 2. With Pose Errors (Translation)
# ==============================================================================
echo "=========================================="
echo "2. Testing Pose Robustness (Translation)"
echo "=========================================="

for TRANS_ERROR in 0 2 5 7; do
    echo "Testing translation error: ${TRANS_ERROR}m"

    python -m mmdet3d eval \
        configs_spd_coop/cooptrack_25m.py \
        --cfg-options test.pose_translation_error=${TRANS_ERROR} \
        --work-dir ${OUTPUT_DIR}/pose_${TRANS_ERROR}m \
        --checkpoint ./checkpoints/cooptrack_25m.pth
done

# ==============================================================================
# 3. With Angle Errors
# ==============================================================================
echo "=========================================="
echo "3. Testing Pose Robustness (Angle)"
echo "=========================================="

for ANGLE_ERROR in 0 2 5; do
    echo "Testing angle error: ${ANGLE_ERROR}deg"

    python -m mmdet3d eval \
        configs_spd_coop/cooptrack_25m.py \
        --cfg-options test.pose_angle_error=${ANGLE_ERROR} \
        --work-dir ${OUTPUT_DIR}/angle_${ANGLE_ERROR}deg \
        --checkpoint ./checkpoints/cooptrack_25m.pth
done

# ==============================================================================
# 4. Ablation: Height-Adaptive Fusion Only
# ==============================================================================
echo "=========================================="
echo "4. Ablation: Height-Adaptive Fusion Only"
echo "=========================================="

python -m mmdet3d eval \
    configs_spd_coop/adaptive_fusion_height_only.py \
    --work-dir ${OUTPUT_DIR}/height_only \
    --checkpoint ./checkpoints/height_only.pth

# Test at different altitudes
python -m mmdet3d eval \
    configs_spd_coop/adaptive_fusion_height_only.py \
    --cfg-options data.alt_root=40 \
    --work-dir ${OUTPUT_DIR}/height_only_40m \
    --checkpoint ./checkpoints/height_only.pth

python -m mmdet3d eval \
    configs_spd_coop/adaptive_fusion_height_only.py \
    --cfg-options data.alt_root=55 \
    --work-dir ${OUTPUT_DIR}/height_only_55m \
    --checkpoint ./checkpoints/height_only.pth

# ==============================================================================
# 5. Ablation: Geometry-Aware Mechanisms Only
# ==============================================================================
echo "=========================================="
echo "5. Ablation: Geometry-Aware Mechanisms Only"
echo "=========================================="

python -m mmdet3d eval \
    configs_spd_coop/adaptive_fusion_geometry_only.py \
    --work-dir ${OUTPUT_DIR}/geometry_only \
    --checkpoint ./checkpoints/geometry_only.pth

# ==============================================================================
# 6. Full Model (Height-Adaptive + Geometry-Aware)
# ==============================================================================
echo "=========================================="
echo "6. Running Full Adaptive Fusion Model"
echo "=========================================="

python -m mmdet3d eval \
    configs_spd_coop/adaptive_fusion_full.py \
    --work-dir ${OUTPUT_DIR}/full_25m \
    --checkpoint ./checkpoints/full_25m.pth

python -m mmdet3d eval \
    configs_spd_coop/adaptive_fusion_full.py \
    --work-dir ${OUTPUT_DIR}/full_40m \
    --checkpoint ./checkpoints/full_40m.pth

python -m mmdet3d eval \
    configs_spd_coop/adaptive_fusion_full.py \
    --work-dir ${OUTPUT_DIR}/full_55m \
    --checkpoint ./checkpoints/full_55m.pth

# ==============================================================================
# 7. Full Model with Pose Errors
# ==============================================================================
echo "=========================================="
echo "7. Testing Full Model with Pose Errors"
echo "=========================================="

for TRANS_ERROR in 0 2 5 7; do
    echo "Full model + translation error: ${TRANS_ERROR}m"

    python -m mmdet3d eval \
        configs_spd_coop/adaptive_fusion_full.py \
        --cfg-options test.pose_translation_error=${TRANS_ERROR} \
        --work-dir ${OUTPUT_DIR}/full_pose_${TRANS_ERROR}m \
        --checkpoint ./checkpoints/full_25m.pth
done

for ANGLE_ERROR in 0 2 5; do
    echo "Full model + angle error: ${ANGLE_ERROR}deg"

    python -m mmdet3d eval \
        configs_spd_coop/adaptive_fusion_full.py \
        --cfg-options test.pose_angle_error=${ANGLE_ERROR} \
        --work-dir ${OUTPUT_DIR}/full_angle_${ANGLE_ERROR}deg \
        --checkpoint ./checkpoints/full_25m.pth
done

# ==============================================================================
# 8. Generate Results Table
# ==============================================================================
echo "=========================================="
echo "8. Generating Results Summary"
echo "=========================================="

python scripts/aggregate_results.py \
    --input-dir ${OUTPUT_DIR} \
    --output ${OUTPUT_DIR}/results_summary.json

echo "=========================================="
echo "Done! Results saved to ${OUTPUT_DIR}"
echo "=========================================="