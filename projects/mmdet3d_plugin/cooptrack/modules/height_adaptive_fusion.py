# -------------------------------------------------------------------------
# AAF: Altitude-Adaptive Fusion with Uncertainty Quantization
# GAF: Geometry-Aware Fusion (Enhanced Version)
# -------------------------------------------------------------------------
"""
Altitude-Adaptive Fusion with Uncertainty Quantization.

Key insight: Aerial-ground perception suffers from dual uncertainties:
1. Viewpoint uncertainty: perspective distortion from altitude disparity
2. Representation uncertainty: feature misalignment between modalities

This module introduces:
1. Uncertainty Quantization: per-query uncertainty prediction
2. Adaptive Fusion: uncertainty-weighted feature combination
3. Theoretical grounding: height-dependent projection error bounds

The Uncertainty Predictor learns to estimate fusion confidence,
enabling adaptive weighting based on feature quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UncertaintyPredictor(nn.Module):
    """
    Predicts per-query uncertainty for adaptive fusion.

    Unlike confidence scores from detection heads, this module learns
    to estimate the uncertainty of cross-modal feature alignment,
    which is not directly predictable from detection confidence.

    Architecture:
    - Feature encoder: extracts uncertainty-relevant patterns
    - Position encoder: height-dependent uncertainty estimation
    - Uncertainty predictor: outputs calibrated uncertainty scores
    """

    def __init__(self, embed_dims=256, hidden_dim=128):
        super(UncertaintyPredictor, self).__init__()

        # Feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Linear(embed_dims, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Position encoder for height-aware uncertainty
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Combined uncertainty predictor
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable temperature for calibration
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.1)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query_feats, ref_pts):
        """
        Args:
            query_feats: [N, embed_dims] query features
            ref_pts: [N, 3] reference points (normalized)

        Returns:
            uncertainty: [N, 1] uncertainty scores in (0, 1)
            log_prob: [N, 1] log probability for uncertainty learning
        """
        feat_encoded = self.feat_encoder(query_feats)
        pos_encoded = self.pos_encoder(ref_pts)

        combined = torch.cat([feat_encoded, pos_encoded], dim=-1)
        uncertainty_logit = self.uncertainty_net(combined)

        # Calibrated uncertainty using temperature scaling
        temperature = torch.exp(self.log_temperature)
        uncertainty = torch.sigmoid(uncertainty_logit / temperature)

        return uncertainty, uncertainty_logit


class GeometryConstraintLayer(nn.Module):
    """
    Geometry-aware constraint layer for robust matching.

    Instead of pure Euclidean distance, we incorporate:
    1. Direction consistency: matched pairs should have similar velocity directions
    2. Scale consistency: matched boxes should have similar sizes
    3. Geometric confidence: position uncertainty based on altitude

    This implements the geometric consistency branch of CAA.
    """

    def __init__(self, embed_dims=256):
        super(GeometryConstraintLayer, self).__init__()

        # Direction encoder (from motion features or bbox velocity)
        self.direction_encoder = nn.Sequential(
            nn.Linear(3, embed_dims // 4),
            nn.ReLU(inplace=True),
        )

        # Scale encoder (from bbox dimensions)
        self.scale_encoder = nn.Sequential(
            nn.Linear(3, embed_dims // 4),
            nn.ReLU(inplace=True),
        )

        # Position uncertainty estimator
        self.pos_uncertainty = nn.Sequential(
            nn.Linear(3, embed_dims // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 4, 1),
            nn.Sigmoid(),
        )

        # Combined geometry score
        self.geometry_score_net = nn.Sequential(
            nn.Linear(embed_dims // 2, embed_dims // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ref_pts, pred_boxes):
        """
        Args:
            ref_pts: [N, 3] reference points
            pred_boxes: [N, 10] predicted boxes (center, dims, rot, vel)

        Returns:
            geo_confidence: [N, 1] geometric confidence scores
        """
        # Extract velocity direction
        velocity = pred_boxes[..., 8:11]  # [N, 3]
        dir_encoded = self.direction_encoder(velocity)

        # Extract scale (dimensions)
        dims = pred_boxes[..., 3:6]  # [N, 3]
        scale_encoded = self.scale_encoder(dims)

        # Estimate position uncertainty based on altitude
        pos_uncertainty = self.pos_uncertainty(ref_pts)

        # Combine into geometry confidence
        combined = torch.cat([dir_encoded, scale_encoded], dim=-1)
        geo_score = self.geometry_score_net(combined)

        return geo_score * pos_uncertainty


class HeightAdaptiveFusion(nn.Module):
    """
    Altitude-Adaptive Fusion with Uncertainty Quantization.

    Key mechanisms:
    1. Height-aware coordinate remapping: compensates perspective distortion
    2. Uncertainty-based adaptive fusion: learns to weight cross-modal features
    3. Geometry-aware matching: incorporates geometric constraints

    Theoretical basis:
    - Projection error bound: ε_view ≤ k · |Δh| / h_ref · d
    - Uncertainty-aware fusion: w = σ(Φ([F; Δh])) · (1 - σ(U))
    """

    def __init__(self, embed_dims=256, h_ref=25.0, hidden_dim_ratio=2):
        super(HeightAdaptiveFusion, self).__init__()
        self.embed_dims = embed_dims
        self.h_ref = h_ref
        hidden_dim = embed_dims * hidden_dim_ratio
        self.pe_dim = 64  # positional encoding dimension for altitude

        # --- 1. Uncertainty Predictor ---
        self.uncertainty_predictor = UncertaintyPredictor(embed_dims, hidden_dim // 2)

        # --- 2. Sinusoidal Positional Encoding for altitude ---
        # Encoding altitude h and deviation Δh into a smooth high-d feature,
        # enabling extrapolation beyond training range (25-55m).
        # Instead of learnable affine W,b, we predict a residual ΔP(h):
        #     P' = P + ΔP(h),   ΔP(h) = MLP([PE(h); PE(Δh)])
        self.register_buffer('pe_freq', 10000 ** (2 * torch.arange(self.pe_dim // 4) / (self.pe_dim // 4)))

        # --- 3. Residual correction MLP (extrapolation-robust) ---
        # Input: [PE(h); PE(Δh)] of dim 2 * (pe_dim//2)*2 = pe_dim, Output: ΔP ∈ R^3
        # This replaces the old learnable affine_W/affine_b which collapsed outside 25-55m.
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.pe_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # output ΔP ∈ R^3 (residual offset)
        )

        # --- 4. Dynamic fusion weight network ---
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(embed_dims + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # --- 5. Feature alignment MLP ---
        self.feat_align_mlp = nn.Sequential(
            nn.Linear(embed_dims + 9, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dims),
        )

        # --- 6. Geometry constraint layer ---
        self.geometry_constraint = GeometryConstraintLayer(embed_dims)

        # --- 7. Feature fusion MLP ---
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dims * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dims),
        )

        # --- 8. Uncertainty-aware residual connection ---
        self.residual_net = nn.Sequential(
            nn.Linear(embed_dims * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dims),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _alt_pe(self, h):
        """
        Sinusoidal positional encoding for altitude.

        Args:
            h: [N, 1] altitude values

        Returns:
            pe: [N, pe_dim] positional encoding
        """
        # h shape: [N, 1]
        pe = h * self.pe_freq.unsqueeze(0)  # [N, pe_dim//4]
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)  # [N, pe_dim//2]
        # Double for odd/even encoding positions
        pe = torch.cat([pe, pe], dim=-1)  # [N, pe_dim]
        return pe

    def _residual_correction(self, ref_pts_veh):
        """
        Apply extrapolation-robust residual correction:
            P' = P + ΔP(h),   ΔP(h) = MLP([PE(h); PE(Δh)])

        Args:
            ref_pts_veh: [N, 3] points in vehicle coordinate frame (absolute)

        Returns:
            corrected_pts: [N, 3] residual-corrected points
            displacement: [N, 3] ΔP = corrected - raw
        """
        altitude = ref_pts_veh[..., 2:3]  # z-coordinate ≈ altitude in veh frame [N, 1]
        height_diff = altitude - self.h_ref  # deviation from reference [N, 1]

        pe_h = self._alt_pe(altitude)         # [N, pe_dim]
        pe_dh = self._alt_pe(height_diff)     # [N, pe_dim]
        pe_input = pe_h + pe_dh               # [N, pe_dim]

        delta_p = self.residual_mlp(pe_input)  # [N, 3] residual offset
        corrected_pts = ref_pts_veh + delta_p  # residual correction
        return corrected_pts, delta_p

    def _loc_norm(self, locs, pc_range):
        """Normalize locations to [0, 1] range."""
        locs_norm = locs.clone()
        locs_norm[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs_norm[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs_norm[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
        return locs_norm

    def _loc_denorm(self, ref_pts, pc_range):
        """Denormalize locations from [0, 1] to absolute coordinates."""
        locs = ref_pts.clone()
        locs[..., 0:1] = locs[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        locs[..., 1:2] = locs[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        locs[..., 2:3] = locs[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        return locs

    def align_reference_points(self, ref_pts_veh, altitude=None):
        """
        Align reference points through AAF's extrapolation-robust residual correction.

        Protocol (坐标系协议 — 死角 2 防御):
        -------------------------------------------------------------------
        1. ref_pts_veh MUST already be in the EGO-VEHICLE coordinate frame.
           This is ensured by apply_pose_transform() which transforms
           infrastructure-side points p_I into ego-vehicle frame:
               p_{I→V} = R · p_I + t
        2. All distance computations in GBA use these aligned points,
           strictly in the ego-vehicle coordinate frame.
        3. The correction is RESIDUAL: P' = P + ΔP(h), where ΔP learns
           "how much to shift given altitude change" rather than an absolute
           remapping. This avoids extrapolation collapse beyond the
           training altitude range (25-55m).
        -------------------------------------------------------------------

        Args:
            ref_pts_veh: [N, 3] infrastructure points in ego-vehicle
                         coordinate frame (absolute, NOT normalized).
                         This is guaranteed by the caller via
                         apply_pose_transform() before entering GBA.
            altitude: [N, 1] or scalar, current drone altitude.
                      If None, inferred from ref_pts_veh[..., 2].

        Returns:
            corrected_pts: [N, 3] residual-corrected points (veh frame)
            displacement: [N, 3] ΔP = corrected - raw (for ACM input)
        """
        if altitude is None:
            altitude = ref_pts_veh[..., 2:3]
        elif isinstance(altitude, (int, float)):
            altitude = torch.full_like(ref_pts_veh[..., :1], altitude)

        # Residual correction: P' = P + ΔP(h), ΔP = MLP(PE(h) + PE(Δh))
        corrected_pts, delta_p = self._residual_correction(ref_pts_veh)

        return corrected_pts, delta_p

    def compute_projection_error_bound(self, height_diff, distance):
        """
        Compute theoretical projection error bound.

        Args:
            height_diff: [N, 1] height difference from reference
            distance: [N, 1] horizontal distance

        Returns:
            error_bound: [N, 1] theoretical error upper bound
        """
        # ε_view ≤ k · |Δh| / h_ref · d
        k = 0.01  # scaling factor
        error_bound = k * torch.abs(height_diff) / self.h_ref * distance
        return error_bound

    def forward(self, inf_instances, veh_instances, veh2inf_rt, inf_pc_range, pc_range):
        """
        AAF forward pass with uncertainty quantization.

        Args:
            inf_instances: Infrastructure (drone) Instances
            veh_instances: Vehicle Instances
            veh2inf_rt: [4,4] transformation matrix
            inf_pc_range: Infrastructure point cloud range
            pc_range: Vehicle point cloud range

        Returns:
            fused_instances: Vehicle Instances after AAF fusion
            matched_veh_idx: Vehicle indices that participated in fusion
            matched_inf_idx: Infrastructure indices that participated in fusion
            uncertainty: Per-query uncertainty scores
        """
        if len(inf_instances) == 0:
            return veh_instances, torch.tensor([], device=veh_instances.query_feats.device), \
                   torch.tensor([], device=inf_instances.query_feats.device), None

        device = inf_instances.query_feats.device
        inf_n = len(inf_instances)
        veh_n = len(veh_instances)

        # === Step 1: Coordinate transformation inf→vehicle ===
        inf_ref_pts = self._loc_denorm(inf_instances.ref_pts, inf_pc_range)
        calib = np.linalg.inv(veh2inf_rt.cpu().numpy().T)
        calib = torch.from_numpy(calib).to(device)
        inf_ref_pts_h = torch.cat([inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])], dim=-1).unsqueeze(-1)
        inf_ref_pts_veh = torch.matmul(calib, inf_ref_pts_h).squeeze(-1)[..., :3]

        # === Step 2: Extrapolation-robust residual correction ===
        # P' = P + ΔP(h),   ΔP(h) = MLP(PE(h) + PE(Δh))
        # This replaces the old affine normalization (P' = P*W + b) which
        # caused extrapolation collapse beyond the training altitude range.
        inf_ref_pts_corrected, _ = self._residual_correction(inf_ref_pts_veh)
        inf_ref_pts_norm = self._loc_norm(inf_ref_pts_corrected, pc_range)

        # === Step 3: Uncertainty prediction for inf queries ===
        inf_uncertainty, inf_uncertainty_logit = self.uncertainty_predictor(
            inf_instances.query_feats, inf_ref_pts_norm
        )

        # === Step 4: Geometry-aware matching ===
        veh_ref_pts = self._loc_denorm(veh_instances.ref_pts, pc_range)
        veh_ref_pts_norm = self._loc_norm(veh_ref_pts, pc_range)

        inf_geo_conf = self.geometry_constraint(inf_ref_pts_norm, inf_instances.pred_boxes)
        inf_pts_exp = inf_ref_pts_norm.unsqueeze(1).expand(inf_n, veh_n, 3)
        veh_pts_exp = veh_ref_pts_norm.unsqueeze(0).expand(inf_n, veh_n, 3)
        distances = torch.sqrt(torch.sum((inf_pts_exp - veh_pts_exp) ** 2, dim=-1))

        # Uncertainty-weighted distance
        uncertainty_exp = inf_uncertainty.unsqueeze(1).expand(inf_n, veh_n)
        geo_conf_exp = inf_geo_conf.unsqueeze(1).expand(inf_n, veh_n)
        weighted_dist = distances * (1 + uncertainty_exp) / (geo_conf_exp + 1e-6)

        # Nearest neighbor matching with confidence filtering
        filter_mask = weighted_dist < 2.0  # relaxed threshold with uncertainty

        matched_inf_idx = []
        matched_veh_idx = []
        for i in range(inf_n):
            if filter_mask[i].any():
                j = distances[i].argmin()
                if j not in matched_veh_idx:
                    matched_inf_idx.append(i)
                    matched_veh_idx.append(j)

        matched_inf_idx = torch.tensor(matched_inf_idx, device=device)
        matched_veh_idx = torch.tensor(matched_veh_idx, device=device)

        if len(matched_inf_idx) == 0:
            return veh_instances, matched_veh_idx, matched_inf_idx, None

        # === Step 5: Cross-agent feature alignment ===
        inf2veh_r = calib[:3, :3].reshape(1, 9).repeat(inf_n, 1)
        inf_query_aligned = self.feat_align_mlp(
            torch.cat([inf_instances.query_feats, inf2veh_r], dim=-1)
        )

        # === Step 6: Dynamic height-weighted feature fusion with uncertainty ===
        altitude = inf_ref_pts_veh[..., 2:3]
        h_diff = altitude - self.h_ref
        h_diff_expanded = h_diff.expand(-1, self.embed_dims)
        feat_concat = torch.cat([inf_instances.query_feats, h_diff_expanded], dim=-1)
        height_weight = self.fusion_weight_net(feat_concat).squeeze(-1)

        # Uncertainty-based adaptive weighting
        uncertainty_matched = inf_uncertainty[matched_inf_idx].squeeze(-1)
        adaptive_weight = height_weight[matched_inf_idx] * (1 - uncertainty_matched)

        # Normalize to [0, 1]
        adaptive_weight = torch.clamp(adaptive_weight, 0, 1)

        # === Step 7: Uncertainty-aware feature fusion ===
        w = adaptive_weight.unsqueeze(-1)
        fused_feats = self.fusion_mlp(
            torch.cat([
                veh_instances.query_feats[matched_veh_idx] * (1 - w),
                inf_query_aligned[matched_inf_idx] * w,
            ], dim=-1)
        )

        # === Step 8: Residual connection with uncertainty gating ===
        residual_input = torch.cat([
            veh_instances.query_feats[matched_veh_idx],
            inf_query_aligned[matched_inf_idx]
        ], dim=-1)

        # Only apply residual when uncertainty is low
        residual_gate = (1 - uncertainty_matched).unsqueeze(-1)
        residual = self.residual_net(residual_input) * residual_gate
        fused_feats = fused_feats + residual

        # === Step 9: Clone and update instances ===
        fused_instances = veh_instances[matched_veh_idx].clone()
        fused_instances.query_feats = fused_feats

        # Store uncertainty for loss computation
        fused_instances.uncertainty = uncertainty_matched

        return fused_instances, matched_veh_idx, matched_inf_idx, uncertainty_matched

    def compute_uncertainty_loss(self, pred_uncertainty, target_uncertainty):
        """
        Compute uncertainty learning loss.

        Uses negative log-likelihood with calibration.

        Args:
            pred_uncertainty: [N, 1] predicted uncertainty
            target_uncertainty: [N, 1] target uncertainty (0=high quality, 1=low quality)

        Returns:
            loss: uncertainty learning loss
        """
        # BCE loss between predicted and target uncertainty
        loss = F.binary_cross_entropy(
            pred_uncertainty,
            target_uncertainty,
            reduction='mean'
        )
        return loss