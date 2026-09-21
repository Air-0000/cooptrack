# -------------------------------------------------------------------------
# Adaptive Fusion for Robust Aerial-Ground Cooperative 3D Detection
# -------------------------------------------------------------------------
"""
Modular Architecture:
1. AAF: Altitude-Adaptive Fusion (主贡献)
2. CAA: Context-Aware Association (辅助)
3. GRU: 时空预测 (辅助)
4. ACM: Adaptive Confidence Modulation (辅助)

Key insight: Aerial-ground perception suffers from geometric uncertainties:
1. Altitude variation: perspective distortion from altitude disparity (25m-55m)
2. Pose errors: translation offset up to 7m, rotation misalignment
3. Angle offsets: dynamic pitch/yaw errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Module 1: AAF - Altitude-Adaptive Fusion (主贡献)
# ============================================================================
class AAF(nn.Module):
    """
    Altitude-Adaptive Fusion Module (主贡献).

    Key mechanisms:
    1. Height-aware coordinate remapping: compensates perspective distortion
    2. Affine normalization: learnable scale transformation
    3. Dynamic height-weighted fusion: adaptively combines features based on altitude

    Unlike OpenCOOD-Air's static CDSC, AAF is end-to-end jointly optimized.
    """

    def __init__(self, embed_dims=256, h_ref=25.0, hidden_dim_ratio=2):
        super(AAF, self).__init__()
        self.embed_dims = embed_dims
        self.h_ref = h_ref
        hidden_dim = embed_dims * hidden_dim_ratio
        self.pe_dim = 64

        # 1. Sinusoidal altitude PE
        self.register_buffer('pe_freq', 10000 ** (2 * torch.arange(self.pe_dim // 4) / (self.pe_dim // 4)))

        # 2. Residual correction MLP (replaces old remap_mlp + affine_W/b)
        # P' = P + ΔP(h), ΔP = MLP(PE(h) + PE(Δh))
        #
        # Output is R^4, not R^3:
        #   [0:3] translation residual ΔP ∈ R^3
        #   [3]   log-scale s along the line of sight (drone optical centre → point)
        #
        # The 4th DOF is the learnable replacement for the hand-crafted
        # `refine_ratio` that the released Griffin codebase ships DISABLED:
        #     ray = P - drone_pos;  P' = drone_pos + ray * refine_ratio
        # A fixed/analytic ratio cannot express altitude-dependent behaviour, so
        # we predict log s from PE(h) instead. exp(s) keeps the scale positive and
        # s = 0 is the identity, preserving DGC's graceful fallback for unseen h.
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.pe_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

        # 3. Dynamic fusion weight network
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(embed_dims + 3, hidden_dim),  # [F; h]
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 4. Feature alignment MLP
        self.feat_align_mlp = nn.Sequential(
            nn.Linear(embed_dims + 9, hidden_dim),  # [F; R]
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dims),
        )

        # 5. Feature fusion MLP
        self.fusion_mlp = nn.Sequential(
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
        pe = h * self.pe_freq.unsqueeze(0)
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)
        pe = torch.cat([pe, pe], dim=-1)
        return pe

    def _residual_correction(self, ref_pts_veh, drone_pos, altitude):
        """
        Altitude-conditioned residual correction with the 4th DOF.

        Args:
            ref_pts_veh: [N, 3] drone reference points in the ego-vehicle frame
            drone_pos:   [3]    drone optical centre in the ego-vehicle frame
                                (calib_inf2veh[:3, 3])
            altitude:    [N, 1] flight altitude h — a per-FRAME scalar broadcast to
                                N queries. This is drone_pos[2], NOT the z coordinate
                                of a reference point (that is the object's height,
                                ~0 for ground objects).

        Returns:
            corrected_pts: [N, 3]
            delta_p:       [N, 3] translation residual
            log_scale:     [N, 1] line-of-sight log-scale
        """
        height_diff = altitude - self.h_ref
        pe_h = self._alt_pe(altitude)
        pe_dh = self._alt_pe(height_diff)
        pe_input = pe_h + pe_dh
        delta = self.residual_mlp(pe_input)

        delta_p = delta[..., :3]
        log_scale = delta[..., 3:4].clamp(-0.5, 0.5)

        # Line-of-sight scaling about the drone optical centre. This is the
        # learnable stand-in for Griffin's disabled analytic `refine_ratio`.
        drone_pos = drone_pos.reshape(1, 3)
        ray = ref_pts_veh - drone_pos
        corrected_pts = drone_pos + ray * torch.exp(log_scale) + delta_p

        return corrected_pts, delta_p, log_scale

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

    def forward(self, inf_instances, veh_instances, veh2inf_rt, inf_pc_range, pc_range):
        """
        AAF forward pass.

        Args:
            inf_instances: Infrastructure (drone) Instances
            veh_instances: Vehicle Instances
            veh2inf_rt: [4,4] transformation matrix
            inf_pc_range: Infrastructure point cloud range
            pc_range: Vehicle point cloud range

        Returns:
            fused_feats: Fused features
            height_weight: Adaptive fusion weights based on altitude
        """
        if len(inf_instances) == 0:
            return veh_instances.query_feats, None

        device = inf_instances.query_feats.device
        inf_n = len(inf_instances)
        veh_n = len(veh_instances)

        # Step 1: Coordinate transformation inf→vehicle
        inf_ref_pts = self._loc_denorm(inf_instances.ref_pts, inf_pc_range)
        calib = np.linalg.inv(veh2inf_rt.cpu().numpy().T)
        calib = torch.from_numpy(calib).to(device=device, dtype=inf_ref_pts.dtype)
        inf_ref_pts_h = torch.cat([inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])], dim=-1).unsqueeze(-1)
        inf_ref_pts_veh = torch.matmul(calib, inf_ref_pts_h).squeeze(-1)[..., :3]

        # Step 1b: flight altitude.
        # h is the DRONE's height in the ego-vehicle frame — one scalar per frame.
        # Using ref_pts_veh[..., 2] here would be wrong: that is the object's
        # height above the ground plane (~0), which carries no altitude signal and
        # would make PE(h) depend on what is being detected rather than on where
        # the drone is flying.
        drone_pos = calib[:3, 3].detach()                       # [3]
        altitude = drone_pos[2].reshape(1, 1).expand(inf_n, 1)  # [N, 1]

        # Step 2: Extrapolation-robust residual correction (ΔP ∈ R^4)
        # P' = drone_pos + (P - drone_pos)·exp(s) + ΔP,
        #      [ΔP; s] = MLP(PE(h) + PE(Δh))
        inf_ref_pts_corrected, delta_p, log_scale = self._residual_correction(
            inf_ref_pts_veh, drone_pos, altitude
        )
        inf_ref_pts_norm = self._loc_norm(inf_ref_pts_corrected, pc_range)

        # Step 3: Cross-agent feature alignment
        inf2veh_r = calib[:3, :3].reshape(1, 9).repeat(inf_n, 1)
        inf_query_aligned = self.feat_align_mlp(
            torch.cat([inf_instances.query_feats, inf2veh_r], dim=-1)
        )

        # Step 4: Dynamic height-weighted feature fusion
        # (altitude already defined in Step 1b as the per-frame flight height)
        h_diff = altitude - self.h_ref  # deviation from reference altitude
        # fusion_weight_net's first layer is Linear(embed_dims + 3, ...), so the
        # altitude deviation must occupy 3 slots. Expanding it to embed_dims
        # produced a 512-wide concat against a 259-wide layer.
        h_diff_expanded = h_diff.expand(-1, 3)
        feat_concat = torch.cat([inf_instances.query_feats, h_diff_expanded], dim=-1)
        height_weight = self.fusion_weight_net(feat_concat).squeeze(-1)  # [N]

        # Step 5: return ALIGNED inf features — not fused ones.
        # Fusion belongs to the caller: AdaptiveFusion matches inf↔veh queries
        # first and only then fuses the matched pairs. The old loop fused every
        # inf query against ALL veh queries, producing [N, M, D], which broke
        # `aaf_feats[matched_inf_idx]` downstream (and crashed on the 2-D/1-D
        # concat when veh_n == 1).
        #
        # altitude and delta_p are returned because ACM needs them:
        #   altitude → the σ ∝ h prior
        #   delta_p  → ACM's geometric (feature-free) input, keeping the
        #              confidence branch gradient-orthogonal to CAA
        return inf_query_aligned, height_weight, inf_ref_pts_norm, altitude, delta_p


# ============================================================================
# Module 2: CAA - Context-Aware Association (辅助)
# ============================================================================
class CAA(nn.Module):
    """
    Context-Aware Association Module (辅助).

    Key mechanisms:
    1. Semantic matching: feature similarity-based association
    2. Geometry-aware matching: direction + scale consistency
    3. Replaces pure 3DIoU matching

    Combines with ACM for robust association under pose errors.
    """

    def __init__(self, embed_dims=256):
        super(CAA, self).__init__()
        self.embed_dims = embed_dims

        # Semantic similarity encoder
        self.semantic_encoder = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
        )

        # Direction encoder (from bbox velocity).
        # pred_boxes is CoopTrack's 10-D code:
        #   [0:3] centre xyz | [3:6] dims wlh | [6:8] sin/cos yaw | [8:10] vel vx,vy
        # so velocity is 2-D (vx, vy), matching spatial_temporal_reason.py.
        # It was previously declared Linear(3, ...) and blew up at runtime.
        self.direction_encoder = nn.Sequential(
            nn.Linear(2, embed_dims // 4),
            nn.ReLU(inplace=True),
        )

        # Scale encoder (from bbox dimensions)
        self.scale_encoder = nn.Sequential(
            nn.Linear(3, embed_dims // 4),
            nn.ReLU(inplace=True),
        )

        # Combined matching score.
        # Its input is cat(dir_encoded, scale_encoded) = embed_dims//4 * 2 =
        # embed_dims//2, NOT embed_dims. The old Linear(embed_dims, ...) expected
        # 256 and crashed on the 128-wide concat.
        self.matching_net = nn.Sequential(
            nn.Linear(embed_dims // 2, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_semantic_similarity(self, feat_a, feat_b):
        """Compute semantic similarity between query features."""
        feat_a_enc = self.semantic_encoder(feat_a)
        feat_b_enc = self.semantic_encoder(feat_b)
        similarity = torch.cosine_similarity(feat_a_enc, feat_b_enc, dim=-1)
        return similarity

    def compute_geometry_score(self, ref_pts, pred_boxes):
        """
        Compute geometric consistency score.

        Args:
            ref_pts: [N, 3] reference points
            pred_boxes: [N, 10] predicted boxes (center, dims, rot, vel)

        Returns:
            geo_score: geometric consistency scores
        """
        # Extract velocity direction
        velocity = pred_boxes[..., 8:10]  # [N, 2] vx, vy — see layout note above
        dir_encoded = self.direction_encoder(velocity)

        # Extract scale (dimensions)
        dims = pred_boxes[..., 3:6]  # [N, 3]
        scale_encoded = self.scale_encoder(dims)

        # Combine into geometry score
        combined = torch.cat([dir_encoded, scale_encoded], dim=-1)
        geo_score = self.matching_net(combined)

        return torch.sigmoid(geo_score)

    def forward(self, inf_feats, veh_feats, inf_pts, veh_pts, inf_boxes, veh_boxes):
        """
        CAA forward pass for cross-modal matching.

        Args:
            inf_feats: [N, D] Infrastructure query features
            veh_feats: [M, D] Vehicle query features
            inf_pts: [N, 3] Infrastructure reference points
            veh_pts: [M, 3] Vehicle reference points
            inf_boxes: [N, 10] Infrastructure predicted boxes
            veh_boxes: [M, 10] Vehicle predicted boxes

        Returns:
            matching_scores: [N, M] matching scores for association
        """
        inf_n = len(inf_feats)
        veh_n = len(veh_feats)

        # Compute semantic similarity
        inf_feats_exp = inf_feats.unsqueeze(1).expand(inf_n, veh_n, self.embed_dims)
        veh_feats_exp = veh_feats.unsqueeze(0).expand(inf_n, veh_n, self.embed_dims)
        semantic_sim = self.compute_semantic_similarity(
            inf_feats_exp.reshape(-1, self.embed_dims),
            veh_feats_exp.reshape(-1, self.embed_dims)
        ).reshape(inf_n, veh_n)

        # Compute geometric consistency
        inf_geo = self.compute_geometry_score(inf_pts, inf_boxes)  # [N, 1]
        veh_geo = self.compute_geometry_score(veh_pts, veh_boxes)  # [M, 1]

        # Geometric score for pairs (combination of direction + scale consistency)
        # inf_geo / veh_geo are [N, 1] and [M, 1]: broadcast straight to [N, M].
        # The old `.unsqueeze(1)` / `.unsqueeze(0)` produced [N,1,1] / [1,M,1],
        # which made expand() raise. Info: veh_geo needs a transpose to [1, M].
        inf_geo_exp = inf_geo.expand(inf_n, veh_n)          # [N, 1] -> [N, M]
        veh_geo_exp = veh_geo.T.expand(inf_n, veh_n)        # [M, 1] -> [1, M] -> [N, M]
        geometry_score = (inf_geo_exp + veh_geo_exp) / 2    # [N, M]

        # Combined matching score: semantic + geometric.
        # NOTE: no squeeze(-1) here — geometry_score is genuinely 2-D [N, M] and
        # squeezing it collapsed the score matrix to [N], silently breaking the
        # element-wise product with semantic_sim.
        matching_scores = semantic_sim * geometry_score     # [N, M]

        return matching_scores


# ============================================================================
# Module 3: ACM - Adaptive Confidence Modulation (辅助)
# ============================================================================
class ACM(nn.Module):
    """
    Adaptive Confidence Modulation Module (辅助).

    Key mechanisms:
    1. Per-query uncertainty prediction
    2. Dynamic 0-1 confidence weighting
    3. Suppress low-quality queries in fusion

    Integrates with AAF and CAA for robust fusion.
    """

    def __init__(self, embed_dims=256, hidden_dim=128, h_ref=25.0, sigma_min=0.1):
        super(ACM, self).__init__()
        self.h_ref = h_ref
        self.sigma_min = sigma_min

        # Displacement encoder (Δp: AAF-corrected - raw coordinate)
        # Takes 3D displacement vector → hidden representation
        # This decouples ACM from CAA: ACM uses geometric uncertainty
        # (how much AAF corrected the point), NOT feature similarity.
        self.disp_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Position encoder (for height-dependent uncertainty)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Altitude encoder — h is what drives the first-order trend σ ∝ h.
        self.alt_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(inplace=True),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(2 * (hidden_dim // 2) + hidden_dim // 4, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # σ head: predicts the DEVIATION from the analytic σ ∝ h trend, not the
        # trend itself. See forward().
        self.sigma_head = nn.Linear(hidden_dim, 1)

        # Confidence predictor (from displacement + position only)
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
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

    def forward(self, displacement, ref_pts, altitude):
        """
        ACM forward pass.

        Uses the DGC displacement (Δp) and the flight altitude h as inputs instead
        of fused features, keeping the confidence branch gradient-orthogonal to CAA.

        Args:
            displacement: [N, 3] DGC displacement = corrected_pts - raw_pts
            ref_pts:      [N, 3] corrected reference points (position context)
            altitude:     [N, 1] flight altitude h (per-frame scalar broadcast to N)

        Returns:
            sigma:      [N, 1] per-query estimate, in metres, of the BEV
                               localisation error that SURVIVES DGC correction
            confidence: [N, 1] confidence scores in (0, 1)
        """
        disp_encoded = self.disp_encoder(displacement)  # [N, hidden_dim/2]
        pos_encoded = self.pos_encoder(ref_pts)          # [N, hidden_dim/2]
        alt_encoded = self.alt_encoder(altitude)         # [N, hidden_dim/4]

        combined = torch.cat([disp_encoded, pos_encoded, alt_encoded], dim=-1)
        feat = self.trunk(self.input_proj(combined))     # [N, hidden_dim]

        # ---- σ: analytic first-order trend + learned deviation ----------------
        # Under the pinhole model the BEV localisation error grows approximately
        # linearly with altitude (σ_pos ∝ h). That trend carries NO learned
        # parameters; the network only learns the multiplicative deviation from it.
        # This is what keeps the head data-efficient and lets the tolerance
        # extrapolate to altitudes outside the training range.
        deviation = F.softplus(self.sigma_head(feat))            # > 0
        h_ratio = (altitude / self.h_ref).clamp(min=1e-3)
        sigma = self.sigma_min + deviation * h_ratio             # [N, 1], metres

        # Temperature scaling for calibration
        temperature = torch.exp(self.log_temperature)
        confidence = self.confidence_net(feat) * temperature

        return sigma, confidence.clamp(0, 1)


# ============================================================================
# Module 4: Dual GRU Spatial-Temporal Prediction (辅助, 隐患 3 修复)
# ============================================================================
class VehicleGRU(nn.Module):
    """Vehicle-side GRU for spatio-temporal prediction."""
    def __init__(self, embed_dims=256, num_history=5):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_history = num_history
        self.gru = nn.GRUCell(input_size=embed_dims, hidden_size=embed_dims)
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dims * num_history, embed_dims), nn.ReLU(inplace=True),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, history_feats):
        B, T, N, D = history_feats.shape
        input_proj = self.input_proj(history_feats.reshape(B, T * N, D))
        hidden_state = torch.zeros(B, N, self.embed_dims, device=history_feats.device)
        for t in range(T):
            h_t = input_proj[:, t * N:(t + 1) * N, :]
            hidden_state = self.gru(h_t, hidden_state)
        return self.output_proj(hidden_state)


class InfrastructureGRU(nn.Module):
    """Infrastructure-side GRU for spatio-temporal prediction.

    Note: NOT shared with VehicleGRU — separate params per agent type.
    This prevents feature space mismatch during cooperative training.
    """
    def __init__(self, embed_dims=256, num_history=5):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_history = num_history
        self.gru = nn.GRUCell(input_size=embed_dims, hidden_size=embed_dims)
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dims * num_history, embed_dims), nn.ReLU(inplace=True),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, history_feats):
        B, T, N, D = history_feats.shape
        input_proj = self.input_proj(history_feats.reshape(B, T * N, D))
        hidden_state = torch.zeros(B, N, self.embed_dims, device=history_feats.device)
        for t in range(T):
            h_t = input_proj[:, t * N:(t + 1) * N, :]
            hidden_state = self.gru(h_t, hidden_state)
        return self.output_proj(hidden_state)


# ============================================================================
# Main Module: AdaptiveFusion (整合)
# ============================================================================
class AdaptiveFusion(nn.Module):
    """
    Integrated Adaptive Fusion Module.

    Combines:
    1. AAF: Altitude-Adaptive Fusion (主贡献)
    2. CAA: Context-Aware Association (辅助)
    3. ACM: Adaptive Confidence Modulation (辅助)
    4. GRU: Spatio-Temporal Prediction (辅助)

    Usage:
        model = AdaptiveFusion(embed_dims=256)
        fused = model(inf_instances, veh_instances, veh2inf_rt, ...)
    """

    def __init__(self, embed_dims=256, h_ref=25.0, hidden_dim_ratio=2, use_gru=True):
        super(AdaptiveFusion, self).__init__()
        self.embed_dims = embed_dims
        self.use_gru = use_gru
        # Stage gating: GRU only active during Stage 3 (cooperative training).
        # In Stage 1 (vehicle) / Stage 2 (infrastructure), GRU forward is a no-op.
        # This prevents feature space mismatch: vehicle and infrastructure learn
        # independent temporal dynamics before being aligned in Stage 3.
        self.training_stage = 3  # 1=veh, 2=inf, 3=cooperative

        # Initialize all modules
        self.aaf = AAF(embed_dims, h_ref, hidden_dim_ratio)
        self.caa = CAA(embed_dims)
        self.acm = ACM(embed_dims, embed_dims // 2)

        if use_gru:
            # Separate GRUs per agent type (NOT shared)
            self.veh_gru = VehicleGRU(embed_dims)
            self.inf_gru = InfrastructureGRU(embed_dims)

        # Feature fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims * hidden_dim_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims * hidden_dim_ratio, embed_dims),
        )

        # Uncertainty-aware residual connection
        self.residual_net = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims * hidden_dim_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims * hidden_dim_ratio, embed_dims),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inf_instances, veh_instances, veh2inf_rt, inf_pc_range, pc_range,
                history_feats=None, missing_mask=None):
        """
        Integrated Adaptive Fusion forward pass.

        Args:
            inf_instances: Infrastructure (drone) Instances
            veh_instances: Vehicle Instances
            veh2inf_rt: [4,4] transformation matrix
            inf_pc_range: Infrastructure point cloud range
            pc_range: Vehicle point cloud range
            history_feats: [B, T, N, D] optional historical features for GRU
            missing_mask: [B, N] optional mask for missing queries

        Returns:
            fused_instances: Fused instances with updated features
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

        # === Step 1: DGC (inside AAF) - altitude-aware residual correction ===
        aaf_feats, height_weight, inf_pts_norm, altitude, delta_p = self.aaf(
            inf_instances, veh_instances, veh2inf_rt, inf_pc_range, pc_range
        )

        # === Step 2: ACM - Adaptive Confidence Modulation ===
        # Δp is the displacement DGC actually applied — a geometric, feature-free
        # signal, which is what keeps this branch orthogonal to CAA's loss.
        # (Previously this read inf_instances.displacement, which was usually never
        #  set and silently fell back to zeros, making the branch input-free.)
        displacement = delta_p
        sigma, confidence = self.acm(displacement, inf_pts_norm, altitude)

        # === Step 3: CAA - Context-Aware Association ===
        # Compute geometric confidence for matching
        inf_geo = self.caa.compute_geometry_score(inf_pts_norm, inf_instances.pred_boxes)

        # Get vehicle reference points
        veh_ref_pts_norm = veh_instances.ref_pts  # Assuming normalized

        # Matching with an uncertainty-aware (Mahalanobis) tolerance
        inf_pts_exp = inf_pts_norm.unsqueeze(1).expand(inf_n, veh_n, 3)
        veh_pts_exp = veh_ref_pts_norm.unsqueeze(0).expand(inf_n, veh_n, 3)
        distances = torch.sqrt(torch.sum((inf_pts_exp - veh_pts_exp) ** 2, dim=-1) + 1e-12)

        # σ is in metres while ref_pts are normalised → rescale by the BEV extent.
        sigma_scale = float(pc_range[3] - pc_range[0])
        # sigma is already [N, 1]; broadcasting to [N, M] needs no extra dim.
        # The previous `.unsqueeze(1)` made it [N, 1, 1] and expand() raised.
        sigma_exp = (sigma / sigma_scale).expand(inf_n, veh_n)
        mahal = distances / (sigma_exp + 1e-6)

        # Both are [N, 1]; broadcast straight to [N, M] (unsqueeze was a bug).
        confidence_exp = confidence.expand(inf_n, veh_n)
        geo_conf_exp = inf_geo.expand(inf_n, veh_n)
        weighted_dist = mahal * (1 + confidence_exp) / (geo_conf_exp + 1e-6)

        # Nearest neighbor matching
        filter_mask = weighted_dist < 2.0

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

        # === Step 4: GRU Spatio-Temporal Prediction (Stage 3 only) ===
        # In Stage 1 (veh-only) and Stage 2 (inf-only), GRU is disabled
        # to prevent feature space mismatch during alignment in Stage 3.
        if self.use_gru and self.training_stage == 3 and history_feats is not None and missing_mask is not None:
            # Use InfrastructureGRU for inf-side prediction (separate params)
            predicted_feats = self.inf_gru(history_feats)
            # Replace missing features with predictions
            inf_query = torch.where(
                missing_mask.unsqueeze(-1).expand(-1, -1, self.embed_dims),
                predicted_feats,
                inf_instances.query_feats
            )
        else:
            inf_query = inf_instances.query_feats

        # Vehicle-side GRU (applied to veh_instances when available)
        veh_query = veh_instances.query_feats
        if self.use_gru and self.training_stage == 3 and hasattr(veh_instances, 'history_feats') \
                and veh_instances.history_feats is not None:
            veh_pred = self.veh_gru(veh_instances.history_feats)
            if hasattr(veh_instances, 'veh_missing_mask') and veh_instances.veh_missing_mask is not None:
                veh_query = torch.where(
                    veh_instances.veh_missing_mask.unsqueeze(-1).expand(-1, self.embed_dims),
                    veh_pred,
                    veh_query
                )

        # === Step 5: Confidence-weighted Feature Fusion ===
        uncertainty_matched = confidence[matched_inf_idx].squeeze(-1)
        adaptive_weight = height_weight[matched_inf_idx] * (1 - uncertainty_matched)
        adaptive_weight = torch.clamp(adaptive_weight, 0, 1)

        w = adaptive_weight.unsqueeze(-1)
        fused_feats = self.fusion_mlp(
            torch.cat([
                veh_instances.query_feats[matched_veh_idx] * (1 - w),
                aaf_feats[matched_inf_idx] * w,
            ], dim=-1)
        )

        # === Step 6: Uncertainty-aware Residual ===
        residual_input = torch.cat([
            veh_instances.query_feats[matched_veh_idx],
            aaf_feats[matched_inf_idx]
        ], dim=-1)
        residual_gate = (1 - uncertainty_matched).unsqueeze(-1)
        residual = self.residual_net(residual_input) * residual_gate
        fused_feats = fused_feats + residual

        # === Step 7: Output ===
        fused_instances = veh_instances[matched_veh_idx].clone()
        fused_instances.query_feats = fused_feats
        fused_instances.confidence = confidence_matched = uncertainty_matched

        return fused_instances, matched_veh_idx, matched_inf_idx, uncertainty_matched

    def compute_loss(self, pred_uncertainty, target_uncertainty):
        """
        Compute uncertainty learning loss (IoU-aware contrastive).

        ACM confidence uses displacement only (Δp).
        It is supervised indirectly by detection quality (IoU-aware weighting).
        This avoids direct gradient conflict with CAA's feature-space loss:
          - CAA maximizes feature distance between different instances (discriminative)
          - ACM minimizes confidence for high-displacement matches (geometric)
          - The two operate in orthogonal spaces — no gradient conflict.

        The target uncertainty is derived from matching quality:
          target_u = 1 - IoU_matched
        where IoU_matched is the 3D IoU between fused detection and ground truth.

        Args:
            pred_uncertainty: [N, 1] predicted uncertainty
            target_uncertainty: [N, 1] target uncertainty from detection IoU
                                (0=high quality match, 1=poor match)

        Returns:
            loss: contrastive uncertainty learning loss
        """
        # Contrastive loss: pull high-IoU pairs toward 0, push low-IoU toward 1
        return F.binary_cross_entropy(
            pred_uncertainty,
            target_uncertainty,
            reduction='mean'
        )


# ============================================================================
# Aliases for backward compatibility
# ============================================================================
# Keep old class names as aliases for existing code
UncertaintyPredictor = ACM  # ACM is the updated version
GeometryConstraintLayer = CAA  # CAA is the expanded version
# WARNING: this shadows the class of the SAME NAME in height_adaptive_fusion.py.
# They are different implementations. The cooperative pipeline uses the other
# one (`from .height_adaptive_fusion import HeightAdaptiveFusion`); this alias
# is kept only for backward compatibility with older configs.
HeightAdaptiveFusion = AdaptiveFusion  # Full integrated module
GRUSpatioTemporalPredictor = VehicleGRU  # Renamed in Dual GRU refactor


# ============================================================================
# Module Summary
# ============================================================================
"""
Module Summary:

┌─────────────────────────────────────────────────────────────────────┐
│                    AdaptiveFusion (Integrated)                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │     AAF      │  │     CAA      │  │     ACM      │  │   GRU   │ │
│  │  (主贡献)    │  │   (辅助)     │  │   (辅助)     │  │  (辅助) │ │
│  │              │  │              │  │              │  │          │ │
│  │ Height-aware │  │ Semantic +   │  │ Confidence  │  │ Temporal │ │
│  │ remapping    │  │ Geometry     │  │ prediction  │  │ modeling │ │
│  │              │  │ matching     │  │             │  │          │ │
│  │ Affine       │  │              │  │ Dynamic     │  │ Packet   │ │
│  │ normalization│  │ Replace      │  │ weighting   │  │ loss     │ │
│  │              │  │ 3DIoU        │  │             │  │ handling │ │
│  │ Dynamic      │  │              │  │ Low-quality │  │          │ │
│  │ fusion weight│  │              │  │ suppression │  │          │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘

Usage:
    # Create model
    model = AdaptiveFusion(embed_dims=256)

    # Or use individual modules
    aaf = AAF(embed_dims=256)
    caa = CAA(embed_dims=256)
    acm = ACM(embed_dims=256)
    gru = GRUSpatioTemporalPredictor(embed_dims=256)
"""
