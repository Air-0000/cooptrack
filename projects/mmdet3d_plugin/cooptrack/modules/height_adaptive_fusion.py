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
from scipy.optimize import linear_sum_assignment


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

        # Learnable temperature for calibration.
        # NOTE: temperature must start at 1.0. The previous init (log 0.1, i.e.
        # temperature=0.1) *sharpened* the sigmoid into a near step function, which
        # is the opposite of what temperature scaling is meant to do.
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

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


class LocalizationSigmaHead(nn.Module):
    """
    Per-query estimate σ_i of the BEV localisation error that SURVIVES DGC
    correction, in metres. This is the quantity that drives the association
    tolerance in CAA (paper Eq. 9) and is supervised by a heteroscedastic NLL.

    Paper Eq. 10:
        σ_i = σ_0 · (h / h_ref) · softplus(φ_θ([f_i^aer; r_i]))
        r_i = ‖P_i^xy − P_nadir^xy‖

    Two deliberate design choices, both taken from the paper:

    1. Input is (f_i^aer, r_i) — the AERIAL QUERY FEATURE and the horizontal
       distance to the drone's ground projection. They are complementary and NOT
       redundant: r_i carries the *geometric* part of the uncertainty (a target
       far from the nadir is seen at a larger slant range and with fewer pixels)
       while f_i^aer carries the *appearance* part (occlusion, motion blur,
       small-object feature degradation). Supplying r_i explicitly releases φ_θ
       from having to re-derive slant range from the feature, so it only has to
       learn a residual correction.
       NOTE: Δp is NOT an input here — Δp belongs to ACM (Eq. 12). Feeding Δp
       to σ would make the tolerance a function of the correction instead of a
       function of the target's difficulty.

    2. The first-order trend σ ∝ h is ANALYTIC and enters MULTIPLICATIVELY. It
       follows from the pinhole model (slant range grows with altitude while
       relative depth error stays roughly constant), so the network only learns
       the multiplicative deviation from it. That is why the head stays
       data-efficient and why the tolerance extrapolates to altitudes outside
       the training range.
    """

    def __init__(self, embed_dims=256, hidden_dim=64, h_ref=25.0, sigma_0=1.0):
        super(LocalizationSigmaHead, self).__init__()
        self.h_ref = h_ref

        # φ_θ: [f_i^aer (embed_dims); r_i (1)] → multiplicative deviation
        self.net = nn.Sequential(
            nn.Linear(embed_dims + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # σ_0 of Eq. 10: the tolerance scale at h = h_ref, in metres.
        self.sigma_0 = nn.Parameter(torch.tensor(float(sigma_0)))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, aerial_feats, r_i, altitude):
        """
        Args:
            aerial_feats: [N, embed_dims] aerial (drone) query features f_i^aer
            r_i:          [N, 1] horizontal distance from the target to the
                          drone's ground projection (nadir), in metres
            altitude:     [N, 1] / [1, 1] flight altitude h in metres

        Returns:
            sigma: [N, 1] localisation error estimate in metres (> 0)
        """
        if altitude is None:
            raise ValueError("LocalizationSigmaHead requires the flight altitude h.")
        if altitude.shape[0] != aerial_feats.shape[0]:
            altitude = altitude.expand(aerial_feats.shape[0], 1)
        if r_i.shape[0] != aerial_feats.shape[0]:
            r_i = r_i.expand(aerial_feats.shape[0], 1)

        x = torch.cat([aerial_feats, r_i], dim=-1)           # [N, embed_dims + 1]
        deviation = F.softplus(self.net(x))                   # > 0
        h_ratio = (altitude / self.h_ref).clamp(min=1e-3)     # h / h_ref
        return self.sigma_0 * h_ratio * deviation             # metres


class DisplacementConfidenceHead(nn.Module):
    """
    ACM of the paper (Eq. 12–13): the confidence branch of UGIM.

        c = σ(φ_conf([Δp; P']))          (Eq. 12)
        c_calibrated = σ(log c / τ)      (Eq. 13, τ learnable, init 1.0)
        ε̂ = σ(ψ([Δp; F_d]))              (learnable floor)
        c̃ = (1 − ε̂) · c

    The single most important property: input is the GEOMETRIC DISPLACEMENT Δp
    produced by DGC, never the query features. This is what keeps the confidence
    branch gradient-orthogonal to CAA's feature-space matching loss. The only
    place features enter is the floor ψ, which bounds the retained fraction from
    below and therefore cannot re-couple the two branches' gradients.
    """

    def __init__(self, embed_dims=256, hidden_dim=128):
        super(DisplacementConfidenceHead, self).__init__()

        self.disp_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # φ_conf → c. Initialised so that c ≈ 0.5 at step 0 (bias 0 + sigmoid),
        # which prevents early-training saturation where (1−c) permanently
        # zeroes the drone branch.
        self.confidence_net = nn.Linear(hidden_dim, 1)

        # ψ → ε̂, the per-query learnable floor. Input is [Δp; F_d] per Eq. 12
        # and Algorithm 1 line 8.
        self.floor_net = nn.Sequential(
            nn.Linear(3 + embed_dims, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Temperature τ of Eq. 13. MUST start at 1.0: an init of 0.1 *sharpens*
        # the sigmoid into a near step function, which is the opposite of what
        # temperature scaling is for. The paper reports τ converging to 2–4,
        # i.e. flattening c.
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, displacement, ref_pts, aerial_feats=None):
        """
        Args:
            displacement: [N, 3] Δp applied by DGC (corrected − raw)
            ref_pts:      [N, 3] corrected reference points P'
            aerial_feats: [N, embed_dims] optional drone features F_d, used only
                          by the floor ψ. When None, ε̂ falls back to 0 and the
                          branch stays strictly feature-free.

        Returns:
            confidence:  [N, 1] c̃ = (1 − ε̂)·c, the discard gate in (0, 1)
            epsilon_hat: [N, 1] the learned per-query floor
        """
        disp_encoded = self.disp_encoder(displacement)
        pos_encoded = self.pos_encoder(ref_pts)
        feat = self.trunk(self.input_proj(torch.cat([disp_encoded, pos_encoded], dim=-1)))

        # Eq. 13: temperature scaling. Applied to the LOGIT, not to the
        # probability — σ(logit / τ) with τ > 1 flattens, τ < 1 sharpens.
        temperature = torch.exp(self.log_temperature)
        c = torch.sigmoid(self.confidence_net(feat) / temperature)

        # Learnable floor (replaces the hard ε = 0.2).
        if aerial_feats is not None:
            epsilon_hat = torch.sigmoid(
                self.floor_net(torch.cat([displacement, aerial_feats], dim=-1))
            )
        else:
            epsilon_hat = torch.zeros_like(c)

        # c̃ = (1 − ε̂)·c  guarantees 1 − c̃ ≥ ε̂, i.e. the retained fraction of
        # the aerial feature can never fall below the learned floor.
        confidence = (1.0 - epsilon_hat) * c
        return confidence, epsilon_hat


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

        # Direction encoder (from bbox velocity).
        # pred_boxes follows CoopTrack's 10-D code (see detectors/cooptrack.py
        # `pred_boxes = torch.zeros((len, 10))`):
        #   [0:3] centre xyz | [3:6] dims wlh | [6:8] sin/cos yaw | [8:10] vel vx,vy
        # so velocity is 2-D, exactly as spatial_temporal_reason.py slices it.
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
        velocity = pred_boxes[..., 8:10]  # [N, 2] vx, vy — see layout note above
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
        # Retained ONLY for the ablation that deliberately re-couples the
        # confidence branch to the feature space. The default pipeline uses
        # `confidence_head` (ACM, Δp-only) — see paper Eq. 12.
        self.uncertainty_predictor = UncertaintyPredictor(embed_dims, hidden_dim // 2)

        # --- 1b. Localisation σ head (paper Eq. 10) ---
        # Produces the per-query σ_i that scales CAA's association tolerance.
        # Input is (f_i^aer, r_i) — the aerial query FEATURE plus the horizontal
        # distance to the drone nadir. NOT Δp: Δp belongs to ACM.
        self.sigma_head = LocalizationSigmaHead(
            embed_dims=embed_dims, hidden_dim=hidden_dim // 4, h_ref=h_ref
        )

        # --- 1c. ACM: Adaptive Confidence Modulation (paper Eq. 12) ---
        # Reads the DGC displacement Δp and never the query features, which is
        # what keeps this branch gradient-orthogonal to CAA's feature-space loss.
        self.confidence_head = DisplacementConfidenceHead(
            embed_dims=embed_dims, hidden_dim=hidden_dim // 2
        )

        # --- 1d. Association hyper-parameters (paper Eq. 9) ---
        # σ_floor keeps the geometric term discriminative even for a query
        # predicted to be highly uncertain, so matching can never collapse into
        # pure semantic similarity. λ_sem weights the semantic term.
        self.sigma_floor = 0.5                                   # metres
        self.lambda_sem = nn.Parameter(torch.tensor(1.0))
        # Pre-solve gate: reject pairs further apart than this many predicted
        # standard deviations. A Hungarian solver always returns min(N, M) pairs,
        # so without a gate it would force matches on genuinely unmatched queries.
        self.mahal_thresh = 2.0

        # --- 2. Sinusoidal Positional Encoding for altitude ---
        # Encoding altitude h and deviation Δh into a smooth high-d feature,
        # enabling extrapolation beyond training range (25-55m).
        # Instead of learnable affine W,b, we predict a residual ΔP(h):
        #     P' = P + ΔP(h),   ΔP(h) = MLP([PE(h); PE(Δh)])
        # PE(h)_{2i} = sin(h / 10000^{2i/d}),  i = 0 .. d/2-1   (paper Eq. 5)
        # => frequency_i = 10000^{-2i/d}, DECREASING with i. The old buffer used
        # 10000^{+2i/(d/4)} — wrong sign *and* wrong denominator — and carried
        # only d/4 frequencies, which is why _alt_pe had to duplicate the
        # sin/cos block to reach d.
        self.register_buffer(
            'pe_freq',
            10000 ** (-2.0 * torch.arange(self.pe_dim // 2) / self.pe_dim),
        )

        # --- 3. Residual correction MLP (extrapolation-robust) ---
        # Input:  PE(h) + PE(Δh), dim pe_dim
        # Output: ΔP ∈ R^3
        #
        #     P' = P + ΔP(h),   ΔP(h) = MLP([PE(h); PE(Δh)])      (paper Eq. 4)
        #
        # Deliberately R^3, not R^4. An earlier revision emitted a 4th DOF `s`
        # and applied a line-of-sight scale
        #     P' = drone_pos + (P − drone_pos)·exp(s) + ΔP,
        # but `s` appears nowhere in the paper and it invalidates Theorem 1: the
        # extrapolation bound is proved for ΔP(h) alone, while the ray term
        # depends on P itself and grows without bound with distance. Removing it
        # restores exact correspondence with Eq. 4 and with the theorem.
        # This also supersedes the old learnable affine_W/affine_b, which
        # collapsed outside 25-55m.
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.pe_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # ΔP ∈ R^3
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
        Sinusoidal positional encoding for altitude (paper Eq. 5):

            PE(h)_{2i}   = sin(h / 10000^{2i/d})
            PE(h)_{2i+1} = cos(h / 10000^{2i/d}),   i = 0 .. d/2-1

        Args:
            h: [N, 1] altitude values

        Returns:
            pe: [N, pe_dim] positional encoding, d INDEPENDENT components
        """
        pe = h * self.pe_freq.unsqueeze(0)                        # [N, d/2]
        return torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)  # [N, d]

    def _residual_correction(self, ref_pts_veh, altitude):
        """
        Apply the extrapolation-robust residual correction of paper Eq. 4:

            P' = P + ΔP(h),      ΔP(h) = MLP([PE(h); PE(Δh)])

        Args:
            ref_pts_veh: [N, 3] points in the ego-vehicle coordinate frame
                         (absolute, not normalised)
            altitude:    [N, 1] (or [1, 1]) FLIGHT altitude h — the drone's height,
                         one scalar per frame broadcast to N queries.
                         NOTE: this must NOT be ref_pts_veh[..., 2]. That is each
                         detected point's height above the ground plane (~0 for a
                         car), which carries no altitude information whatsoever.

        Returns:
            corrected_pts: [N, 3] residual-corrected points
            delta_p:       [N, 3] ΔP = corrected − raw (the ACM input)
        """
        if altitude.ndim == 2 and altitude.shape[0] == 1 and ref_pts_veh.shape[0] != 1:
            altitude = altitude.expand(ref_pts_veh.shape[0], 1)

        height_diff = altitude - self.h_ref      # [N, 1]

        pe_h = self._alt_pe(altitude)            # [N, pe_dim]
        pe_dh = self._alt_pe(height_diff)        # [N, pe_dim]
        pe_input = pe_h + pe_dh                  # [N, pe_dim]

        delta_p = self.residual_mlp(pe_input)    # [N, 3]
        corrected_pts = ref_pts_veh + delta_p

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
            altitude: [N, 1] / [1, 1] / scalar — the DRONE FLIGHT ALTITUDE h.
                      This is required. Inferring it from ref_pts_veh[..., 2]
                      would be wrong: that is the object's height above the
                      ground plane, not the drone's height.
        Returns:
            corrected_pts: [N, 3] residual-corrected points (veh frame)
            delta_p:       [N, 3] ΔP = corrected − raw (for ACM input)
        """
        if altitude is None:
            # Fail loudly. Silently substituting ref_pts_veh[..., 2] here is what
            # made the module altitude-blind in the first place: for ground objects
            # that value is ~0 regardless of whether the drone flies at 25m or 55m,
            # so PE(h) would encode a constant and ΔP(h) would collapse to a bias.
            raise ValueError(
                "align_reference_points requires the drone flight altitude `h`. "
                "Pass altitude=calib_inf2veh[2, 3] (the drone's height in the "
                "ego-vehicle frame); do not derive it from ref_pts_veh[..., 2]."
            )
        if isinstance(altitude, (int, float)):
            altitude = torch.full_like(ref_pts_veh[..., :1], float(altitude))
        elif isinstance(altitude, torch.Tensor) and altitude.ndim == 0:
            altitude = torch.full_like(ref_pts_veh[..., :1], float(altitude.item()))

        # Residual correction: ΔP = MLP(PE(h) + PE(Δh))   (paper Eq. 4)
        corrected_pts, delta_p = self._residual_correction(ref_pts_veh, altitude)

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
        calib = torch.from_numpy(calib).to(device=device, dtype=inf_ref_pts.dtype)
        inf_ref_pts_h = torch.cat([inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])], dim=-1).unsqueeze(-1)
        inf_ref_pts_veh = torch.matmul(calib, inf_ref_pts_h).squeeze(-1)[..., :3]

        # === Step 1b: flight altitude ===
        # h is the DRONE's height in the ego-vehicle frame — one scalar per frame.
        # (Previously this module used inf_ref_pts_veh[..., 2:3], the z coordinate
        #  of each detected point. For ground objects that is ~0 whether the drone
        #  flies at 25m or 55m, so PE(h) encoded a constant and ΔP(h) collapsed to
        #  a per-query bias — the module was altitude-blind by construction.)
        drone_pos = calib[:3, 3].detach()                        # [3]
        altitude = drone_pos[2].reshape(1, 1).expand(inf_n, 1)   # [N, 1]

        # === Step 2: Extrapolation-robust residual correction (paper Eq. 4) ===
        # P' = P + ΔP(h),   ΔP(h) = MLP(PE(h) + PE(Δh))
        # This replaces the old affine normalization (P' = P*W + b) which
        # caused extrapolation collapse beyond the training altitude range.
        inf_ref_pts_corrected, delta_p = self._residual_correction(
            inf_ref_pts_veh, altitude
        )
        inf_ref_pts_norm = self._loc_norm(inf_ref_pts_corrected, pc_range)

        # === Step 2b: per-query σ (paper Eq. 10) ===
        # σ_i = σ_0 · (h/h_ref) · softplus(φ_θ([f_i^aer; r_i])),
        # r_i = ‖P_i^xy − P_nadir^xy‖ — horizontal distance from the corrected
        # target to the drone's ground projection (optical centre, z dropped).
        nadir_xy = drone_pos[:2].reshape(1, 2)
        r_i = torch.norm(
            inf_ref_pts_corrected[..., :2] - nadir_xy, dim=-1, keepdim=True
        )                                                             # [N, 1]
        inf_sigma = self.sigma_head(inf_instances.query_feats, r_i, altitude)

        # === Step 3: ACM — confidence from Δp ONLY (paper Eq. 12) ===
        # The main branch never sees query features; that is what keeps it
        # gradient-orthogonal to CAA. Features reach the floor ψ only
        # (Algorithm 1, line 8).
        inf_confidence, inf_epsilon = self.confidence_head(
            delta_p, inf_ref_pts_corrected, inf_instances.query_feats
        )

        # === Step 4: Association on corrected coordinates (paper Eq. 9) ===
        # D_match = ‖P'_i − P'_j‖² / (σ_i² + σ_floor²) + λ_sem (1 − S_sem),
        # solved by HUNGARIAN — not greedy nearest-neighbour, because only a
        # globally optimal assignment lets the per-query tolerance matter.
        veh_ref_pts = self._loc_denorm(veh_instances.ref_pts, pc_range)
        veh_ref_pts_norm = self._loc_norm(veh_ref_pts, pc_range)

        inf_pts_exp = inf_ref_pts_norm.unsqueeze(1).expand(inf_n, veh_n, 3)
        veh_pts_exp = veh_ref_pts_norm.unsqueeze(0).expand(inf_n, veh_n, 3)
        sq_dist = torch.sum((inf_pts_exp - veh_pts_exp) ** 2, dim=-1)    # [N, M]

        # Unit note: sq_dist lives on NORMALISED reference points, so σ (metres)
        # must be rescaled by the same BEV extent to stay commensurate.
        sigma_scale = float(pc_range[3] - pc_range[0])
        sigma_n = (inf_sigma / sigma_scale).clamp(min=1e-6)               # [N, 1]
        sigma_floor_n = self.sigma_floor / sigma_scale

        geo_term = sq_dist / (sigma_n.pow(2) + sigma_floor_n ** 2)        # [N, M]

        inf_norm = F.normalize(inf_instances.query_feats, dim=-1)
        veh_norm = F.normalize(veh_instances.query_feats, dim=-1)
        sem_sim = inf_norm @ veh_norm.T                                   # [N, M]
        sem_term = self.lambda_sem * (1.0 - sem_sim)

        cost = geo_term + sem_term                                        # [N, M]

        # Pre-solve gate: a Hungarian solver always returns min(N, M) pairs, so
        # implausible candidates must be made prohibitively expensive first.
        mahal = torch.sqrt(sq_dist) / (sigma_n + 1e-6)                    # [N, M]
        gate = ((mahal < self.mahal_thresh) & (sem_sim > 0.0)).cpu().numpy()
        cost_np = cost.detach().cpu().numpy().copy()
        cost_np[~gate] = 1e6

        row_ind, col_ind = linear_sum_assignment(cost_np)
        keep = cost_np[row_ind, col_ind] < 1e5
        matched_inf_idx = torch.tensor(row_ind[keep], device=device)
        matched_veh_idx = torch.tensor(col_ind[keep], device=device)

        if len(matched_inf_idx) == 0:
            return veh_instances, matched_veh_idx, matched_inf_idx, None

        # === Step 5: Cross-agent feature alignment ===
        inf2veh_r = calib[:3, :3].reshape(1, 9).repeat(inf_n, 1)
        inf_query_aligned = self.feat_align_mlp(
            torch.cat([inf_instances.query_feats, inf2veh_r], dim=-1)
        )

        # === Step 6: Dynamic height-weighted fusion (paper Eq. 6) ===
        # F_fused = α(1 − c̃)·F'_d + (1 − α)·F_v
        # (altitude already defined in Step 1b as the per-frame flight height)
        h_diff = altitude - self.h_ref
        # fusion_weight_net's first layer is Linear(embed_dims + 3, ...), so the
        # altitude deviation must occupy 3 slots. Expanding it to embed_dims
        # produced a 512-wide concat against a 259-wide layer.
        h_diff_expanded = h_diff.expand(-1, 3)
        feat_concat = torch.cat([inf_instances.query_feats, h_diff_expanded], dim=-1)
        height_weight = self.fusion_weight_net(feat_concat).squeeze(-1)    # α

        confidence_matched = inf_confidence[matched_inf_idx].squeeze(-1)   # c̃
        adaptive_weight = height_weight[matched_inf_idx] * (1 - confidence_matched)
        adaptive_weight = torch.clamp(adaptive_weight, 0, 1)

        # === Step 7: Confidence-weighted feature fusion ===
        w = adaptive_weight.unsqueeze(-1)
        fused_feats = self.fusion_mlp(
            torch.cat([
                veh_instances.query_feats[matched_veh_idx] * (1 - w),
                inf_query_aligned[matched_inf_idx] * w,
            ], dim=-1)
        )

        # === Step 8: Residual connection with confidence gating ===
        residual_input = torch.cat([
            veh_instances.query_feats[matched_veh_idx],
            inf_query_aligned[matched_inf_idx]
        ], dim=-1)

        # Only apply the residual when the aerial query is trustworthy (c̃ low).
        residual_gate = (1 - confidence_matched).unsqueeze(-1)
        residual = self.residual_net(residual_input) * residual_gate
        fused_feats = fused_feats + residual

        # === Step 9: Clone and update instances ===
        fused_instances = veh_instances[matched_veh_idx].clone()
        fused_instances.query_feats = fused_feats

        # Stored for loss computation: c̃ is what the IoU-aware BCE supervises.
        fused_instances.uncertainty = confidence_matched
        fused_instances.epsilon = inf_epsilon[matched_inf_idx].squeeze(-1)

        return fused_instances, matched_veh_idx, matched_inf_idx, confidence_matched

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

    # ------------------------------------------------------------------
    # σ: prediction and heteroscedastic NLL supervision
    # ------------------------------------------------------------------
    def predict_sigma(self, aerial_feats, r_i, altitude):
        """
        Per-query σ_i of the BEV localisation error surviving DGC correction
        (paper Eq. 10).

        Args:
            aerial_feats: [N, embed_dims] aerial query features f_i^aer
            r_i:          [N, 1] horizontal distance to the drone nadir, metres
            altitude:     [N, 1] or [1, 1] flight altitude h in metres

        Returns:
            sigma: [N, 1] in metres
        """
        return self.sigma_head(aerial_feats, r_i, altitude)

    def compute_sigma_nll_loss(self, sigma, corrected_pts, gt_pts, mask=None):
        """
        Heteroscedastic negative log-likelihood for σ.

        Supervising σ against the residual localisation error is what makes the
        association tolerance *calibrated* rather than hand-set: σ_i is trained to
        be the error it actually has, in metres, so `d_pos / σ_i` really is a
        "how many standard deviations apart" test.

            L = ½ (‖P'_d − P_gt‖ / σ)² + log σ

        The log σ term is what stops the trivial solution σ → ∞.

        Args:
            sigma:         [N, 1] predicted σ in metres
            corrected_pts: [N, 3] DGC-corrected drone points (veh frame, absolute)
            gt_pts:        [N, 3] matched ground-truth centres (same frame)
            mask:          optional [N] 0/1 weights

        Returns:
            loss: scalar
        """
        err = torch.norm(corrected_pts - gt_pts, dim=-1)      # [N]
        s = sigma.squeeze(-1).clamp(min=1e-6)                 # [N]

        loss = 0.5 * (err / s).pow(2) + torch.log(s)          # [N]

        if mask is not None:
            mask = mask.to(loss.dtype)
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss.mean()