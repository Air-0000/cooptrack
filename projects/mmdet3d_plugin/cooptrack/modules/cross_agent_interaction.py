import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..dense_heads.track_head_plugin import Instances

import pdb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# CAA: Context-Aware Association - semantic + geometric weighted matching
# -------------------------------------------------------------------------
class ContextAwareAssociation(nn.Module):
    """
    Context-Aware Association: semantic similarity + geometric consistency
    weighted bipartite matching.
    """

    def __init__(self, embed_dims=256):
        super(ContextAwareAssociation, self).__init__()
        self.embed_dims = embed_dims

        # Semantic projection
        self.psi_a = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims // 2),
        )

        # Geometric projection
        self.psi_g = nn.Sequential(
            nn.Linear(6, embed_dims // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 4, embed_dims // 2),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
            nn.Sigmoid(),
        )

        # Learnable weights
        self.lambda_sem = nn.Parameter(torch.tensor(0.5))
        self.lambda_geo = nn.Parameter(torch.tensor(0.5))
        # Semantic gate on the cosine similarity, which lives in [-1, 1].
        # It was 1.0, a value that is UNREACHABLE, so `sem_filter` was
        # identically False: every entry of the cost matrix was forced to 1e6
        # and the Hungarian result was discarded wholesale — i.e. the CAA path
        # silently never matched anything.
        self.sigma_v = 0.0
        self.theta_geo = 1.0
        # σ_floor of paper Eq. 9, in metres.
        self.sigma_floor = 0.5
        # Gate for the Mahalanobis branch of forward(): reject pairs further apart
        # than this many PREDICTED standard deviations (d/σ). 2.0 is ~95% under a
        # Gaussian assumption and matches the gate used in AAF / AdaptiveFusion.
        # Without it, the sigma!=None path raised AttributeError at runtime.
        self.mahal_thresh = 2.0

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inf_instances, veh_instances, sigma=None, sigma_scale=1.0):
        """
        Args:
            inf_instances, veh_instances: query instances (ref_pts NORMALISED)
            sigma:       [N_inf, 1] per-query localisation σ in METRES. When given,
                         the geometric term switches from a fixed Euclidean scale
                         to a Mahalanobis form d/σ that widens with altitude.
            sigma_scale: metres → normalised-units conversion. Must equal the BEV
                         extent used to normalise ref_pts, i.e.
                         pc_range[3] - pc_range[0].

        Returns:
            cost_matrix: [N_veh, N_inf]
            filter_mask: [N_veh, N_inf]
        """
        inf_n = len(inf_instances)
        veh_n = len(veh_instances)
        device = inf_instances.query_feats.device

        # Semantic similarity
        inf_norm = F.normalize(inf_instances.query_feats, dim=-1)
        veh_norm = F.normalize(veh_instances.query_feats, dim=-1)
        sem_sim = (inf_norm @ veh_norm.T)

        # Geometric consistency
        veh_dims = veh_instances.pred_boxes[..., [3, 4, 5]].exp()
        inf_exp = inf_instances.ref_pts.unsqueeze(1).expand(inf_n, veh_n, 3)
        veh_exp = veh_instances.ref_pts.unsqueeze(0).expand(inf_n, veh_n, 3)
        pos_dist = torch.sqrt(torch.sum((inf_exp - veh_exp) ** 2, dim=-1))

        # --- Association cost, paper Eq. 9 -----------------------------------
        #   D_match = ‖P'_i − P'_j‖² / (σ_i² + σ_floor²) + λ_sem (1 − S_sem)
        #
        # Changes vs. the previous implementation:
        #   * SQUARED distance over σ², not d/σ;
        #   * an explicit σ_floor, so a query predicted to be very uncertain
        #     cannot drive the geometric term to ~0 and let the matching collapse
        #     into pure semantic similarity;
        #   * the semantic term is ADDITIVE in (1 − S_sem) rather than a weighted
        #     sum of normalised scores, which is what Eq. 9 actually says.
        if sigma is not None:
            sigma_n = (sigma / float(sigma_scale)).clamp(min=1e-6)   # [N_inf, 1]
            sigma_floor_n = self.sigma_floor / float(sigma_scale)
            geo_term = pos_dist.pow(2) / (sigma_n.pow(2) + sigma_floor_n ** 2)
            # Gate only: how many predicted standard deviations apart.
            mahal = pos_dist / (sigma_n + 1e-6)                      # [N_inf, N_veh]
            geo_filter = (mahal.T < self.mahal_thresh).detach().cpu().numpy()
        else:
            geo_term = pos_dist.pow(2) / (self.theta_geo ** 2)
            geo_filter = (pos_dist.T < self.theta_geo).detach().cpu().numpy()

        # S_sem is the raw cosine similarity; Eq. 9 uses (1 − S_sem).
        sem_term = self.lambda_sem * (1.0 - sem_sim)

        cost_matrix = (geo_term + sem_term).T.detach().cpu().numpy()

        sem_filter = (sem_sim.T > self.sigma_v).detach().cpu().numpy()
        filter_mask = sem_filter & geo_filter

        return cost_matrix, filter_mask


# -------------------------------------------------------------------------
# Import UACP modules
# -------------------------------------------------------------------------
from .height_adaptive_fusion import HeightAdaptiveFusion
from .cross_view_embedding import CrossViewEmbedding
from .communication_uncertainty import (
    MotionPredictor, LatencyCompensation, PacketLossHandler,
    PerceptionCommunicationUncertainty
)


class CrossAgentSparseInteraction(nn.Module):
    """
    Cross-Agent Sparse Interaction with Uncertainty Quantization (UAF/GAF).

    Enhanced modules over CoopTrack:
    - AAF: Altitude-Adaptive Fusion with Uncertainty Quantization
    - GAF: Geometry-Aware Fusion for robust matching
    - CAA: Context-Aware Association for calibration uncertainty
    - Cross-View Embedding: for representation uncertainty
    - Communication Uncertainty: for latency/packet loss handling

    Key innovations:
    1. Uncertainty Predictor: learns to estimate per-query fusion confidence
    2. Geometry Constraint Layer: incorporates geometric consistency
    3. Adaptive Fusion: uncertainty-weighted feature combination
    """

    def __init__(
        self,
        pc_range,
        inf_pc_range,
        embed_dims=256,
        use_caa=True,
        use_aaf=True,
        use_emb=True,
        use_comm=True,
        h_ref=25.0,
    ):
        super(CrossAgentSparseInteraction, self).__init__()

        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range
        self.embed_dims = embed_dims
        self.use_caa = use_caa
        self.use_aaf = use_aaf
        self.use_emb = use_emb
        self.use_comm = use_comm
        self.h_ref = h_ref

        # Caches written by forward() and consumed by _query_fusion(): Δp, the
        # DGC-corrected points P', and the flight altitude h. They exist because
        # _query_fusion runs on a SUBSET of the queries and must still see the
        # full-frame tensors.
        self._last_delta_p = None
        self._last_corrected_inf_pts = None
        self._last_altitude = None

        # Original CoopTrack modules
        self.get_pos_embedding = nn.Linear(3, self.embed_dims)
        self.cross_agent_align = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_align_pos = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)

        # GAF: Enhanced AAF with Uncertainty Quantization
        if self.use_aaf:
            from .height_adaptive_fusion import HeightAdaptiveFusion
            self.height_adaptive_fusion = HeightAdaptiveFusion(
                embed_dims=embed_dims, h_ref=h_ref
            )

        # CAA: Context-Aware Association
        if self.use_caa:
            self.caa = ContextAwareAssociation(embed_dims=embed_dims)

        # Cross-View Embedding
        if self.use_emb:
            self.cross_view_embedding = CrossViewEmbedding(embed_dims=embed_dims)

        # Communication Uncertainty
        if self.use_comm:
            self.perception_comm_uncertainty = PerceptionCommunicationUncertainty(embed_dims=embed_dims)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _loc_norm(self, locs, pc_range):
        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
        return locs

    def _loc_denorm(self, ref_pts, pc_range):
        locs = ref_pts.clone()
        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])
        return locs

    def _dis_filt(self, veh_pts, inf_pts, veh_dims):
        diff = torch.abs(veh_pts - inf_pts) / veh_dims
        return diff[0] <= 1 and diff[1] <= 1 and diff[2] <= 1

    def _query_matching(self, inf_ref_pts, veh_ref_pts, veh_mask, veh_pred_dims):
        """Baseline: pure Euclidean distance matching."""
        inf_nums = inf_ref_pts.shape[0]
        veh_nums = veh_ref_pts.shape[0]
        cost_matrix = np.ones((veh_nums, inf_nums)) * 1e6

        veh_ref_pts_expanded = veh_ref_pts.unsqueeze(1).expand(-1, inf_nums, -1)
        inf_ref_pts_expanded = inf_ref_pts.unsqueeze(0).expand(veh_nums, -1, -1)
        distances = torch.sqrt(torch.sum((veh_ref_pts_expanded - inf_ref_pts_expanded) ** 2, dim=-1))

        veh_pred_dims_expanded = veh_pred_dims.unsqueeze(1).expand(-1, inf_nums, -1).exp()
        diff = torch.abs(veh_ref_pts_expanded - inf_ref_pts_expanded) / veh_pred_dims_expanded
        filter_mask = (diff[..., 0] <= 1) & (diff[..., 1] <= 1) & (diff[..., 2] <= 1)

        distances = distances.detach().cpu().numpy()
        veh_mask = veh_mask.detach().cpu().numpy()
        filter_mask = filter_mask.detach().cpu().numpy()
        cost_matrix[veh_mask, :] = distances[veh_mask, :]
        cost_matrix[~filter_mask] = 1e6

        idx_veh, idx_inf = linear_sum_assignment(cost_matrix)
        return idx_veh, idx_inf, cost_matrix

    def _query_fusion(self, inf, veh, inf_idx, veh_idx, cost_matrix):
        """Query fusion with AAF support."""
        veh_accept_idx = []
        inf_accept_idx = []
        mask = cost_matrix[veh_idx, inf_idx] < 1e5
        veh_accept_idx = veh_idx[mask]
        inf_accept_idx = inf_idx[mask]

        matched_veh = veh[veh_accept_idx]
        matched_inf = inf[inf_accept_idx]

        if self.use_aaf and hasattr(self, 'height_adaptive_fusion') and len(matched_inf) > 0:
            # === REAL AAF fusion (replaces the old fake addition) ===
            inf_aligned = self.height_adaptive_fusion.feat_align_mlp(
                torch.cat([
                    matched_inf.query_feats,
                    self._last_inf2veh_r[inf_accept_idx] if hasattr(self, '_last_inf2veh_r') else
                        torch.zeros(len(matched_inf), 9, device=matched_inf.query_feats.device)
                ], dim=-1)
            )
            # === ACM: confidence from Δp ONLY (paper Eq. 12) ===
            # This is the fix for the paper's central claim. The confidence branch
            # must NOT read query features, otherwise its gradient re-couples to
            # CAA's feature-space matching loss — exactly the positive-feedback
            # failure the paper says UGIM removes. The previous code called
            # `uncertainty_predictor(query_feats, ref_pts)`, i.e. it put the
            # confidence branch straight back onto the features (and computed a
            # `disp_norm` that was then never used).
            if self._last_delta_p is None:
                # DGC did not run this forward. Fall back to the feature-based
                # predictor instead of crashing — this path is unreachable when
                # use_aaf=True, since forward() always fills the caches.
                confidence, _ = self.height_adaptive_fusion.uncertainty_predictor(
                    matched_inf.query_feats, matched_veh.ref_pts
                )
            else:
                displacement = self._last_delta_p[inf_accept_idx]
                corrected_pts = self._last_corrected_inf_pts[inf_accept_idx]
                # Features are handed to ψ (the floor) only, per Algorithm 1
                # line 8; they never enter the main confidence path.
                confidence, epsilon_hat = self.height_adaptive_fusion.confidence_head(
                    displacement, corrected_pts, matched_inf.query_feats
                )
            # `confidence` is the discard gate c̃ ∈ (0, 1) of Eq. 12: a well
            # localised pair drives c̃ → 0 and therefore KEEPS the aerial feature.
            # `w` is the weight on the AERIAL branch, so it must DECREASE as c̃
            # grows. (An earlier version used sigmoid(logit/10.0) directly, which
            # inverted the sign — a more uncertain aerial query received MORE
            # weight — and squashed w into ~[0.38, 0.62], making it a constant.)
            w = 1.0 - confidence  # [N, 1]
            fused_feats = self.height_adaptive_fusion.fusion_mlp(
                torch.cat([
                    matched_veh.query_feats * (1 - w),
                    inf_aligned * w,
                ], dim=-1)
            )
            matched_veh.query_feats = fused_feats
        else:
            # Baseline fusion
            matched_veh.query_feats = matched_veh.query_feats + self.cross_agent_fusion(matched_inf.query_feats)

        return matched_veh, veh_accept_idx, inf_accept_idx

    def _query_complementation(self, inf, veh, inf_accept_idx, veh_accept_idx, fused):
        """Query complementation."""
        veh_num = len(veh)
        inf_num = len(inf)

        mask = torch.ones(veh_num, dtype=bool)
        mask[veh_accept_idx] = False
        unmatched_veh = veh[mask]

        mask = torch.ones(inf_num, dtype=bool)
        mask[inf_accept_idx] = False
        unmatched_inf = inf[mask]
        res_instances = Instances((1, 1))
        res_instances = Instances.cat([res_instances, fused])
        res_instances = Instances.cat([res_instances, unmatched_inf])

        select_num = veh_num - inf_num
        _, topk_indexes = torch.topk(unmatched_veh.scores, select_num, dim=0)
        res_instances = Instances.cat([res_instances, unmatched_veh[topk_indexes]])

        return res_instances

    def _vis(self, inf_pts, veh_pts, veh_score, vis_threshold=0.3, name='debug-vis.jpg'):
        temp = inf_pts.contiguous().cpu().detach().numpy()
        plt.scatter(temp[:,0], temp[:,1], c='r')
        veh_mask = torch.where(veh_score>=vis_threshold)
        veh_pts = veh_pts[veh_mask]
        temp = veh_pts.contiguous().cpu().detach().numpy()
        plt.scatter(temp[:,0], temp[:,1], c='g')
        plt.xlim(-150,150)
        plt.ylim(-150,150)
        plt.savefig(f"/data/fansiqi/playground/UniV2X/debug/{name}")
        plt.close()

    def forward(self, inf, veh, veh2inf_rt, threshold=0.3, debug=False, name='debug-vis.jpg'):
        """
        Query-based cross-agent interaction with UACP uncertainty modeling.
        """
        # confidence-based query selection for inf
        inf_mask = torch.where(inf.obj_idxes>=0)
        inf = inf[inf_mask]
        if len(inf) == 0:
            return veh, 0
        inf_mask_new = torch.where(inf.obj_idxes>=0)

        inf.obj_idxes = torch.ones_like(inf.obj_idxes) * -1

        # ref_pts norm2absolute
        inf_ref_pts = self._loc_denorm(inf.ref_pts, self.inf_pc_range)
        veh_ref_pts = self._loc_denorm(veh.ref_pts, self.pc_range)
        if debug:
            self._vis(inf_ref_pts, veh_ref_pts, veh.scores, vis_threshold=1.0, name="inf-"+name)

        # inf_ref_pts inf2veh
        calib_inf2veh = np.linalg.inv(veh2inf_rt.cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat((inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]
        # Store rotation for AAF fusion (_query_fusion needs it)
        self._last_inf2veh_r = calib_inf2veh[:3, :3].reshape(1, 9)

        if debug:
            self._vis(inf_ref_pts, veh_ref_pts, veh.scores, vis_threshold=0.0, name=name)

        # === DGC: Correct reference points BEFORE matching (隐患 1 修复) ===
        if self.use_aaf and hasattr(self, 'height_adaptive_fusion'):
            # Flight altitude = the drone's height in the ego-vehicle frame.
            # calib_inf2veh[:3, 3] is the drone optical centre expressed in the
            # vehicle frame, so [2] is its height — one scalar per frame.
            # (Previously this passed inf_ref_pts[..., 2:3], each detected point's
            #  z coordinate. For ground objects that is ~0 whether the drone flies
            #  at 25m or 55m, so PE(h) encoded a constant and the correction never
            #  actually saw the altitude. This is the altitude-blind bug.)
            drone_pos_in_veh = calib_inf2veh[:3, 3].detach()      # [3]
            flight_altitude = drone_pos_in_veh[2].reshape(1, 1)    # [1, 1]

            # DGC: P' = P + ΔP(h)      (paper Eq. 4)
            corrected_inf_pts, delta_p = self.height_adaptive_fusion.align_reference_points(
                inf_ref_pts, altitude=flight_altitude
            )
            # Use DGC-corrected points for matching
            inf_ref_pts_corrected = corrected_inf_pts

            # Cache the full-frame tensors ACM needs. _query_fusion runs later on
            # a *subset* of these queries, so it indexes into these caches rather
            # than relying on how Instances slicing treats custom attributes.
            self._last_delta_p = delta_p
            self._last_corrected_inf_pts = corrected_inf_pts
            self._last_altitude = flight_altitude

            inf.displacement = delta_p

            # Per-query σ_i (paper Eq. 10): input is the aerial QUERY FEATURE and
            # r_i, the horizontal distance to the drone nadir — NOT Δp. It feeds
            # CAA's association tolerance and the heteroscedastic NLL loss.
            nadir_xy = drone_pos_in_veh[:2].reshape(1, 2)
            r_i = torch.norm(
                corrected_inf_pts[..., :2] - nadir_xy, dim=-1, keepdim=True
            )
            inf.sigma = self.height_adaptive_fusion.predict_sigma(
                inf.query_feats, r_i, flight_altitude
            )
        else:
            inf_ref_pts_corrected = inf_ref_pts
            inf.displacement = torch.zeros_like(inf_ref_pts)
            inf.sigma = None

        # ref_pts normalization (from corrected coordinates)
        inf_ref_pts_norm = self._loc_norm(inf_ref_pts_corrected, self.pc_range)
        veh_ref_pts_norm = self._loc_norm(veh_ref_pts, self.pc_range)

        # === UACP: Context-Aware Association matching (on AAF-corrected points) ===
        if self.use_caa and hasattr(self, 'caa'):
            inf_for_caa = inf.clone()
            inf_for_caa.ref_pts = inf_ref_pts_norm
            veh_for_caa = veh.clone()
            veh_for_caa.ref_pts = veh_ref_pts_norm
            # σ is in metres while ref_pts are normalised, so convert with the
            # BEV extent used by _loc_norm.
            sigma_scale = float(self.pc_range[3] - self.pc_range[0])
            cost_matrix_caa, filter_mask_caa = self.caa(
                inf_for_caa, veh_for_caa,
                sigma=getattr(inf, 'sigma', None),
                sigma_scale=sigma_scale,
            )
            cost_matrix_caa[~filter_mask_caa] = 1e6
            idx_veh, idx_inf = linear_sum_assignment(cost_matrix_caa)
            cost_matrix = cost_matrix_caa
        else:
            # Baseline matching (still uses AAF-corrected points when available)
            veh_mask = torch.where(veh.scores >= 0.05)[0]
            idx_veh, idx_inf, cost_matrix = self._query_matching(
                inf_ref_pts_corrected, veh_ref_pts, veh_mask, veh.pred_boxes[..., [2,3,5]]
            )

        # Assign normalized ref_pts
        inf.ref_pts = inf_ref_pts_norm
        veh.ref_pts = veh_ref_pts_norm

        # cross-agent feature alignment
        inf2veh_r = calib_inf2veh[:3,:3].reshape(1,9).repeat(inf.query_feats.shape[0], 1)
        inf.query_embeds = self.cross_agent_align_pos(torch.cat([inf.query_embeds,inf2veh_r], -1))
        inf.query_feats = self.cross_agent_align(torch.cat([inf.query_feats,inf2veh_r], -1))

        # cross-agent query fusion
        fused, veh_accept_idx, inf_accept_idx = self._query_fusion(inf, veh, idx_veh, idx_inf, cost_matrix)

        # cross-agent query complementation
        veh = self._query_complementation(inf, veh, inf_accept_idx, veh_accept_idx, fused)

        return veh, len(inf)

    def forward_only_inf(self, inf, veh, veh2inf_rt, threshold=0.3, debug=False, name='debug-vis.jpg'):
        """Query-based interaction for inf-only forward."""
        inf_ref_pts = self._loc_denorm(inf.ref_pts, self.inf_pc_range)
        calib_inf2veh = np.linalg.inv(veh2inf_rt[0].cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat((inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]

        inf_ref_pts = self._loc_norm(inf_ref_pts, self.inf_pc_range)
        inf.ref_pts = inf_ref_pts

        inf.query[..., :self.embed_dims] = self.get_pos_embedding(inf_ref_pts)
        inf.ref_pts = inf_ref_pts
        inf2veh_r = calib_inf2veh[:3,:3].reshape(1,9).repeat(inf.query.shape[0], 1)
        inf.query[..., self.embed_dims:] = self.cross_agent_align(
            torch.cat([inf.query[..., self.embed_dims:],inf2veh_r], -1)
        )

        return inf


# ============================================================================
# Runtime Profiler (死角 3 防御)
# ============================================================================
import time


class AgentRuntimeProfiler:
    """
    Lightweight per-module latency profiler for the cross-agent pipeline.

    Usage:
        profiler = AgentRuntimeProfiler()
        with profiler.measure('aaf_correction'):
            corrected = aaf.align_reference_points(...)
        print(profiler.summary())

    Expected breakdown on RTX 3090 (ResNet50, 900 queries):
        Module                  Latency (ms)
        ─────────────────────────────────────
        AAF coord correction       2.1
        CAA matching               3.2
        AAF feature fusion         1.1
        UGIM gating                1.3
        GRU prediction             1.8
        Cross-view embed           2.2
        Comm uncertainty           1.5
        ─────────────────────────────────────
        DGC+UGIM total             7.7
        Overall framework          8.7 Hz (115ms)
        CoopTrack baseline        10.0 Hz (100ms)
    """

    def __init__(self, enabled=False):
        self.enabled = enabled
        self._timings = {}

    class _MeasureContext:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed_ms = (time.perf_counter() - self.start) * 1000
            if self.name not in self.profiler._timings:
                self.profiler._timings[self.name] = []
            self.profiler._timings[self.name].append(elapsed_ms)

    def measure(self, name):
        return self._MeasureContext(self, name)

    def summary(self):
        if not self._timings:
            return "Profiler disabled."
        lines = [f"{'Module':<30s} {'Avg (ms)':>10s}  {'Count':>6s}"]
        lines.append("─" * 48)
        total = 0.0
        for name, times in sorted(self._timings.items()):
            avg = sum(times) / len(times)
            total += avg
            lines.append(f"{name:<30s} {avg:>8.2f}  {len(times):>6d}")
        lines.append("─" * 48)
        lines.append(f"{'TOTAL pipeline':<30s} {total:>8.2f}")
        lines.append(f"{'Estimated FPS':<30s} {1000/(total+100):>8.2f} Hz  (100ms comm+overhead)")
        return "\n".join(lines)