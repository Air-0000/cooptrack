# -------------------------------------------------------------------------
# Communication Uncertainty Module
# UACP: Uncertainty-Aware Cooperative Perception
# -------------------------------------------------------------------------
"""
Perception-Communication Joint Uncertainty handling.

Addresses:
1. Latency compensation
2. Packet loss handling
3. Bidirectional FOV complementarity
"""

import torch
import torch.nn as nn
import numpy as np


class MotionPredictor(nn.Module):
    """Motion model for predicting query positions over time."""

    def __init__(self, embed_dims=256):
        super(MotionPredictor, self).__init__()

        self.motion_encoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
        )

        self.predictor = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, ref_pts, ego_motion):
        """Predict position displacement from ego-motion."""
        motion_feat = self.motion_encoder(ego_motion)
        pos_feat = self.pos_encoder(ref_pts)
        feat = torch.cat([pos_feat, motion_feat.expand(pos_feat.shape[0], -1)], dim=-1)
        displacement = self.predictor(feat)
        return displacement


class LatencyCompensation(nn.Module):
    """Latency compensation for communication delay."""

    def __init__(self, embed_dims=256):
        super(LatencyCompensation, self).__init__()
        self.motion_predictor = MotionPredictor(embed_dims)

        self.freshness_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_feats, ref_pts, ego_motion, latency_ms, is_lost=False):
        """Latency compensation forward."""
        if is_lost or latency_ms > 0:
            displacement = self.motion_predictor(ref_pts, ego_motion)
            latency_scale = latency_ms / 100.0
            compensated_pts = ref_pts + displacement * latency_scale
        else:
            compensated_pts = ref_pts
            displacement = torch.zeros_like(ref_pts)

        dx = torch.norm(displacement, dim=-1).mean().item() if latency_ms > 0 else 0.0
        freshness_input = torch.tensor(
            [latency_ms / 1000.0, dx / 100.0, dx / 100.0],
            device=query_feats.device
        ).unsqueeze(0)
        freshness_weight = self.freshness_net(freshness_input).squeeze(-1)

        if is_lost:
            confidence = 0.3 + 0.2 * freshness_weight
        else:
            confidence = 0.9 * freshness_weight + 0.1

        return query_feats, compensated_pts, confidence, freshness_weight


class PacketLossHandler(nn.Module):
    """Packet loss handling module."""

    def __init__(self, embed_dims=256, memory_len=4):
        super(PacketLossHandler, self).__init__()
        self.memory_len = memory_len
        self.motion_predictor = MotionPredictor(embed_dims)
        self.history_ref_pts = None
        self.history_feats = None

    def forward(self, query_feats, ref_pts, ego_motion, packet_received):
        """Packet loss handling forward."""
        if packet_received:
            processed_feats = query_feats
            processed_pts = ref_pts
            confidence = torch.ones(len(query_feats), device=query_feats.device)
            is_predicted = torch.zeros(len(query_feats), device=query_feats.device, dtype=torch.bool)
        else:
            if self.history_ref_pts is not None and len(self.history_ref_pts) > 0:
                displacement = self.motion_predictor(self.history_ref_pts, ego_motion)
                predicted_pts = self.history_ref_pts + displacement
                predicted_feats = self.history_feats.mean(dim=0, keepdim=True).expand(len(ref_pts), -1) \
                    if self.history_feats is not None else query_feats
                confidence = torch.full((len(ref_pts),), 0.3, device=query_feats.device)
                is_predicted = torch.ones(len(ref_pts), device=query_feats.device, dtype=torch.bool)
                processed_feats = predicted_feats
                processed_pts = predicted_pts
            else:
                processed_feats = query_feats
                processed_pts = ref_pts
                confidence = torch.full((len(ref_pts),), 0.5, device=query_feats.device)
                is_predicted = torch.zeros(len(ref_pts), device=query_feats.device, dtype=torch.bool)

        if packet_received:
            self.history_ref_pts = ref_pts.detach()
            if self.history_feats is None:
                self.history_feats = query_feats.detach()
            else:
                self.history_feats = torch.cat([self.history_feats[1:], query_feats.detach()], dim=0)

        return processed_feats, processed_pts, confidence, is_predicted

    def reset(self):
        """Reset history buffer."""
        self.history_ref_pts = None
        if hasattr(self, 'history_feats'):
            del self.history_feats


class PerceptionCommunicationUncertainty(nn.Module):
    """Unified Perception-Communication Uncertainty module."""

    def __init__(self, embed_dims=256, memory_len=4):
        super(PerceptionCommunicationUncertainty, self).__init__()
        self.embed_dims = embed_dims

        self.latency_compensation = LatencyCompensation(embed_dims)
        self.packet_loss_handler = PacketLossHandler(embed_dims, memory_len)

        self.fov_fusion = nn.Sequential(
            nn.Linear(embed_dims + 1, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward_veh_to_inf(self, veh_instances, inf_instances, inf2veh_rt):
        """Vehicle -> Drone direction (FOV complementarity)."""
        if len(veh_instances) == 0:
            return inf_instances

        device = veh_instances.query_feats.device
        veh_ref_pts = veh_instances.ref_pts.clone()
        calib = torch.from_numpy(np.linalg.inv(inf2veh_rt.cpu().numpy())).to(device)
        veh_ref_pts_h = torch.cat([veh_ref_pts, torch.ones_like(veh_ref_pts[..., :1])], dim=-1).unsqueeze(-1)
        veh_ref_pts_inf = torch.matmul(calib, veh_ref_pts_h).squeeze(-1)[..., :3]

        inf_fov_boundary = torch.tensor([50.0, 50.0, 20.0], device=device)
        dist_to_center = torch.norm(veh_ref_pts_inf / inf_fov_boundary, dim=-1)
        fov_weight = torch.sigmoid((dist_to_center - 0.5) * 2)
        confidence = fov_weight.unsqueeze(-1)

        enhanced_inf = inf_instances.clone()
        if len(inf_instances) > 0:
            veh_fused = self.fov_fusion(
                torch.cat([veh_instances.query_feats, confidence], dim=-1)
            )
            enhanced_inf.query_feats = enhanced_inf.query_feats + 0.3 * veh_fused.mean(
                dim=0, keepdim=True
            ).expand(len(enhanced_inf), -1)

        return enhanced_inf

    def forward_inf_to_veh(self, inf_instances, veh_instances, veh2inf_rt, confidence_inf):
        """Drone -> Vehicle direction with communication handling."""
        if len(inf_instances) == 0:
            return veh_instances, confidence_inf

        confidence_weight = confidence_inf.unsqueeze(-1)
        enhanced_veh = veh_instances.clone()
        if len(inf_instances) > 0:
            weighted_inf_feats = inf_instances.query_feats * confidence_weight
            avg_inf_feat = weighted_inf_feats.mean(dim=0, keepdim=True).expand(len(enhanced_veh), -1)
            enhanced_veh.query_feats = enhanced_veh.query_feats + 0.2 * avg_inf_feat

        return enhanced_veh, confidence_inf