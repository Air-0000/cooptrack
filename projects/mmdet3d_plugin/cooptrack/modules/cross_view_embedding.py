# -------------------------------------------------------------------------
# Cross-View Embedding: 表征不确定性建模
# UACP: Uncertainty-Aware Cooperative Perception
# -------------------------------------------------------------------------
"""
Cross-View Joint Embedding for aerial-ground cooperative perception.

Reduces representation uncertainty by learning a shared embedding
space where corresponding vehicle-drone queries are close.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewEmbedding(nn.Module):
    """
    Cross-View Joint Embedding module.

    Args:
        embed_dims (int): Input feature dimension. Default: 256
        proj_dims (int): Embedding space dimension. Default: 128
        temperature (float): InfoNCE temperature. Default: 0.1
    """

    def __init__(self, embed_dims=256, proj_dims=128, temperature=0.1):
        super(CrossViewEmbedding, self).__init__()
        self.embed_dims = embed_dims
        self.proj_dims = proj_dims
        self.temperature = temperature

        # Shared projection network
        self.proj_net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, proj_dims),
        )

        # Per-instance quality predictor
        self.quality_net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, 1),
            nn.Sigmoid(),
        )

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, veh_feats, inf_feats, matched_mask=None):
        """
        Cross-view embedding forward.

        Args:
            veh_feats: [N_veh, 256] vehicle query features
            inf_feats: [N_inf, 256] drone query features
            matched_mask: [N_veh, N_inf] bool, True=matched pairs

        Returns:
            loss_contrastive: Contrastive learning loss
            loss_quality: Representation quality loss
            embedding_feats: Projected embeddings [N_veh + N_inf, proj_dims]
            quality_scores: Per-instance quality [N_veh + N_inf, 1]
        """
        device = veh_feats.device
        N_veh = veh_feats.shape[0]
        N_inf = inf_feats.shape[0]

        # Project to shared embedding space
        proj_veh = F.normalize(self.proj_net(veh_feats), dim=-1)
        proj_inf = F.normalize(self.proj_net(inf_feats), dim=-1)

        # Compute similarity matrix
        tau = torch.exp(self.log_temperature)
        sim_matrix = (proj_veh @ proj_inf.T) / tau

        # InfoNCE contrastive loss
        if matched_mask is not None:
            matched_mask_float = matched_mask.float()
            pos_sim = (sim_matrix * matched_mask_float).sum(-1) / (matched_mask_float.sum(-1) + 1e-8)
            neg_mask = 1.0 - matched_mask_float
            neg_sim = (sim_matrix * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-8)
            loss_contrastive = -pos_sim + torch.log(torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8).mean()
        else:
            pos_idx = torch.arange(min(N_veh, N_inf), device=device)
            pos_sim = sim_matrix[pos_idx, pos_idx]
            neg_sim = (sim_matrix.sum() - pos_sim.sum()) / (N_veh * N_inf - min(N_veh, N_inf))
            loss_contrastive = -pos_sim.mean() + torch.log(torch.exp(pos_sim) + torch.exp(neg_sim * N_veh) + 1e-8).mean()

        # Per-instance representation quality
        quality_veh = self.quality_net(veh_feats)
        quality_inf = self.quality_net(inf_feats)
        quality_scores = torch.cat([quality_veh, quality_inf], dim=0)

        # Quality loss
        if matched_mask is not None:
            matched_quality = (quality_veh * quality_inf * matched_mask_float).sum(-1) / (matched_mask_float.sum(-1) + 1e-8)
            loss_quality = 1.0 - matched_quality.mean()
        else:
            matched_quality = (quality_veh * quality_inf).diag().mean()
            loss_quality = 1.0 - matched_quality

        embedding_feats = torch.cat([proj_veh, proj_inf], dim=0)

        return loss_contrastive, loss_quality, embedding_feats, quality_scores