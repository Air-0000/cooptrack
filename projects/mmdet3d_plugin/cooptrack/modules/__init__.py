from .transformer import PerceptionTransformer
from .spatial_cross_attention import SpatialCrossAttention, MSDeformableAttention3D
from .temporal_self_attention import TemporalSelfAttention
from .encoder import BEVFormerEncoder, BEVFormerLayer
from .decoder import DetectionTransformerDecoder
from .cross_agent_interaction import CrossAgentSparseInteraction, ContextAwareAssociation
from .cross_lane_interaction import CrossLaneInteraction
from .spatial_temporal_reason import SpatialTemporalReasoner, pos2posemb3d
from .pf_temporal_transformer import TemporalTransformer
from .pf_petr_transformer import PETRTransformer, PETRTransformerDecoder, PETRMultiheadAttention, PETRTransformerDecoderLayer
from .motion_extractor import MotionExtractor
from .latent_transformation import LatentTransformation

# UACP modules
from .height_adaptive_fusion import HeightAdaptiveFusion
from .cross_view_embedding import CrossViewEmbedding
from .communication_uncertainty import (
    MotionPredictor,
    LatencyCompensation,
    PacketLossHandler,
    PerceptionCommunicationUncertainty,
)

# Adaptive Fusion (DGC + UGIM integrated module, optional advanced usage)
from .adaptive_fusion import (
    AdaptiveFusion,
    AAF,
    CAA as AdaptiveCAA,           # Note: cross_agent_interaction also exports a CAA
    ACM,
    VehicleGRU,
    InfrastructureGRU,
)