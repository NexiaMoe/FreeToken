"""GLM-4.7-Flash (``glm4_moe_lite``).

The decoder graph is GLM-5.2's (``glm_moe_dsa``) unchanged -- MLA attention whose DSA
indexer is config-gated off, the sigmoid/``noaux_tc`` sparse block, the dense-prefix
switch and the LM head all key off ``ModelConfig`` fields this package's parser fills in.
Only the config parsing and the checkpoint reader are specific to this family, so those
are the only modules here.
"""

from freetoken.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM as Glm4MoeLiteForCausalLM

from .config import parse_config
from .weight import (
    iter_weights,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
)

__all__ = [
    "parse_config",
    "Glm4MoeLiteForCausalLM",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
