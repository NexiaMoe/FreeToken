from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

from .mlp import LagunaGatedMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class LagunaSparseBlock(BaseOP):
    """Laguna sparse MoE block: top-k sigmoid-routed experts + one always-on shared expert.

    Routing matches HF ``LagunaTopKRouter``: fp32 router logits, sigmoid, add
    ``e_score_correction_bias`` for *selection only* (aux-loss-free load balancing,
    arXiv:2408.15664), gather the unbiased sigmoid scores, renormalize.

    HF scales the routed *output* by ``moe_routed_scaling_factor`` and adds the unscaled
    shared expert. The routed output is linear in the routing weights, so folding the scale
    into ``topk_weights`` here is identical arithmetic with one fewer full-width multiply
    (the same folding GLM-4 uses).
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        # Selection bias; an fp32 buffer in HF. Stored in the model dtype and upcast at use,
        # which is exact enough for a top-k comparison.
        self.e_score_correction_bias = torch.empty(config.num_experts)

        # The offload cache indexes experts by *MoE* layer (global layer minus the leading
        # dense layers), matching how the expert banks are packed.
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
        )
        self.shared_expert = LagunaGatedMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            quant=config.dense_quant,
        )

    def _route(self, hidden_states: torch.Tensor) -> TopK:
        # HF computes the router logits in fp32; the gate is tiny so we match it exactly.
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.float()
        _, topk_ids = torch.topk(scores_for_choice, self.top_k, dim=-1)
        topk_weights = scores.gather(-1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32).contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Shared expert first: the fused MoE kernels may write into ``hidden_states`` in
        # place, which would corrupt the shared expert's input (HF also evaluates it first).
        shared = self.shared_expert.forward(hidden_states)
        topk_weights, topk_ids = self._route(hidden_states)
        out = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        return (out + shared).view(num_tokens, hidden_dim)


__all__ = ["LagunaSparseBlock"]
