from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    RMSNorm,
)
from freetoken.layers.rotary import get_rope
from freetoken.models.config import FullAttentionGroupConfig, SWAAttentionGroupConfig
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class LagunaAttention(BaseOP):
    """Laguna attention for one full-context or sliding-window layer.

    Matches HF ``LagunaAttention``:
    - bias-free q/k/v/o, explicit ``head_dim``, per-head qk-norm (RMSNorm over ``head_dim``)
      applied *before* rope;
    - the layer's own rope -- YaRN over the first 64 of 128 dims on full layers, plain rope
      over all 128 on sliding layers -- taken from the layer's attention group;
    - per-head output gating: ``attn_out *= softplus(g_proj(x))`` broadcast across
      ``head_dim``, applied *before* ``o_proj``. The gate is computed in fp32 (softplus is
      not symmetric around its bf16 rounding) and cast back, as HF does.

    Query-head count is per attention group (48 full, 64 sliding); KV heads are uniform.
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        group = config.attention_group_for_layer(layer_id)
        if not isinstance(group, (FullAttentionGroupConfig, SWAAttentionGroupConfig)):
            raise ValueError(f"LagunaAttention does not support {group.kind!r} layers")
        self.layer_id = layer_id
        self.head_dim = group.head_dim
        self.num_kv_heads = group.num_kv_heads
        self.num_qo_heads = config.num_qo_heads_for_layer(layer_id)
        self.q_dim = self.num_qo_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.qkv_proj = LinearQKVMerged(
            config.hidden_size,
            self.head_dim,
            self.num_qo_heads,
            self.num_kv_heads,
            has_bias=config.has_attn_bias,
        )
        # One gate scalar per query head -> column-parallel over the same head partition
        # as q (checkpoint shape ``[num_heads, hidden]``).
        self.g_proj = LinearColParallelMerged(
            config.hidden_size, [self.num_qo_heads], has_bias=False
        )
        self.o_proj = LinearOProj(self.q_dim, config.hidden_size, has_bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_spec = AttentionSpec(
            sliding_window=(
                group.sliding_window if isinstance(group, SWAAttentionGroupConfig) else None
            ),
            sm_scale=config.attn_sm_scale,
        )
        rotary_config = group.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=(
                tuple(rotary_config.scaling.items()) if rotary_config.scaling else None
            ),
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        T = x.shape[0]

        qkv = self.qkv_proj.forward(x)
        q, k, v = qkv.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        del qkv
        # softplus in fp32 on the pre-attention hidden state, exactly as HF does.
        gate = F.softplus(self.g_proj.forward(x).float()).to(x.dtype)
        del x

        self.q_norm.forward_inplace(q.view(T, self.num_qo_heads, self.head_dim))
        self.k_norm.forward_inplace(k.view(T, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)

        o = ctx.attn_backend.forward(
            q.view(T, self.num_qo_heads, self.head_dim),
            k,
            v,
            self.layer_id,
            ctx.batch,
            attn_spec=self.attn_spec,
        )
        o = o.view(T, self.num_qo_heads, self.head_dim) * gate.unsqueeze(-1)
        return self.o_proj.forward(o.view(T, self.q_dim))


__all__ = ["LagunaAttention"]
