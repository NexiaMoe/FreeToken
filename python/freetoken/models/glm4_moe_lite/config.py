"""Engine-facing config for GLM-4.7-Flash (``glm4_moe_lite``).

Same decoder as GLM-5.2 (``glm_moe_dsa``): DeepSeek-class Multi-head Latent Attention
plus a sigmoid/``noaux_tc`` sparse MoE with a shared expert, after a leading dense layer.
The difference is what is *absent* and how it is quantized:

- **No DSA indexer.** The checkpoint carries no ``index_head_dim``/``index_topk``/
  ``indexer_types``, so ``GlmMoeDsaArgs`` reads them as 0 and the attention group is a
  plain latent-KV MLA group (``AttnType.MLA`` -> ``MLAKVCache``, served by the all-Triton
  ``dsa`` backend's identity-selection path). ``GlmMoeDsaAttention`` builds no indexer.
- **Quantization is per-component, not per-layer.** The routed experts, the shared
  experts and the leading dense MLP are compressed-tensors NVFP4; MLA, the router,
  ``lm_head``, norms and embeddings stay BF16 (the checkpoint's ``ignore`` list). Unlike
  GLM-5.2 nothing is requantized at load, so there are no FREETOKEN_GLM_*_FP8 switches
  here -- the modes are read off the checkpoint.
- **The trailing MTP layer is dropped.** ``num_hidden_layers`` counts only the real
  layers, so ``range(num_layers)`` already excludes it; the loader skips its tensors.
"""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_compressed_tensors_nvfp4,
)
from freetoken.models.glm_moe_dsa.args import load_args


def _weight_schemes(quant: Any) -> set[tuple]:
    """The distinct weight-quantization schemes an export declares, as comparable tuples."""
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    groups = get("config_groups") or {}
    schemes = set()
    for group in groups.values() if isinstance(groups, dict) else []:
        w = (group or {}).get("weights") or {}
        schemes.add((
            int(w.get("num_bits", 0) or 0),
            str(w.get("type", "")).lower(),
            int(w.get("group_size", 0) or 0),
            str(w.get("strategy", "")).lower(),
        ))
    return schemes


def _expert_quant(hf_config: Any) -> str:
    """``nvfp4`` for a uniformly-NVFP4 compressed-tensors export, else a hard failure.

    A ``mixed-precision`` export (unsloth's, which keeps the experts of layers 1/39/46 at
    FP8 while the rest are NVFP4) cannot be represented: the offload cache is ONE
    fixed-geometry slot pool shared by every layer -- a slot holds an arbitrary layer's
    expert, so every layer's bank row must have the same shape and dtype.

    Checked BEFORE ``detect_compressed_tensors_nvfp4``, which only asks whether *some*
    group is NVFP4 and so answers True for a mixed export -- exactly the silent
    acceptance that would surface much later as an unreadable bank-shape assert.
    """
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        raise NotImplementedError(
            "glm4_moe_lite: only the NVFP4 checkpoints are served; the BF16 checkpoint's "
            "routed experts have no expert-bank loader"
        )
    schemes = _weight_schemes(quant)
    if len(schemes) > 1:
        raise NotImplementedError(
            f"glm4_moe_lite: mixed-precision export declares {len(schemes)} weight schemes "
            f"{sorted(schemes)}. FreeToken serves the uniformly-NVFP4 exports: one "
            "offload-cache slot pool is shared by every layer, so expert layers quantized "
            "differently from each other cannot be represented"
        )
    if detect_compressed_tensors_nvfp4(hf_config):
        return "nvfp4"
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    raise NotImplementedError(
        f"glm4_moe_lite: unsupported quantization format {str(get('format') or '')!r}; "
        "FreeToken serves the compressed-tensors NVFP4 exports"
    )


def parse_config(hf_config: Any) -> ModelConfig:
    args = load_args(hf_config)
    # Latent-KV MLA: the pool stores one ckv (kv_lora_rank) | kpe (qk_rope_head_dim) row
    # per token; the model absorbs kv_b into Q/O.
    latent_dim = args.kv_lora_rank + args.qk_rope_head_dim  # 576
    num_layers = int(hf_config.num_hidden_layers)  # excludes the trailing MTP layer

    if args.index_head_dim or args.index_topk or args.indexer_types:
        raise NotImplementedError(
            "glm4_moe_lite: this checkpoint carries a DSA indexer; serve it as "
            "GlmMoeDsaForCausalLM (glm_moe_dsa) instead"
        )

    rotary_config = RotaryConfig(
        head_dim=args.qk_head_dim,
        rotary_dim=args.qk_rope_head_dim,
        max_position=args.max_position,
        base=args.rope_theta,
        scaling=None,  # rope_type "default"; interleaved rope applied inside the model
    )
    expert_quant = _expert_quant(hf_config)

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,  # single shared MLA latent
        head_dim=latent_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=tuple(range(num_layers)),
                num_kv_heads=1,
                head_dim=latent_dim,
                rotary_config=rotary_config,
                mla=True,
                # No indexer -> AttnType.MLA (plain latent pool), not AttnType.DSA.
                index_head_dim=0,
                num_index_layers=0,
            ),
        ),
        num_experts=(
            getattr(hf_config, "n_routed_experts", None)
            or getattr(hf_config, "num_local_experts", None)
            or getattr(hf_config, "num_experts", 0)
        ),
        num_experts_per_tok=hf_config.num_experts_per_tok,
        moe_intermediate_size=(
            getattr(hf_config, "moe_intermediate_size", 0) or hf_config.intermediate_size
        ),
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "glm4_moe_lite"),
        architectures=getattr(hf_config, "architectures", ["Glm4MoeLiteForCausalLM"]),
        moe_enabled=True,
        expert_quant=expert_quant,
        # The shared experts and the leading dense MLP ship packed FP4 like the routed
        # experts; MLA and lm_head are in the checkpoint's ignore list and stay BF16.
        dense_quant=expert_quant,
        attn_quant="none",
        lm_head_quant="none",
        first_k_dense_replace=int(getattr(hf_config, "first_k_dense_replace", 0)),
        n_shared_experts=int(getattr(hf_config, "n_shared_experts", 0)),
        routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
        n_group=int(getattr(hf_config, "n_group", 1)),
        topk_group=int(getattr(hf_config, "topk_group", 1)),
        attn_sm_scale=args.qk_head_dim**-0.5,
        has_attn_bias=bool(getattr(hf_config, "attention_bias", False)),
        glm_dsa_args=args,
    )


__all__ = ["parse_config"]
