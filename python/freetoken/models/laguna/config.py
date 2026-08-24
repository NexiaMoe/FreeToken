from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
    detect_compressed_tensors_nvfp4,
)

_FULL = "full_attention"
_SWA = "sliding_attention"

# YaRN scalars copied verbatim from ``rope_parameters`` into ``RotaryConfig.scaling``;
# ``layers/rotary.py`` reads exactly these names. ``partial_rotary_factor`` is consumed
# here (it becomes ``rotary_dim``) and deliberately not forwarded.
_ROPE_SCALAR_KEYS = (
    "factor",
    "beta_fast",
    "beta_slow",
    "original_max_position_embeddings",
    "attention_factor",
    "mscale",
    "mscale_all_dim",
    "truncate",
)


def _rotary_config(rope: dict, head_dim: int, max_position: int) -> RotaryConfig:
    """One attention type's rope. Laguna's full layers run YaRN over a half-width rotary
    (``partial_rotary_factor`` 0.5 -> 64 of 128 dims); its sliding layers run plain rope at
    a different theta over the full head."""
    partial = float(rope.get("partial_rotary_factor", 1.0))
    rope_type = rope.get("rope_type", "default")
    scaling = None
    if rope_type not in (None, "default"):
        scaling = {"rope_type": rope_type}
        scaling.update({k: rope[k] for k in _ROPE_SCALAR_KEYS if k in rope})
    return RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(head_dim * partial),
        max_position=max_position,
        base=float(rope["rope_theta"]),
        scaling=scaling,
    )


def _uniform(values: list[int], layer_ids: tuple[int, ...], what: str) -> int:
    """The single value shared by ``layer_ids``; Laguna's geometry is per attention *type*,
    so a group whose layers disagree cannot be expressed as one attention group."""
    distinct = {values[i] for i in layer_ids}
    if len(distinct) != 1:
        raise ValueError(
            f"Laguna: {what} must be uniform within an attention group, got {distinct}"
        )
    return distinct.pop()


def _first_k_dense_replace(mlp_layer_types: list[str], num_layers: int) -> int:
    """Length of the leading dense-MLP run. The expert banks (and the offload cache) are
    indexed by MoE layer, which only works when the dense layers are a contiguous prefix."""
    if mlp_layer_types is None:
        return 0
    dense = [i for i, kind in enumerate(mlp_layer_types) if kind == "dense"]
    if dense != list(range(len(dense))):
        raise ValueError(
            f"Laguna: dense MLP layers must be a contiguous prefix, got layers {dense}"
        )
    if len(dense) >= num_layers:
        raise ValueError("Laguna: checkpoint has no sparse MoE layers")
    return len(dense)


def _expert_quant(hf_config: Any) -> str:
    """``nvfp4`` for the NVFP4 checkpoint, ``none`` for BF16.

    The INT4 checkpoint is compressed-tensors too, but ``pack-quantized`` group-128
    *integer* experts -- a format FreeToken has no kernels for. Reject it here instead of
    letting it fall through as an unquantized checkpoint and die on a missing ``.weight``.
    """
    if detect_compressed_tensors_nvfp4(hf_config):
        return "nvfp4"
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return "none"
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    groups = get("config_groups") or {}
    for group in groups.values() if isinstance(groups, dict) else []:
        weights = (group or {}).get("weights") or {}
        if str(weights.get("type", "")).lower() == "int":
            raise ValueError(
                "unsupported compressed-tensors integer scheme "
                f"(num_bits={weights.get('num_bits')}, group_size={weights.get('group_size')}, "
                f"strategy={weights.get('strategy')!r}); FreeToken serves the BF16 and NVFP4 "
                "Laguna checkpoints"
            )
    return "none"


def parse_config(hf_config: Any) -> ModelConfig:
    """Parse a ``LagunaConfig`` (poolside Laguna XS 2.x) into FreeToken's :class:`ModelConfig`.

    Laguna specifics handled here:
    - 3:1 sliding/full attention layout from ``layer_types``, each type with its own rope
      (YaRN + half-width rotary on full layers, plain rope on sliding layers) and its own
      *query*-head count (48 full / 64 sliding, 8 KV heads throughout).
    - one leading dense MLP layer, then sigmoid-routed MoE layers with an
      ``e_score_correction_bias`` selection bias, ``routed_scaling_factor`` and one shared
      expert.
    - NVFP4 (compressed-tensors) routed experts and shared experts; attention, the dense
      layer-0 MLP, the router and lm_head stay BF16 (the checkpoint's ``ignore`` list).
    """
    num_layers = int(hf_config.num_hidden_layers)
    head_dim = int(
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    max_position = int(hf_config.max_position_embeddings)

    layer_types = list(getattr(hf_config, "layer_types", None) or [_FULL] * num_layers)
    if len(layer_types) != num_layers:
        raise ValueError(
            f"Laguna: layer_types has {len(layer_types)} entries, expected {num_layers}"
        )
    full_ids = tuple(i for i, kind in enumerate(layer_types) if kind == _FULL)
    swa_ids = tuple(i for i, kind in enumerate(layer_types) if kind == _SWA)
    if len(full_ids) + len(swa_ids) != num_layers:
        raise ValueError(f"Laguna: unsupported layer_types {sorted(set(layer_types))}")

    heads_per_layer = list(
        getattr(hf_config, "num_attention_heads_per_layer", None)
        or [hf_config.num_attention_heads] * num_layers
    )
    _WHAT = "num_attention_heads_per_layer"
    full_qo_heads = _uniform(heads_per_layer, full_ids, _WHAT) if full_ids else 0
    swa_qo_heads = _uniform(heads_per_layer, swa_ids, _WHAT) if swa_ids else 0

    rope_params = dict(getattr(hf_config, "rope_parameters", None) or {})
    full_rope = rope_params.get(_FULL, rope_params)
    swa_rope = rope_params.get(_SWA, rope_params)
    full_rotary = _rotary_config(full_rope, head_dim, max_position)
    swa_rotary = _rotary_config(swa_rope, head_dim, max_position)

    num_kv_heads = int(getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads))
    sliding_window = int(getattr(hf_config, "sliding_window", 0) or 0)
    if swa_ids and sliding_window <= 0:
        raise ValueError("Laguna: sliding layers need a positive sliding_window")

    groups: list[FullAttentionGroupConfig | SWAAttentionGroupConfig] = []
    if full_ids:
        groups.append(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=full_rotary,
                num_qo_heads=full_qo_heads,
            )
        )
    if swa_ids:
        groups.append(
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=swa_rotary,
                sliding_window=sliding_window,
                num_qo_heads=swa_qo_heads,
            )
        )

    if getattr(hf_config, "moe_apply_router_weight_on_input", False):
        raise ValueError("Laguna: moe_apply_router_weight_on_input=True is not supported")
    if float(getattr(hf_config, "moe_router_logit_softcapping", 0.0) or 0.0) != 0.0:
        raise ValueError("Laguna: moe_router_logit_softcapping is not supported")
    if getattr(hf_config, "swa_attention_sink_enabled", False):
        raise ValueError("Laguna: swa_attention_sink_enabled is not supported")

    expert_quant = _expert_quant(hf_config)
    num_experts = int(getattr(hf_config, "num_experts", 0) or 0)
    if num_experts > 0 and expert_quant != "nvfp4":
        raise NotImplementedError(
            "Laguna: only the NVFP4 checkpoint (poolside/Laguna-XS-2.1-NVFP4) is served "
            "today; the BF16 checkpoint's routed experts have no expert-bank loader"
        )
    moe_intermediate_size = int(getattr(hf_config, "moe_intermediate_size", 0) or 0)
    shared_intermediate = int(
        getattr(hf_config, "shared_expert_intermediate_size", 0) or moe_intermediate_size
    )

    return ModelConfig(
        num_layers=num_layers,
        # The widest group: every model-agnostic consumer that sizes per-launch buffers off
        # this scalar must cover the sliding layers. Per-layer geometry comes from the
        # attention groups (``num_qo_heads_for_layer``).
        num_qo_heads=max(full_qo_heads, swa_qo_heads),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=int(hf_config.hidden_size),
        vocab_size=int(hf_config.vocab_size),
        intermediate_size=int(hf_config.intermediate_size),
        rms_norm_eps=float(hf_config.rms_norm_eps),
        rotary_config=full_rotary,
        hidden_act=getattr(hf_config, "hidden_act", "silu"),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        num_experts=num_experts,
        num_experts_per_tok=int(getattr(hf_config, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=moe_intermediate_size,
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "laguna"),
        architectures=getattr(hf_config, "architectures", ["LagunaForCausalLM"]),
        moe_enabled=num_experts > 0,
        expert_quant=expert_quant,
        # The shared experts are NVFP4 like the routed ones; attention, the dense layer-0
        # MLP, the router and lm_head are in the checkpoint's ``ignore`` list and stay BF16.
        dense_quant=expert_quant,
        attn_quant="none",
        lm_head_quant="none",
        shared_expert_intermediate_size=shared_intermediate,
        use_qk_norm=True,
        first_k_dense_replace=_first_k_dense_replace(
            getattr(hf_config, "mlp_layer_types", None), num_layers
        ),
        n_shared_experts=1 if shared_intermediate > 0 else 0,
        routed_scaling_factor=float(getattr(hf_config, "moe_routed_scaling_factor", 1.0)),
        has_attn_bias=bool(getattr(hf_config, "attention_bias", False)),
        attention_groups=tuple(groups),
    )


__all__ = ["parse_config"]
