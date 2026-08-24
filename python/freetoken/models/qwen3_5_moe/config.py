from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_compressed_tensors_nvfp4,
)


def _quant_accessor(hf_config: Any):
    """A ``get(key, default=None)`` accessor over the HF ``quantization_config`` (dict or
    object), or ``None`` when the model has no quant config."""
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return None
    return quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))


def _fp8_block_quant(hf_config: Any) -> tuple[str, tuple[int, int] | None]:
    """Detect DeepSeek-V3-style 128x128 block-fp8 from HF ``quantization_config``.

    Returns ``("fp8_block", (block_n, block_k))`` for a block-fp8 checkpoint (weights
    fp8-e4m3 + per-block ``weight_scale_inv``, dynamic activation), else ``("none", None)``.
    The quantization_config sits on the top-level hf_config (not ``text_config``).
    """
    get = _quant_accessor(hf_config)
    if get is None:
        return "none", None
    method = str(get("quant_method") or get("quant_algo") or "").lower()
    block = get("weight_block_size")
    if method == "fp8" and block:
        bs = tuple(int(x) for x in block)
        assert bs == (128, 128), f"only 128x128 block-fp8 is supported, got {bs}"
        return "fp8_block", bs
    return "none", None


def _expert_quant(hf_config: Any) -> str:
    """Quantization format of the *routed* experts (the only weights served from the
    offload cache). The nvidia/modelopt checkpoints are either plain NVFP4 (``quant_algo``
    ``NVFP4``) or ``MIXED_PRECISION`` (per-layer ``quantized_layers`` map); in the mixed
    case the routed experts carry their own ``W4A16_NVFP4``/``FP8`` algo. Dense quantized
    weights (attention/shared-expert/lm_head) are handled separately by dequant-at-load."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    algo = str(get("quant_algo") or get("quant_method") or "").lower()
    if "fp4" in algo:
        return "nvfp4"
    if "mixed" in algo:
        layers = get("quantized_layers") or {}
        for name, spec in (layers.items() if isinstance(layers, dict) else []):
            if name.endswith(".mlp.experts") or ".mlp.experts." in name:
                expert_algo = str((spec or {}).get("quant_algo", "")).lower()
                if "fp4" in expert_algo:
                    return "nvfp4"
                if "fp8" in expert_algo:
                    return "fp8"
    return "none"


# Detection now lives in models/config.py (shared with muse_glimmer); weight.py imports
# it under this name.
_compressed_tensors_nvfp4 = detect_compressed_tensors_nvfp4

def _compressed_tensors_linear_attn_out_quant(hf_config: Any) -> str:
    """Return GatedDeltaNet ``out_proj`` storage format for a compressed export.

    An exact ``linear_attn.out_proj`` ignore entry means this projection remains bf16.
    Broad ``linear_attn`` entries are not enough: exporters can retain those labels while
    storing the output projection as native NVFP4.
    """
    get = _quant_accessor(hf_config)
    ignored = () if get is None else (get("ignore") or ())
    if isinstance(ignored, str):
        ignored = (ignored,)
    return (
        "none"
        if any(str(name).endswith(".linear_attn.out_proj") for name in ignored)
        else "nvfp4"
    )


def _lm_head_quant(hf_config: Any) -> str:
    """Whether the checkpoint stores ``lm_head`` as NVFP4. modelopt MIXED_PRECISION lists it in
    the per-layer ``quantized_layers`` map (``W4A16_NVFP4``); pure-NVFP4 checkpoints have no
    per-layer map and leave lm_head bf16. Returns ``"nvfp4"`` or ``"none"``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if name == "lm_head" or name.endswith(".lm_head"):
            if "fp4" in str((spec or {}).get("quant_algo", "")).lower():
                return "nvfp4"
    return "none"


def _dense_mlp_quant(hf_config: Any) -> str:
    """NVFP4 on the *dense* (non-MoE) decoder MLP. modelopt MIXED_PRECISION dense checkpoints
    (e.g. Qwen3.6-27B-NVFP4) list ``.mlp.{gate,up,down}_proj`` as ``W4A16_NVFP4`` in
    ``quantized_layers``; MoE checkpoints have ``.mlp.experts.*`` / ``.mlp.shared_expert.*``
    instead (covered by ``expert_quant``). ``endswith(".mlp.gate_proj")`` matches only the bare
    dense MLP -- not ``.mlp.shared_expert.gate_proj`` nor ``.mlp.experts.N.gate_proj``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if name.endswith((".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")):
            if "fp4" in str((spec or {}).get("quant_algo", "")).lower():
                return "nvfp4"
    return "none"


def _shared_expert_quant(hf_config: Any) -> str | None:
    """What the MoE *shared expert* is actually stored as, from the modelopt
    ``quantized_layers`` map: ``"nvfp4"``, ``"fp8_pertensor"``, or ``None`` when the
    checkpoint says nothing about it.

    Most NVFP4 exports quantize the shared expert exactly like the routed experts, which
    is why this used to be inferred from ``expert_quant``. A MIXED_PRECISION export need
    not agree: apodex/Apodex-1.1-mini-NVFP4 lists ``.mlp.experts`` as NVFP4 while every
    ``.mlp.shared_expert.*`` entry is FP8. Inferring FP4 there builds an
    ``Nvfp4DenseColMerged`` the loader can never fill.
    """
    get = _quant_accessor(hf_config)
    if get is None:
        return None
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return None
    for name, spec in layers.items():
        if ".mlp.shared_expert." in name or name.endswith(".mlp.shared_expert"):
            algo = str((spec or {}).get("quant_algo", "")).lower()
            if "fp4" in algo:
                return "nvfp4"
            if "fp8" in algo:
                return "fp8_pertensor"
    return None


def _attn_quant(hf_config: Any) -> str:
    """Per-tensor FP8 on the *dense* attention/GDN projections. The modelopt
    ``MIXED_PRECISION`` checkpoints tag ``self_attn.{q,k,v,o}_proj`` and
    ``linear_attn.{in_proj_qkv,in_proj_z,out_proj}`` with ``quant_algo`` ``FP8`` (fp8-e4m3
    weight + a scalar ``weight_scale``; W8A16). Returns ``"fp8_pertensor"`` when present,
    else ``"none"`` (NVFP4 dense weights -- shared_expert/lm_head -- stay dequant-at-load)."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        algo = str((spec or {}).get("quant_algo", "")).lower()
        if algo == "fp8" and (".self_attn." in name or ".linear_attn." in name):
            return "fp8_pertensor"
    return "none"


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = round(head_dim * partial)

    # For text-only with the default rope type, partial NeoX rope needs no scaling dict
    # (the mRoPE params reduce to standard partial rope for text). Avoid carrying the
    # unhashable ``mrope_section`` list into get_rope's cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    expert_quant, weight_block_size = _fp8_block_quant(hf_config)
    if expert_quant == "none":
        expert_quant = _expert_quant(hf_config)  # nvfp4 / mixed-precision modelopt
    # Dense attention/GDN quant is independent of the routed experts (block-fp8 already
    # quantizes both, so only probe for per-tensor FP8 when experts aren't block-fp8).
    attn_quant = "none" if expert_quant == "fp8_block" else _attn_quant(hf_config)
    linear_attn_out_quant = attn_quant
    # NVFP4 checkpoints usually store the dense MLP projections (shared_expert; dense
    # non-MoE MLP) as packed FP4 exactly like the routed experts, so inferring FP4 from
    # expert_quant is the right default. A MIXED_PRECISION export can disagree, and when
    # its quantized_layers map names the shared expert explicitly that map wins: apodex's
    # Qwen3.5 derivative pairs NVFP4 routed experts with an FP8 shared expert, and
    # building Nvfp4DenseColMerged there leaves gate_up_proj.weight unfillable.
    # The lm_head is detected separately (only the mixed checkpoint quantizes it).
    declared_shared = _shared_expert_quant(hf_config)
    if declared_shared is not None:
        dense_quant = declared_shared
    else:
        dense_quant = "nvfp4" if expert_quant == "nvfp4" else _dense_mlp_quant(hf_config)
    lm_head_quant = _lm_head_quant(hf_config)

    # Compressed-tensors NVFP4 quantizes routed experts in the same native layout as
    # dense linears. Routed MoE checkpoints must use native FP4 expert banks; dense
    # variants have no banks. Some exports intentionally leave GDN out_proj bf16.
    if _compressed_tensors_nvfp4(hf_config):
        if getattr(text, "num_experts", 0):
            expert_quant = "nvfp4"
        attn_quant = "nvfp4"
        dense_quant = "nvfp4"
        lm_head_quant = "none"
        linear_attn_out_quant = _compressed_tensors_linear_attn_out_quant(hf_config)

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate=True,
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", False)),
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen3_5_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        vision_config=None,  # text-only milestone
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        attn_quant=attn_quant,
        linear_attn_out_quant=linear_attn_out_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
    )


__all__ = ["parse_config"]
