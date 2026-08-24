"""Laguna XS 2.x config parsing, quant dispositions and checkpoint-key contract.

Drives ``laguna.parse_config`` off a synthetic HF config shaped exactly like
poolside/Laguna-XS-2.1-NVFP4's and checks the decisions that are unique to this family:
the 3:1 full/sliding split with a *different query-head count per attention type*, the
per-type rope (YaRN + half-width rotary on full layers, plain rope on sliding), the single
leading dense MLP layer, and which components stay BF16 on an otherwise-NVFP4 checkpoint.
"""

from __future__ import annotations

import pytest

from freetoken.attention.base import AttnType
from freetoken.models.config import FullAttentionGroupConfig, SWAAttentionGroupConfig
from freetoken.models.laguna.config import parse_config

NUM_LAYERS = 40
FULL_QO_HEADS = 48
SWA_QO_HEADS = 64


class _Cfg:
    """Attribute-access shim over a dict (what AutoConfig hands parse_config)."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, value)


def _nvfp4_quant_config() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "format": "nvfp4-pack-quantized",
                "targets": ["re:.*experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)$"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "strategy": "tensor_group",
                },
            }
        },
    }


def _int4_quant_config() -> dict:
    """The INT4 sibling: group-128 *integer* pack-quantized, mixed 4-bit/8-bit by layer."""
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "targets": ["re:.*layers\\.([1-9]|[12]\\d|30)\\..*(gate_proj|up_proj|down_proj)$"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "group_size": 128,
                    "strategy": "group",
                    "symmetric": True,
                },
            },
            "group_1": {
                "format": "pack-quantized",
                "targets": ["re:.*layers\\.3[1-9]\\..*(gate_proj|up_proj|down_proj)$"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "group_size": 128,
                    "strategy": "group",
                    "symmetric": True,
                },
            },
        },
    }


def _hf_config(quant: dict | None = "nvfp4") -> _Cfg:
    # 3:1 with the full layer leading: 0, 4, 8, ... 36.
    layer_types = [
        "full_attention" if i % 4 == 0 else "sliding_attention" for i in range(NUM_LAYERS)
    ]
    data = {
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 100352,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "gating": "per-head",
        "sliding_window": 512,
        "layer_types": layer_types,
        "num_attention_heads_per_layer": [
            FULL_QO_HEADS if kind == "full_attention" else SWA_QO_HEADS for kind in layer_types
        ],
        "mlp_layer_types": ["dense"] + ["sparse"] * (NUM_LAYERS - 1),
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "norm_topk_prob": True,
        "moe_routed_scaling_factor": 2.5,
        "moe_apply_router_weight_on_input": False,
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 500000.0,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 64.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    }
    if quant == "nvfp4":
        data["quantization_config"] = _nvfp4_quant_config()
    elif quant is not None:
        data["quantization_config"] = quant
    return _Cfg(data)


# ---------------------------------------------------------------------------- geometry


def test_attention_groups_split_3_to_1_with_per_type_query_heads():
    cfg = parse_config(_hf_config())
    full, swa = cfg.attention_groups
    assert isinstance(full, FullAttentionGroupConfig)
    assert isinstance(swa, SWAAttentionGroupConfig)
    assert full.layer_ids == tuple(range(0, NUM_LAYERS, 4))
    assert len(swa.layer_ids) == 30
    assert (full.num_qo_heads, swa.num_qo_heads) == (FULL_QO_HEADS, SWA_QO_HEADS)
    assert (full.num_kv_heads, swa.num_kv_heads) == (8, 8)
    assert swa.sliding_window == 512


def test_model_wide_query_head_count_is_the_widest_group():
    """Every model-agnostic consumer sizes per-launch buffers off this scalar, so it has to
    cover the sliding layers; the narrower full layers come from the group."""
    cfg = parse_config(_hf_config())
    assert cfg.num_qo_heads == SWA_QO_HEADS
    assert [cfg.num_qo_heads_for_layer(i) for i in range(5)] == [48, 64, 64, 64, 48]


def test_kv_cache_specs_carry_the_per_group_query_heads():
    specs = {spec.name: spec for spec in parse_config(_hf_config()).kv_cache_group_specs()}
    assert specs["full"].num_qo_heads == FULL_QO_HEADS
    assert specs["swa"].num_qo_heads == SWA_QO_HEADS
    assert specs["full"].attn_type is AttnType.FULL
    assert specs["swa"].attn_type is AttnType.SWA
    assert specs["swa"].sliding_window == 512


def test_uniform_models_still_report_one_query_head_count():
    """The per-group field defaults to 0 -> inherit, so no existing parser has to change."""
    from freetoken.models.qwen3_moe.config import parse_config as qwen_parse

    cfg = qwen_parse(
        _Cfg(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "rope_theta": 1000000.0,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 768,
                "norm_topk_prob": True,
                "tie_word_embeddings": False,
            }
        )
    )
    assert cfg.num_qo_heads == 32
    assert all(cfg.num_qo_heads_for_layer(i) == 32 for i in range(cfg.num_layers))
    assert all(spec.num_qo_heads == 32 for spec in cfg.kv_cache_group_specs())


# -------------------------------------------------------------------------------- rope


def test_full_layers_get_yarn_over_a_half_width_rotary():
    full, swa = parse_config(_hf_config()).attention_groups
    rope = full.rotary_config
    assert (rope.head_dim, rope.rotary_dim) == (128, 64)  # partial_rotary_factor 0.5
    assert rope.base == 500000.0
    # Every scalar layers/rotary.py's yarn arm reads must survive the copy.
    assert rope.scaling["rope_type"] == "yarn"
    assert rope.scaling["factor"] == 32.0
    assert rope.scaling["beta_fast"] == 64.0
    assert rope.scaling["beta_slow"] == 1.0
    assert rope.scaling["original_max_position_embeddings"] == 8192
    assert rope.scaling["attention_factor"] == pytest.approx(1.3465735902799727)
    # ...and partial_rotary_factor must NOT: it is consumed into rotary_dim, and get_rope's
    # yarn arm would not know what to do with it.
    assert "partial_rotary_factor" not in rope.scaling


def test_sliding_layers_get_plain_rope_at_their_own_theta():
    _full, swa = parse_config(_hf_config()).attention_groups
    rope = swa.rotary_config
    assert (rope.head_dim, rope.rotary_dim) == (128, 128)
    assert rope.base == 10000.0
    assert rope.scaling is None


def test_rope_scaling_is_hashable_for_the_get_rope_cache():
    """``get_rope`` is ``functools.cache``d on ``tuple(scaling.items())``."""
    for group in parse_config(_hf_config()).attention_groups:
        scaling = group.rotary_config.scaling
        if scaling is not None:
            hash(tuple(scaling.items()))


# ------------------------------------------------------------------------------ MoE/quant


def test_one_leading_dense_layer_then_sparse():
    cfg = parse_config(_hf_config())
    assert cfg.first_k_dense_replace == 1
    assert cfg.num_moe_layers == 39
    assert cfg.moe_enabled and cfg.is_moe


def test_router_and_shared_expert_scalars():
    cfg = parse_config(_hf_config())
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 512
    assert cfg.shared_expert_intermediate_size == 512
    assert cfg.n_shared_experts == 1
    assert cfg.routed_scaling_factor == 2.5
    assert cfg.norm_topk_prob is True
    # The selection bias is a block-owned buffer, not a router Linear bias.
    assert cfg.has_router_bias is False
    assert cfg.use_qk_norm is True


def test_only_the_experts_are_nvfp4():
    """The checkpoint's ``ignore`` list keeps attention, the dense layer-0 MLP, the router
    and lm_head in BF16; the routed *and* shared experts are NVFP4."""
    cfg = parse_config(_hf_config())
    assert cfg.expert_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.attn_quant == "none"
    assert cfg.lm_head_quant == "none"


def test_int4_checkpoint_is_rejected_with_its_actual_scheme():
    with pytest.raises(ValueError, match="integer scheme"):
        parse_config(_hf_config(quant=_int4_quant_config()))


def test_bf16_checkpoint_is_rejected_until_it_has_an_expert_bank_loader():
    with pytest.raises(NotImplementedError, match="NVFP4 checkpoint"):
        parse_config(_hf_config(quant=None))


# ---------------------------------------------------------------------------- validation


def test_a_non_contiguous_dense_prefix_is_rejected():
    cfg = _hf_config()
    cfg.mlp_layer_types = ["dense", "sparse", "dense"] + ["sparse"] * (NUM_LAYERS - 3)
    with pytest.raises(ValueError, match="contiguous prefix"):
        parse_config(cfg)


def test_query_heads_that_disagree_inside_one_group_are_rejected():
    """Attention groups are per *type*; a group whose layers differ cannot be expressed."""
    cfg = _hf_config()
    heads = list(cfg.num_attention_heads_per_layer)
    heads[4] = 32  # another full layer, different width
    cfg.num_attention_heads_per_layer = heads
    with pytest.raises(ValueError, match="uniform within an attention group"):
        parse_config(cfg)


def test_unsupported_router_softcapping_is_rejected():
    cfg = _hf_config()
    cfg.moe_router_logit_softcapping = 30.0
    with pytest.raises(ValueError, match="softcapping"):
        parse_config(cfg)


def test_unsupported_attention_sinks_are_rejected():
    cfg = _hf_config()
    cfg.swa_attention_sink_enabled = True
    with pytest.raises(ValueError, match="sink"):
        parse_config(cfg)


# ------------------------------------------------------------------- checkpoint key contract


def test_selection_bias_moves_off_the_experts_module():
    """HF parks ``e_score_correction_bias`` under ``mlp.experts`` so accelerate's hooks keep
    it with the gate; FreeToken's sparse block owns it directly (the experts are an
    offload-cache handle, not a module)."""
    from freetoken.models.laguna.weight import _rename

    assert (
        _rename("model.layers.7.mlp.experts.e_score_correction_bias")
        == "model.layers.7.mlp.e_score_correction_bias"
    )


def test_fp8_kv_cache_scalars_are_dropped():
    from freetoken.models.laguna.weight import _rename

    assert _rename("model.layers.7.self_attn.k_scale") is None
    assert _rename("model.layers.7.self_attn.v_scale") is None
    assert _rename("model.layers.7.self_attn.q_proj.weight") is not None


def test_dense_mlp_fusion_does_not_swallow_the_shared_expert():
    """``.mlp.gate_proj`` must match only the dense layer-0 MLP: the shared expert is NVFP4
    and fuses through a different table, so a loose suffix match would mix precisions."""
    from freetoken.models.laguna.weight import _BF16_FUSE, _NVFP4_FUSE

    dense_parts = _BF16_FUSE[".mlp.gate_up_proj"]
    shared_base = "model.layers.5.mlp.shared_expert.gate_proj"
    assert not any(shared_base.endswith(part) for part in dense_parts)
    assert "model.layers.0.mlp.gate_proj".endswith(dense_parts[0])
    assert shared_base.endswith(_NVFP4_FUSE[".mlp.shared_expert.gate_up_proj"][0])


def test_routed_experts_never_enter_the_dense_pass():
    from freetoken.models.laguna.weight import _ROUTED_EXPERT_RE

    assert _ROUTED_EXPERT_RE.search("model.layers.5.mlp.experts.31.gate_proj.weight_packed")
    assert not _ROUTED_EXPERT_RE.search("model.layers.5.mlp.shared_expert.gate_proj.weight_packed")
    assert not _ROUTED_EXPERT_RE.search("model.layers.5.mlp.experts.e_score_correction_bias")


def test_expert_bank_pattern_matches_every_routed_tensor_kind():
    from freetoken.models.laguna.weight import _NVFP4_EXPERT_KEY_RE, _NVFP4_SOURCE_SPEC

    for kind in ("weight_packed", "weight_scale", "weight_global_scale"):
        match = _NVFP4_EXPERT_KEY_RE.match(f"model.layers.9.mlp.experts.31.down_proj.{kind}")
        assert match is not None, kind
        assert (match["layer"], match["expert"], match["proj"]) == ("9", "31", "down_proj")
    # W4A4 calibration scales are not bank inputs.
    assert _NVFP4_EXPERT_KEY_RE.match(
        "model.layers.9.mlp.experts.31.down_proj.input_global_scale"
    ) is None
    # Banks are indexed by MoE layer, so the leading dense layer shifts every index down.
    cfg = parse_config(_hf_config())
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(1, cfg) == 0
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(39, cfg) == 38
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(0, cfg) is None


# ------------------------------------------------------------------------------- wiring


def test_architecture_is_registered_with_the_hooks_the_engine_resolves():
    from freetoken.models.register import _load_attr, get_model_spec

    spec = get_model_spec("LagunaForCausalLM")
    assert spec.module == "freetoken.models.laguna"
    for attr in (
        spec.model_cls,
        spec.parse_config,
        spec.iter_weights,
        "load_nvfp4_expert_sources",
        "load_nvfp4_expert_sources_parallel",
    ):
        assert callable(_load_attr(spec.module, attr)), attr


def test_aot_row_matches_the_parsed_geometry():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS

    cfg = parse_config(_hf_config())
    (row,) = [m for m in SUPPORTED_MODELS if m.architecture == "LagunaForCausalLM"]
    assert row.hidden_size == cfg.hidden_size
    assert row.top_k == cfg.num_experts_per_tok
    assert row.moe_intermediate_size == cfg.moe_intermediate_size
    # Both attention groups share the KV geometry, so one store shape covers the model.
    assert row.kv_groups == ((cfg.num_kv_heads, cfg.head_dim),)
    assert set(row.expert_formats) >= {"nvfp4"}
