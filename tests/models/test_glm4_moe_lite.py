"""GLM-4.7-Flash (``glm4_moe_lite``) config parsing and checkpoint-key contract.

The decoder graph is GLM-5.2's, reused wholesale; what is specific to this family and
therefore worth pinning is the *resolution*: no DSA indexer (so a plain latent-KV MLA
group rather than a DSA one), which components are NVFP4 versus BF16, the trailing MTP
layer being excluded from both the layer count and the expert banks, and the rejection of
an export whose expert layers are not uniformly quantized.
"""

from __future__ import annotations

import pytest

from freetoken.attention.base import AttnType
from freetoken.models.glm4_moe_lite.config import parse_config

NUM_LAYERS = 47  # 1 dense + 46 MoE; the MTP block is layer 47 and is not counted


class _Cfg:
    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, value)


def _nvfp4_quant() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": ["lm_head", "re:.*embed.*", "re:.*gate$", "re:.*self_attn.*"],
        "config_groups": {
            "group_0": {
                "format": "nvfp4-pack-quantized",
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "strategy": "tensor_group",
                },
            }
        },
    }


def _mixed_quant() -> dict:
    """unsloth's export: FP8 on layers 1/39/46, NVFP4 elsewhere."""
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": ["re:.*layers\\.(1|39|46)\\.mlp\\.experts\\.\\d+\\..*"],
                "weights": {"num_bits": 8, "type": "float", "strategy": "channel"},
            },
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "targets": ["re:.*mlp\\.experts\\.\\d+\\..*"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "strategy": "tensor_group",
                },
            },
        },
    }


def _hf_config(quant: dict | None = "nvfp4", **overrides) -> _Cfg:
    data = {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "model_type": "glm4_moe_lite",
        "hidden_size": 2048,
        "intermediate_size": 10240,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "vocab_size": 154880,
        "max_position_embeddings": 202752,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "attention_bias": False,
        # MLA
        "q_lora_rank": 768,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "rope_interleave": True,
        "rope_parameters": {
            "rope_type": "default", "rope_theta": 1000000, "partial_rotary_factor": 1.0,
        },
        # MoE
        "n_routed_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1536,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.8,
        "num_nextn_predict_layers": 1,
    }
    data.update(overrides)
    if quant == "nvfp4":
        data["quantization_config"] = _nvfp4_quant()
    elif quant is not None:
        data["quantization_config"] = quant
    return _Cfg(data)


# ----------------------------------------------------------------------------- attention


def test_no_indexer_resolves_to_plain_latent_mla_not_dsa():
    """The checkpoint carries no index_* fields, so the group must be MLA: a DSA group
    would make the pool factory allocate an index-key slab that has no weights."""
    cfg = parse_config(_hf_config())
    (group,) = cfg.attention_groups
    assert group.mla is True
    assert (group.index_head_dim, group.num_index_layers) == (0, 0)
    assert cfg.attn_type_for_layer(0) is AttnType.MLA
    assert group.num_kv_heads == 1


def test_latent_kv_geometry():
    cfg = parse_config(_hf_config())
    # one latent row per token: ckv (512) | kpe (64)
    assert cfg.head_dim == 576
    assert cfg.attention_groups[0].head_dim == 576
    assert cfg.num_qo_heads == 20


def test_softmax_scale_is_over_the_full_qk_head_dim():
    """qk_head_dim is 192+64=256; scaling off the latent 576 or the v_head 256-by-accident
    would silently change every logit."""
    cfg = parse_config(_hf_config())
    assert cfg.attn_sm_scale == pytest.approx(256**-0.5)


def test_rope_covers_only_the_rope_half():
    rope = parse_config(_hf_config()).rotary_config
    assert rope.rotary_dim == 64
    assert rope.head_dim == 256
    assert rope.base == 1000000
    assert rope.scaling is None  # rope_type "default"; interleave handled in the module


def test_interleaved_rope_reaches_the_args_payload():
    args = parse_config(_hf_config()).glm_dsa_args
    assert args.rope_interleave is True
    assert (args.qk_nope_head_dim, args.qk_rope_head_dim, args.v_head_dim) == (192, 64, 256)
    assert args.qk_head_dim == 256


def test_a_checkpoint_with_a_dsa_indexer_is_routed_elsewhere():
    cfg = _hf_config()
    cfg.index_head_dim = 128
    cfg.index_topk = 2048
    cfg.indexer_types = ["full"] * NUM_LAYERS
    with pytest.raises(NotImplementedError, match="glm_moe_dsa"):
        parse_config(cfg)


# --------------------------------------------------------------------------------- MoE


def test_dense_prefix_and_moe_layer_count_exclude_the_mtp_block():
    cfg = parse_config(_hf_config())
    assert cfg.num_layers == NUM_LAYERS  # MTP layer is not one of these
    assert cfg.first_k_dense_replace == 1
    assert cfg.num_moe_layers == 46
    assert cfg.attention_groups[0].layer_ids == tuple(range(NUM_LAYERS))


def test_router_scalars():
    cfg = parse_config(_hf_config())
    assert (cfg.num_experts, cfg.num_experts_per_tok) == (64, 4)
    assert cfg.moe_intermediate_size == 1536
    assert cfg.intermediate_size == 10240  # the leading dense layer is much wider
    assert cfg.routed_scaling_factor == 1.8
    assert cfg.n_shared_experts == 1
    assert cfg.n_group == cfg.topk_group == 1  # group-limited routing is a no-op
    assert cfg.norm_topk_prob is True


# ------------------------------------------------------------------------------- quant


def test_experts_shared_and_dense_mlp_are_nvfp4_while_attention_stays_bf16():
    cfg = parse_config(_hf_config())
    assert cfg.expert_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"  # shared experts + the leading dense MLP
    assert cfg.attn_quant == "none"  # MLA is in the checkpoint's ignore list
    assert cfg.lm_head_quant == "none"


def test_a_mixed_precision_export_is_rejected_with_the_reason():
    """One offload-cache slot pool is shared by every layer, so expert layers quantized
    differently from each other cannot be represented."""
    with pytest.raises(NotImplementedError, match="mixed-precision|one offload-cache"):
        parse_config(_hf_config(quant=_mixed_quant()))


def test_an_unquantized_checkpoint_is_rejected():
    with pytest.raises(NotImplementedError, match="NVFP4 checkpoints"):
        parse_config(_hf_config(quant=None))


# ------------------------------------------------------------ checkpoint key contract


def test_selection_bias_moves_off_the_router_linear():
    from freetoken.models.glm4_moe_lite.weight import _rename

    assert (
        _rename("model.layers.5.mlp.gate.e_score_correction_bias")
        == "model.layers.5.mlp.e_score_correction_bias"
    )
    assert _rename("model.layers.5.mlp.gate.weight") == "model.layers.5.mlp.gate.weight"


def test_expert_bank_indices_skip_the_dense_prefix_and_the_mtp_layer():
    """The MTP block carries a full 64-expert set of its own; packing it would both shift
    every bank index and overrun the bank list."""
    from freetoken.models.glm4_moe_lite.weight import _NVFP4_SOURCE_SPEC

    cfg = parse_config(_hf_config())
    to_bank = _NVFP4_SOURCE_SPEC.layer_to_bank
    assert to_bank(0, cfg) is None  # dense prefix
    assert to_bank(1, cfg) == 0
    assert to_bank(46, cfg) == 45
    assert to_bank(47, cfg) is None  # MTP


def test_expert_pattern_matches_the_compressed_tensors_kinds():
    from freetoken.models.glm4_moe_lite.weight import _NVFP4_EXPERT_KEY_RE

    for kind in ("weight_packed", "weight_scale", "weight_global_scale"):
        m = _NVFP4_EXPERT_KEY_RE.match(f"model.layers.9.mlp.experts.31.down_proj.{kind}")
        assert m is not None, kind
        assert (m["layer"], m["expert"], m["proj"]) == ("9", "31", "down_proj")
    # W4A4 calibration scales are not bank inputs.
    assert _NVFP4_EXPERT_KEY_RE.match(
        "model.layers.9.mlp.experts.31.down_proj.input_global_scale"
    ) is None
    # The shared expert is a resident dense linear, not a bank input.
    assert _NVFP4_EXPERT_KEY_RE.match(
        "model.layers.9.mlp.shared_experts.down_proj.weight_packed"
    ) is None


def test_routed_experts_never_enter_the_dense_pass():
    from freetoken.models.glm4_moe_lite.weight import _ROUTED_EXPERT_RE

    assert _ROUTED_EXPERT_RE.search("model.layers.5.mlp.experts.31.gate_proj.weight_packed")
    assert not _ROUTED_EXPERT_RE.search(
        "model.layers.5.mlp.shared_experts.gate_proj.weight_packed"
    )


# ------------------------------------------------------------------------------ wiring


def test_architecture_is_registered_with_the_hooks_the_engine_resolves():
    from freetoken.models.register import _load_attr, get_model_spec

    spec = get_model_spec("Glm4MoeLiteForCausalLM")
    assert spec.module == "freetoken.models.glm4_moe_lite"
    for attr in (
        spec.model_cls,
        spec.parse_config,
        spec.iter_weights,
        "load_nvfp4_expert_sources",
        "load_nvfp4_expert_sources_parallel",
    ):
        assert callable(_load_attr(spec.module, attr)), attr


def test_nvfp4_dense_projections_are_constructible():
    """``_make_proj`` gained an nvfp4 arm for this family; without it the shared experts
    and the dense MLP would silently build bf16 buffers and fail the strict load."""
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear
    from freetoken.models.glm_moe_dsa.attention import _make_proj

    assert isinstance(_make_proj("nvfp4", 2048, 1536), Nvfp4DenseLinear)


def test_aot_row_matches_the_parsed_geometry():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS

    cfg = parse_config(_hf_config())
    (row,) = [m for m in SUPPORTED_MODELS if m.architecture == "Glm4MoeLiteForCausalLM"]
    assert row.hidden_size == cfg.hidden_size
    assert row.top_k == cfg.num_experts_per_tok
    assert row.moe_intermediate_size == cfg.moe_intermediate_size
    # MLA writes its latent through torch scatter, not store_cache -> no paged-KV groups.
    assert row.kv_groups == ()
    assert set(row.expert_formats) >= {"nvfp4"}
