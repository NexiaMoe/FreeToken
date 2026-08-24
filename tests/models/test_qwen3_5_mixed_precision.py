"""Qwen3.5 derivatives whose components are NOT all quantized the same way.

``dense_quant``/``attn_quant`` used to be inferred as "whatever the routed experts are",
which holds for every single-scheme NVFP4 export. Two real checkpoints break it, one per
quantization dialect:

* apodex/Apodex-1.1-mini-NVFP4 -- modelopt MIXED_PRECISION. Its ``quantized_layers`` map
  lists ``.mlp.experts`` as NVFP4 and every ``.mlp.shared_expert.*`` entry as FP8.
* primitive-ai/Ornith-1.5-35B-A3B-agentic-NVFP4-FP8 -- compressed-tensors
  ``mixed-precision``. It has no ``quantized_layers``; the split lives in the group
  ``targets``, an NVFP4 group for the routed experts and a per-tensor FP8 group for
  attention, GDN and the shared expert.

Both build FP4 layers for FP8 tensors under the old inference, and the second is worse:
``detect_compressed_tensors_nvfp4`` answers True whenever SOME group is NVFP4, so the
compressed-tensors branch forced attention to FP4 as well.
"""

from __future__ import annotations

import pytest

from freetoken.models.qwen3_5_moe.config import _shared_expert_quant, parse_config


class _Cfg:
    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            nested = isinstance(value, dict) and key == "text_config"
            setattr(self, key, _Cfg(value) if nested else value)


def _quantized_layers(shared_algo: str | None, expert_algo: str = "NVFP4") -> dict:
    layers = {
        "model.language_model.layers.0.linear_attn.out_proj": {"quant_algo": "FP8"},
        "model.language_model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
        "model.language_model.layers.0.mlp.experts": {"quant_algo": expert_algo, "group_size": 16},
    }
    if shared_algo is not None:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            layers[f"model.language_model.layers.0.mlp.shared_expert.{proj}"] = {
                "quant_algo": shared_algo
            }
    return layers


def _hf_config(shared_algo: str | None, expert_algo: str = "NVFP4") -> _Cfg:
    return _Cfg({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "quantization_config": {
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt_mixed",
            "producer": {"name": "modelopt", "version": "0.44.0"},
            "quantized_layers": _quantized_layers(shared_algo, expert_algo),
            "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
            "ignore": ["mtp*"],
        },
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "rope_theta": 1000000.0,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "norm_topk_prob": True,
            "tie_word_embeddings": False,
            "mlp_only_layers": [],
            # Qwen3.5 is a GDN hybrid: 3 linear-attention layers per full-attention one.
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
        },
    })


# ------------------------------------------------------------------------- detection


def test_an_fp8_shared_expert_is_detected():
    assert _shared_expert_quant(_hf_config("FP8")) == "fp8_pertensor"


def test_an_nvfp4_shared_expert_is_detected():
    assert _shared_expert_quant(_hf_config("W4A16_NVFP4")) == "nvfp4"


def test_a_silent_checkpoint_returns_none_so_the_old_inference_still_applies():
    assert _shared_expert_quant(_hf_config(None)) is None


# ------------------------------------------------------------------------ resolution


def test_an_fp8_shared_expert_does_not_inherit_the_routed_experts_nvfp4():
    """The regression: routed experts NVFP4, shared expert FP8."""
    cfg = parse_config(_hf_config("FP8"))
    assert cfg.expert_quant == "nvfp4"  # routed experts are still FP4 banks
    assert cfg.dense_quant == "fp8_pertensor"  # ...but the shared expert is not


def test_an_nvfp4_shared_expert_still_resolves_to_nvfp4():
    cfg = parse_config(_hf_config("W4A16_NVFP4"))
    assert cfg.expert_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"


def test_a_checkpoint_that_says_nothing_keeps_the_previous_behaviour():
    """No shared-expert entry -> infer from the routed experts, as before."""
    cfg = parse_config(_hf_config(None))
    assert cfg.expert_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"


# ---------------------------------------------------------------------------- wiring


def test_the_shared_expert_builds_the_fp8_kernel_for_an_fp8_checkpoint():
    from freetoken.kernel.triton.fp8_pertensor_linear import (
        Fp8PerTensorColMerged,
        Fp8PerTensorLinear,
    )
    from freetoken.models.qwen3_5_moe.moe import _SharedExpert

    cfg = parse_config(_hf_config("FP8"))
    shared = _SharedExpert(cfg, cfg.hidden_size, cfg.shared_expert_intermediate_size)
    assert isinstance(shared.gate_up_proj, Fp8PerTensorColMerged)
    assert isinstance(shared.down_proj, Fp8PerTensorLinear)


def test_the_shared_expert_still_builds_nvfp4_for_an_nvfp4_checkpoint():
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear
    from freetoken.models.qwen3_5_moe.moe import _SharedExpert

    cfg = parse_config(_hf_config("W4A16_NVFP4"))
    shared = _SharedExpert(cfg, cfg.hidden_size, cfg.shared_expert_intermediate_size)
    assert isinstance(shared.gate_up_proj, Nvfp4DenseColMerged)
    assert isinstance(shared.down_proj, Nvfp4DenseLinear)


def test_gate_and_up_fuse_when_the_shared_expert_is_per_tensor_fp8():
    """Without the fusion pair the loader emits gate_proj/up_proj standalone and the model,
    which always wants a merged gate_up_proj, dies on the missing key. Each part keeps its
    own scalar scale, broadcast into one piecewise per-output-row vector."""
    import torch

    from freetoken.models.qwen3_5_moe.weight import _PT_FP8_FUSE, _pt_fp8_fuse

    assert ".mlp.shared_expert.gate_up_proj" in _PT_FP8_FUSE

    buf: dict = {}
    base = "model.layers.3.mlp.shared_expert"
    w = torch.zeros(512, 2048, dtype=torch.float8_e4m3fn)
    assert _pt_fp8_fuse(f"{base}.gate_proj", w, torch.tensor(0.25), None, buf) == []
    emit = _pt_fp8_fuse(f"{base}.up_proj", w, torch.tensor(0.5), None, buf)

    assert emit is not None and not buf
    keys = dict(emit)
    fused = f"{base}.gate_up_proj"
    assert keys[f"{fused}.weight"].shape == (1024, 2048)
    scale = keys[f"{fused}.weight_scale"]
    assert scale.shape == (1024,)
    assert torch.equal(scale[:512], torch.full((512,), 0.25))
    assert torch.equal(scale[512:], torch.full((512,), 0.5))


def test_the_dense_mlp_pair_does_not_capture_the_shared_expert():
    """``.mlp.gate_proj`` must not match ``.mlp.shared_expert.gate_proj``, or the two
    fusions would race for the same parts."""
    from freetoken.models.qwen3_5_moe.weight import _PT_FP8_FUSE

    shared_parts = _PT_FP8_FUSE[".mlp.shared_expert.gate_up_proj"]
    assert "model.layers.0.mlp.shared_expert.gate_proj".endswith(shared_parts[0])
    for fused, parts in _PT_FP8_FUSE.items():
        if fused == ".mlp.shared_expert.gate_up_proj":
            continue
        assert not any(
            "model.layers.0.mlp.shared_expert.gate_proj".endswith(p) for p in parts
        ), fused


# ============================ compressed-tensors mixed-precision ======================
# No quantized_layers map: the per-component split lives in each group's ``targets``.


def _ct_config(groups: dict) -> _Cfg:
    base = _hf_config(None)
    base.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": groups,
        "ignore": ["model.visual.blocks.0.attn.qkv"],
    }
    return base


_CT_FP8 = {"num_bits": 8, "type": "float", "group_size": None, "strategy": "tensor"}
_CT_NVFP4 = {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"}
_CT_SPLIT = {
    "group_0": {
        "weights": _CT_FP8,
        "targets": [
            r"re:.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"re:.*\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
            r"re:.*\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$",
        ],
    },
    "group_1": {
        "weights": _CT_NVFP4,
        "targets": [r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
    },
}


def test_ct_mixed_resolves_each_component_from_its_own_group():
    """Ornith-agentic: NVFP4 routed experts, per-tensor FP8 attention/GDN/shared expert."""
    cfg = parse_config(_ct_config(_CT_SPLIT))
    assert cfg.expert_quant == "nvfp4"
    assert cfg.dense_quant == "fp8_pertensor"
    assert cfg.attn_quant == "fp8_pertensor"
    assert cfg.linear_attn_out_quant == "fp8_pertensor"


def test_ct_single_group_targeting_every_linear_is_unchanged():
    """Muse-Glimmer / Qwen3.6-27B shape: one NVFP4 group over ``Linear``. Everything must
    stay NVFP4, exactly as before the per-component resolution."""
    cfg = parse_config(_ct_config({"group_0": {"weights": _CT_NVFP4, "targets": ["Linear"]}}))
    assert cfg.expert_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.attn_quant == "nvfp4"


def test_ct_component_probe_reads_the_targets():
    from freetoken.models.qwen3_5_moe.config import _ct_component_quant

    cfg = _ct_config(_CT_SPLIT)
    assert _ct_component_quant(cfg, "experts") == "nvfp4"
    assert _ct_component_quant(cfg, "attn") == "fp8_pertensor"
    assert _ct_component_quant(cfg, "shared_expert") == "fp8_pertensor"


def test_ct_component_probe_is_silent_when_nothing_matches():
    from freetoken.models.qwen3_5_moe.config import _ct_component_quant

    only_experts = {"group_0": {"weights": _CT_NVFP4, "targets": [r"re:.*\.mlp\.experts\..*"]}}
    cfg = _ct_config(only_experts)
    assert _ct_component_quant(cfg, "experts") == "nvfp4"
    assert _ct_component_quant(cfg, "attn") is None  # -> caller falls back


def test_shared_expert_helper_falls_back_to_ct_targets():
    """The modelopt map wins when present; otherwise the CT targets answer."""
    assert _shared_expert_quant(_ct_config(_CT_SPLIT)) == "fp8_pertensor"
    assert _shared_expert_quant(_hf_config("FP8")) == "fp8_pertensor"
