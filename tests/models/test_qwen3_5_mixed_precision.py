"""modelopt MIXED_PRECISION Qwen3.5 derivatives whose shared expert disagrees with the
routed experts.

``dense_quant`` used to be inferred as "whatever the routed experts are", which holds for
every NVFP4 export seen so far. apodex/Apodex-1.1-mini-NVFP4 breaks it: its
``quantized_layers`` map lists ``.mlp.experts`` as NVFP4 and every
``.mlp.shared_expert.*`` entry as FP8. Inferring FP4 there builds an
``Nvfp4DenseColMerged`` whose ``gate_up_proj.weight`` the loader can never fill --
KeyError at load, long after the checkpoint has been read.
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
