from __future__ import annotations

from types import SimpleNamespace

import safetensors.torch
import torch

from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks
from freetoken.models.qwen3_5_moe.config import parse_config
from freetoken.models.qwen3_5_moe.weight import _NVFP4_SOURCE_SPEC, _ct_nvfp4_fuse


def _config(*, ignored: list[str] | None = None) -> SimpleNamespace:
    text = SimpleNamespace(
        head_dim=16,
        hidden_size=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        max_position_embeddings=128,
        num_hidden_layers=2,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        norm_topk_prob=False,
        intermediate_size=16,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        vocab_size=32,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
    )
    return SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        model_type="qwen3_5_moe",
        text_config=text,
        quantization_config={
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                        "strategy": "tensor_group",
                    }
                }
            },
            "ignore": ignored or [],
        },
    )


def test_compressed_moe_config_selects_native_experts_and_projection_override():
    kat = parse_config(_config())
    frontis = parse_config(
        _config(ignored=["model.language_model.layers.0.linear_attn.out_proj"])
    )

    assert kat.expert_quant == frontis.expert_quant == "nvfp4"
    assert kat.attn_quant == frontis.attn_quant == "nvfp4"
    assert kat.linear_attn_out_quant == "nvfp4"
    assert frontis.linear_attn_out_quant == "none"


def test_compressed_shared_expert_fuses_gate_and_up_native_nvfp4():
    part = (
        torch.zeros((2, 1), dtype=torch.uint8),
        torch.zeros((2, 1), dtype=torch.float8_e4m3fn),
        torch.ones(2, dtype=torch.float16),
    )
    buf: dict = {}

    assert _ct_nvfp4_fuse("model.layers.0.mlp.shared_expert.gate_proj", part, buf) == []
    emitted = _ct_nvfp4_fuse("model.layers.0.mlp.shared_expert.up_proj", part, buf)

    assert [name for name, _ in emitted] == [
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight_scale",
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight_global",
    ]
    assert not buf


def test_single_file_compressed_experts_load_native_banks(tmp_path):
    base = "model.language_model.layers.0.mlp.experts.0"
    tensors: dict[str, torch.Tensor] = {}
    for projection, packed_value, scale_value, global_value in (
        ("gate_proj", 11, 1.0, 8.0),
        ("up_proj", 22, 2.0, 4.0),
        ("down_proj", 33, 3.0, 2.0),
    ):
        tensors[f"{base}.{projection}.weight_packed"] = torch.full(
            (16, 8), packed_value, dtype=torch.uint8
        )
        tensors[f"{base}.{projection}.weight_scale"] = torch.full(
            (16, 1), scale_value, dtype=torch.float32
        ).to(torch.float8_e4m3fn)
        tensors[f"{base}.{projection}.weight_global_scale"] = torch.tensor(
            [global_value], dtype=torch.float32
        )
        tensors[f"{base}.{projection}.input_global_scale"] = torch.tensor(
            [99.0], dtype=torch.float32
        )
    safetensors.torch.save_file(tensors, str(tmp_path / "model.safetensors"))

    config = SimpleNamespace(
        num_moe_layers=1,
        num_experts=1,
        hidden_size=16,
        moe_intermediate_size=16,
    )
    captured: dict[int, dict[str, torch.Tensor]] = {}

    def sink(layer_id, layer_banks):
        captured[layer_id] = {name: bank.tensor.clone() for name, bank in layer_banks.items()}
        for bank in layer_banks.values():
            bank.release()

    load_nvfp4_expert_source_banks(
        str(tmp_path),
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=lambda _path: None,
        primary=False,
        layer_sink=sink,
    )

    banks = captured[0]
    assert torch.equal(
        banks["gate_up_packed"][0, :16], torch.full((16, 8), 11, dtype=torch.uint8)
    )
    assert torch.equal(
        banks["gate_up_packed"][0, 16:], torch.full((16, 8), 22, dtype=torch.uint8)
    )
    assert torch.equal(
        banks["down_packed"][0], torch.full((16, 8), 33, dtype=torch.uint8)
    )
    assert torch.all(banks["gate_up_global"][0, :16] == 0.125)
    assert torch.all(banks["gate_up_global"][0, 16:] == 0.25)
    assert torch.all(banks["down_global"][0] == 0.5)
