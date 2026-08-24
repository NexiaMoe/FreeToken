"""Laguna checkpoint reader.

Two disjoint passes, matching how the engine loads a MoE model:

* :func:`iter_weights` -- the resident (dense) weights. Attention (q/k/v/o + the per-head
  gate), the leading dense layer's MLP, the router, norms, embeddings and lm_head are BF16
  in every Laguna checkpoint; each MoE layer's shared expert is native NVFP4
  (``weight_packed`` + fp8 block ``weight_scale`` + scalar ``weight_global_scale``) and is
  kept that way for the W4A16 dense kernels.
* :func:`load_nvfp4_expert_sources` / :func:`load_nvfp4_expert_sources_parallel` -- the
  routed-expert host banks for the offload cache, built by the shared NVFP4 bank loader.
  These tensors never enter the dense iterator.

The FP8 KV-cache calibration scalars (``self_attn.{k,v}_scale``) are dropped: FreeToken's
KV pools are BF16, which stores K/V at strictly higher precision than the scheme they
describe.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    CT_SCALE_SUFFIXES,
    ShardReader,
    ct_bf16_fuse,
    ct_nvfp4_fuse,
    drop_page_cache,
    nvfp4_parts_ct,
)
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Routed experts: ``model.layers.<L>.mlp.experts.<E>.<proj>.<kind>``. Matched here to keep
# them out of the dense pass; the bank loader matches them with the spec pattern below.
_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")

_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_packed|weight_scale|weight_global_scale)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    # Experts exist only on layers [first_k_dense_replace, num_layers); pack by MoE-layer
    # index so the banks have no holes for the leading dense layer.
    layer_to_bank=lambda layer, config: (
        layer - config.first_k_dense_replace
        if layer >= config.first_k_dense_replace
        else None
    ),
    desc="Laguna NVFP4 experts",
)

# Unused per-layer FP8 KV-cache scalars (see module docstring).
_DROP_SUFFIXES = (".self_attn.k_scale", ".self_attn.v_scale")

# fused model buffer suffix -> ordered checkpoint part suffixes, matched on the base name
# (no trailing ``.weight`` / ``.weight_packed``). ``.mlp.gate_proj`` only matches the dense
# layer-0 MLP: the shared expert's base ends ``.shared_expert.gate_proj``.
_BF16_FUSE: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
}
_NVFP4_FUSE: dict[str, tuple[str, ...]] = {
    ".mlp.shared_expert.gate_up_proj": (
        ".mlp.shared_expert.gate_proj",
        ".mlp.shared_expert.up_proj",
    ),
}


def _rename(raw_name: str) -> str | None:
    """Checkpoint name -> model buffer name; ``None`` drops the tensor."""
    if raw_name.endswith(_DROP_SUFFIXES):
        return None
    # HF keeps the aux-loss-free selection bias under the experts module; FreeToken's
    # sparse block owns it directly (the experts are an offload-cache handle, not a module).
    if raw_name.endswith(".mlp.experts.e_score_correction_bias"):
        return raw_name.replace(
            ".mlp.experts.e_score_correction_bias", ".mlp.e_score_correction_bias"
        )
    return raw_name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Resident Laguna weights. Routed experts are excluded unconditionally -- they are
    served from the offload cache's NVFP4 banks, never as resident tensors."""
    if get_tp_info().size > 1:
        raise NotImplementedError("laguna weight loading currently supports TP=1 only")
    if not include_non_moe:
        return

    tp_info = get_tp_info()
    bf16_buf: dict[str, dict[int, torch.Tensor]] = {}
    nvfp4_buf: dict[str, dict[int, tuple]] = {}

    # Scale lookups go through the shard-map reader: a ``weight_packed`` and its scales can
    # land in different shards.
    reader = ShardReader(model_path, device)
    try:
        for file in tqdm(
            reader.files(),
            desc="Loading Laguna weights",
            disable=not tp_info.is_primary(),
        ):
            for raw_name in reader.names_in(file):
                if _ROUTED_EXPERT_RE.search(raw_name):
                    continue
                if raw_name.endswith(CT_SCALE_SUFFIXES):
                    continue  # consumed with weight_packed (or unused W4A4 input scales)
                name = _rename(raw_name)
                if name is None:
                    continue

                if raw_name.endswith(".weight_packed"):  # native NVFP4 (shared expert)
                    base = name[: -len(".weight_packed")]
                    parts = nvfp4_parts_ct(reader, raw_name[: -len(".weight_packed")])
                    emit = ct_nvfp4_fuse(base, parts, nvfp4_buf, _NVFP4_FUSE)
                    if emit is not None:
                        yield from emit
                    else:  # standalone: shared_expert.down_proj
                        w, s, g = parts
                        yield base + ".weight", w
                        yield base + ".weight_scale", s
                        yield base + ".weight_global", g
                    continue

                tensor = reader.get_tensor(raw_name)
                if name.endswith(".weight"):
                    emit = ct_bf16_fuse(name[: -len(".weight")], tensor, bf16_buf, _BF16_FUSE)
                    if emit is not None:
                        yield from emit
                        continue
                yield name, tensor
    finally:
        reader.close()

    assert not bf16_buf, f"Laguna: incomplete bf16 fusions: {list(bf16_buf.keys())}"
    assert not nvfp4_buf, f"Laguna: incomplete NVFP4 fusions: {list(nvfp4_buf.keys())}"


def load_nvfp4_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Pinned CPU NVFP4 banks for the routed experts (serial per-shard read)."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks

    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str,
    config: ModelConfig,
    *,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Same banks via the common chunked multi-threaded O_DIRECT reader. Laguna stores one
    tensor per (expert, projection) -- 119,808 of them -- so the parallel path matters."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
