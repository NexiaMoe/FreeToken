"""GLM-4.7-Flash (``glm4_moe_lite``) checkpoint reader.

Two disjoint passes:

* :func:`iter_weights` -- the resident weights. MLA (q_a/q_b/kv_a/kv_b/o + the two
  layernorms), the router, norms, embeddings and ``lm_head`` are BF16 and stream through
  verbatim; the leading dense MLP and every layer's shared expert are native NVFP4
  (``weight_packed`` + fp8 block ``weight_scale`` + scalar ``weight_global_scale``) and
  are kept packed for the W4A16 dense kernels. Unlike GLM-5.2 nothing is requantized at
  load. No projection is fused: ``GlmDsaGatedMLP`` keeps ``gate_proj``/``up_proj``
  separate, so every tensor maps 1:1.
* :func:`load_nvfp4_expert_sources` / ``_parallel`` -- the routed-expert host banks,
  built by the shared NVFP4 bank loader. These never enter the dense iterator.

The trailing MTP layer (``num_nextn_predict_layers``) is skipped wholesale: it is a
self-contained block (its own ``embed_tokens``, ``enorm``/``hnorm``, ``eh_proj``,
attention, experts and ``shared_head``) that this server does not run.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    CT_SCALE_SUFFIXES,
    ShardReader,
    drop_page_cache,
    nvfp4_parts_ct,
)
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")

# compressed-tensors spellings (``weight_packed``/``weight_global_scale``). The shared
# bank loader also accepts the modelopt pair, but this family only ships the CT one.
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_packed|weight_scale|weight_global_scale)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    # Experts live on layers [first_k_dense_replace, num_layers); pack by MoE-layer index
    # so the banks have no hole for the leading dense layer. ``layer >= num_layers``
    # drops the MTP layer, which carries a full expert set of its own.
    layer_to_bank=lambda layer, config: (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    ),
    desc="GLM-4.7-Flash NVFP4 experts",
)


def _mtp_prefix(config: ModelConfig) -> str:
    return f"model.layers.{config.num_layers}."


def _rename(raw_name: str) -> str | None:
    """Checkpoint name -> model buffer name; ``None`` drops the tensor."""
    # HF parks the aux-loss-free selection bias under the router Linear; FreeToken's
    # sparse block owns it directly (the experts are an offload-cache handle).
    if raw_name.endswith(".mlp.gate.e_score_correction_bias"):
        return raw_name.replace(
            ".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias"
        )
    return raw_name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("glm4_moe_lite weight loading currently supports TP=1 only")
    assert not include_moe_experts, (
        "glm4_moe_lite stores routed experts as NVFP4 and only supports the offload "
        "backend; they load through load_nvfp4_expert_sources, not this iterator"
    )
    if not include_non_moe:
        return

    from freetoken.utils import cached_load_hf_config

    from .config import parse_config

    config = parse_config(cached_load_hf_config(model_path))
    mtp_prefix = _mtp_prefix(config)
    tp_info = get_tp_info()

    reader = ShardReader(model_path, device)
    try:
        for file in tqdm(
            reader.files(),
            desc="Loading GLM-4.7-Flash weights",
            disable=not tp_info.is_primary(),
        ):
            for raw_name in reader.names_in(file):
                if raw_name.startswith(mtp_prefix):
                    continue  # MTP block: not served
                if _ROUTED_EXPERT_RE.search(raw_name):
                    continue  # offload-cache banks
                if raw_name.endswith(CT_SCALE_SUFFIXES):
                    continue  # consumed with weight_packed (or unused W4A4 input scales)

                name = _rename(raw_name)
                if name is None:
                    continue

                if raw_name.endswith(".weight_packed"):  # native NVFP4 dense projection
                    base = name[: -len(".weight_packed")]
                    w, s, g = nvfp4_parts_ct(reader, raw_name[: -len(".weight_packed")])
                    yield base + ".weight", w
                    yield base + ".weight_scale", s
                    yield base + ".weight_global", g
                    continue

                yield name, reader.get_tensor(raw_name)
    finally:
        reader.close()


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
    """Same banks via the common chunked multi-threaded O_DIRECT reader."""
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
