from __future__ import annotations

import collections
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from freetoken.utils import download_hf_weight
from tqdm import tqdm

LayerToBank = Callable[[int, object], int | None]
DropPageCache = Callable[[str], None]


@dataclass(frozen=True)
class Nvfp4ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str

_PACKED_KINDS = frozenset(("weight", "weight_packed"))
_GLOBAL_KINDS = frozenset(("weight_scale_2", "weight_global_scale"))


def _source_weight_map(model_path: str) -> dict[str, str]:
    """Tensor name -> absolute source shard, including unsharded checkpoints."""
    folder = download_hf_weight(model_path)
    index_path = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            return {
                name: os.path.join(folder, shard)
                for name, shard in json.load(f)["weight_map"].items()
            }

    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    files = [path for path in files if not path.endswith("consolidated.safetensors")] or files
    weight_map: dict[str, str] = {}
    for path in files:
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                weight_map[name] = path
    if not weight_map:
        raise ValueError(f"No safetensors weights found in {folder!r}")
    return weight_map


def _native_global_scale(kind: str, tensor: torch.Tensor) -> torch.Tensor:
    """ModelOpt globals are dequant scales; compressed-tensors stores their reciprocal."""
    if kind == "weight_global_scale":
        return (1.0 / tensor.reshape(1).to(torch.float32)).to(torch.float16)
    return tensor.to(torch.float16)


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def _bank_layer(spec: Nvfp4ExpertSourceSpec, layer: int, config) -> int | None:
    bank_layer = spec.layer_to_bank(layer, config)
    if bank_layer is None:
        return None
    num_layers = _num_moe_layers(config)
    if bank_layer < 0 or bank_layer >= num_layers:
        raise ValueError(
            f"{spec.desc}: bank layer {bank_layer} for checkpoint layer {layer} "
            f"is outside [0, {num_layers})"
        )
    return bank_layer


def _alloc_nvfp4_host_banks(num_layers: int, E: int, H: int, I: int):
    """6 NVFP4 source banks, one ``[E, ...]`` tensor per layer (independent allocations),
    unpinned (pin-after-fill): register only after fill to skip cudaHostAlloc's slow
    commit. Caller fills each layer's ``.tensor`` then pins it (per-layer, via
    ``PinPipeline``, as its writes complete)."""
    from freetoken.moe.host_banks import alloc_layer_banks

    fp8 = torch.float8_e4m3fn
    return alloc_layer_banks({
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), fp8),
        "gate_up_global": ((E, 2 * I), torch.float16),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), fp8),
        "down_global": ((E, H), torch.float16),
    }, num_layers)


def load_nvfp4_expert_source_banks(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Build the 6 native NVFP4 source banks by streaming checkpoint shards (serial per-shard read).

    Gate/up are fused on the output-row axis and down stays separate. ModelOpt stores
    ``weight`` + ``weight_scale`` + dequant ``weight_scale_2``; compressed-tensors stores
    ``weight_packed`` + ``weight_scale`` + quant-side ``weight_global_scale``, whose reciprocal
    is written into the per-output-row FP16 ``*_global`` banks. The marlin/b12x backends repack
    these source banks and fold globals into per-expert alphas.

    ``layer_sink=None`` (serving): pin each bank layer as its writes complete, via an
    internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter; for
    marlin/b12x the provider wraps it in a per-layer repacking sink first): the
    completion tracker fires into it instead -- nothing here is pinned, and the sink
    may release banks it has written out, so the returned tensors are only valid
    until then (the caller owns that tradeoff).
    """
    weight_map = _source_weight_map(model_path)

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    for path in sorted(set(weight_map.values())):
        drop_page_cache(path)

    weight_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    global_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    for name, path in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        bank_layer = _bank_layer(spec, layer, config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = match.group("kind")
        if kind in _GLOBAL_KINDS:
            global_shards[path].append((name, match, bank_layer))
        elif kind in _PACKED_KINDS or kind == "weight_scale":
            weight_shards[path].append((name, match, bank_layer))
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for path in sorted(global_shards):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name, match, _bank_layer_id in global_shards[path]:
                key = (
                    int(match.group("layer")),
                    int(match.group("expert")),
                    match.group("proj"),
                )
                globals_map[key] = _native_global_scale(
                    match.group("kind"), f.get_tensor(name)
                )
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, _hb, sink)
        placed = 0
        for path in tqdm(sorted(weight_shards), desc=f"Loading {spec.desc}", disable=not primary):
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, match, bank_layer_id in weight_shards[path]:
                    layer = int(match.group("layer"))
                    expert = int(match.group("expert"))
                    proj = match.group("proj")
                    role = spec.proj_to_role[proj]
                    kind = match.group("kind")
                    tensor = f.get_tensor(name)
                    if kind in _PACKED_KINDS:
                        if role == "gate":
                            gate_up_packed[bank_layer_id][expert, :I] = tensor
                        elif role == "up":
                            gate_up_packed[bank_layer_id][expert, I:] = tensor
                        elif role == "down":
                            down_packed[bank_layer_id][expert] = tensor
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    else:
                        global_scale = globals_map[(layer, expert, proj)]
                        if role == "gate":
                            gate_up_scale[bank_layer_id][expert, :I] = tensor
                            gate_up_global[bank_layer_id][expert, :I] = global_scale
                        elif role == "up":
                            gate_up_scale[bank_layer_id][expert, I:] = tensor
                            gate_up_global[bank_layer_id][expert, I:] = global_scale
                        elif role == "down":
                            down_scale[bank_layer_id][expert] = tensor
                            down_global[bank_layer_id][expert] = global_scale
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    tracker.note(bank_layer_id)
                    placed += 1
            drop_page_cache(path)
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


def load_nvfp4_expert_source_banks_parallel(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Parallel counterpart of :func:`load_nvfp4_expert_source_banks`.

    Packed weights/scales use chunked multi-threaded O_DIRECT reads; tiny ModelOpt or
    compressed-tensors global scales remain serial. ``layer_sink`` follows the serial path.
    """
    from freetoken.models.weight import iter_expert_tensors_parallel

    weight_map = _source_weight_map(model_path)

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    weight_info: dict[str, tuple[re.Match[str], int]] = {}  # name -> (match, bank_layer)
    global_names_by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name, path in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        kind = match.group("kind")
        if kind in _GLOBAL_KINDS:
            global_names_by_shard[path].append(name)
        elif kind in _PACKED_KINDS or kind == "weight_scale":
            weight_info[name] = (match, bank_layer)
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    # Pass 1: tiny per-tensor global scales (serial; data is scalar-per-expert).
    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for path in sorted(global_names_by_shard):
        drop_page_cache(path)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name in global_names_by_shard[path]:
                m = spec.key_pattern.match(name)
                globals_map[(int(m.group("layer")), int(m.group("expert")), m.group("proj"))] = (
                    _native_global_scale(m.group("kind"), f.get_tensor(name))
                )
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    # Pass 2: bulk packed weights/scales via the common parallel reader; place by name.
    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, _hb, sink)
        placed = 0
        for name, tensor in iter_expert_tensors_parallel(
            model_path, lambda n: n in weight_info, workers=workers, chunk=chunk
        ):
            match, bank_layer_id = weight_info[name]
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            proj = match.group("proj")
            role = spec.proj_to_role[proj]
            kind = match.group("kind")
            if kind in _PACKED_KINDS:
                if role == "gate":
                    gate_up_packed[bank_layer_id][expert, :I] = tensor
                elif role == "up":
                    gate_up_packed[bank_layer_id][expert, I:] = tensor
                else:
                    down_packed[bank_layer_id][expert] = tensor
            else:
                g = globals_map[(layer, expert, proj)]
                if role == "gate":
                    gate_up_scale[bank_layer_id][expert, :I] = tensor
                    gate_up_global[bank_layer_id][expert, :I] = g
                elif role == "up":
                    gate_up_scale[bank_layer_id][expert, I:] = tensor
                    gate_up_global[bank_layer_id][expert, I:] = g
                else:
                    down_scale[bank_layer_id][expert] = tensor
                    down_global[bank_layer_id][expert] = g
            tracker.note(bank_layer_id)
            placed += 1
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


__all__ = [
    "Nvfp4ExpertSourceSpec",
    "load_nvfp4_expert_source_banks",
    "load_nvfp4_expert_source_banks_parallel",
]
