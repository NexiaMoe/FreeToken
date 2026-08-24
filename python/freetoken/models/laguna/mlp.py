from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearRowParallel,
    silu_and_mul,
)
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class LagunaGatedMLP(BaseOP):
    """SwiGLU MLP with a fused ``gate_up_proj``.

    Serves two roles with different precisions in the same checkpoint, hence the explicit
    ``quant`` flag rather than a config lookup: every MoE layer's always-on shared expert is
    NVFP4 (W4A16, like the routed experts), while the leading dense layer's MLP is in the
    checkpoint's quantization ``ignore`` list and stays BF16.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, *, quant: str = "none"):
        if quant == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear

            self.gate_up_proj = Nvfp4DenseColMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = Nvfp4DenseLinear(intermediate_size, hidden_size, has_bias=False)
        else:
            self.gate_up_proj = LinearColParallelMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


__all__ = ["LagunaGatedMLP"]
