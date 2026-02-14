from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_min_bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to 3-bit precision along the last dimension.
    Always quantize group_size value together and store their absolute value first.
    To keep things simple, we require x to be a 1D tensor, and the size divisible by group_size.
    Return the quantized tensor and scaling factor.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)

    # Quantize to 3 bits (0-7)
    # Range is [0, 1], mapped to [0, 7]
    x_quant_val = (x_norm * 7).round().clamp(0, 7).to(torch.uint8)

    # Pack 8 values into 3 bytes
    # Input x_quant_val is flattened list of values. 
    # Since group_size=16 (divisible by 8), we can reshape freely.
    N = x_quant_val.numel()
    assert N % 8 == 0
    x_reshaped = x_quant_val.view(-1, 8)

    # Use int32 for bitwise ops to avoid overflow
    cols = [x_reshaped[:, i].to(torch.int32) for i in range(8)]

    # Byte 0: v0(3), v1(3), v2_hi(2)
    b0 = (cols[0] << 5) | (cols[1] << 2) | (cols[2] >> 1)

    # Byte 1: v2_lo(1), v3(3), v4(3), v5_hi(1)
    b1 = ((cols[2] & 1) << 7) | (cols[3] << 4) | (cols[4] << 1) | (cols[5] >> 2)

    # Byte 2: v5_lo(2), v6(3), v7(3)
    b2 = ((cols[5] & 3) << 6) | (cols[6] << 3) | cols[7]

    packed = torch.stack([b0, b1, b2], dim=1).to(torch.uint8)

    # Reshape to (groups, bytes_per_group)
    # 16 values -> 6 bytes
    bytes_per_group = (group_size * 3) // 8

    return packed.view(-1, bytes_per_group), normalization.to(torch.float16)


def block_dequantize_min_bit(x_quant_packed: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """
    The reverse operation of block_quantize_min_bit.
    """
    assert x_quant_packed.dim() == 2

    normalization = normalization.to(torch.float32)

    # Infer group size
    # size * 8 / 3 = group_size
    packed_width = x_quant_packed.shape[1]
    group_size = (packed_width * 8) // 3

    # Unpack 3 bytes -> 8 values
    packed_view = x_quant_packed.view(-1, 3)
    b = [packed_view[:, i].to(torch.int32) for i in range(3)]

    # v0: b0 >> 5
    v0 = (b[0] >> 5) & 7
    v1 = (b[0] >> 2) & 7
    v2 = ((b[0] & 3) << 1) | ((b[1] >> 7) & 1)
    v3 = (b[1] >> 4) & 7
    v4 = (b[1] >> 1) & 7
    v5 = ((b[1] & 1) << 2) | ((b[2] >> 6) & 3)
    v6 = (b[2] >> 3) & 7
    v7 = b[2] & 7

    unpacked = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1)

    # Reshape and scale
    x_quant = unpacked.view(-1, group_size).to(torch.float32)
    x_norm = x_quant / 7.0
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class LinearMinimumBit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        # Let's store all the required information to load the weights from a checkpoint
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # self.register_buffer is used to store the weights in the model, but not as parameters
        # This makes sure weights are put on the correct device when calling `model.to(device)`.
        # persistent=False makes sure the buffer is not saved or loaded. The bignet has a parameters
        # called "weight" that we need to quantize when the model is loaded.
        self.register_buffer(
            "weight_q3",
            torch.zeros(out_features * in_features // group_size, (group_size * 3) // 8, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        # Register a hook to load the weights from a checkpoint. This function reaches deep into
        # PyTorch internals. It makes sure that LinearMinimumBit._load_state_dict_pre_hook is called
        # every time the model is loaded from a checkpoint. We will quantize the weights in that function.
        self._register_load_state_dict_pre_hook(LinearMinimumBit._load_state_dict_pre_hook, with_module=True)
        # Add in an optional bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            # Load the original weights and remove them from the state_dict (mark them as loaded)
            weight = state_dict[f"{prefix}weight"]  # noqa: F841
            del state_dict[f"{prefix}weight"]
            # Quantize the weights and store them in self.weight_q3 and self.weight_norm
            q3, norm = block_quantize_min_bit(weight.flatten(), self._group_size)
            self.weight_q3.copy_(q3)
            self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Dequantize and call the layer
            weight_dequant = block_dequantize_min_bit(self.weight_q3, self.weight_norm)
            # unflatten weight dequant to (out_features, in_features)
            weight_dequant = weight_dequant.view(self._shape)
            return torch.nn.functional.linear(x, weight_dequant, self.bias)


class BigNetMinBit(torch.nn.Module):
    """
    A BigNet where all weights are in min_bit precision. Use the LinearMinimumBit module for this.
    It is fine to keep all computation in float32.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                LinearMinimumBit(channels, channels),
                torch.nn.ReLU(),
                LinearMinimumBit(channels, channels),
                torch.nn.ReLU(),
                LinearMinimumBit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNetMinBit:
    net = BigNetMinBit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
