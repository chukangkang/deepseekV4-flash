import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, fp4_act_quant, int4_act_quant, fp8_gemm, fp4_gemm, int4_gemm, sparse_attn, hc_split_sinkhorn


world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32
expert_int4 = False
turbo_quant_enabled = False
turbo_quant_bits = 4
tp_group = None


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4", "int4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.
    swiglu_limit: float = 0.
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # TurboQuant: training-free low-bit KV cache compression
    turbo_quant: bool = False
    turbo_quant_bits: int = 4


class DynamicKVCache:
    def __init__(self, max_batch_size: int, head_dim: int, min_capacity: int = 1, growth_factor: int = 2, block_len: int = 256):
        self.max_batch_size = max_batch_size
        self.head_dim = head_dim
        self.min_capacity = max(1, min_capacity)
        self.growth_factor = max(2, growth_factor)
        self.block_len = max(1, block_len)
        self.storage: Optional[torch.Tensor] = None
        self.flat_storage: Optional[torch.Tensor] = None
        self.capacity = 0
        self.num_blocks = 0

    def _round_capacity(self, length: int) -> int:
        return ((max(1, length) + self.block_len - 1) // self.block_len) * self.block_len

    def _blocks_for(self, length: int) -> int:
        return max(1, (max(1, length) + self.block_len - 1) // self.block_len)

    def _next_capacity(self, required_len: int) -> int:
        capacity = self._round_capacity(max(self.min_capacity, self.capacity, self.block_len))
        while capacity < required_len:
            capacity = self._round_capacity(max(capacity * self.growth_factor, required_len))
        return capacity

    def logical_block_table(self, required_len: int) -> torch.Tensor:
        needed_blocks = self._blocks_for(required_len)
        return torch.arange(needed_blocks, dtype=torch.int32, device=self.storage.device if self.storage is not None else None)

    def logical_view(self, length: Optional[int] = None) -> Optional[torch.Tensor]:
        if self.flat_storage is None:
            return None
        if length is None:
            return self.flat_storage
        return self.flat_storage[:, :length]

    def ensure(self, batch_size: int, required_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        batch_capacity = max(self.max_batch_size, batch_size)
        required_len = max(1, required_len)
        should_realloc = (
            self.storage is None or
            self.storage.device != device or
            self.storage.dtype != dtype or
            self.storage.size(0) < batch_capacity or
            self.capacity < required_len
        )
        if not should_realloc:
            return self.flat_storage
        new_capacity = self._next_capacity(required_len)
        new_num_blocks = self._blocks_for(new_capacity)
        new_storage = torch.zeros(batch_capacity, new_num_blocks, self.block_len, self.head_dim, dtype=dtype, device=device)
        new_flat_storage = new_storage.view(batch_capacity, new_num_blocks * self.block_len, self.head_dim)
        if self.flat_storage is not None and self.storage is not None and self.storage.device == device and self.storage.dtype == dtype:
            copy_batch = min(self.flat_storage.size(0), new_flat_storage.size(0))
            copy_len = min(self.flat_storage.size(1), new_flat_storage.size(1))
            new_flat_storage[:copy_batch, :copy_len].copy_(self.flat_storage[:copy_batch, :copy_len])
        self.storage = new_storage
        self.flat_storage = new_flat_storage
        self.capacity = new_capacity
        self.num_blocks = new_num_blocks
        return self.flat_storage

    def reset(self, release: bool = False):
        if release:
            self.storage = None
            self.flat_storage = None
            self.capacity = 0
            self.num_blocks = 0


class TurboQuantKVCache:
    """TurboQuant-inspired low-bit KV cache compression.
    Quantizes KV entries to N-bit (default 4) per-token asymmetric quantization on write,
    dequantizes to BF16 on read. Reduces KV memory by ~4x (BF16→4-bit) or ~5.3x (BF16→3-bit).

    Storage layout:
      quantized: [batch, capacity, head_dim] in uint8 (packed: 2 values per byte for 4-bit, 8 values per 3 bytes for 3-bit)
      scales:    [batch, capacity] in float16  (per-token max abs value)
      zeros:     [batch, capacity] in float16  (per-token min value)
    """

    def __init__(self, max_batch_size: int, head_dim: int, bits: int = 4,
                 min_capacity: int = 1, growth_factor: int = 2, block_len: int = 256):
        self.max_batch_size = max_batch_size
        self.head_dim = head_dim
        self.bits = bits
        self.n_levels = (1 << bits) - 1  # 15 for 4-bit, 7 for 3-bit
        self.min_capacity = max(1, min_capacity)
        self.growth_factor = max(2, growth_factor)
        self.block_len = max(1, block_len)
        # For 4-bit: pack 2 per byte → head_dim//2 bytes per token
        # For 3-bit: pack 8 per 3 bytes → head_dim*3//8 bytes per token
        if bits == 4:
            self.packed_dim = head_dim // 2
        elif bits == 3:
            assert head_dim % 8 == 0, "head_dim must be divisible by 8 for 3-bit packing"
            self.packed_dim = head_dim * 3 // 8
        else:
            raise ValueError(f"Unsupported bits={bits}, must be 3 or 4")
        self.quantized: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self.zeros: Optional[torch.Tensor] = None
        self.capacity = 0
        # BF16 view returned to caller (lazily allocated)
        self._bf16_view: Optional[torch.Tensor] = None
        self._bf16_capacity = 0

    def _round_capacity(self, length: int) -> int:
        return ((max(1, length) + self.block_len - 1) // self.block_len) * self.block_len

    def _next_capacity(self, required_len: int) -> int:
        capacity = self._round_capacity(max(self.min_capacity, self.capacity, self.block_len))
        while capacity < required_len:
            capacity = self._round_capacity(max(capacity * self.growth_factor, required_len))
        return capacity

    def ensure(self, batch_size: int, required_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Allocates quantized storage and returns a BF16 view tensor for the caller.
        The BF16 view is a scratch buffer — writes to it are quantized via write_slice()."""
        batch_cap = max(self.max_batch_size, batch_size)
        required_len = max(1, required_len)
        need_realloc = (
            self.quantized is None or
            self.quantized.device != device or
            self.quantized.size(0) < batch_cap or
            self.capacity < required_len
        )
        if need_realloc:
            new_cap = self._next_capacity(required_len)
            new_q = torch.zeros(batch_cap, new_cap, self.packed_dim, dtype=torch.uint8, device=device)
            new_s = torch.ones(batch_cap, new_cap, dtype=torch.float16, device=device)
            new_z = torch.zeros(batch_cap, new_cap, dtype=torch.float16, device=device)
            if self.quantized is not None and self.quantized.device == device:
                cb = min(self.quantized.size(0), batch_cap)
                cl = min(self.capacity, new_cap)
                new_q[:cb, :cl].copy_(self.quantized[:cb, :cl])
                new_s[:cb, :cl].copy_(self.scales[:cb, :cl])
                new_z[:cb, :cl].copy_(self.zeros[:cb, :cl])
            self.quantized = new_q
            self.scales = new_s
            self.zeros = new_z
            self.capacity = new_cap
        # Ensure BF16 view is large enough
        if self._bf16_view is None or self._bf16_view.size(0) < batch_cap or self._bf16_capacity < required_len or self._bf16_view.device != device:
            self._bf16_view = torch.zeros(batch_cap, self.capacity, self.head_dim, dtype=dtype, device=device)
            self._bf16_capacity = self.capacity
        return self._bf16_view

    def _quantize_4bit(self, x: torch.Tensor) -> tuple:
        """x: [..., head_dim] → packed uint8 [..., head_dim//2], scale, zero"""
        xf = x.float()
        vmin = xf.amin(dim=-1, keepdim=True)
        vmax = xf.amax(dim=-1, keepdim=True)
        scale = (vmax - vmin).clamp(min=1e-12) / self.n_levels
        qi = torch.round((xf - vmin) / scale).clamp(0, self.n_levels).to(torch.uint8)
        # Pack 2 per byte: low nibble = even, high nibble = odd
        packed = qi[..., 0::2] | (qi[..., 1::2] << 4)
        return packed, scale.squeeze(-1).half(), vmin.squeeze(-1).half()

    def _dequantize_4bit(self, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Unpack uint8 → 4-bit → float → apply scale+zero"""
        lo = (packed & 0x0F).to(torch.float32)
        hi = ((packed >> 4) & 0x0F).to(torch.float32)
        vals = torch.stack([lo, hi], dim=-1).flatten(-2)  # [..., head_dim]
        return (vals * scale.unsqueeze(-1).float() + zero.unsqueeze(-1).float()).to(dtype)

    def _quantize_3bit(self, x: torch.Tensor) -> tuple:
        """x: [..., head_dim] → packed uint8 [..., head_dim*3//8], scale, zero"""
        xf = x.float()
        vmin = xf.amin(dim=-1, keepdim=True)
        vmax = xf.amax(dim=-1, keepdim=True)
        scale = (vmax - vmin).clamp(min=1e-12) / self.n_levels
        qi = torch.round((xf - vmin) / scale).clamp(0, self.n_levels).to(torch.uint8)
        # Pack 8 values into 3 bytes: bits [b0*3..b0*3+2] → byte positions
        shape = qi.shape[:-1]
        qi = qi.view(*shape, -1, 8)  # [..., groups, 8]
        b = qi.to(torch.uint32)
        byte0 = (b[..., 0]) | (b[..., 1] << 3) | ((b[..., 2] & 0x3) << 6)
        byte1 = (b[..., 2] >> 2) | (b[..., 3] << 1) | (b[..., 4] << 4) | ((b[..., 5] & 0x1) << 7)
        byte2 = (b[..., 5] >> 1) | (b[..., 6] << 2) | (b[..., 7] << 5)
        packed = torch.stack([byte0, byte1, byte2], dim=-1).to(torch.uint8).flatten(-2)
        return packed, scale.squeeze(-1).half(), vmin.squeeze(-1).half()

    def _dequantize_3bit(self, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Unpack 3-bit packed uint8 → float"""
        shape = packed.shape[:-1]
        packed = packed.view(*shape, -1, 3).to(torch.uint32)  # [..., groups, 3]
        b0, b1, b2 = packed[..., 0], packed[..., 1], packed[..., 2]
        v0 = b0 & 0x7
        v1 = (b0 >> 3) & 0x7
        v2 = ((b0 >> 6) | (b1 << 2)) & 0x7
        v3 = (b1 >> 1) & 0x7
        v4 = (b1 >> 4) & 0x7
        v5 = ((b1 >> 7) | (b2 << 1)) & 0x7
        v6 = (b2 >> 2) & 0x7
        v7 = (b2 >> 5) & 0x7
        vals = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).flatten(-2).float()
        return (vals * scale.unsqueeze(-1).float() + zero.unsqueeze(-1).float()).to(dtype)

    def write_slice(self, batch_slice, pos_slice, data: torch.Tensor):
        """Quantize and store data at the given position."""
        if self.bits == 4:
            packed, s, z = self._quantize_4bit(data)
        else:
            packed, s, z = self._quantize_3bit(data)
        self.quantized[batch_slice, pos_slice] = packed
        self.scales[batch_slice, pos_slice] = s
        self.zeros[batch_slice, pos_slice] = z

    def read_slice(self, batch_slice, pos_slice, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Dequantize and return data from the given position."""
        packed = self.quantized[batch_slice, pos_slice]
        s = self.scales[batch_slice, pos_slice]
        z = self.zeros[batch_slice, pos_slice]
        if self.bits == 4:
            return self._dequantize_4bit(packed, s, z, dtype)
        else:
            return self._dequantize_3bit(packed, s, z, dtype)

    def reset(self, release: bool = False):
        if release:
            self.quantized = None
            self.scales = None
            self.zeros = None
            self._bf16_view = None
            self.capacity = 0
            self._bf16_capacity = 0


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each rank holds vocab_size // world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y, group=tp_group)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatches to int4_gemm / fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    For quantized weights, x is first quantized to FP8 via act_quant."""
    assert bias is None

    if weight.dtype == torch.uint8 and hasattr(weight, 'scale'):
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return int4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        return F.linear(x, weight)


class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or default_dtype
        if dtype == torch.uint8:
            # INT4: weight is [out, in//2] in uint8 (2 signed INT4 per byte)
            # Scale is [out, in//32] in float32
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.uint8), requires_grad=False)
            scale_out_features = out_features
            scale_in_features = in_features // fp4_block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32), requires_grad=False)
        elif dtype == torch.float4_e2m1fn_x2:
            # FP4: weight is [out, in//2] in float4_e2m1fn_x2, logically [out, in] in fp4
            # Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))
            scale_out_features = out_features
            scale_in_features = in_features // fp4_block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            y = y.float()
            dist.all_reduce(y, group=tp_group)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling.
    When original_seq_len > 0, applies frequency interpolation with a smooth
    linear ramp between beta_fast and beta_slow correction ranges."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


@lru_cache(maxsize=32)
def get_freqs_cis_cached(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device_type: str, device_index: int) -> torch.Tensor:
    freqs_cis = precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow)
    device = torch.device(device_type, device_index) if device_index >= 0 else torch.device(device_type)
    return freqs_cis.to(device=device)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if freqs_cis.ndim == 2:
        # Shared freqs: [seqlen, rd//2] → add batch dim
        if x.ndim == 3:
            freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
        else:
            freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    else:
        # Per-item freqs: [bsz, seqlen, rd//2] → add head dim for 4D x
        if x.ndim == 4:
            freqs_cis = freqs_cis.unsqueeze(2)
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Applies randomized Hadamard rotation to spread information across dims before FP8 quant."""
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size),  torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def get_window_topk_idxs_per_item(window_size: int, start_pos: torch.Tensor):
    """Per-item window indices for decode (seqlen=1). start_pos: [bsz] tensor.
    Returns [bsz, 1, win] with per-item circular window positions."""
    bsz = start_pos.size(0)
    win = window_size
    device = start_pos.device
    sp_mod = start_pos % win  # [bsz]
    idx = torch.arange(win, device=device).unsqueeze(0)  # [1, win]
    # Wrap-around: [(sp%w+1)%w, (sp%w+2)%w, ..., sp%w]
    result = (sp_mod.unsqueeze(1) + 1 + idx) % win  # [bsz, win]
    # For partial windows (start_pos < win - 1): [0, 1, ..., sp, -1, ...]
    partial = start_pos < (win - 1)  # [bsz]
    if partial.any():
        linear_idx = idx.expand(bsz, -1)
        partial_result = torch.where(
            linear_idx <= start_pos.unsqueeze(1),
            linear_idx,
            torch.full((1,), -1, device=device, dtype=linear_idx.dtype),
        )
        result = torch.where(partial.unsqueeze(1), partial_result, result)
    return result.unsqueeze(1)  # [bsz, 1, win]


def get_compress_topk_idxs_per_item(ratio: int, start_pos: torch.Tensor, offset: int):
    """Per-item compress indices for decode (seqlen=1). start_pos: [bsz] tensor.
    Returns [bsz, 1, max_compressed] with per-item compressed KV positions."""
    device = start_pos.device
    max_compressed = ((start_pos.max() + 1) // ratio).item()
    if max_compressed == 0:
        return torch.full((start_pos.size(0), 1, 0), -1, device=device, dtype=torch.long)
    idx = torch.arange(max_compressed, device=device).unsqueeze(0)  # [1, max_compressed]
    per_item_len = (start_pos + 1) // ratio  # [bsz]
    result = torch.where(
        idx < per_item_len.unsqueeze(1),
        idx + offset,
        torch.full((1,), -1, device=device, dtype=idx.dtype),
    )
    return result.unsqueeze(1)  # [bsz, 1, max_compressed]


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.
    When overlap=True (ratio==4), uses overlapping windows for smoother compression boundaries."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        # wkv and wgate in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        # When overlap, the first half of dims is for overlapping compression, second half for normal.
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache
        # State buffers for decode-phase incremental compression.
        # With overlap: state[:, :ratio] = overlapping window, state[:, ratio:] = current window.
        self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis: torch.Tensor = None

    def reset_cache(self):
        self.kv_cache = None
        self.freqs_cis = None
        self.kv_state.zero_()
        self.score_state.fill_(float("-inf"))

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        per_item = isinstance(start_pos, torch.Tensor)
        # compression need fp32
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if not per_item and start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            if not should_compress:
                return
            kv = self.norm(kv.to(dtype))
            freqs_cis = self.freqs_cis[:cutoff:ratio]
            apply_rotary_emb(kv[..., -rd:], freqs_cis)
            if self.rotate:
                kv = rotate_activation(kv)
                (int4_act_quant if expert_int4 else fp4_act_quant)(kv, fp4_block_size, True)
            else:
                act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
            self.kv_cache[:bsz, :seqlen // ratio] = kv
            return kv
        elif per_item:
            # Per-item decode: start_pos is Tensor[bsz], seqlen==1
            sp_cpu = start_pos.tolist()
            sp_mod = [s % ratio for s in sp_cpu]
            ape_idx = torch.tensor(sp_mod, device=x.device, dtype=torch.long)
            kv_sq = kv.squeeze(1)        # [bsz, coff*d]
            score_sq = score.squeeze(1) + self.ape[ape_idx]   # [bsz, coff*d]
            batch_idx = torch.arange(bsz, device=x.device)
            if overlap:
                write_pos = ratio + ape_idx
            else:
                write_pos = ape_idx
            self.kv_state[batch_idx, write_pos] = kv_sq
            self.score_state[batch_idx, write_pos] = score_sq
            # Find which items need compression
            compress_mask = [((s + 1) % ratio == 0) for s in sp_cpu]
            if not any(compress_mask):
                return
            for i in range(bsz):
                if not compress_mask[i]:
                    continue
                sp_i = sp_cpu[i]
                if overlap:
                    kv_st = torch.cat([self.kv_state[i:i+1, :ratio, :d], self.kv_state[i:i+1, ratio:, d:]], dim=1)
                    sc_st = torch.cat([self.score_state[i:i+1, :ratio, :d], self.score_state[i:i+1, ratio:, d:]], dim=1)
                    compressed = (kv_st * sc_st.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[i, :ratio] = self.kv_state[i, ratio:]
                    self.score_state[i, :ratio] = self.score_state[i, ratio:]
                else:
                    compressed = (self.kv_state[i:i+1] * self.score_state[i:i+1].softmax(dim=1)).sum(dim=1, keepdim=True)
                compressed = self.norm(compressed.to(dtype))
                freqs_ci = self.freqs_cis[sp_i + 1 - ratio].unsqueeze(0)
                apply_rotary_emb(compressed[..., -rd:], freqs_ci)
                if self.rotate:
                    compressed = rotate_activation(compressed)
                    (int4_act_quant if expert_int4 else fp4_act_quant)(compressed, fp4_block_size, True)
                else:
                    act_quant(compressed[..., :-rd], 64, scale_fmt, scale_dtype, True)
                self.kv_cache[i, sp_i // ratio] = compressed.squeeze(1)
            return
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
            if not should_compress:
                return
            kv = self.norm(kv.to(dtype))
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
            apply_rotary_emb(kv[..., -rd:], freqs_cis)
            if self.rotate:
                kv = rotate_activation(kv)
                (int4_act_quant if expert_int4 else fp4_act_quant)(kv, fp4_block_size, True)
            else:
                act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
            return kv


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.kv_cache_mgr = DynamicKVCache(args.max_batch_size, self.head_dim, min_capacity=max(1, args.window_size // compress_ratio if compress_ratio else 1))
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis = None

    def reset_cache(self, release: bool = False):
        self.kv_cache_mgr.reset(release)
        self.kv_cache = None
        self.freqs_cis = None
        self.compressor.reset_cache()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos, offset: int):
        bsz, seqlen, _ = x.size()
        per_item = isinstance(start_pos, torch.Tensor)
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if per_item:
            # Per-item decode: seqlen==1, different start_pos per item
            freqs_cis = self.freqs_cis[start_pos].unsqueeze(1)  # [bsz, 1, rd//2]
            end_pos_max = (start_pos.max() + seqlen).item()
            end_pos_items = start_pos + seqlen  # [bsz]
        else:
            freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
            end_pos_max = start_pos + seqlen
        required_len = max(1, (end_pos_max + ratio - 1) // ratio)
        self.kv_cache = self.kv_cache_mgr.ensure(bsz, required_len, x.device, x.dtype)
        self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # use fp4/int4 simulation for q and kv in indexer
        (int4_act_quant if expert_int4 else fp4_act_quant)(q, fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos_max // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1:
            dist.all_reduce(index_score, group=tp_group)
        if per_item:
            # Per-item: mask out positions beyond each item's compressed range
            max_compressed = end_pos_max // ratio
            if max_compressed > 0:
                compress_range = torch.arange(max_compressed, device=x.device).unsqueeze(0)  # [1, max_c]
                per_item_range = (end_pos_items // ratio).unsqueeze(1)  # [bsz, 1]
                beyond_mask = compress_range >= per_item_range  # [bsz, max_c]
                index_score[:, 0] += torch.where(beyond_mask, float("-inf"), 0.0)
            topk_idxs = index_score.topk(min(self.index_topk, max(1, end_pos_max // ratio)), dim=-1)[1]
            topk_idxs += offset
        elif start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
            topk_idxs = index_score.topk(min(self.index_topk, end_pos_max // ratio), dim=-1)[1]
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = index_score.topk(min(self.index_topk, end_pos_max // ratio), dim=-1)[1]
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression.
    Uses low-rank Q projection (wq_a -> q_norm -> wq_b) and grouped low-rank O projection."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        self.use_turbo_quant = args.turbo_quant
        if self.use_turbo_quant:
            self.kv_cache_mgr = TurboQuantKVCache(args.max_batch_size, self.head_dim, bits=args.turbo_quant_bits,
                                                   min_capacity=max(args.window_size, 1 + args.window_size))
        else:
            self.kv_cache_mgr = DynamicKVCache(args.max_batch_size, self.head_dim, min_capacity=max(args.window_size, 1 + args.window_size))
        self.kv_cache: Optional[torch.Tensor] = None
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        self.original_seq_len = original_seq_len
        self.rope_theta = rope_theta
        self.rope_factor = args.rope_factor
        self.beta_fast = args.beta_fast
        self.beta_slow = args.beta_slow
        self.max_seq_len = args.max_seq_len
        self.freqs_cis: Optional[torch.Tensor] = None

    def reset_cache(self, release: bool = False):
        self.kv_cache_mgr.reset(release)
        self.kv_cache = None
        self.freqs_cis = None
        if self.compress_ratio:
            self.compressor.reset_cache()
            if self.indexer is not None:
                self.indexer.reset_cache(release)

    def forward(self, x: torch.Tensor, start_pos):
        bsz, seqlen, _ = x.size()
        device = x.device
        device_index = device.index if device.index is not None else -1
        per_item = isinstance(start_pos, torch.Tensor)
        self.freqs_cis = get_freqs_cis_cached(
            self.rope_head_dim,
            self.max_seq_len,
            self.original_seq_len,
            self.rope_theta,
            self.rope_factor,
            self.beta_fast,
            self.beta_slow,
            device.type,
            device_index,
        )
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if per_item:
            freqs_cis = self.freqs_cis[start_pos].unsqueeze(1)  # [bsz, 1, rd//2]
            sp_max = start_pos.max().item()
            compressed_required = 0 if not ratio else max(1, (sp_max // ratio) + 1)
        else:
            freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
            sp_max = start_pos
            compressed_required = 0 if not ratio else max(1, (start_pos // ratio) + 1 if start_pos > 0 else seqlen // ratio)
        self.kv_cache = self.kv_cache_mgr.ensure(bsz, win + compressed_required, device, x.dtype)
        if self.compress_ratio:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for positional precision
        act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if per_item:
            topk_idxs = get_window_topk_idxs_per_item(win, start_pos)
            if self.compress_ratio:
                offset = win
                if self.indexer is not None:
                    compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
                else:
                    compress_topk_idxs = get_compress_topk_idxs_per_item(ratio, start_pos, offset)
                topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        else:
            topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
            if self.compress_ratio:
                offset = kv.size(1) if start_pos == 0 else win
                if self.indexer is not None:
                    compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
                else:
                    compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
                topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        tq = self.use_turbo_quant
        if not per_item and start_pos == 0:
            if seqlen <= win:
                if tq:
                    self.kv_cache_mgr.write_slice(slice(None, bsz), slice(None, seqlen), kv)
                else:
                    self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                win_kv_a, win_kv_b = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
                if tq:
                    self.kv_cache_mgr.write_slice(slice(None, bsz), slice(cutoff, win), win_kv_a)
                    self.kv_cache_mgr.write_slice(slice(None, bsz), slice(None, cutoff), win_kv_b)
                else:
                    self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = win_kv_a, win_kv_b
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
                if tq:
                    # Sync compressor's writes to quantized storage
                    comp_len = self.kv_cache[:bsz, win:].size(1)
                    if comp_len > 0:
                        self.kv_cache_mgr.write_slice(slice(None, bsz), slice(win, win + comp_len), self.kv_cache[:bsz, win:win + comp_len])
            # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        elif per_item:
            # Per-item decode: scatter KV to per-item window positions
            sp_cpu = start_pos.tolist()
            batch_idx = torch.arange(bsz, device=device)
            win_pos = start_pos % win  # [bsz]
            if tq:
                for i in range(bsz):
                    self.kv_cache_mgr.write_slice(i, int(sp_cpu[i]) % win, kv[i, 0])
            else:
                self.kv_cache[batch_idx, win_pos] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
                if tq:
                    for i in range(bsz):
                        if (sp_cpu[i] + 1) % ratio == 0:
                            comp_pos = win + sp_cpu[i] // ratio
                            self.kv_cache_mgr.write_slice(i, comp_pos, self.kv_cache[i, comp_pos])
            if tq:
                cache_len = self.kv_cache.size(1)
                kv_for_attn = self.kv_cache_mgr.read_slice(slice(None, bsz), slice(None, cache_len), x.dtype)
                o = sparse_attn(q, kv_for_attn, self.attn_sink, topk_idxs, self.softmax_scale)
            else:
                o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            if tq:
                self.kv_cache_mgr.write_slice(slice(None, bsz), start_pos % win, kv.squeeze(1))
            else:
                self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
                if tq:
                    # Sync compressor's write (to BF16 view) back to quantized storage
                    comp_pos = win + start_pos // ratio
                    self.kv_cache_mgr.write_slice(slice(None, bsz), comp_pos, self.kv_cache[:bsz, comp_pos])
            if tq:
                cache_len = self.kv_cache.size(1)
                kv_for_attn = self.kv_cache_mgr.read_slice(slice(None, bsz), slice(None, cache_len), x.dtype)
                o = sparse_attn(q, kv_for_attn, self.attn_sink, topk_idxs, self.softmax_scale)
            else:
                o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # NOTE: wo_a is FP8 in checkpoint; could do FP8 einsum here for better perf,
        # but using BF16 for simplicity.
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """MoE gating: computes expert routing scores and selects top-k experts.
    Supports hash-based routing (first n_hash_layers) where expert indices are
    predetermined per token ID, and score-based routing (remaining layers)."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        # Bias shifts scores for expert selection (topk) but does not affect routing weights.
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability."""
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Mixture-of-Experts: gate routes each token to top-k routed experts + 1 shared expert.
    Experts are sharded across TP ranks; each rank handles n_routed_experts // world_size experts."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = torch.uint8 if args.expert_dtype == "int4" else (torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit) if self.experts_start_idx <= i < self.experts_end_idx else None
                                       for i in range(self.n_routed_experts)])
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        # MiniMax-style overlap: launch shared expert on a separate CUDA stream
        # so it runs in parallel with routed experts + all_reduce
        main_stream = torch.cuda.current_stream()
        if not hasattr(self, '_shared_stream'):
            self._shared_stream = torch.cuda.Stream(device=x.device)
        self._shared_stream.wait_stream(main_stream)
        with torch.cuda.stream(self._shared_stream):
            shared_out = self.shared_experts(x)
        # Routed experts on main stream
        n_tokens = x.size(0)
        if n_tokens <= 2:
            # Fast path for single/few-token decode: directly iterate activated experts
            # Avoids bincount (CUDA sync) + 32 torch.where kernel launches per layer
            indices_list = indices.tolist()
            for row in range(n_tokens):
                for col, eid in enumerate(indices_list[row]):
                    if self.experts_start_idx <= eid < self.experts_end_idx:
                        y[row] += self.experts[eid](x[row:row+1], weights[row:row+1, col:col+1])
        else:
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                expert = self.experts[i]
                idx, top = torch.where(indices == i)
                y[idx] += expert(x[idx], weights[idx, top, None])
        if world_size > 1:
            dist.all_reduce(y, group=tp_group)
        # Wait for shared expert to finish then add
        main_stream.wait_stream(self._shared_stream)
        y += shared_out
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def reset_cache(self, release: bool = False):
        self.attn.reset_cache(release)

    def hc_pre(self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
        residual = x
        # x: [b,s,hc,d], fn: [mix_hc,hc*d], scale: [3], base: [mix_hc], y: [b,s,hc,d]
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        _dbg = os.environ.get("DEBUG_SYNC")
        def _sync(tag):
            if _dbg:
                torch.cuda.synchronize()
                print(f"[Block {self.layer_id}] {tag} ok", flush=True)
        _sync("enter")
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        _sync("hc_pre_attn")
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        _sync("attn")
        x = self.hc_post(x, residual, post, comb)
        _sync("hc_post_attn")

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        _sync("hc_pre_ffn")
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        _sync("ffn")
        x = self.hc_post(x, residual, post, comb)
        _sync("hc_post_ffn")
        return x


class ParallelHead(nn.Module):

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = (vocab_size // world_size)
        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

    def get_logits(self, x):
        return F.linear(x[:, -1].float(), self.weight)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm):
        # x: [b,s,hc,d]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x))
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits, group=tp_group)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class MTPBlock(Block):

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
        self.embed: ParallelEmbedding = None
        self.head: ParallelHead = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, start_pos, input_ids: torch.Tensor) -> torch.Tensor:
        # x: [b,s,hc,d]
        assert self.embed is not None and self.head is not None
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        logits = self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits


class Transformer(nn.Module):
    """Full DeepSeek-V4 model: embed -> HC-expand -> N blocks -> HC-head -> logits.
    Sets global state (world_size, rank, default_dtype, scale_fmt, scale_dtype) in __init__.
    Supports Pipeline Parallelism (PP) via pp_rank/pp_size kwargs:
      - pp_size=1 (default): all layers on all ranks (pure TP)
      - pp_size=2: stage 0 gets embed + layers[0:split], stage 1 gets layers[split:] + head
    When pp_size>1, tp_world_size/tp_rank/tp_grp must be provided."""
    def __init__(self, args: ModelArgs, pp_rank: int = 0, pp_size: int = 1,
                 tp_world_size: int = 0, tp_rank: int = 0, tp_grp=None):
        global world_size, rank, default_dtype, scale_fmt, scale_dtype, tp_group, expert_int4, turbo_quant_enabled, turbo_quant_bits
        if pp_size == 1:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
        else:
            assert tp_world_size > 0
            world_size = tp_world_size
            rank = tp_rank
            tp_group = tp_grp
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        expert_int4 = args.expert_dtype == "int4"
        turbo_quant_enabled = args.turbo_quant
        turbo_quant_bits = args.turbo_quant_bits
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.dim = args.dim
        # Pipeline stage layer ranges (ceil division: first stage gets extra layer if odd)
        n_layers = args.n_layers
        if pp_size > 1:
            layers_per_stage = (n_layers + pp_size - 1) // pp_size
            self.layer_start = pp_rank * layers_per_stage
            self.layer_end = min(n_layers, (pp_rank + 1) * layers_per_stage)
        else:
            self.layer_start = 0
            self.layer_end = n_layers
        # Embedding only on stage 0
        if pp_rank == 0:
            self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        else:
            self.embed = None
        # Layers for this stage
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.layer_start, self.layer_end):
            self.layers.append(Block(layer_id, args))
        # Head/norm/MTP only on last stage
        if pp_rank == pp_size - 1:
            self.norm = RMSNorm(args.dim, self.norm_eps)
            self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
            self.mtp = torch.nn.ModuleList()
            for layer_id in range(args.n_mtp_layers):
                self.mtp.append(MTPBlock(n_layers + layer_id, args))
            self.hc_mult = hc_mult = args.hc_mult
            hc_dim = hc_mult * args.dim
            with set_dtype(torch.float32):
                self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
                self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
                self.hc_head_scale = nn.Parameter(torch.empty(1))
            # Embed reference needed by MTP
            if pp_size > 1:
                self.mtp_embed = ParallelEmbedding(args.vocab_size, args.dim)
            for m in self.mtp:
                m.embed = self.mtp_embed if pp_size > 1 else self.embed
                m.head = self.head
        else:
            self.norm = None
            self.head = None
            self.mtp = torch.nn.ModuleList()
            self.hc_mult = args.hc_mult
            with set_dtype(torch.float32):
                self.hc_head_fn = None
                self.hc_head_base = None
                self.hc_head_scale = None

    def reset_caches(self, release: bool = False):
        for layer in self.layers:
            layer.reset_cache(release)
        for layer in self.mtp:
            layer.reset_cache(release)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos = 0, hidden_states: torch.Tensor = None):
        _dbg = os.environ.get("DEBUG_SYNC")
        if self.embed is not None:
            h = self.embed(input_ids)
            # Expand to hc_mult copies for Hyper-Connections
            h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
            if _dbg:
                torch.cuda.synchronize()
                print(f"[Transformer] embed ok, h {h.shape}", flush=True)
        else:
            assert hidden_states is not None, "Stage > 0 requires hidden_states input"
            h = hidden_states
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        if self.head is not None:
            if _dbg:
                torch.cuda.synchronize()
                print(f"[Transformer] all layers done, running head", flush=True)
            logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
            if _dbg:
                torch.cuda.synchronize()
                print(f"[Transformer] head ok", flush=True)
            return logits
        return h


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(n_hash_layers=0)
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)

    print(model(x).size())
    for i in range(128, 150):
        print(i, model(x[:, 0:1], i).size())

    h = torch.randn(2, 128, args.hc_mult, args.dim)
    mtp = model.mtp[0]
    print(mtp(h, 0, x).size())
    print(mtp(h[:, 0:1], 1, x[:, 0:1]).size())
