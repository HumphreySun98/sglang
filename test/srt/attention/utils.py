"""
Shared infrastructure for attention backend correctness unit tests.

Design:
- MockModelRunner: minimal ModelRunner stub backed by real pool objects.
- make_decode_batch / make_extend_batch: explicit per-mode ForwardBatch factories.
- prefill_kv_cache: seeds the KV pool with random prefix data before a forward pass.
- reconstruct_dense_kv: reads K, V back from the paged pool after a forward pass.
- hf_sdpa_reference / hf_swa_reference: HF-equivalent reference kernels (no network).
- run_attn_forward: sets the ForwardContext and calls the attention layer forward.
- assert_close: numerical comparison helper.

Q, K, V inputs are treated as post-RoPE for RadixAttention tests. For model-specific
attention classes (MLA etc.) that apply RoPE internally, the caller passes hidden
states and shares weights between sglang and the HF reference.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context


# ---------------------------------------------------------------------------
# Hardcoded model configs (no checkpoint download needed)
# ---------------------------------------------------------------------------

@dataclass
class AttentionConfig:
    """Minimal config for one attention layer, copied from a real model card."""
    num_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    context_len: int
    sliding_window_size: Optional[int] = None
    page_size: int = 16
    name: str = ""


# GPT-2 small — standard MHA
GPT2_CONFIG = AttentionConfig(
    num_heads=12, num_kv_heads=12, head_dim=64, v_head_dim=64,
    context_len=2048, name="gpt2",
)
# LLaMA-3-8B — GQA
LLAMA3_CONFIG = AttentionConfig(
    num_heads=32, num_kv_heads=8, head_dim=128, v_head_dim=128,
    context_len=8192, name="llama3",
)
# Gemma-3 / Mistral style — SWA
GEMMA_SWA_CONFIG = AttentionConfig(
    num_heads=8, num_kv_heads=4, head_dim=256, v_head_dim=256,
    context_len=8192, sliding_window_size=4096, name="gemma_swa",
)

# page_size=1 variants for FA3/FA4: production TokenToKVPoolAllocator hardcodes
# page_size=1, so slot_id == page_id and FA3's page-table division is a no-op.
GPT2_CONFIG_PS1 = dataclasses.replace(GPT2_CONFIG, page_size=1)
LLAMA3_CONFIG_PS1 = dataclasses.replace(LLAMA3_CONFIG, page_size=1)
GEMMA_SWA_CONFIG_PS1 = dataclasses.replace(GEMMA_SWA_CONFIG, page_size=1)


# ---------------------------------------------------------------------------
# Mock model runner
# ---------------------------------------------------------------------------

class MockModelConfig:
    def __init__(self, cfg: AttentionConfig) -> None:
        self.num_attention_heads = cfg.num_heads
        self._num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.v_head_dim = cfg.v_head_dim
        self.swa_v_head_dim = None
        self.context_len = cfg.context_len
        self.attention_arch = AttentionArch.MHA
        self.is_multimodal = False
        self.is_encoder_decoder = False
        self.is_local_attention_model = False
        # Minimal hf_config stub — only architectures list is read by backends.
        self.hf_config = SimpleNamespace(architectures=[])
        # hf_text_config stub — read by FA3/FA4 for head counts and softcap.
        self.hf_text_config = SimpleNamespace(
            num_attention_heads=cfg.num_heads,
            attn_logit_softcapping=None,
        )

    def get_num_kv_heads(self, tp_size: int = 1) -> int:
        return self._num_kv_heads // tp_size


class MockServerArgs:
    def __init__(self, page_size: int = 16) -> None:
        self.page_size = page_size
        self.speculative_num_draft_tokens = 0
        self.speculative_num_steps = 0
        self.triton_attention_num_kv_splits = 8
        self.triton_attention_split_tile_size = None
        self.enable_deterministic_inference = False
        self.disable_cuda_graph = True
        self.chunked_prefill_size = -1
        self.enable_mis = False
        self.disable_piecewise_cuda_graph = True
        # Fields read by FlashInfer backend.
        self.dllm_algorithm = None
        # Fields read by FA3/FA4 backend.
        self.kv_cache_dtype = "auto"
        self.speculative_eagle_topk = 0
        self.is_embedding = False
        self.disable_radix_cache = False
        self.enable_dp_attention = False


class MockModelRunner:
    """
    Lightweight stub of ModelRunner for attention backend unit tests.

    Creates real ReqToTokenPool and MHATokenToKVPool so backends can do
    genuine index lookups and KV writes. All fields that backends read
    during __init__ and init_forward_metadata are set here. Missing fields
    surface as AttributeError at test time rather than silently returning None.
    """

    def __init__(
        self,
        cfg: AttentionConfig,
        num_layers: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        max_num_reqs: int = 128,
        max_total_tokens: int = 8192,
    ) -> None:
        self.device = device
        self.gpu_id = 0
        self.tp_size = 1
        self.tp_rank = 0
        self.page_size = cfg.page_size
        self.sliding_window_size = cfg.sliding_window_size
        self.dtype = dtype
        self.kv_cache_dtype = dtype
        self.use_mla_backend = False
        # Linear/hybrid attention flags — None disables their special paths
        self.hybrid_gdn_config = None
        self.kimi_linear_config = None
        self.linear_attn_model_spec = None

        self.model_config = MockModelConfig(cfg)
        self.server_args = MockServerArgs(page_size=cfg.page_size)

        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=cfg.context_len,
            device=device,
            enable_memory_saver=False,
        )
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_tokens,
            page_size=cfg.page_size,
            dtype=dtype,
            head_num=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            layer_num=num_layers,
            device=device,
            enable_memory_saver=False,
            v_head_dim=cfg.v_head_dim,
        )
        # Alias used by some backends (e.g. triton_backend stores it separately)
        self.token_to_kv_pool_allocator = self.token_to_kv_pool
        # CP (context parallelism) size — backends read this; 1 = no CP.
        self.attn_cp_size = 1


# ---------------------------------------------------------------------------
# ForwardBatch factories
# ---------------------------------------------------------------------------

def make_decode_batch(
    seq_lens: List[int],
    model_runner: MockModelRunner,
    layer_id: int = 0,
) -> Tuple[ForwardBatch, torch.Tensor]:
    """
    Build a ForwardBatch for DECODE mode and pre-fill the KV cache with random
    prefix data.

    Returns (forward_batch, kv_slot_map) where kv_slot_map[i, :seq_lens[i]]
    gives the pool slot indices for request i (used by reconstruct_dense_kv).

    Fields set: forward_mode, batch_size, input_ids, req_pool_indices,
    seq_lens, out_cache_loc, seq_lens_sum, positions.
    """
    device = model_runner.device
    dtype = model_runner.dtype
    bsz = len(seq_lens)
    cfg = model_runner.model_config
    num_kv_heads = cfg._num_kv_heads
    head_dim = cfg.head_dim
    v_head_dim = cfg.v_head_dim

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    total_tokens = int(seq_lens_t.sum().item())

    # Assign contiguous pool slots starting at 1 (slot 0 is padding).
    # Request i owns slots [offset_i, offset_i + seq_len_i).
    offsets = [0] + list(seq_lens_t.cumsum(0).cpu().tolist())
    all_slots = torch.arange(1, 1 + total_tokens, dtype=torch.int32, device=device)

    # Populate req_to_token: row r+1 holds the slot indices for request r.
    req_pool_indices = torch.arange(1, 1 + bsz, dtype=torch.int32, device=device)
    for r in range(bsz):
        sl = seq_lens[r]
        model_runner.req_to_token_pool.req_to_token[r + 1, :sl] = all_slots[
            offsets[r] : offsets[r] + sl
        ]

    # out_cache_loc: the last slot of each request (the new decode token).
    out_cache_loc = torch.tensor(
        [offsets[r] + seq_lens[r] - 1 + 1 for r in range(bsz)],  # +1 for 1-based slots
        dtype=torch.int32, device=device,
    )

    # Seed prefix K, V (all tokens except the last) with random data.
    k_buf = model_runner.token_to_kv_pool.k_buffer[layer_id]
    v_buf = model_runner.token_to_kv_pool.v_buffer[layer_id]
    for r in range(bsz):
        prefix_len = seq_lens[r] - 1
        if prefix_len > 0:
            prefix_slots = all_slots[offsets[r] : offsets[r] + prefix_len]
            k_buf[prefix_slots] = torch.randn(
                prefix_len, num_kv_heads, head_dim, dtype=dtype, device=device
            )
            v_buf[prefix_slots] = torch.randn(
                prefix_len, num_kv_heads, v_head_dim, dtype=dtype, device=device
            )

    positions = seq_lens_t.long() - 1

    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=bsz,
        input_ids=torch.zeros(bsz, dtype=torch.int32, device=device),
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens_t,
        seq_lens_cpu=seq_lens_t.cpu(),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=total_tokens,
        positions=positions,
    )

    # Build a slot map [bsz, max_seq_len] for reconstruct_dense_kv.
    max_sl = max(seq_lens)
    kv_slot_map = torch.zeros(bsz, max_sl, dtype=torch.int32, device=device)
    for r in range(bsz):
        kv_slot_map[r, : seq_lens[r]] = all_slots[offsets[r] : offsets[r] + seq_lens[r]]
    # The last slot (decode token) will be written by the forward pass.
    # out_cache_loc already points there; kv_slot_map already includes it.

    return batch, kv_slot_map


def make_extend_batch(
    prefix_lens: List[int],
    extend_lens: List[int],
    model_runner: MockModelRunner,
    layer_id: int = 0,
) -> Tuple[ForwardBatch, torch.Tensor]:
    """
    Build a ForwardBatch for EXTEND mode and pre-fill prefix KV data.

    Returns (forward_batch, kv_slot_map) where kv_slot_map[i, :seq_len_i]
    gives the pool slot indices for request i (prefix + extend).

    Fields set: forward_mode, batch_size, input_ids, req_pool_indices,
    seq_lens, out_cache_loc, seq_lens_sum, positions,
    extend_seq_lens, extend_prefix_lens, extend_num_tokens, extend_start_loc.
    """
    assert len(prefix_lens) == len(extend_lens)
    device = model_runner.device
    dtype = model_runner.dtype
    bsz = len(prefix_lens)
    cfg = model_runner.model_config
    num_kv_heads = cfg._num_kv_heads
    head_dim = cfg.head_dim
    v_head_dim = cfg.v_head_dim

    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    total_prefix = sum(prefix_lens)
    total_extend = sum(extend_lens)
    total_tokens = total_prefix + total_extend

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    extend_lens_t = torch.tensor(extend_lens, dtype=torch.int32, device=device)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)

    # Assign slots: first all prefix slots, then all extend slots.
    # Slot 0 is the padding slot; use 1-based indexing.
    prefix_offsets = [0] + list(
        torch.tensor(prefix_lens).cumsum(0).tolist()
    )
    extend_offsets = [0] + list(
        torch.tensor(extend_lens).cumsum(0).tolist()
    )

    prefix_slots = torch.arange(1, 1 + total_prefix, dtype=torch.int32, device=device)
    extend_slots = torch.arange(
        1 + total_prefix, 1 + total_tokens, dtype=torch.int32, device=device
    )

    # Populate req_to_token: prefix slots then extend slots interleaved per request.
    req_pool_indices = torch.arange(1, 1 + bsz, dtype=torch.int32, device=device)
    max_sl = max(seq_lens)
    kv_slot_map = torch.zeros(bsz, max_sl, dtype=torch.int32, device=device)
    for r in range(bsz):
        p, e = prefix_lens[r], extend_lens[r]
        p_slots = prefix_slots[prefix_offsets[r] : prefix_offsets[r] + p]
        e_slots = extend_slots[extend_offsets[r] : extend_offsets[r] + e]
        model_runner.req_to_token_pool.req_to_token[r + 1, :p] = p_slots
        model_runner.req_to_token_pool.req_to_token[r + 1, p : p + e] = e_slots
        kv_slot_map[r, :p] = p_slots
        kv_slot_map[r, p : p + e] = e_slots

    # out_cache_loc: extend slot indices (flattened), written by the forward pass.
    out_cache_loc = extend_slots

    # Seed prefix K, V with random data.
    k_buf = model_runner.token_to_kv_pool.k_buffer[layer_id]
    v_buf = model_runner.token_to_kv_pool.v_buffer[layer_id]
    for r in range(bsz):
        p = prefix_lens[r]
        if p > 0:
            p_slots = prefix_slots[prefix_offsets[r] : prefix_offsets[r] + p]
            k_buf[p_slots] = torch.randn(
                p, num_kv_heads, head_dim, dtype=dtype, device=device
            )
            v_buf[p_slots] = torch.randn(
                p, num_kv_heads, v_head_dim, dtype=dtype, device=device
            )

    # positions: for request r, new tokens are at positions prefix_r, prefix_r+1, ...
    positions_list = []
    for r in range(bsz):
        p, e = prefix_lens[r], extend_lens[r]
        positions_list.extend(range(p, p + e))
    positions = torch.tensor(positions_list, dtype=torch.long, device=device)

    extend_start_loc = torch.zeros(bsz, dtype=torch.int32, device=device)
    extend_start_loc[1:] = extend_lens_t[:-1].cumsum(0)

    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bsz,
        input_ids=torch.zeros(total_extend, dtype=torch.int32, device=device),
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens_t,
        seq_lens_cpu=seq_lens_t.cpu(),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=total_tokens,
        positions=positions,
        extend_seq_lens=extend_lens_t,
        extend_prefix_lens=prefix_lens_t,
        extend_num_tokens=total_extend,
        extend_start_loc=extend_start_loc,
        extend_seq_lens_cpu=extend_lens,
        extend_prefix_lens_cpu=prefix_lens,
        is_extend_in_batch=True,
        all_extend_in_batch=True,
    )

    return batch, kv_slot_map


def make_split_prefill_batch(
    prefix_lens: List[int],
    extend_lens: List[int],
    model_runner: MockModelRunner,
    layer_id: int = 0,
) -> Tuple[ForwardBatch, torch.Tensor]:
    """
    Build a ForwardBatch for SPLIT_PREFILL mode (chunked prefill).

    SPLIT_PREFILL is handled by the same extend codepath (is_extend() returns True)
    but uses a different forward_mode flag. The batch is constructed identically to an
    extend batch, then forward_mode is overridden to SPLIT_PREFILL.
    """
    batch, kv_slot_map = make_extend_batch(
        prefix_lens, extend_lens, model_runner, layer_id=layer_id
    )
    batch.forward_mode = ForwardMode.SPLIT_PREFILL
    return batch, kv_slot_map


def make_mixed_batch(
    decode_seq_lens: List[int],
    extend_prefix_lens: List[int],
    extend_lens: List[int],
    model_runner: MockModelRunner,
    layer_id: int = 0,
) -> Tuple[ForwardBatch, torch.Tensor]:
    """
    Build a ForwardBatch for MIXED mode (some decode + some extend requests).

    Decode requests are represented as extend_len=1 with prefix_len=seq_len-1.
    The combined batch is built via make_extend_batch and then forward_mode is
    set to MIXED so backends exercise that branch.

    Returns (forward_batch, kv_slot_map) using the same layout as make_extend_batch.
    """
    decode_prefix_lens = [s - 1 for s in decode_seq_lens]
    decode_extend_lens = [1] * len(decode_seq_lens)

    all_prefix_lens = decode_prefix_lens + extend_prefix_lens
    all_extend_lens = decode_extend_lens + extend_lens

    batch, kv_slot_map = make_extend_batch(
        all_prefix_lens, all_extend_lens, model_runner, layer_id=layer_id
    )
    batch.forward_mode = ForwardMode.MIXED
    return batch, kv_slot_map


# ---------------------------------------------------------------------------
# KV reconstruction
# ---------------------------------------------------------------------------

def reconstruct_dense_kv(
    forward_batch: ForwardBatch,
    model_runner: MockModelRunner,
    layer_id: int,
    kv_slot_map: torch.Tensor,
    seq_lens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read K and V back from the paged KV pool into dense tensors.

    Must be called AFTER the forward pass so newly written slots are included.

    Returns:
        k_list: list of [seq_len_i, num_kv_heads, head_dim] tensors
        v_list: list of [seq_len_i, num_kv_heads, v_head_dim] tensors
    """
    k_buf = model_runner.token_to_kv_pool.k_buffer[layer_id]
    v_buf = model_runner.token_to_kv_pool.v_buffer[layer_id]
    k_list, v_list = [], []
    for r, sl in enumerate(seq_lens):
        slots = kv_slot_map[r, :sl]
        k_list.append(k_buf[slots].float())
        v_list.append(v_buf[slots].float())
    return k_list, v_list


# ---------------------------------------------------------------------------
# HF reference attention kernels
# ---------------------------------------------------------------------------

def _expand_kv_heads(k: torch.Tensor, v: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand GQA K, V from num_kv_heads to num_heads via repeat_interleave."""
    num_kv_heads = k.shape[-2]
    if num_kv_heads == num_heads:
        return k, v
    factor = num_heads // num_kv_heads
    # k shape: [..., num_kv_heads, head_dim]
    k = k.repeat_interleave(factor, dim=-2)
    v = v.repeat_interleave(factor, dim=-2)
    return k, v


def hf_sdpa_reference(
    q_list: List[torch.Tensor],  # per-request [seq_q, num_heads, head_dim]
    k_list: List[torch.Tensor],  # per-request [seq_k, num_kv_heads, head_dim]
    v_list: List[torch.Tensor],  # per-request [seq_k, num_kv_heads, v_head_dim]
    scaling: float,
    causal: bool = True,
    prefix_lens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Standard SDPA reference for MHA / GQA (no sliding window).

    For DECODE (seq_q=1 per request), causal=False is equivalent since q
    attends to all k positions regardless.

    For EXTEND (seq_q>1), uses an explicit causal+prefix mask so that
    q[t] attends to k[0..prefix_len+t].

    Returns concatenated output [total_q_tokens, num_heads, v_head_dim] (float32).
    """
    outputs = []
    for i, (q, k, v) in enumerate(zip(q_list, k_list, v_list)):
        num_heads = q.shape[-2]
        k_exp, v_exp = _expand_kv_heads(k.float(), v.float(), num_heads)

        seq_q = q.shape[0]
        seq_k = k.shape[0]

        # q: [seq_q, H, D] → [1, H, seq_q, D]
        q4 = q.float().transpose(0, 1).unsqueeze(0)
        k4 = k_exp.transpose(0, 1).unsqueeze(0)
        v4 = v_exp.transpose(0, 1).unsqueeze(0)

        if seq_q == 1:
            # decode: attend to all keys
            out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False, scale=scaling)
        else:
            # extend: explicit mask — q at absolute position (prefix_len + t) attends to k[0..prefix_len+t]
            p = prefix_lens[i] if prefix_lens is not None else 0
            row = torch.arange(seq_q, device=q.device).unsqueeze(1) + p  # absolute pos
            col = torch.arange(seq_k, device=q.device).unsqueeze(0)
            mask = (col <= row).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_q, seq_k]
            out = F.scaled_dot_product_attention(
                q4, k4, v4,
                attn_mask=mask,
                is_causal=False,
                scale=scaling,
            )

        # out: [1, H, seq_q, v_head_dim] → [seq_q, H, v_head_dim]
        outputs.append(out.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)


def hf_swa_reference(
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    scaling: float,
    window_size: int,
    prefix_lens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    SWA reference: causal + sliding-window mask.
    q[t] at absolute position p+t can attend to k[s] if p+t - window_size < s <= p+t.
    """
    outputs = []
    for i, (q, k, v) in enumerate(zip(q_list, k_list, v_list)):
        num_heads = q.shape[-2]
        k_exp, v_exp = _expand_kv_heads(k.float(), v.float(), num_heads)

        seq_q = q.shape[0]
        seq_k = k.shape[0]

        p = prefix_lens[i] if prefix_lens is not None else 0

        q4 = q.float().transpose(0, 1).unsqueeze(0)
        k4 = k_exp.transpose(0, 1).unsqueeze(0)
        v4 = v_exp.transpose(0, 1).unsqueeze(0)

        row = torch.arange(seq_q, device=q.device).unsqueeze(1) + p
        col = torch.arange(seq_k, device=q.device).unsqueeze(0)
        causal_mask = col <= row
        window_mask = (row - col) < window_size
        mask = (causal_mask & window_mask).unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask, is_causal=False, scale=scaling,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Backend construction helpers
# ---------------------------------------------------------------------------

def build_triton_backend(model_runner: MockModelRunner):
    """Construct TritonAttnBackend with get_attention_tp_size patched to 1."""
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    with patch("sglang.srt.layers.attention.triton_backend.get_attention_tp_size", return_value=1):
        return TritonAttnBackend(model_runner)


def build_torch_native_backend(model_runner: MockModelRunner):
    """Construct TorchNativeAttnBackend (no patching needed)."""
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
    return TorchNativeAttnBackend(model_runner)


def build_flashinfer_backend(model_runner: MockModelRunner, force_paged: bool = False):
    """Construct FlashInferAttnBackend with get_attention_tp_size patched to 1.

    Args:
        force_paged: When True, sets SGLANG_FLASHINFER_USE_PAGED=1 to force the
            non-ragged paged extend path (required for SWA extend with prefix, where
            the ragged path incorrectly reads uninitialized extend slots from the cache).
    """
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.environ import envs
    _tp1 = patch("sglang.srt.layers.attention.flashinfer_backend.get_attention_tp_size", return_value=1)
    with _tp1:
        if force_paged:
            envs.SGLANG_FLASHINFER_USE_PAGED.set(True)
            try:
                return FlashInferAttnBackend(model_runner)
            finally:
                envs.SGLANG_FLASHINFER_USE_PAGED.clear()
        return FlashInferAttnBackend(model_runner)


def build_fa_backend(model_runner: MockModelRunner, fa_version: int = 3):
    """Construct FlashAttentionBackend (fa3 or fa4)."""
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    return FlashAttentionBackend(model_runner, fa_impl_ver=fa_version)


def build_flex_backend(model_runner: MockModelRunner):
    """Construct TorchFlexAttnBackend (no SWA support)."""
    from sglang.srt.layers.attention.torch_flex_backend import TorchFlexAttnBackend
    return TorchFlexAttnBackend(model_runner)


# ---------------------------------------------------------------------------
# Forward runner
# ---------------------------------------------------------------------------

def run_attn_forward(
    layer: RadixAttention,
    backend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """
    Set the ForwardContext, call init_forward_metadata, then run the layer forward.
    Returns output reshaped to [num_tokens, num_heads, v_head_dim] (float32).
    sglang returns a flat [num_tokens, num_heads * v_head_dim] tensor.
    """
    ctx = ForwardContext(attn_backend=backend)
    with forward_context(ctx):
        backend.init_forward_metadata(forward_batch)
        out = layer(q, k, v, forward_batch)
    # Reshape from flat [N, H*Dv] to [N, H, Dv] to match the HF reference shape.
    num_tokens = out.shape[0]
    return out.float().view(num_tokens, layer.tp_q_head_num, layer.v_head_dim)


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------

def assert_close(
    ref: torch.Tensor,
    out: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    msg: str = "",
) -> None:
    ref = ref.float().cpu()
    out = out.float().cpu()
    max_diff = (ref - out).abs().max().item()
    mean_diff = (ref - out).abs().mean().item()
    assert torch.allclose(ref, out, atol=atol, rtol=rtol), (
        f"{msg} max_diff={max_diff:.4e} mean_diff={mean_diff:.4e} "
        f"atol={atol} rtol={rtol}"
    )
