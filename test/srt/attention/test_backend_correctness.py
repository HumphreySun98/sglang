"""
Phase 2: Attention backend correctness unit tests.

For each (attention_type, backend, forward_mode, input_shape) combination:
  1. Build a MockModelRunner with a hardcoded model config.
  2. Construct the backend and a RadixAttention layer.
  3. Build a synthetic ForwardBatch with random KV prefix data.
  4. Run the sglang attention forward pass.
  5. Reconstruct dense K, V from the paged pool.
  6. Compare against an HF SDPA reference on the same Q, K, V.

No server launch, no checkpoint download. Q/K/V are treated as post-RoPE inputs.
"""

import math
import unittest

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.test.test_utils import CustomTestCase

from .utils import (
    GEMMA_SWA_CONFIG,
    GEMMA_SWA_CONFIG_PS1,
    GPT2_CONFIG,
    GPT2_CONFIG_PS1,
    LLAMA3_CONFIG,
    LLAMA3_CONFIG_PS1,
    AttentionConfig,
    MockModelRunner,
    assert_close,
    build_fa_backend,
    build_flex_backend,
    build_flashinfer_backend,
    build_torch_native_backend,
    build_triton_backend,
    hf_sdpa_reference,
    hf_swa_reference,
    make_decode_batch,
    make_extend_batch,
    reconstruct_dense_kv,
    run_attn_forward,
)

DEVICE = "cuda"
DTYPE = torch.float16
LAYER_ID = 0
SEED = 42


def _make_layer(cfg: AttentionConfig) -> RadixAttention:
    scaling = 1.0 / math.sqrt(cfg.head_dim)
    return RadixAttention(
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        scaling=scaling,
        num_kv_heads=cfg.num_kv_heads,
        layer_id=LAYER_ID,
        v_head_dim=cfg.v_head_dim,
        sliding_window_size=cfg.sliding_window_size if cfg.sliding_window_size else -1,
    ).to(DEVICE)


def _rand(shape, dtype=DTYPE, device=DEVICE):
    return torch.randn(*shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Standard MHA (GPT-2 config) — triton backend
# ---------------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonMHADecode(CustomTestCase):
    """Triton backend, standard MHA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)

        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)

        # q, k, v for the new decode token (post-RoPE)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])

        out = run_attn_forward(layer, backend, q, k, v, batch)

        # Reconstruct full K, V after forward (includes newly written decode token).
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)

        q_list = [q[r : r + 1] for r in range(bsz)]  # [1, H, D] per request
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)

        assert_close(ref, out, atol=2e-2, rtol=2e-2,
                     msg=f"MHA decode seq_lens={seq_lens}")

    def test_bsz1_short(self):
        self._run([8])

    def test_bsz2_short(self):
        self._run([8, 12])

    def test_bsz4_varied(self):
        self._run([4, 8, 16, 32])

    def test_bsz1_long(self):
        self._run([256])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonMHAExtend(CustomTestCase):
    """Triton backend, standard MHA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)

        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)

        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])

        out = run_attn_forward(layer, backend, q, k, v, batch)

        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)

        # Split q into per-request tensors for the reference.
        bsz = len(prefix_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]

        ref = hf_sdpa_reference(
            q_list, k_list, v_list,
            scaling=layer.scaling,
            prefix_lens=prefix_lens,
        )

        assert_close(ref, out, atol=2e-2, rtol=2e-2,
                     msg=f"MHA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([4, 8], [4, 4])

    def test_long_prefix(self):
        self._run([64, 64], [16, 16])

    def test_bsz1_extend(self):
        self._run([16], [8])


# ---------------------------------------------------------------------------
# GQA (LLaMA-3 config) — triton backend
# ---------------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonGQADecode(CustomTestCase):
    """Triton backend, GQA (num_kv_heads < num_heads), DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)

        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])

        out = run_attn_forward(layer, backend, q, k, v, batch)

        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)

        assert_close(ref, out, atol=2e-2, rtol=2e-2,
                     msg=f"GQA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([16])

    def test_bsz4(self):
        self._run([8, 16, 32, 64])

    def test_bsz1_long(self):
        self._run([512])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonGQAExtend(CustomTestCase):
    """Triton backend, GQA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)

        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)

        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])

        out = run_attn_forward(layer, backend, q, k, v, batch)

        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)

        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]

        ref = hf_sdpa_reference(
            q_list, k_list, v_list,
            scaling=layer.scaling,
            prefix_lens=prefix_lens,
        )

        assert_close(ref, out, atol=2e-2, rtol=2e-2,
                     msg=f"GQA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([16, 32], [8, 8])

    def test_long(self):
        self._run([128], [32])


# ---------------------------------------------------------------------------
# SWA (Gemma-3 style) — triton backend
# ---------------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonSWADecode(CustomTestCase):
    """Triton backend, sliding-window attention, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GEMMA_SWA_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)

        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])

        out = run_attn_forward(layer, backend, q, k, v, batch)

        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]

        # For decode, q is at position seq_len-1; use prefix_len = seq_len-1 for mask.
        prefix_lens = [s - 1 for s in seq_lens]
        ref = hf_swa_reference(
            q_list, k_list, v_list,
            scaling=layer.scaling,
            window_size=cfg.sliding_window_size,
            prefix_lens=prefix_lens,
        )

        assert_close(ref, out, atol=2e-2, rtol=2e-2,
                     msg=f"SWA decode seq_lens={seq_lens}")

    def test_within_window(self):
        # seq_len < window_size: SWA == full causal
        self._run([32])

    def test_bsz2_within_window(self):
        self._run([16, 64])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonSWAExtend(CustomTestCase):
    """Triton backend, sliding-window attention, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GEMMA_SWA_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)

        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)

        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])

        out = run_attn_forward(layer, backend, q, k, v, batch)

        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)

        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]

        ref = hf_swa_reference(
            q_list, k_list, v_list,
            scaling=layer.scaling,
            window_size=cfg.sliding_window_size,
            prefix_lens=prefix_lens,
        )

        assert_close(ref, out, atol=2e-2, rtol=2e-2,
                     msg=f"SWA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([8, 16], [4, 4])


# ---------------------------------------------------------------------------
# TorchNative backend — MHA, GQA, SWA
# ---------------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTorchNativeMHADecode(CustomTestCase):
    """TorchNative backend, standard MHA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_torch_native_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"TorchNative MHA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTorchNativeMHAExtend(CustomTestCase):
    """TorchNative backend, standard MHA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_torch_native_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"TorchNative MHA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([4, 8], [4, 4])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTorchNativeGQADecode(CustomTestCase):
    """TorchNative backend, GQA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_torch_native_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"TorchNative GQA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([16])

    def test_bsz4(self):
        self._run([8, 16, 32, 64])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTorchNativeGQAExtend(CustomTestCase):
    """TorchNative backend, GQA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_torch_native_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"TorchNative GQA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([16, 32], [8, 8])


# ---------------------------------------------------------------------------
# FlashInfer backend — MHA, GQA, SWA
# ---------------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFlashInferMHADecode(CustomTestCase):
    """FlashInfer backend, standard MHA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flashinfer_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlashInfer MHA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])

    def test_bsz1_long(self):
        self._run([256])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFlashInferMHAExtend(CustomTestCase):
    """FlashInfer backend, standard MHA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flashinfer_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlashInfer MHA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([4, 8], [4, 4])

    def test_bsz1_extend(self):
        self._run([16], [8])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFlashInferGQADecode(CustomTestCase):
    """FlashInfer backend, GQA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flashinfer_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlashInfer GQA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([16])

    def test_bsz4(self):
        self._run([8, 16, 32, 64])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFlashInferGQAExtend(CustomTestCase):
    """FlashInfer backend, GQA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flashinfer_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlashInfer GQA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([16, 32], [8, 8])

    def test_long(self):
        self._run([128], [32])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFlashInferSWADecode(CustomTestCase):
    """FlashInfer backend, sliding-window attention, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GEMMA_SWA_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        # force_paged: SWA needs the paged extend path so all KV is written before read.
        cls.backend = build_flashinfer_backend(cls.runner, force_paged=True)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        prefix_lens = [s - 1 for s in seq_lens]
        ref = hf_swa_reference(q_list, k_list, v_list, scaling=layer.scaling,
                               window_size=cfg.sliding_window_size, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlashInfer SWA decode seq_lens={seq_lens}")

    def test_within_window(self):
        self._run([32])

    def test_bsz2(self):
        self._run([16, 64])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFlashInferSWAExtend(CustomTestCase):
    """FlashInfer backend, sliding-window attention, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GEMMA_SWA_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        # force_paged: SWA needs the paged extend path so all KV is written before read.
        cls.backend = build_flashinfer_backend(cls.runner, force_paged=True)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_swa_reference(q_list, k_list, v_list, scaling=layer.scaling,
                               window_size=cfg.sliding_window_size, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlashInfer SWA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([8, 16], [4, 4])


# ---------------------------------------------------------------------------
# FA3 backend (SM90+) — MHA, GQA, SWA
# ---------------------------------------------------------------------------

_HAS_FA3 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func as _fa3_probe  # noqa: F401
    _FA3_AVAILABLE = True
except Exception:
    _FA3_AVAILABLE = False


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3MHADecode(CustomTestCase):
    """FA3 backend, standard MHA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG_PS1  # page_size=1 matches production TokenToKVPoolAllocator
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FA3 MHA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])

    def test_bsz1_long(self):
        self._run([256])


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3MHAExtend(CustomTestCase):
    """FA3 backend, standard MHA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG_PS1
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FA3 MHA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([4, 8], [4, 4])

    def test_bsz1_extend(self):
        self._run([16], [8])


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3GQADecode(CustomTestCase):
    """FA3 backend, GQA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG_PS1
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FA3 GQA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([16])

    def test_bsz4(self):
        self._run([8, 16, 32, 64])


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3GQAExtend(CustomTestCase):
    """FA3 backend, GQA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG_PS1
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FA3 GQA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([16, 32], [8, 8])


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3SWADecode(CustomTestCase):
    """FA3 backend, sliding-window attention, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GEMMA_SWA_CONFIG_PS1
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        prefix_lens = [s - 1 for s in seq_lens]
        ref = hf_swa_reference(q_list, k_list, v_list, scaling=layer.scaling,
                               window_size=cfg.sliding_window_size, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FA3 SWA decode seq_lens={seq_lens}")

    def test_within_window(self):
        self._run([32])

    def test_bsz2(self):
        self._run([16, 64])


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3SWAExtend(CustomTestCase):
    """FA3 backend, sliding-window attention, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GEMMA_SWA_CONFIG_PS1
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_swa_reference(q_list, k_list, v_list, scaling=layer.scaling,
                               window_size=cfg.sliding_window_size, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FA3 SWA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([8, 16], [4, 4])


# ---------------------------------------------------------------------------
# FlexAttention backend — MHA, GQA (no SWA: sliding window not supported)
# ---------------------------------------------------------------------------

_HAS_FLEX = torch.cuda.is_available()
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_probe  # noqa: F401
    _FLEX_AVAILABLE = True
except Exception:
    _FLEX_AVAILABLE = False


@unittest.skipIf(not _HAS_FLEX or not _FLEX_AVAILABLE, "flex_attention requires CUDA and PyTorch >= 2.5")
class TestFlexAttnMHADecode(CustomTestCase):
    """FlexAttention backend, standard MHA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flex_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlexAttn MHA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])


@unittest.skipIf(not _HAS_FLEX or not _FLEX_AVAILABLE, "flex_attention requires CUDA and PyTorch >= 2.5")
class TestFlexAttnMHAExtend(CustomTestCase):
    """FlexAttention backend, standard MHA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flex_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlexAttn MHA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([4, 8], [4, 4])


@unittest.skipIf(not _HAS_FLEX or not _FLEX_AVAILABLE, "flex_attention requires CUDA and PyTorch >= 2.5")
class TestFlexAttnGQADecode(CustomTestCase):
    """FlexAttention backend, GQA, DECODE mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flex_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        bsz = len(seq_lens)
        batch, kv_slot_map = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
        q = _rand([bsz, cfg.num_heads, cfg.head_dim])
        k = _rand([bsz, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([bsz, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        q_list = [q[r : r + 1] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlexAttn GQA decode seq_lens={seq_lens}")

    def test_bsz1(self):
        self._run([16])

    def test_bsz4(self):
        self._run([8, 16, 32, 64])


@unittest.skipIf(not _HAS_FLEX or not _FLEX_AVAILABLE, "flex_attention requires CUDA and PyTorch >= 2.5")
class TestFlexAttnGQAExtend(CustomTestCase):
    """FlexAttention backend, GQA, EXTEND mode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flex_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, prefix_lens, extend_lens):
        cfg, runner, backend, layer = self.cfg, self.runner, self.backend, self.layer
        total_extend = sum(extend_lens)
        bsz = len(prefix_lens)
        batch, kv_slot_map = make_extend_batch(prefix_lens, extend_lens, runner, layer_id=LAYER_ID)
        q = _rand([total_extend, cfg.num_heads, cfg.head_dim])
        k = _rand([total_extend, cfg.num_kv_heads, cfg.head_dim])
        v = _rand([total_extend, cfg.num_kv_heads, cfg.v_head_dim])
        out = run_attn_forward(layer, backend, q, k, v, batch)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k_list, v_list = reconstruct_dense_kv(batch, runner, LAYER_ID, kv_slot_map, seq_lens)
        ext_offsets = [0] + list(torch.tensor(extend_lens).cumsum(0).tolist())
        q_list = [q[ext_offsets[r] : ext_offsets[r + 1]] for r in range(bsz)]
        ref = hf_sdpa_reference(q_list, k_list, v_list, scaling=layer.scaling, prefix_lens=prefix_lens)
        assert_close(ref, out, atol=2e-2, rtol=2e-2, msg=f"FlexAttn GQA extend prefix={prefix_lens} extend={extend_lens}")

    def test_no_prefix(self):
        self._run([0, 0], [8, 8])

    def test_with_prefix(self):
        self._run([16, 32], [8, 8])


if __name__ == "__main__":
    unittest.main()
