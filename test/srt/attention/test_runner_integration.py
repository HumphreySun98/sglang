"""
Phase 3: Runner integration tests — CUDA graph consistency.

Step A (eager correctness) is already proven by Phase 2.
Step B (graph consistency): capture a CUDA graph with the same inputs as the
eager run and verify the replay output matches the eager output.

Pattern for each test:
  1. make_decode_batch → pre-populate KV prefix.
  2. Eager: init_forward_metadata → layer(q, k, v, batch) → eager_out.
  3. Graph:
     a. init_cuda_graph_state (allocate static buffers)
     b. init_forward_metadata_capture_cuda_graph (fill static buffers; outside graph)
     c. Warmup: two eager forwards with same q, k, v, batch (outside graph)
     d. Capture: torch.cuda.graph → layer(q, k, v, batch) → graph_out_buf
     e. Replay: init_forward_metadata_replay_cuda_graph; g.replay()
     f. Compare graph_out_buf ≈ eager_out.

q, k, v, batch are the SAME objects in both eager and graph runs, so any
differences expose graph bookkeeping bugs rather than numerical tolerances.
"""

import math
import unittest

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.test_utils import CustomTestCase

from .utils import (
    GEMMA_SWA_CONFIG,
    GEMMA_SWA_CONFIG_PS1,
    GPT2_CONFIG,
    GPT2_CONFIG_PS1,
    LLAMA3_CONFIG,
    LLAMA3_CONFIG_PS1,
    MockModelRunner,
    assert_close,
    build_fa_backend,
    build_flashinfer_backend,
    build_triton_backend,
    make_decode_batch,
)

DEVICE = "cuda"
DTYPE = torch.float16
LAYER_ID = 0
SEED = 42


def _make_layer(cfg) -> RadixAttention:
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


def _run_graph_decode_consistency(
    layer: RadixAttention,
    backend,
    batch: ForwardBatch,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eager_out: torch.Tensor,
):
    """
    Capture a CUDA graph for a decode forward and assert replay == eager_out.

    batch, q, k, v are the SAME objects used for the eager baseline so the
    graph sees identical KV cache state and inputs.  The graph path exercises:
      init_cuda_graph_state → init_forward_metadata_capture_cuda_graph
      → (warmup ×2) → CUDAGraph.capture → init_forward_metadata_replay_cuda_graph
      → g.replay().
    """
    bsz = batch.batch_size
    num_heads = layer.tp_q_head_num
    v_head_dim = layer.v_head_dim

    # Allocate static backend buffers.
    backend.init_cuda_graph_state(max_bs=bsz, max_num_tokens=bsz)

    # Fill static metadata buffers (outside graph context).
    backend.init_forward_metadata_capture_cuda_graph(
        bs=bsz,
        num_tokens=bsz,
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
    )

    # Warmup — two passes outside the graph to warm compiled kernels.
    for _ in range(2):
        torch.cuda.synchronize()
        with forward_context(ForwardContext(attn_backend=backend)):
            _ = layer(q, k, v, batch)
        backend.on_after_cuda_graph_warmup()

    # Capture — only the layer forward is recorded in the graph.
    graph_out_buf = torch.zeros(
        bsz, num_heads * v_head_dim, dtype=DTYPE, device=DEVICE
    )
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        with forward_context(ForwardContext(attn_backend=backend)):
            cap_out = layer(q, k, v, batch)
        graph_out_buf.copy_(cap_out)
    torch.cuda.synchronize()

    # Replay with same batch → same kv_indices → same output.
    backend.init_forward_metadata_replay_cuda_graph(
        bs=bsz,
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        seq_lens_sum=int(batch.seq_lens.sum().item()),
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=batch.seq_lens_cpu,
    )
    g.replay()
    torch.cuda.synchronize()

    graph_out = graph_out_buf.float().view(bsz, num_heads, v_head_dim)
    assert_close(
        eager_out, graph_out, atol=5e-3, rtol=5e-3,
        msg=f"Graph vs eager decode bsz={bsz}",
    )


def _run_decode(backend, layer, runner, seq_lens):
    """Eager decode pass; returns (batch, q, k, v, eager_out [float32])."""
    cfg = runner.model_config
    bsz = len(seq_lens)
    batch, _ = make_decode_batch(seq_lens, runner, layer_id=LAYER_ID)
    q = _rand([bsz, cfg.num_attention_heads, cfg.head_dim])
    k = _rand([bsz, cfg._num_kv_heads, cfg.head_dim])
    v = _rand([bsz, cfg._num_kv_heads, cfg.v_head_dim])
    with forward_context(ForwardContext(attn_backend=backend)):
        backend.init_forward_metadata(batch)
        eager_flat = layer(q, k, v, batch)
    num_heads = layer.tp_q_head_num
    v_head_dim = layer.v_head_dim
    eager_out = eager_flat.float().view(bsz, num_heads, v_head_dim)
    return batch, q, k, v, eager_out


# ---------------------------------------------------------------------------
# Triton backend — CUDA graph consistency, DECODE
# ---------------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonDecodeGraphConsistency(CustomTestCase):
    """Triton backend: graph capture+replay output matches eager decode output."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        batch, q, k, v, eager_out = _run_decode(
            self.backend, self.layer, self.runner, seq_lens
        )
        _run_graph_decode_consistency(self.layer, self.backend, batch, q, k, v, eager_out)

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonGQADecodeGraphConsistency(CustomTestCase):
    """Triton backend, GQA: graph capture+replay matches eager decode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        batch, q, k, v, eager_out = _run_decode(
            self.backend, self.layer, self.runner, seq_lens
        )
        _run_graph_decode_consistency(self.layer, self.backend, batch, q, k, v, eager_out)

    def test_bsz1(self):
        self._run([16])

    def test_bsz4(self):
        self._run([8, 16, 32, 64])


# ---------------------------------------------------------------------------
# Triton backend — CUDA graph consistency, SWA DECODE
# ---------------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTritonSWADecodeGraphConsistency(CustomTestCase):
    """Triton backend, SWA: graph capture+replay matches eager decode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GEMMA_SWA_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_triton_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        batch, q, k, v, eager_out = _run_decode(
            self.backend, self.layer, self.runner, seq_lens
        )
        _run_graph_decode_consistency(self.layer, self.backend, batch, q, k, v, eager_out)

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])


# ---------------------------------------------------------------------------
# FlashInfer backend — CUDA graph consistency, DECODE
# ---------------------------------------------------------------------------

@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFlashInferDecodeGraphConsistency(CustomTestCase):
    """FlashInfer backend: graph capture+replay output matches eager decode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flashinfer_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        batch, q, k, v, eager_out = _run_decode(
            self.backend, self.layer, self.runner, seq_lens
        )
        _run_graph_decode_consistency(self.layer, self.backend, batch, q, k, v, eager_out)

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestFlashInferGQADecodeGraphConsistency(CustomTestCase):
    """FlashInfer backend, GQA: graph capture+replay matches eager decode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_flashinfer_backend(cls.runner)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        batch, q, k, v, eager_out = _run_decode(
            self.backend, self.layer, self.runner, seq_lens
        )
        _run_graph_decode_consistency(self.layer, self.backend, batch, q, k, v, eager_out)

    def test_bsz1(self):
        self._run([16])

    def test_bsz4(self):
        self._run([8, 16, 32, 64])


# ---------------------------------------------------------------------------
# FA3 backend — CUDA graph consistency, DECODE (SM90+ only)
# ---------------------------------------------------------------------------

_HAS_FA3 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func as _fa3_probe  # noqa: F401
    _FA3_AVAILABLE = True
except Exception:
    _FA3_AVAILABLE = False


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3DecodeGraphConsistency(CustomTestCase):
    """FA3 backend: graph capture+replay output matches eager decode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GPT2_CONFIG_PS1
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        batch, q, k, v, eager_out = _run_decode(
            self.backend, self.layer, self.runner, seq_lens
        )
        _run_graph_decode_consistency(self.layer, self.backend, batch, q, k, v, eager_out)

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3GQADecodeGraphConsistency(CustomTestCase):
    """FA3 backend, GQA: graph capture+replay matches eager decode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = LLAMA3_CONFIG_PS1
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        batch, q, k, v, eager_out = _run_decode(
            self.backend, self.layer, self.runner, seq_lens
        )
        _run_graph_decode_consistency(self.layer, self.backend, batch, q, k, v, eager_out)

    def test_bsz1(self):
        self._run([16])

    def test_bsz4(self):
        self._run([8, 16, 32, 64])


@unittest.skipIf(not _HAS_FA3 or not _FA3_AVAILABLE, "FA3 requires SM90+ and sgl_kernel.flash_attn")
class TestFA3SWADecodeGraphConsistency(CustomTestCase):
    """FA3 backend, SWA: graph capture+replay matches eager decode."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cfg = GEMMA_SWA_CONFIG_PS1
        cls.cfg = cfg
        cls.runner = MockModelRunner(cfg, device=DEVICE, dtype=DTYPE)
        cls.backend = build_fa_backend(cls.runner, fa_version=3)
        cls.layer = _make_layer(cfg)

    def _run(self, seq_lens):
        batch, q, k, v, eager_out = _run_decode(
            self.backend, self.layer, self.runner, seq_lens
        )
        _run_graph_decode_consistency(self.layer, self.backend, batch, q, k, v, eager_out)

    def test_bsz1(self):
        self._run([8])

    def test_bsz4(self):
        self._run([4, 8, 16, 32])


if __name__ == "__main__":
    unittest.main()
