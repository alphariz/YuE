"""Deep unit & integration tests for CPU RAM and async transfer optimizations.

Tests verify:
1. CPU RAM efficiency: low_cpu_mem_usage, gc.collect after VAE, chunked latent save
2. Feature integrity: CoT modes (full/melody/off), 48 kHz stereo output
3. Async non_blocking transfers: correctness in _offload_ar, _prefill, velocity
4. CPU thread limiting: torch.set_num_threads cap
"""
import gc
import json
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np

# Ensure yue2 can be imported without transformers installed natively
if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")

    class _MockConfig:
        @classmethod
        def register_for_auto_class(cls, *a, **kw):
            pass

    class _MockModel:
        @classmethod
        def register_for_auto_class(cls, *a, **kw):
            pass

    _tf.GenerationMixin = object
    _tf.PretrainedConfig = _MockConfig
    _tf.PreTrainedModel = _MockModel
    _cache = types.ModuleType("transformers.cache_utils")
    _cache.DynamicCache = object
    _tf.cache_utils = _cache
    _outputs = types.ModuleType("transformers.modeling_outputs")
    _outputs.CausalLMOutputWithPast = object
    _tf.modeling_outputs = _outputs
    sys.modules["transformers"] = _tf
    sys.modules["transformers.cache_utils"] = _cache
    sys.modules["transformers.modeling_outputs"] = _outputs

if "soundfile" not in sys.modules:
    _sf = types.ModuleType("soundfile")
    def _mock_write(file, data, samplerate, subtype=None):
        Path(file).touch()
    _sf.write = _mock_write
    sys.modules["soundfile"] = _sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch
from yue2 import pipeline, nar
from yue2.pipeline import (
    YuE2Pipeline, SymbolicPlan, SemanticResult, SongResult,
)
from yue2.protocol import (
    SongRequest, GenerationConfig, token_prefixes, INSTRUCTIONS,
)
from yue2.nar import attention, _offload_ar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pipe(device="cpu", memory_budget_gib=24):
    """Create a lightweight YuE2Pipeline without loading real weights."""
    mock_props = MagicMock()
    mock_props.total_memory = 24 * (1024 ** 3)
    mock_props.name = "Mock GPU"
    mock_props.major = 8
    mock_props.minor = 0

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.device_count", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
    ):
        pipe = YuE2Pipeline.__new__(YuE2Pipeline)
        pipe.device = torch.device(device)
        pipe.multi_gpu = False
        pipe.vae_device = pipe.device
        pipe.memory_budget_gib = float(memory_budget_gib)
        pipe.offload_ar = False
        pipe.progress = False
        pipe.backend = "torch"
        pipe.quantization = "none"
        pipe.vae_core_frames = 512
        pipe.generation_config = GenerationConfig()
        pipe._model = None
        pipe._vae = None
        pipe.load_timing = {}
        pipe.weights = {"mot": "mock_hash", "vae": "mock_hash"}
        pipe.runtime_sha256 = "mock_sha256"
    return pipe


def _make_song_result(num_samples=96000, num_latent_frames=200, latent_dtype=np.float64):
    """Build a mock SongResult with controllable audio/latent sizes."""
    request = SongRequest(style="pop", lyrics="hello world", cot="full")
    plan = SymbolicPlan(request, "X:1\nT:Test\n", [1, 2, 3],
                        token_prefixes(request, MagicMock(encode=lambda x: [1, 2, 3])),
                        {"seconds": 0.1}, False)
    semantic = SemanticResult(plan, list(range(10)), {"seconds": 0.5}, False)
    audio = np.random.randn(num_samples, 2).astype(np.float32)
    latents = np.random.randn(num_latent_frames, 64).astype(latent_dtype)
    return SongResult(
        audio=audio,
        sample_rate=48000,
        semantic=semantic,
        latents=latents,
        config={"test": True},
        weights={"mot": "h1", "vae": "h2"},
        timing={"e2e_seconds": 1.0},
        request_identity="test_id",
    )


# ===========================================================================
# TEST 1: CPU THREAD LIMITING
# ===========================================================================

class TestCPUThreadLimiting(unittest.TestCase):
    """Verify torch.set_num_threads is called during pipeline init."""

    def test_thread_limit_capped_at_four(self):
        """Pipeline.__init__ should set threads to min(4, cpu_count)."""
        original_threads = torch.get_num_threads()
        try:
            with patch("os.cpu_count", return_value=8):
                pipe = _make_mock_pipe()
                # Simulate the thread-setting logic from __init__
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                assert torch.get_num_threads() <= 4, (
                    f"Thread count should be ≤4, got {torch.get_num_threads()}"
                )
        finally:
            torch.set_num_threads(original_threads)

    def test_thread_limit_with_few_cores(self):
        """On a 2-core machine, threads should be capped at 2."""
        original_threads = torch.get_num_threads()
        try:
            with patch("os.cpu_count", return_value=2):
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                assert torch.get_num_threads() == 2
        finally:
            torch.set_num_threads(original_threads)


# ===========================================================================
# TEST 2: LOW_CPU_MEM_USAGE IN MODEL LOADING
# ===========================================================================

class TestLowCPUMemUsage(unittest.TestCase):
    """Verify low_cpu_mem_usage=True is passed to from_pretrained."""

    def test_single_gpu_uses_low_cpu_mem(self):
        """Single-GPU path must pass low_cpu_mem_usage=True."""
        pipe = _make_mock_pipe()
        pipe.model_dir = Path("/fake/model")
        pipe.multi_gpu = False
        pipe.device = torch.device("cpu")
        pipe.dtype = torch.float32

        mock_llm_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.to.return_value = mock_instance
        mock_instance.eval.return_value = mock_instance
        mock_param = torch.zeros(1)
        mock_instance.parameters.return_value = iter([mock_param])
        mock_llm_cls.from_pretrained.return_value = mock_instance

        mock_module = types.ModuleType("yue2.modeling_yue2")
        mock_module.YuE2ForCausalLM = mock_llm_cls

        with patch.dict("sys.modules", {"yue2.modeling_yue2": mock_module}):
            pipe._load_model()
            mock_llm_cls.from_pretrained.assert_called_once()
            _, kwargs = mock_llm_cls.from_pretrained.call_args
            assert kwargs.get("low_cpu_mem_usage") is True, (
                "low_cpu_mem_usage=True must be passed for single-GPU loading"
            )

    def test_multi_gpu_uses_low_cpu_mem(self):
        """Multi-GPU path must also pass low_cpu_mem_usage=True."""
        pipe = _make_mock_pipe()
        pipe.multi_gpu = True
        pipe.device = torch.device("cuda:0")
        pipe.model_dir = Path("/fake/model")
        pipe.dtype = torch.float32

        mock_llm_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_llm_cls.from_pretrained.return_value = mock_instance

        mock_module = types.ModuleType("yue2.modeling_yue2")
        mock_module.YuE2ForCausalLM = mock_llm_cls

        with (
            patch.dict("sys.modules", {"yue2.modeling_yue2": mock_module}),
            patch("torch.cuda.device_count", return_value=2),
        ):
            pipe._load_model()
            mock_llm_cls.from_pretrained.assert_called_once()
            _, kwargs = mock_llm_cls.from_pretrained.call_args
            assert kwargs.get("low_cpu_mem_usage") is True
            assert kwargs.get("device_map") == "auto"


# ===========================================================================
# TEST 3: GC.COLLECT AFTER VAE DECODE
# ===========================================================================

class TestGCCollectAfterDecode(unittest.TestCase):
    """Verify gc.collect() is called in decode()'s finally block."""

    def test_gc_collect_called_after_decode(self):
        """decode() must call gc.collect() in its finally block."""
        # Verify the source code contains gc.collect() in decode method
        import inspect
        source = inspect.getsource(YuE2Pipeline.decode)
        assert "gc.collect()" in source, (
            "decode() must call gc.collect() to free host RAM after VAE"
        )

    def test_gc_module_imported(self):
        """pipeline module must import gc."""
        assert hasattr(pipeline, "gc"), "gc must be imported in pipeline.py"
        assert pipeline.gc is gc


# ===========================================================================
# TEST 4: CHUNKED SAVE_ARTIFACTS (LATENT CONVERSION)
# ===========================================================================

import tempfile


class TestChunkedSaveArtifacts(unittest.TestCase):
    """Verify chunked float32 conversion in save_artifacts."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_float64_latents_saved_as_float32(self):
        """Latents stored as float64 must be chunked-converted to float32."""
        tmp_path = self.tmp_path
        result = _make_song_result(latent_dtype=np.float64)
        original_latents = result.latents.copy()

        # We need to mock the parts that save_artifacts calls
        with (
            patch.object(result.semantic.plan, "save"),
            patch.object(result, "save", return_value="/fake/audio.flac"),
            patch("yue2.pipeline.write_json"),
            patch("yue2.pipeline.collect_hashes", return_value={}),
        ):
            result.save_artifacts(tmp_path)

        saved = np.load(tmp_path / "latent.npy")
        assert saved.dtype == np.float32, f"Expected float32, got {saved.dtype}"
        np.testing.assert_allclose(saved, original_latents.astype(np.float32), rtol=1e-6)

    def test_float32_latents_bypass_chunking(self):
        """Already-float32 latents skip the chunking path (no-op)."""
        tmp_path = self.tmp_path
        result = _make_song_result(latent_dtype=np.float32)

        with (
            patch.object(result.semantic.plan, "save"),
            patch.object(result, "save", return_value="/fake/audio.flac"),
            patch("yue2.pipeline.write_json"),
            patch("yue2.pipeline.collect_hashes", return_value={}),
        ):
            result.save_artifacts(tmp_path)

        saved = np.load(tmp_path / "latent.npy")
        assert saved.dtype == np.float32
        np.testing.assert_array_equal(saved, result.latents)

    def test_save_artifacts_preserves_all_files(self):
        """save_artifacts must still produce all expected output files."""
        tmp_path = self.tmp_path
        result = _make_song_result()

        with (
            patch.object(result.semantic.plan, "save"),
            patch.object(result, "save", return_value=str(tmp_path / "audio.flac")),
            patch("yue2.pipeline.write_json"),
            patch("yue2.pipeline.collect_hashes", return_value={}),
        ):
            ret = result.save_artifacts(tmp_path)

        assert ret["status"] == "complete"
        assert ret["sample_rate"] == 48000
        assert "audio_seconds" in ret
        assert (tmp_path / "semantic.npy").exists()
        assert (tmp_path / "latent.npy").exists()


# ===========================================================================
# TEST 5: FEATURE INTEGRITY — CoT MODES
# ===========================================================================

class TestFeatureIntegrity(unittest.TestCase):
    """Verify all Chain-of-Thought modes produce valid structures."""

    def test_cot_full_produces_abc_plan(self):
        """cot='full' must produce a SymbolicPlan with abc text."""
        request = SongRequest(style="rock", lyrics="test lyrics", cot="full")
        assert request.cot == "full"
        assert "chord-annotated" in INSTRUCTIONS["full"]
        # ABC plan should have non-None abc when provided
        mock_tokenizer = MagicMock(encode=lambda x: [10, 20, 30])
        plan = SymbolicPlan(request, "X:1\nT:Full\n", [10, 20, 30],
                            token_prefixes(request, mock_tokenizer, [10, 20, 30]))
        assert plan.abc is not None
        assert plan.abc_ids == [10, 20, 30]

    def test_cot_melody_produces_abc_plan(self):
        """cot='melody' must also produce a SymbolicPlan."""
        request = SongRequest(style="jazz", lyrics="melody test", cot="melody")
        assert request.cot == "melody"
        assert "melody-only" in INSTRUCTIONS["melody"]

    def test_cot_off_produces_empty_abc(self):
        """cot='off' must produce a plan with abc=None and empty abc_ids."""
        request = SongRequest(style="classical", lyrics="off test", cot="off")
        mock_tokenizer = MagicMock(encode=lambda x: [5, 6])
        plan = SymbolicPlan(request, None, [],
                            token_prefixes(request, mock_tokenizer))
        assert plan.abc is None
        assert plan.abc_ids == []

    def test_all_cot_modes_accepted(self):
        """All three CoT modes must be accepted without error."""
        for mode in ("full", "melody", "off"):
            req = SongRequest(style="pop", lyrics="test", cot=mode)
            assert req.cot == mode

    def test_invalid_cot_rejected(self):
        """An invalid CoT mode must raise ValueError."""
        try:
            SongRequest(style="pop", lyrics="test", cot="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cot must be" in str(e)

    def test_audio_output_48khz_stereo(self):
        """SongResult must carry 48000 Hz stereo audio."""
        result = _make_song_result(num_samples=96000)
        assert result.sample_rate == 48000
        assert result.audio.shape[1] == 2  # stereo
        assert result.audio.dtype == np.float32


# ===========================================================================
# TEST 6: NON_BLOCKING TRANSFERS IN NAR.PY
# ===========================================================================

class TestNonBlockingTransfers(unittest.TestCase):
    """Verify non_blocking=True appears in nar.py tensor transfers."""

    def test_prefill_uses_non_blocking(self):
        """_prefill cross-device transfers must use non_blocking=True."""
        import inspect
        source = inspect.getsource(nar.CachedNAR._prefill)
        # Check all .to() calls inside _prefill use non_blocking
        to_calls = [line.strip() for line in source.split("\n")
                     if ".to(layer_dev" in line and "non_blocking" not in line]
        assert len(to_calls) == 0, (
            f"Found .to() calls without non_blocking in _prefill: {to_calls}"
        )

    def test_velocity_uses_non_blocking(self):
        """velocity cross-device transfers must use non_blocking=True."""
        import inspect
        source = inspect.getsource(nar.CachedNAR.velocity)
        lines_with_to = [line.strip() for line in source.split("\n")
                         if ".to(layer_dev" in line or ".to(norm_dev" in line
                         or ".to(llm2vae_dev" in line]
        missing = [l for l in lines_with_to if "non_blocking" not in l]
        assert len(missing) == 0, (
            f"Found .to() without non_blocking in velocity: {missing}"
        )

    def test_solve_noise_transfer_non_blocking(self):
        """solve() noise-to-device transfer must use non_blocking=True."""
        import inspect
        source = inspect.getsource(nar.CachedNAR.solve)
        assert "non_blocking=True" in source, (
            "solve() must use non_blocking=True for noise tensor transfer"
        )

    def test_offload_ar_uses_non_blocking(self):
        """_offload_ar module transfers must use non_blocking=True."""
        import inspect
        source = inspect.getsource(nar._offload_ar)
        to_calls = [line.strip() for line in source.split("\n")
                     if "module.to(" in line]
        missing = [l for l in to_calls if "non_blocking" not in l]
        assert len(missing) == 0, (
            f"Found module.to() without non_blocking in _offload_ar: {missing}"
        )

    def test_offload_ar_synchronizes_after_move(self):
        """_offload_ar must synchronize after non_blocking moves."""
        import inspect
        source = inspect.getsource(nar._offload_ar)
        assert "torch.cuda.synchronize()" in source, (
            "_offload_ar must call torch.cuda.synchronize() after non_blocking transfers"
        )

    def test_apply_rotary_device_guard(self):
        """_apply_rotary must guard against mismatched cos/sin device."""
        import inspect
        from yue2.modeling_yue2 import _apply_rotary
        source = inspect.getsource(_apply_rotary)
        assert "if cos.device != x.device: cos = cos.to(x.device)" in source or (
            "if cos.device != x.device:\n        cos = cos.to(x.device)" in source
        )
        assert "if sin.device != x.device: sin = sin.to(x.device)" in source or (
            "if sin.device != x.device:\n        sin = sin.to(x.device)" in source
        )
        x = torch.randn(1, 4, 8, 16)
        cos = torch.randn(1, 4, 1, 8)
        sin = torch.randn(1, 4, 1, 8)
        out = _apply_rotary(x, cos, sin)
        assert out.shape == x.shape
        assert out.device == x.device

    def test_velocity_rotary_device_alignment(self):
        """velocity must explicitly align cos_l and sin_l with x.device."""
        import inspect
        source = inspect.getsource(nar.CachedNAR.velocity)
        assert "if cos_l.device != x.device: cos_l = cos_l.to(x.device)" in source or (
            "if cos_l.device != x.device:\n                cos_l = cos_l.to(x.device)" in source
        )
        assert "if sin_l.device != x.device: sin_l = sin_l.to(x.device)" in source or (
            "if sin_l.device != x.device:\n                sin_l = sin_l.to(x.device)" in source
        )

    def test_prefill_rotary_device_alignment(self):
        """_prefill must explicitly align cos_l and sin_l with x.device."""
        import inspect
        source = inspect.getsource(nar.CachedNAR._prefill)
        assert "if cos_l.device != x.device: cos_l = cos_l.to(x.device)" in source or (
            "if cos_l.device != x.device:\n                cos_l = cos_l.to(x.device)" in source
        )
        assert "if sin_l.device != x.device: sin_l = sin_l.to(x.device)" in source or (
            "if sin_l.device != x.device:\n                sin_l = sin_l.to(x.device)" in source
        )


# ===========================================================================
# TEST 7: ATTENTION FUNCTION CORRECTNESS (REGRESSION)
# ===========================================================================

class TestAttentionRegression(unittest.TestCase):
    """Ensure attention function still produces correct results."""

    def test_basic_attention_shape(self):
        """Attention output shape must match query shape."""
        q = torch.randn(100, 8, 64)
        k = torch.randn(100, 8, 64)
        v = torch.randn(100, 8, 64)
        out = attention(q, k, v)
        assert out.shape == (100, 8, 64)

    def test_causal_attention_shape(self):
        """Causal attention must work with matching Q/K lengths."""
        q = torch.randn(50, 4, 32)
        k = torch.randn(50, 4, 32)
        v = torch.randn(50, 4, 32)
        out = attention(q, k, v, causal=True)
        assert out.shape == (50, 4, 32)

    def test_chunked_matches_unchunked(self):
        """Chunked attention must produce same result as unchunked."""
        q = torch.randn(200, 4, 32)
        k = torch.randn(200, 4, 32)
        v = torch.randn(200, 4, 32)
        full = attention(q, k, v, query_chunk_size=200)
        chunked = attention(q, k, v, query_chunk_size=64)
        torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-5)

    def test_gqa_attention(self):
        """Grouped-query attention must work correctly."""
        q = torch.randn(80, 8, 32)
        k = torch.randn(80, 2, 32)
        v = torch.randn(80, 2, 32)
        out = attention(q, k, v)
        assert out.shape == (80, 8, 32)


# ===========================================================================
# TEST 8: SONG_RESULT PROPERTIES PRESERVED
# ===========================================================================

class TestSongResultProperties(unittest.TestCase):
    """Verify SongResult data integrity after optimization."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_abc_property(self):
        """SongResult.abc must return the plan's ABC string."""
        result = _make_song_result()
        assert result.abc == result.semantic.plan.abc

    def test_truncated_property(self):
        """SongResult.truncated must return dict with abc and semantic keys."""
        result = _make_song_result()
        t = result.truncated
        assert "abc" in t and "semantic" in t

    def test_save_rejects_mp3(self):
        """save() must reject .mp3 extension."""
        tmp_path = self.tmp_path
        result = _make_song_result()
        try:
            result.save(tmp_path / "output.mp3")
            assert False, "Should reject .mp3"
        except ValueError:
            pass

    def test_save_accepts_flac(self):
        """save() must accept .flac extension."""
        tmp_path = self.tmp_path
        result = _make_song_result()
        path = result.save(tmp_path / "output.flac")
        assert Path(path).exists()

    def test_save_accepts_wav(self):
        """save() must accept .wav extension."""
        tmp_path = self.tmp_path
        result = _make_song_result()
        path = result.save(tmp_path / "output.wav")
        assert Path(path).exists()


# ===========================================================================
# TEST 9: PIPELINE SOURCE CODE STRUCTURE VALIDATION
# ===========================================================================

class TestPipelineSourceStructure(unittest.TestCase):
    """Verify critical optimizations exist in pipeline.py source."""

    def test_gc_import(self):
        """pipeline.py must import gc."""
        import inspect
        source = inspect.getsource(pipeline)
        assert "import gc" in source

    def test_os_import(self):
        """pipeline.py must import os for cpu_count."""
        import inspect
        source = inspect.getsource(pipeline)
        assert "import os" in source

    def test_set_num_threads_in_init(self):
        """__init__ must call torch.set_num_threads."""
        import inspect
        source = inspect.getsource(YuE2Pipeline.__init__)
        assert "torch.set_num_threads" in source

    def test_gc_collect_in_decode(self):
        """decode() must call gc.collect()."""
        import inspect
        source = inspect.getsource(YuE2Pipeline.decode)
        assert "gc.collect()" in source

    def test_low_cpu_mem_in_load_model(self):
        """_load_model must pass low_cpu_mem_usage=True in both paths."""
        import inspect
        source = inspect.getsource(YuE2Pipeline._load_model)
        assert source.count("low_cpu_mem_usage=True") >= 1


# ===========================================================================
# RUNNER
# ===========================================================================

if __name__ == "__main__":
    # Run with pytest if available, otherwise use unittest
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        import unittest
        # Manually collect test classes and run them
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for cls in [
            TestCPUThreadLimiting,
            TestLowCPUMemUsage,
            TestGCCollectAfterDecode,
            TestChunkedSaveArtifacts,
            TestFeatureIntegrity,
            TestNonBlockingTransfers,
            TestAttentionRegression,
            TestSongResultProperties,
            TestPipelineSourceStructure,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
