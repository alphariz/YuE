"""Memory-bounded acoustic flow matching with one AR prefill per original chunk.

Only PyTorch is required. The reference 32-step midpoint solver, full-song CPU
FP32 noise draw, boundary positions, and original context chunks are preserved.
Attention query tiling changes temporary storage, never the visible key set.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END, chunk_ranges


@dataclass
class Chunk:
    ar_tokens: list[int]
    noise: torch.Tensor
    nar_cond_end: int = 0


def _integers(values, name):
    result = list(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError(f"{name} must be a nonempty sequence of integer token IDs")
    return [int(v) for v in result]


def song_chunks(prefix, codec, seed, context=CONTEXT):
    """Draw the complete noise tensor once, then take views at historical cuts."""
    prefix = _integers(prefix, "prefix")
    codec = _integers(codec, "codec")
    if min(prefix) < 0 or min(codec) < 0 or max(codec) >= CODEC_SIZE:
        raise ValueError("Token IDs are outside their allowed vocabulary")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(context, bool) or not isinstance(context, Integral) or not 1 <= context <= CONTEXT:
        raise ValueError(f"context must be an integer in 1..{CONTEXT}")
    ranges = chunk_ranges(len(codec), len(prefix), int(context))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
    return [Chunk(prefix + [value + CODEC_OFFSET for value in codec[a:b]] + [MUSIC_END], noise[a:b])
            for a, b in ranges]


def attention(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
    """Attend [tokens, heads, dim] tensors without materializing a song mask.

    CPU/MPS bound the number of query rows for a potential math SDPA fallback.
    CUDA normally uses PyTorch's fused SDPA without an external flash package.
    """
    if backend not in {"sdpa", "math", "flash"}:
        raise ValueError("attention must be sdpa, math, or flash")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Expected Q/K/V [tokens, heads, dim] with matching K/V")
    if min(q.shape) < 1 or min(k.shape) < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid attention lengths or grouped-query head count")
    if causal and len(q) != len(k):
        raise ValueError("Causal prefill requires matching Q/K sequence lengths")
    if backend == "flash" and q.device.type != "cuda":
        raise ValueError("Explicit flash SDPA requires a CUDA device")
    if query_chunk_size is not None and (isinstance(query_chunk_size, bool) or
                                        not isinstance(query_chunk_size, Integral) or query_chunk_size < 1):
        raise ValueError("query_chunk_size must be a positive integer")
    if query_chunk_size is not None:
        block = query_chunk_size
    elif q.device.type == "cuda" and backend != "math":
        try:
            total_gib = torch.cuda.get_device_properties(q.device).total_memory / (1024 ** 3)
            block = 512 if total_gib < 18.0 else len(q)
        except Exception:
            block = 512
    else:
        block = 256
    query = q.transpose(0, 1).unsqueeze(0)
    key = k.transpose(0, 1).unsqueeze(0)
    value = v.transpose(0, 1).unsqueeze(0)
    grouped = query.shape[1] != key.shape[1]
    if grouped and q.device.type == "mps":
        groups = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
        grouped = False
    context = nullcontext()
    if backend != "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        context = sdpa_kernel(SDPBackend.MATH if backend == "math" else SDPBackend.FLASH_ATTENTION)
    outputs = []
    with context:
        for start in range(0, len(q), block):
            end = min(start + block, len(q))
            used_key = key[..., :end, :] if causal else key
            used_value = value[..., :end, :] if causal else value
            # is_causal on a rectangular Q/K uses an upper-left triangle, so a
            # later query block needs its absolute query positions explicitly.
            mask = None
            if causal and start:
                mask = (torch.arange(end, device=q.device)[None, :] <=
                        torch.arange(start, end, device=q.device)[:, None])
            outputs.append(F.scaled_dot_product_attention(
                query[..., start:end, :], used_key, used_value,
                attn_mask=mask, is_causal=causal and start == 0, enable_gqa=grouped,
            ))
    return torch.cat(outputs, dim=-2)[0].transpose(0, 1)


def _sync_device(device):
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.current_stream(device).synchronize()


class CachedNAR:
    """One original acoustic chunk; AR prefix KV is invariant during the ODE."""

    def __init__(self, model, chunk: Chunk, attention="sdpa", query_chunk_size=None):
        self.model, self.chunk = model, chunk
        self.backend, self.query_chunk_size = attention, query_chunk_size
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        if chunk.noise.ndim != 2 or chunk.noise.shape[1] != 64 or len(chunk.noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        if not torch.isfinite(chunk.noise).all():
            raise ValueError("Acoustic noise contains non-finite values")
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(chunk.noise) + 2
        if self.ar_length < 1 or min(chunk.ar_tokens) < 0 or max(chunk.ar_tokens) >= model.config.vocab_size:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.config.max_position_embeddings:
            raise ValueError("Original acoustic chunk exceeds the model context")
        if chunk.nar_cond_end < 0:
            raise ValueError("nar_cond_end must be nonnegative")
        self.visible_length = min(chunk.nar_cond_end, self.ar_length) if chunk.nar_cond_end else self.ar_length
        positions = torch.arange(self.ar_length, self.ar_length + self.nar_length, device=self.device)[None]
        self.cos, self.sin = model.model.rotary_emb(positions)
        local = torch.arange(self.nar_length, device=self.device).clamp(max=model.config.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache = []
        self._prefill()

    def _attention(self, q, k, v, causal=False):
        return attention(q, k, v, causal=causal, backend=self.backend, query_chunk_size=self.query_chunk_size)

    @torch.inference_mode()
    def _prefill(self):
        backbone = self.model.model
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = backbone.rotary_emb(positions)
        x = backbone.embed_tokens(ids)
        for layer in backbone.layers:
            layer_dev = next(layer.parameters()).device
            if x.device != layer_dev:
                x = x.to(layer_dev, non_blocking=True)
            cos_l = cos.to(layer_dev, non_blocking=True) if cos.device != layer_dev else cos
            sin_l = sin.to(layer_dev, non_blocking=True) if sin.device != layer_dev else sin
            if cos_l.device != x.device:
                cos_l = cos_l.to(x.device)
            if sin_l.device != x.device:
                sin_l = sin_l.to(x.device)
            _sync_device(layer_dev)
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos_l, sin_l)
            # Clone only for restricted visibility; a slice would retain the
            # storage of invisible codec tokens for every layer.
            cached = (k[0, :self.visible_length], v[0, :self.visible_length])
            if self.visible_length != self.ar_length:
                cached = tuple(t.clone() for t in cached)
            self.cache.append(cached)
            h = self._attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def velocity(self, state, raw_t):
        model = self.model
        if tuple(state.shape) != tuple(self.chunk.noise.shape):
            raise ValueError("ODE state shape changed")
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = model._shift_t_value(raw_t, self.device, self.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb
        for layer_idx, layer in enumerate(model.model.layers):
            layer_dev = next(layer.parameters()).device
            if x.device != layer_dev:
                x = x.to(layer_dev)
            ar_k, ar_v = self.cache[layer_idx]
            if ar_k.device != layer_dev:
                ar_k = ar_k.to(layer_dev)
            if ar_v.device != layer_dev:
                ar_v = ar_v.to(layer_dev)
            cos_l = self.cos.to(layer_dev, non_blocking=True) if self.cos.device != layer_dev else self.cos
            sin_l = self.sin.to(layer_dev, non_blocking=True) if self.sin.device != layer_dev else self.sin
            if cos_l.device != x.device:
                cos_l = cos_l.to(x.device)
            if sin_l.device != x.device:
                sin_l = sin_l.to(x.device)
            _sync_device(layer_dev)
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), cos_l, sin_l)
            if ar_k.device != k[0].device:
                ar_k = ar_k.to(k[0].device)
            if ar_v.device != v[0].device:
                ar_v = ar_v.to(v[0].device)
            _sync_device(layer_dev)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = self._attention(q[0], k, v)
            if h.device != layer_dev:
                h = h.to(layer_dev)
            attn_out = layer.nar_self_attn.o_proj(h.flatten(1)[None])
            if attn_out.device != x.device:
                attn_out = attn_out.to(x.device)
            x = x + attn_out
            mlp_out = layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
            if mlp_out.device != x.device:
                mlp_out = mlp_out.to(x.device)
            x = x + mlp_out
        norm_dev = next(model.model.norm.parameters()).device
        if x.device != norm_dev:
            x = x.to(norm_dev, non_blocking=True)
        llm2vae_dev = next(model.llm2vae.parameters()).device
        _sync_device(norm_dev)
        normed = model.model.norm(x)
        if normed.device != llm2vae_dev:
            normed = normed.to(llm2vae_dev, non_blocking=True)
        _sync_device(llm2vae_dev)
        return model.llm2vae(normed)[0, 1:-1]

    @torch.inference_mode()
    def solve(self, steps=32, cancelled: Callable[[], bool] | None = None,
              on_progress: Callable[[int, int], None] | None = None):
        """Solve a chunk, reporting each submitted midpoint step without syncing.

        CUDA work may still be executing when ``on_progress`` runs. The existing
        CPU result transfer completes that work before this method returns.
        Callback exceptions propagate to the caller.
        """
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        state = self.chunk.noise.to(device=self.device, dtype=self.dtype, non_blocking=True)
        dt = 1.0 / steps
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            mid = state - first * (dt / 2)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            state = state - self.velocity(mid, raw_mid) * dt
            if on_progress is not None:
                on_progress(step + 1, int(steps))
        result = state.float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self):
        self.cache.clear()
        self.cos = self.sin = self.pos_emb = None


@contextmanager
def _offload_ar(model, enabled):
    """Temporarily move unused AR modules; this model cannot serve concurrently."""
    modules = [model.model.embed_tokens, model.lm_head]
    for layer in model.model.layers:
        modules.extend((layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp))
    moved = []
    try:
        if enabled:
            for module in modules:
                device = next(module.parameters()).device
                if device.type != "cpu":
                    module.to(device="cpu", non_blocking=True)
                    moved.append((module, device))
            # Synchronize before proceeding to ensure all transfers complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        yield
    finally:
        for module, device in moved:
            module.to(device=device, non_blocking=True)
        # Synchronize after restore to ensure weights are on-device before use
        if moved and torch.cuda.is_available():
            torch.cuda.synchronize()


@torch.inference_mode()
def synthesize(model, prefix: Sequence[int], codec: Sequence[int], seed: int,
               steps=32, context=CONTEXT, attention="sdpa", offload_ar=False,
               cancelled=None, query_chunk_size=None,
               on_progress: Callable[[int, int], None] | None = None):
    """Return CPU FP32 [frames,64] latents, solving original chunks serially.

    Defaults preserve the release protocol. Explicit steps/context overrides
    belong in the caller's effective configuration record. ``offload_ar`` is
    an optional memory tradeoff and requires exclusive access to ``model``.
    Progress counts submitted midpoint steps across all original chunks; it
    introduces no device synchronization. Callback exceptions propagate after
    the current chunk's cache is released and any offloaded weights restored.
    """
    if model.training:
        raise ValueError("synthesize requires model.eval()")
    if query_chunk_size is None:
        try:
            param = next(model.parameters(), None)
            if param is not None and param.device.type == "cuda":
                total_gib = torch.cuda.get_device_properties(param.device).total_memory / (1024 ** 3)
                if total_gib < 18.0:
                    query_chunk_size = 512
        except Exception:
            pass
    chunks = song_chunks(prefix, codec, seed, context)
    output = []
    for chunk_index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        engine = CachedNAR(model, chunk, attention, query_chunk_size)
        # Drop the prefix cache before restoring AR weights, including on
        # cancellation/failure, to keep the restoration memory peak bounded.
        with _offload_ar(model, offload_ar):
            try:
                progress = None
                if on_progress is not None:
                    def progress(completed, total):
                        on_progress(chunk_index * total + completed, total * len(chunks))
                output.append(engine.solve(steps, cancelled, on_progress=progress))
            finally:
                engine.close()
        del engine
    return torch.cat(output, dim=0)
