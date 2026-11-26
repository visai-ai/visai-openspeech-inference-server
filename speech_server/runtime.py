"""ONNX Runtime engine: sessions, batching with length bucketing, long-form windows, CTC / greedy / beam decoding.

Depends on onnxruntime, numpy and sentencepiece only (soundfile for file input). On CUDA every large tensor
(encoder output, cross K/V, the self-attention cache) stays on the device through IO binding; per decoder step the
host sends the (B, 1) token ids and reads back (B,) next ids.

Greedy is plain greedy: argmax every step, no loop fallback, no length heuristics; a row stops at <|eot|> or after
max_new_tokens (default 448; at most the model's text-position limit). See the README's "Limitations".
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from speech_server import audio as audiolib
from speech_server.spec import ModelSpec
from speech_server.text import Detokenizer

_DT = {"fp32": np.float32, "fp16": np.float16, "int8": np.float32}
STREAM_MODES = ("ctc-stream", "dec-stream")      # the chunk encoder modes (fp32, greedy, one stream per input)
MODES = ("decoder", "ctc") + STREAM_MODES


class InputError(ValueError):
    """A request the engine refuses because of its own input or options (bad option value, stream too long, a
    feature this export lacks). Its message is ours and safe to return to the client (the server's `detail`)."""


class TooLong(InputError):
    """The input is longer than this server takes at all (its length limit, or more decoded audio than the whole
    in-flight budget): the server's 413, never a retryable 503."""
DEFAULT_MAX_NEW = 448


def _canon_providers(providers) -> tuple:
    """Provider list in a comparable form: option order does not matter, cache paths are resolved."""
    out = []
    for p in providers or []:
        if isinstance(p, tuple):
            opts = {k: (os.path.realpath(v) if k.endswith("_path") and isinstance(v, str) else str(v))
                    for k, v in (p[1] or {}).items()}
            out.append((p[0], tuple(sorted(opts.items()))))
        else:
            out.append((str(p), ()))
    return tuple(out)


_STATS_LOCK = threading.Lock()   # Stats.add vs Stats.snapshot (GET /stats): never a half-added batch


@dataclass
class Stats:
    audio_seconds: float = 0.0
    windows: int = 0
    encode_s: float = 0.0
    decode_s: float = 0.0
    steps: int = 0
    tokens: int = 0
    wall_s: float = 0.0
    batches: int = 0
    extra: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        """The counters as one consistent set (taken under the lock add() holds)."""
        with _STATS_LOCK:
            return {k: getattr(self, k) for k in ("audio_seconds", "windows", "encode_s", "decode_s", "steps",
                                                  "tokens", "wall_s", "batches")}

    def add(self, o: "Stats"):
        with _STATS_LOCK:
            for k in ("audio_seconds", "windows", "encode_s", "decode_s", "steps", "tokens", "wall_s", "batches"):
                setattr(self, k, getattr(self, k) + getattr(o, k))


class _KV:
    """Self-attention cache in two preallocated ping-pong buffer pairs (k, v), each L x Bmax x H x Tmax x dh: the
    decoder step reads past (L, B, H, t, dh) from one pair and writes present (L, B, H, t + U, dh) into the other, both
    bound by pointer (contiguous prefixes), so the loop allocates nothing for the cache however long it runs."""

    def __init__(self, eng, Bmax: int, Tmax: int):
        self.e, self.B, self.t, self.cur = eng, Bmax, 0, 0
        n = eng.L * Bmax * eng.H * Tmax * eng.dh
        ov = eng.ort.OrtValue.ortvalue_from_shape_and_type
        self.buf = [[ov([n], eng.kv_dtype, eng.dev, 0) for _ in range(2)] for _ in range(2)]   # [pair][k, v]

    def _shape(self, B, t):
        return [self.e.L, B, self.e.H, t, self.e.dh]

    def _bind(self, io, names, pair, B, t, out: bool):
        for name, ov in zip(names, self.buf[pair]):
            f = io.bind_output if out else io.bind_input
            f(name, self.e.dev, 0, self.e.kv_dtype, self._shape(B, t), ov.data_ptr())

    def bind_step(self, io, U: int):
        self._bind(io, ("past_k", "past_v"), self.cur, self.B, self.t, False)
        self._bind(io, ("present_k", "present_v"), 1 - self.cur, self.B, self.t + U, True)

    def advance(self, U: int):
        self.cur, self.t = 1 - self.cur, self.t + U

    def gather(self, idx: np.ndarray):
        """Rows idx of the past (beam reorder / dropping finished rows) into the other pair."""
        g = self.e.gather
        io = g.io_binding()
        self._bind(io, ("x", "x2"), self.cur, self.B, self.t, False)
        io.bind_cpu_input("idx", idx.astype(np.int64))
        self._bind(io, ("y", "y2"), 1 - self.cur, len(idx), self.t, True)
        g.run_with_iobinding(io)
        self.cur, self.B = 1 - self.cur, len(idx)


class _KV2:
    """Per-layer cache for the v2 step graph: two ping-pong pairs (k, v) of flat device buffers; layer l of a pair
    is the region [l * R, (l + 1) * R) with R = Bcap * H * Tcap * dh elements, and its (B, H, t, dh) tensor is that
    region's contiguous prefix. Capacity starts from the CTC hypothesis length and grows (copy) if a row runs past it,
    so the cap never limits decoding."""

    def __init__(self, eng, Bcap: int, Tcap: int):
        self.e, self.B, self.t, self.cur = eng, Bcap, 0, 0
        self.es = np.dtype(eng.kv_dtype).itemsize
        self._alloc(Bcap, Tcap)

    def _alloc(self, Bcap, Tcap):
        e = self.e
        self.Bcap, self.Tcap = Bcap, Tcap
        self.R = Bcap * e.H * Tcap * e.dh
        ov = e.ort.OrtValue.ortvalue_from_shape_and_type
        self.buf = [[ov([e.L * self.R], e.kv_dtype, e.dev, 0) for _ in range(2)] for _ in range(2)]

    def _bind(self, io, prefix, pair, B, t, out, buf=None, R=None):
        e = self.e
        buf, R = buf or self.buf, R or self.R
        f = io.bind_output if out else io.bind_input
        for kv, name in ((0, "k"), (1, "v")):
            base = buf[pair][kv].data_ptr()
            for l in range(e.L):
                f(f"{prefix}_{name}{l}", e.dev, 0, e.kv_dtype, [B, e.H, t, e.dh], base + l * R * self.es)

    def bind_prefill(self, io, U):
        self._bind(io, "present", 1 - self.cur, self.B, U, True)

    def bind_step(self, io):
        if self.t + 1 > self.Tcap:
            self.grow(min(self.e.Tmax, max(self.Tcap * 2, self.t + 1)))
        self._bind(io, "past", self.cur, self.B, self.t, False)
        self._bind(io, "present", 1 - self.cur, self.B, self.t + 1, True)

    def advance(self, U):
        self.cur, self.t = 1 - self.cur, self.t + U

    def _gather(self, idx, dst_buf, dst_R, dst_pair):
        e = self.e
        io = e.gather_layers.io_binding()
        names_in = [f"x{i}" for i in range(2 * e.L)]
        names_out = [f"y{i}" for i in range(2 * e.L)]
        for j, (kv, l) in enumerate([(0, l) for l in range(e.L)] + [(1, l) for l in range(e.L)]):
            io.bind_input(names_in[j], e.dev, 0, e.kv_dtype, [self.B, e.H, self.t, e.dh],
                          self.buf[self.cur][kv].data_ptr() + l * self.R * self.es)
            io.bind_output(names_out[j], e.dev, 0, e.kv_dtype, [len(idx), e.H, self.t, e.dh],
                           dst_buf[dst_pair][kv].data_ptr() + l * dst_R * self.es)
        io.bind_cpu_input("idx", idx.astype(np.int64))
        e.gather_layers.run_with_iobinding(io)

    def gather(self, idx: np.ndarray):
        self._gather(idx, self.buf, self.R, 1 - self.cur)
        self.cur, self.B = 1 - self.cur, len(idx)

    def grow(self, Tcap):
        old_buf = self.buf
        e = self.e
        self.Bcap = self.B                        # rows still decoding only (finished rows were compacted away)
        R = self.Bcap * e.H * Tcap * e.dh
        ov = e.ort.OrtValue.ortvalue_from_shape_and_type
        new = [[ov([e.L * R], e.kv_dtype, e.dev, 0) for _ in range(2)] for _ in range(2)]
        self._gather(np.arange(self.B), new, R, 0)
        self.buf, self.R, self.Tcap, self.cur = new, R, Tcap, 0
        del old_buf


class Engine:
    def __init__(self, model_dir: str, device: str = "cuda", provider: str | None = None, precision: str = "fp16",
                 threads: int | None = None, mode: str = "decoder", decode: str = "greedy", beam: int = 4,
                 batch_size: int = 64, max_batch_seconds: float | None = 600.0, lang: str | None = None,
                 max_new_tokens: int | None = None, max_seconds: float = 30.0, min_seconds: float = 20.0,
                 len_alpha: float = 1.0, compact: bool = True, graph_opt: str = "all",
                 cudnn_search: str = "HEURISTIC", decode_threads: int | None = None, sort: str = "length",
                 itn: bool = False, pnc: bool = False, graphs: str = "v2", encoder_graph: str = "auto",
                 kv_margin: int = 32, step_graph: str = "lean", decoder_precision: str | None = None,
                 cuda_opts: dict | None = None, decode_batch: int = 256, kv_budget_mb: float = 6144,
                 gpu_mem_limit_mb: float | None = None, gpu_arena_shrink_every: int = 8, cut: str = "energy", cut_min_blank_ms: int = 240,
                 timestamps: str = "none", lookahead: int | None = None, endpoint_frames: int = 8,
                 max_segment_frames: int = 375, rebase_frames: int = 7500, dec_format: str | None = None,
                 stream_share=None, enc_share=None, full_encoder: bool = False, word_boost: bool = True,
                 boost_per_char: float | None = None):
        import onnxruntime as ort
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device {device!r}: cpu | cuda")
        if decode not in ("greedy", "beam"):
            raise ValueError(f"decode {decode!r}: greedy | beam")
        if decode == "beam" and not (isinstance(beam, int) and 1 <= beam <= 16):
            raise ValueError(f"beam {beam!r}: an integer from 1 to 16")
        if precision not in _DT:
            raise ValueError(f"precision {precision!r}: fp32 | fp16 | int8")
        if precision == "int8" and device != "cpu":
            raise ValueError("--precision int8 is a CPU-only package (dynamic MatMul quantisation)")
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: decoder | ctc | ctc-stream | dec-stream")
        if mode in ("ctc", "ctc-stream") and decode == "beam":
            raise ValueError("--mode ctc decodes greedily (no CTC beam yet)")
        if mode == "dec-stream" and decode == "beam":
            raise ValueError("--mode dec-stream decodes greedily (the streaming decoder has no beam)")
        if mode in STREAM_MODES and precision != "fp32":
            raise ValueError(f"--mode {mode}: the streaming chunk graphs are fp32 only (use --precision fp32)")
        self._sessions = {}               # (graph, session settings) -> session; shared with an encoder donor
        self._sess_lock = threading.RLock()   # guards _sessions (shared with the donor too)
        self.timestamps = timestamps
        _check_ts(mode, timestamps)
        self.ort = ort
        self.model_dir, self.device, self.precision, self.mode, self.decode_kind = (
            model_dir, device, precision, mode, decode)
        # another streaming Engine on the same package (the server's ctc-stream / dec-stream pair): its front end and
        # chunk graph sessions are reused instead of loaded again (they are stateless: state lives in each stream)
        self._stream_share = stream_share
        # encoder sharing between offline engines (the server's --share-encoder, the default): `enc_share` is another
        # decoder / ctc Engine on the same package, precision and device whose encoder session this one reuses (one
        # copy of the encoder weights); `full_encoder` makes a ctc Engine load the full encoder graph (enc + CTC ids)
        # instead of the CTC-only one, so a decoder Engine can share it. Without either, ctc loads the CTC-only graph.
        if mode not in ("decoder", "ctc") and (enc_share is not None or full_encoder):
            raise ValueError("enc_share / full_encoder are for the offline decoder and ctc modes")
        self.spec = ModelSpec.load(os.path.join(model_dir, "spec.json"))
        self.detok = Detokenizer(model_dir, self.spec)
        self.beam, self.batch_size, self.max_batch_seconds = beam, batch_size, max_batch_seconds
        self.lang, self.itn, self.pnc = lang, itn, pnc
        # default 448 generated tokens incl. <|eot|>; never above the model's text-position limit
        self.max_new = min(max_new_tokens or DEFAULT_MAX_NEW, self.spec.decoder.max_text_pos)
        self.max_seconds, self.min_seconds, self.len_alpha, self.compact, self.sort = (
            max_seconds, min_seconds, len_alpha, compact, sort)
        self.dev = "cuda" if device == "cuda" else "cpu"
        self.dtype = _DT[precision]
        provider = provider or ("CUDA" if device == "cuda" else "CPU")
        from speech_server import device as devlib
        self.profile = devlib.profile(devlib.compute_capability(0) if device == "cuda" else None)
        if provider == "TensorRT" and not self.profile["tensorrt"]:
            raise NotImplementedError(f"TensorRT EP is not available for {self.profile['name']} (TensorRT 10.5+ dropped "
                                      "sm70; onnxruntime-gpu 1.21 links a newer TensorRT): use --provider CUDA")
        if provider == "TensorRT":
            # sm75+: TensorRT first, CUDA for any node TensorRT does not take (engine cache next to the model)
            self.providers = [("TensorrtExecutionProvider", {
                "device_id": 0, "trt_fp16_enable": str(precision == "fp16"), "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": os.path.join(model_dir, "trt_cache")}),
                ("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        elif provider == "CUDA":
            opts = {"device_id": 0, "cudnn_conv_algo_search": cudnn_search, "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_use_max_workspace": "1", "cudnn_conv1d_pad_to_nc1d": "1"}
            opts.update({k: str(v) for k, v in (cuda_opts or {}).items()})
            self.providers = [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
            devlib.load_cuda_libs()               # nvidia-*-cu12 wheels, wherever installed
            try:
                ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
            except Exception:   # noqa: BLE001 (older ORT, or libraries already on the loader path)
                pass
        else:
            self.providers = ["CPUExecutionProvider"]
        self.want = self.providers[0][0] if isinstance(self.providers[0], tuple) else self.providers[0]
        # GPU: give the encoder's unused arena memory back every Nth run (--gpu-arena-shrink-every, default 8).
        # Encoder batches have many different padded shapes; with kSameAsRequested the arena otherwise fragments over
        # a long run until a ~0.5-1 GB request no longer fits the encoder's share of --gpu-mem-limit-mb (measured:
        # out of memory after ~30-40 calls of 64 at 16000 MB fp32; none with N = 1, 4, 8 or 16). Memory only: the
        # numbers computed are the same. Cost on a V100 at fp32: ~5.5% throughput at N = 1, ~1.5% at N = 8.
        self._enc_ro, self._enc_calls = None, [0]   # [runs of this encoder session], shared with enc_share engines
        n = 8 if gpu_arena_shrink_every is None else gpu_arena_shrink_every
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"gpu_arena_shrink_every must be a whole number >= 0 (got {n!r})")
        self.arena_shrink_every = n                  # 1 = every run, 0 = never
        if self.dev == "cuda" and self.arena_shrink_every:
            self._enc_ro = ort.RunOptions()
            self._enc_ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        self.threads, self.decode_threads, self.graph_opt = threads, decode_threads or threads, graph_opt
        self.gpu_mem_limit_mb = gpu_mem_limit_mb   # hard GPU memory cap (sessions + KV); None = no cap
        self.cut, self.cut_min_blank_ms = cut, cut_min_blank_ms   # long-form cut signal: "energy" | "ctc"
        p = precision
        cands = self.profile["encoder_graphs"] if encoder_graph == "auto" else [encoder_graph]
        self.enc, errs = None, []
        if mode in STREAM_MODES:                  # chunk graphs + front end only (stream.py)
            cands = []
            self._init_stream(lookahead, threads)
            if mode == "dec-stream":              # + the streaming decoder's graphs (streaming-decoder packages)
                self._init_dec_stream(threads, dec_format, endpoint_frames, max_segment_frames, rebase_frames)
        elif lookahead is not None:
            raise ValueError("--lookahead is for --mode ctc-stream / dec-stream")
        # the full encoder graph (enc + enc_lens + ctc_ids) for decoder mode, and for ctc mode when it is shared or
        # asked for; the CTC-only graph (enc_lens + ctc_ids) otherwise
        # word boost in ctc mode needs CTC log-probs (ctc_lp on the encoder output), so the full encoder graph
        if mode == "ctc" and word_boost and self._has(f"ctc_lp.{'fp32' if precision == 'int8' else precision}") \
                and any(self._has(f"{g}.{precision}") for g in (self.profile["encoder_graphs"] if encoder_graph == "auto"
                                                                 else [encoder_graph])):
            full_encoder = True
        self.enc_full = mode == "decoder" or (mode == "ctc" and (full_encoder or enc_share is not None))
        self.enc_shared = False
        sh = enc_share
        if sh is not None and mode in ("decoder", "ctc"):
            if not (getattr(sh, "enc", None) is not None and getattr(sh, "enc_full", False)
                    and os.path.realpath(sh.model_dir) == os.path.realpath(model_dir)
                    and sh.precision == precision and sh.want == self.want and sh.mode in ("decoder", "ctc")
                    and int(getattr(sh, "threads", 0) or 0) == int(self.threads or 0)
                    and getattr(sh, "graph_opt", None) == self.graph_opt
                    and _canon_providers(getattr(sh, "providers", [])) == _canon_providers(self.providers)
                    and getattr(sh, "gpu_mem_limit_mb", None) == self.gpu_mem_limit_mb
                    and (encoder_graph == "auto" or encoder_graph == getattr(sh, "encoder_graph", None))
                    and getattr(sh, "arena_shrink_every", None) == self.arena_shrink_every):
                raise ValueError("enc_share: the other engine must hold the full encoder of the same model package, "
                                 "precision, device and session settings (threads, graph_opt, provider options, "
                                 "GPU memory cap, encoder graph, arena shrink interval)")
            self.enc, self.encoder_graph, self.enc_shared = sh.enc, sh.encoder_graph, True
            # the decoder graphs are stateless sessions too (cross_kv, ctc_lp, gather, prefill / step): an engine that
            # shares the encoder also reuses every graph the donor already opened with the same settings (e.g. the
            # top-k graphs a greedy engine opened for word boost, for a beam engine); calls run under one lock
            self._sessions, self._sess_lock = sh._sessions, sh._sess_lock
            self._enc_calls = sh._enc_calls          # "every Nth run of this encoder", whichever engine runs it
            cands = []
        for g in cands:                   # first graph that exists, loads and runs a probe batch
            role = f"{g}{'' if self.enc_full else '_ctc'}.{p}"
            if not self._has(role):
                continue
            sess = None
            try:
                sess = self._session(role, threads)
                self._probe(sess, mode)
                self.enc, self.encoder_graph = sess, g
                break
            except Exception as e:  # noqa: BLE001 (e.g. a contrib-op kernel missing for this architecture)
                for k in [k for k, v in self._sessions.items() if v is sess]:
                    del self._sessions[k]            # a graph that failed its probe is not kept in memory
                errs.append(f"{role}: {type(e).__name__}: {str(e)[:160]}")
        if self.enc is None and mode not in STREAM_MODES:
            raise RuntimeError("no usable encoder graph: " + "; ".join(errs or ["none exported"]))
        self.fallbacks = errs
        self.cross = self.step = self.gather = None
        # the v2 decoder graph family: any prefill variant (a trimmed greedy package may ship only the argmax one)
        self.dec_prec = decoder_precision or (p if p != "int8" else "int8")
        # the v2 decoder graph family: any prefill variant at the encoder's or the decoder's precision (a trimmed
        # greedy package may ship only the argmax one; a mixed package only the decoder precision's)
        self.v2 = graphs == "v2" and any(self._has(f"dec_prefill{x}.{q}") for x in ("", "_argmax", "_topk")
                                         for q in {p, self.dec_prec})
        if not self.v2:
            self.dec_prec = p
        self.prefill = self.gather_layers = None
        self.beam_topk = False
        self.cross3 = False
        # decoder-mode token timestamps need ctc_lp (CTC log-probs of the encoder output): loaded now when exported
        # (int8 deployments use ctc_lp.fp32: the int8 graphs take and give fp32 tensors); capability flag for /help
        self._ctc_lp = None
        lp_role = f"ctc_lp.{'fp32' if precision == 'int8' else precision}"
        if (self._has(lp_role) and mode == "decoder") or \
                (word_boost and mode in ("ctc", "ctc-stream") and self._has(lp_role)):
            self._ctc_lp = self._session(lp_role, decode_threads or threads)
        self.token_timestamps = mode != "decoder" or self._ctc_lp is not None
        if mode in STREAM_MODES:
            self.v2 = False
        if mode not in ("ctc",) + STREAM_MODES:   # CTC and streaming modes never load the offline decoder graphs
            gp = "fp16" if p == "fp16" else "fp32"
            if self.v2:
                dp = self.dec_prec
                gp = "fp16" if dp == "fp16" else "fp32"
                cross = "cross_kv2_in16.fp32" if (p == "fp16" and dp == "fp32") else \
                    f"cross_kv3.{dp}" if self._has(f"cross_kv3.{dp}") else f"cross_kv2.{dp}"
                self.cross3 = cross.startswith("cross_kv3")
                self.cross = self._session(cross, self.decode_threads)
                sfx = "_argmax" if decode == "greedy" else ""
                self.beam_topk = (decode == "beam" and beam <= 16 and self._has(f"dec_lean_topk.{dp}")
                                  and self._has(f"dec_prefill_topk.{dp}"))
                if self.beam_topk:                  # in-graph top-16: (B, 16) to the host per step, not (B, V)
                    sfx = "_topk"
                sg = step_graph if self._has(f"{step_graph}{sfx}.{dp}") else "dec_step1"
                if step_graph == "lean":
                    sg = "dec_lean" if self._has(f"dec_lean{sfx}.{dp}") else "dec_step1"
                self.prefill = self._session(f"dec_prefill{sfx}.{dp}", self.decode_threads)
                self.step = self._session(f"{sg}{sfx}.{dp}", self.decode_threads)
                self.gather_layers = self._session(f"gather_layers.{gp}", self.decode_threads)
            else:
                self.cross = self._session(f"cross_kv.{p}", self.decode_threads)
                self.step = self._session(f"{'dec_step_argmax' if decode == 'greedy' else 'dec_step'}.{p}",
                                          self.decode_threads)
            if self._has(f"gather.{gp}"):
                self.gather = self._session(f"gather.{gp}", self.decode_threads)
            elif decode == "beam":                # beam search reorders rows on the device with this graph
                raise FileNotFoundError(f"{self.model_dir}: --decode beam needs the gather.{gp} graph, which this "
                                        "model package lacks")
        self.kv_dtype = np.float16 if self.dec_prec == "fp16" else np.float32
        d = self.spec.decoder
        self.L, self.H, self.dh = d.n_layers, d.n_heads, d.d_model // d.n_heads
        self.kv_margin = kv_margin
        self.decode_batch = decode_batch          # > 0: pool encoder batches into decode batches of up to this many rows
        self.kv_budget_mb = kv_budget_mb          # pooled decode batches are split to keep the KV buffers under this
        if gpu_mem_limit_mb:
            # one budget: decoder KV + cross K/V buffers get 25%; the encoder arena (50%) bounds the padded seconds
            # per encoder batch (~12 MB per padded second at fp16 on this model, after ~0.7 GB of weights)
            self.kv_budget_mb = min(kv_budget_mb, 0.25 * gpu_mem_limit_mb)
            per_s = 12.0 * (2.0 if self.precision == "fp32" else 1.0)
            cap = max(30.0, (0.45 * gpu_mem_limit_mb - 700.0 * (2.0 if self.precision == "fp32" else 1.0)) / per_s)
            self.max_batch_seconds = min(self.max_batch_seconds or cap, cap)
        self.Tmax = 64 + self.max_new
        if timestamps == "token" and not self.token_timestamps:
            raise ValueError("--timestamps token in decoder mode needs the ctc_lp graph, which this model package lacks")
        # word boost (boost.py): what this engine needs to bias its decoding, loaded now; _boost_why says why not
        from speech_server import boost as boostlib
        self.word_boost = bool(word_boost)
        self.boost_per_char = boostlib.BONUS_PER_CHAR if boost_per_char is None else float(boost_per_char)
        self._bprefill = self._bstep = None
        self._ttext = None
        self._boost_why = None if word_boost else "word boost is turned off on this server"
        if word_boost:
            if mode == "dec-stream":
                self._boost_why = "word_boost is not available for dec-stream (its graphs return no alternatives)"
            elif mode == "decoder":
                dp = self.dec_prec
                need = [f"dec_prefill_topk.{dp}", f"dec_lean_topk.{dp}"]
                if not self.v2 or not all(self._has(r) for r in need) or self.gather is None:
                    self._boost_why = ("word_boost needs the decoder's top-k graphs (" + ", ".join(need) +
                                       ") and the gather graph, which this model package lacks")
                elif self.beam_topk and step_graph == "lean":   # a top-k beam engine already holds these graphs
                    self._bprefill, self._bstep = self.prefill, self.step
                else:
                    self._bprefill = self._session(need[0], self.decode_threads)
                    self._bstep = self._session(need[1], self.decode_threads)
            elif self._ctc_lp is None or (mode == "ctc" and not self.enc_full):
                self._boost_why = "word_boost in CTC modes needs the ctc_lp graph, which this model package lacks"
            if self._boost_why is None:
                self._ttext = boostlib.TokenText(self.detok.sp, self.spec.vocab.text0, self.spec.vocab.padded)
                self._row_ids = np.array(self.spec.vocab.ctc_ids or list(range(self.spec.vocab.padded)), np.int64)
        self._boost_cache = boostlib.CACHE
        # one lock for the engines that share sessions (enc_share): calls into the shared graphs never overlap
        self.lock = enc_share.lock if (self.enc_shared and getattr(enc_share, "lock", None) is not None) \
            else threading.Lock()
        self.stats = Stats()

    # ------------------------------------------------------------ streaming CTC (mode ctc-stream)
    def _init_stream(self, lookahead, threads):
        from speech_server.stream import StreamingEncoder
        e = self.spec.encoder
        if e.kind != "fastconformer_chunked" or not e.chunk_sizes:
            raise ValueError(f"{self.model_dir}: not a streaming package (encoder.kind {e.kind!r}); mode {self.mode} "
                             "needs a streaming model package")
        sh = self._stream_share
        if sh is not None and not (getattr(sh, "stream_enc", None) and os.path.realpath(sh.model_dir)
                                   == os.path.realpath(self.model_dir)):
            sh = None
        self.stream_fe = sh.stream_fe if sh is not None else self._session("frontend.fp32", threads)
        chunks = list(e.chunk_sizes)
        # dec-stream may be served at fewer look-aheads than the export has (spec stream_dec.lookaheads: the ones the
        # streaming decoder was validated at); ctc-stream on the same export keeps all of them
        allow = (self.spec.stream_dec or {}).get("lookaheads") if self.mode == "dec-stream" else None
        if allow is not None:
            if (not isinstance(allow, list) or not allow
                    or any(isinstance(x, bool) or not isinstance(x, int) or x + 1 not in chunks for x in allow)):
                raise ValueError(f"{self.model_dir}: spec stream_dec.lookaheads {allow!r} must be a non-empty list of "
                                 f"exported look-aheads {[c - 1 for c in chunks]}")
            chunks = [c for c in chunks if c - 1 in allow]
        self.stream_enc = {c - 1: sh.stream_enc[c - 1] if sh is not None and c - 1 in sh.stream_enc
                           else StreamingEncoder(self._session(f"encoder_chunk_c{c}.fp32", threads), c)
                           for c in chunks}
        self.stream_shared = sh is not None
        self.lookaheads = sorted(self.stream_enc)
        self.lookahead = self.lookaheads[-1] if lookahead is None else self._check_la(lookahead)

    def _init_dec_stream(self, threads, fmt, endpoint_frames, max_segment_frames, rebase_frames):
        from speech_server.stream import MAX_FRAMES, DecoderGraphs
        sd = self.spec.stream_dec or {}
        need = [f"{r}.fp32" for r in ("dec_stream_cross", "dec_stream_prefill", "dec_stream_step")]
        if not sd or not all(self._has(r) for r in need):
            raise ValueError(f"{self.model_dir}: no streaming decoder in this export (spec stream_dec / dec_stream "
                             "graphs); mode dec-stream needs a streaming-decoder model package")
        self.dec_format = fmt or sd.get("format")
        if self.dec_format not in ("in", "out"):
            raise ValueError(f"streaming decoder format {self.dec_format!r}: in | out")
        self.wait_id = int(sd.get("wait_id", 37))
        for name, v, lo in (("endpoint_frames", endpoint_frames, 1), ("max_segment_frames", max_segment_frames, 1),
                            ("rebase_frames", rebase_frames, 0)):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
                raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
        if rebase_frames + max_segment_frames + max(self.spec.encoder.chunk_sizes) > MAX_FRAMES:
            raise ValueError(f"rebase_frames + max_segment_frames must stay under {MAX_FRAMES} encoder frames (RoPE "
                             "positions are fp32 and rebased only at segment resets)")
        self.endpoint_frames, self.max_segment_frames, self.rebase_frames = endpoint_frames, max_segment_frames, \
            rebase_frames
        d = self.spec.decoder
        self.dec_graphs = DecoderGraphs(*[self._session(r, threads) for r in need], d.n_layers, d.max_prompt_pos,
                                        d.max_text_pos)

    def dec_session(self, lookahead=None, live: bool = True, opt: dict | None = None):
        """A new DecSession (stream.py) at this look-ahead with the prompt of `opt` (lang / itn / pnc, see
        options()); live=True feeds audio through the incremental front end (feed_audio), False takes mel."""
        from speech_server.stream import DecSession
        la = self.lookahead if lookahead is None else self._check_la(lookahead)
        o = opt or self.options()
        prompt = self.spec.prompt(o["lang"], o["itn"], o["pnc"])
        return DecSession(self.stream_enc[la], self.dec_graphs, prompt, self.spec.vocab.eot, self.wait_id,
                          self.dec_format, self.max_new, self.endpoint_frames, self.max_segment_frames,
                          self.rebase_frames, self.spec.encoder.rope_base, self.spec.vocab.blank,
                          self.stream_fe if live else None)

    def dec_token_times(self, toks) -> list[dict]:
        """DecSession tokens [(id, emit frame)] -> token_times: the decoder has no frame alignment, so start / end are
        null (aligned false) and emit is when the stream had the token (the end of the chunk it was emitted on)."""
        fs = 1.0 / self.spec.frames_per_second
        return [dict(self._tt(v, None, None, False), emit=round(em * fs, 3)) for v, em in toks]

    def dec_result(self, ds) -> dict:
        """text / tokens / segments (one per decoder segment: start, end, text, reason) / token_times / guards of a
        DecSession."""
        fs = 1.0 / self.spec.frames_per_second
        ids = [t[0] for t in ds.tokens]
        segs = [{"start": round(g["start"] * fs, 3), "end": round(g["end"] * fs, 3),
                 "text": self.detok.decode(g["tokens"]), "reason": g["reason"]} for g in ds.seg_log]
        return {"text": self.detok.decode(ids), "tokens": ids, "segments": segs,
                "token_times": self.dec_token_times(ds.tokens), "guards": ds.guards()}

    def _check_la(self, la) -> int:
        """An int (not a bool / float / string) among the exported look-aheads."""
        if isinstance(la, bool) or not isinstance(la, (int, np.integer)) or int(la) not in self.stream_enc:
            raise InputError(f"lookahead must be one of {self.lookaheads} (an integer, encoder frames of 80 ms)")
        return int(la)

    def ctc_stream(self, lookahead=None, live: bool = True, opt: dict | None = None):
        """A new CtcStream (stream.py) at this look-ahead; live=True feeds audio through the incremental
        front end (feed_audio), False takes mel (feed_mel). opt (from options()): its word boost, if any."""
        from speech_server.stream import CtcStream
        la = self.lookahead if lookahead is None else self._check_la(lookahead)
        bo = (opt or {}).get("boost")
        fusion = None
        if bo is not None:
            fusion = {"boost": bo, "beam": opt["ctc_beam"], "lp": self._ctc_lp, "rows": self._row_ids,
                      "text": self._ttext}
        return CtcStream(self.stream_enc[la], self.spec.vocab.blank, self.stream_fe if live else None, fusion=fusion)

    def stream_token_times(self, toks) -> list[dict]:
        fs = 1.0 / self.spec.frames_per_second
        return [dict(self._tt(v, a * fs, b * fs, True), emit=round(em * fs, 3)) for v, a, b, em in toks]

    def _stream_one(self, x: np.ndarray, opt: dict, st: Stats) -> dict:
        """One file through the streaming encoder: the whole file's log-mel (exact, as the training eval), then mel
        chunks with the state fed back."""
        from speech_server.stream import MAX_FRAMES, stream_frames
        t0 = time.perf_counter()
        if self.mode == "ctc-stream" and stream_frames(len(x), self.spec.frontend.hop_length) > MAX_FRAMES:
            raise InputError(f"ctc-stream: {len(x) / audiolib.SR:.0f} s is longer than {MAX_FRAMES} encoder frames "
                             f"({MAX_FRAMES * 0.08:.0f} s; RoPE positions are fp32, no rebase in this engine)")
        if self.mode == "dec-stream":
            return self._dec_stream_one(x, opt, st, t0)
        cs = self.ctc_stream(opt.get("lookahead"), live=False, opt=opt)
        cs.keep_greedy = cs.fusion is not None   # an upload can fall back to the default tokens (no listed phrase)
        mel = self.stream_fe.run(["mel"], {"audio": np.asarray(x, np.float32)[None],
                                           "audio_lens": np.array([len(x)], np.int64)})[0][0]
        cs.feed_mel(mel)
        cs.finish()
        if cs.fusion is not None and not cs.hit:  # no listed phrase written: exactly the default output
            cs.tokens = cs.greedy_tokens
        st.encode_s += time.perf_counter() - t0
        st.audio_seconds += len(x) / audiolib.SR
        st.windows += 1
        ids = [t[0] for t in cs.tokens]
        dur = round(len(x) / audiolib.SR, 3)
        text = self.detok.decode(ids)
        st.tokens += len(ids)
        return {"text": text, "tokens": ids, "windows": 1, "segments": [{"start": 0.0, "end": dur, "text": text}],
                "token_times": self.stream_token_times(cs.tokens), "lookahead": cs.e.chunk - 1}

    def _dec_stream_one(self, x: np.ndarray, opt: dict, st: Stats, t0: float) -> dict:
        """One file through the streaming decoder session: the whole file's log-mel, then mel chunks (the last one
        decoded as final), endpoint resets and RoPE rebases as live."""
        ds = self.dec_session(opt.get("lookahead"), live=False, opt=opt)
        mel = self.stream_fe.run(["mel"], {"audio": np.asarray(x, np.float32)[None],
                                           "audio_lens": np.array([len(x)], np.int64)})[0][0]
        ds.feed_mel(mel)
        ds.finish()
        st.encode_s += time.perf_counter() - t0
        st.audio_seconds += len(x) / audiolib.SR
        st.windows += 1
        r = self.dec_result(ds)
        st.tokens += len(r["tokens"])
        r.update(windows=1, lookahead=ds.e.chunk - 1)
        return r

    def _stream_files(self, sources, langs, options, block_seconds):
        from speech_server.stream import MAX_FRAMES, stream_frames
        hop = self.spec.frontend.hop_length
        for i, src in enumerate(sources):
            with self.lock:
                t0 = time.perf_counter()
                st = Stats()
                opt = self.options(i, langs, options)
                blocks, n = [], 0
                import contextlib
                with contextlib.closing(audiolib.stream(src, block_seconds=block_seconds)) as gen:
                    for blk in gen:
                        n += len(blk)
                        if self.mode == "ctc-stream" and stream_frames(n, hop) > MAX_FRAMES:   # exact, as array / live
                            raise InputError(f"ctc-stream: input longer than {MAX_FRAMES} encoder frames "
                                             f"({MAX_FRAMES * 0.08:.0f} s)")
                        blocks.append(blk)
                x = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
                r = self._stream_one(x, opt, st)
                r.update(index=i, duration=round(len(x) / audiolib.SR, 3))
                st.wall_s = time.perf_counter() - t0
                self.stats.add(st)
                self.last = st
            yield r

    # ------------------------------------------------------------ sessions
    def _has(self, role: str) -> bool:
        """The package lists the graph and its file is there (a trimmed package may list graphs it does not ship)."""
        name = self.spec.graphs.get(role)
        return isinstance(name, str) and os.path.isfile(os.path.join(self.model_dir, name))

    def _enc_run(self, io):
        """One encoder run; every arena_shrink_every-th run (GPU) also shrinks the encoder's arena afterwards."""
        ro = None
        if self._enc_ro is not None:
            self._enc_calls[0] += 1                  # under self.lock, which sharing engines share
            if self._enc_calls[0] % self.arena_shrink_every == 0:
                ro = self._enc_ro
        self.enc.run_with_iobinding(io, ro)

    def _session(self, role: str, threads: int | None):
        with self._sess_lock:             # one session per key, also when engines are built from several threads
            return self._session_locked(role, threads)

    def _session_locked(self, role: str, threads: int | None):
        ort = self.ort
        name = self.spec.graphs.get(role)
        if name is None:
            raise FileNotFoundError(f"{self.model_dir}: no graph for {role} in this model package")
        # everything that makes a session differ: file, threads, optimisation, device and provider options, memory cap
        key = (name, int(threads or 0), self.graph_opt, _canon_providers(self.providers), self.gpu_mem_limit_mb)
        if key in self._sessions:
            return self._sessions[key]
        so = ort.SessionOptions()
        so.graph_optimization_level = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                       "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                                       "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                       "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[self.graph_opt]
        if threads:
            so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.set_denormal_as_zero", "1")
        so.log_severity_level = 3
        providers = self.providers
        if self.gpu_mem_limit_mb and self.dev == "cuda":
            # hard cap per session arena: the encoder takes most, decoder graphs a small share (the KV buffers are
            # capped separately by kv_budget_mb, which is set to a share of the same total)
            frac = (0.45 if role.startswith("encoder") else 0.10 if role.startswith("cross") else
                    0.06 if role.startswith(("dec_step", "dec_lean", "dec_prefill")) else 0.02)
            p0 = dict(providers[0][1], gpu_mem_limit=str(int(self.gpu_mem_limit_mb * frac * 2 ** 20)))
            providers = [(providers[0][0], p0)] + list(providers[1:])
        s = ort.InferenceSession(os.path.join(self.model_dir, name), sess_options=so, providers=providers)
        if s.get_providers()[0] != self.want:     # never fall back silently
            raise RuntimeError(f"{role}: asked for {self.want}, ORT gave {s.get_providers()}")
        self._sessions[key] = s
        return s

    def _ctc_cut(self, x: np.ndarray, lo: int, hi: int):
        """Long-form cut from the model's own CTC head: encode the look-ahead window (<= max_seconds), take the
        longest run of blank frames whose centre lies in [lo, hi) samples, cut at its centre; None (energy fallback)
        if no run reaches cut_min_blank_ms. One extra encoder pass per window (quality experiment)."""
        io = self.enc.io_binding()
        io.bind_cpu_input("audio", x[None, :].astype(np.float32))
        io.bind_cpu_input("audio_lens", np.array([len(x)], np.int64))
        for o in self.enc.get_outputs():
            io.bind_output(o.name, "cpu")
        self._enc_run(io)
        names = [o.name for o in self.enc.get_outputs()]
        ctc = io.get_outputs()[names.index("ctc_ids")].numpy()[0]
        fr = audiolib.SR * self.spec.encoder.subsampling * self.spec.frontend.hop_length // audiolib.SR  # samples/frame
        blank = ctc == self.spec.vocab.blank
        best, best_c, run = 0, None, 0
        for j, b in enumerate(blank):
            run = run + 1 if b else 0
            c = (j - run // 2) * fr + fr // 2
            if run > best and lo <= c < hi:
                best, best_c = run, c
        if best_c is None or best * fr * 1000 // audiolib.SR < self.cut_min_blank_ms:
            return None
        return best_c

    def _windower(self):
        return audiolib.Windower(self.max_seconds, self.min_seconds,
                                 cut_fn=self._ctc_cut if self.cut == "ctc" else None)

    def _probe(self, sess, mode):
        io = sess.io_binding()
        io.bind_cpu_input("audio", np.zeros((1, 16000), np.float32))
        io.bind_cpu_input("audio_lens", np.array([16000], np.int64))
        for o in sess.get_outputs():
            io.bind_output(o.name, "cpu")
        sess.run_with_iobinding(io)

    # ------------------------------------------------------------ public API
    def options(self, i: int = 0, langs: list | None = None, options=None) -> dict:
        """Per-input options {lang, itn, pnc, timestamps, lookahead}: `options` (one dict for every input, or a list
        with one dict per input), then langs[i], then the engine's defaults. Values are checked, not guessed.
        timestamps: none | segment (decoder: <|timestamps|> prompt, "timed_segments") | token (CTC times,
        "token_times"); lookahead: ctc-stream only."""
        o = options[i] if isinstance(options, (list, tuple)) else (options or {})
        unknown = set(o) - {"lang", "itn", "pnc", "timestamps", "lookahead", "word_boost", "boost", "ctc_beam"}
        if unknown:
            raise InputError(f"unknown options {sorted(unknown)}")
        # an explicit "lang" key wins (None = automatic, the server's ?lang=auto); otherwise langs[i] / the engine's
        lang = o["lang"] if "lang" in o else ((langs[i] if langs else None) or self.lang)
        r = {"lang": lang, "itn": bool(self.itn if o.get("itn") is None else o["itn"]),
             "pnc": bool(self.pnc if o.get("pnc") is None else o["pnc"]),
             "timestamps": o.get("timestamps") or self.timestamps, "lookahead": o.get("lookahead")}
        _check_ts(self.mode, r["timestamps"])
        if r["timestamps"] == "token" and not self.token_timestamps:
            raise InputError("timestamps=token is not available for this model in decoder mode (its package has no "
                             "ctc_lp graph)")
        if r["lookahead"] is not None and self.mode not in STREAM_MODES:
            raise InputError("lookahead is for mode ctc-stream / dec-stream")
        if self.mode in STREAM_MODES:
            r["lookahead"] = self.lookahead if r["lookahead"] is None else self._check_la(r["lookahead"])
        r["boost"], r["ctc_beam"] = self._boost_options(o)
        return r

    def _boost_options(self, o: dict):
        """(Boost or None, CTC beam) from the word_boost / boost / ctc_beam options; InputError when this engine
        cannot boost or a value is invalid. No list (or an empty one): (None, 1): decoding is exactly the default."""
        from speech_server import boost as boostlib
        wb, scale, beam = o.get("word_boost"), o.get("boost"), o.get("ctc_beam")
        if beam is not None and self.mode not in ("ctc", "ctc-stream"):
            raise InputError("ctc_beam is for mode ctc / ctc-stream")
        if not wb:
            if scale is not None or beam is not None:
                raise InputError("boost / ctc_beam apply only together with word_boost")
            return None, 1
        if self._boost_why is not None:
            raise InputError(self._boost_why)
        try:
            phrases = boostlib.parse_list(wb, boostlib.MAX_PHRASES)
            scale = 1.0 if scale is None else boostlib.parse_scale(scale)
        except boostlib.BoostError as e:
            raise InputError(str(e)) from None
        if beam is None:
            beam = boostlib.CTC_BEAM_DEFAULT
        if isinstance(beam, bool) or not isinstance(beam, int) or not 1 <= beam <= boostlib.CTC_BEAM_MAX:
            raise InputError(f"ctc_beam must be an integer from 1 to {boostlib.CTC_BEAM_MAX}")
        auto, key = self._boost_cache.get(phrases, self.boost_per_char)
        return boostlib.Boost(auto, scale, boostlib.MAX_HITS, key), (beam if self.mode in ("ctc", "ctc-stream") else 1)

    @staticmethod
    def _pkey(o: dict):
        """Decoder prompt key: (lang, itn, pnc, <|timestamps|>, word boost, CTC beam) - items batch / pool only with an
        equal key (the prompt is built from its first four fields)."""
        return (o["lang"], o["itn"], o["pnc"], o["timestamps"] == "segment", o.get("boost"), o.get("ctc_beam", 1))

    def transcribe(self, arrays: list[np.ndarray], langs: list | None = None, options=None) -> list[dict]:
        """Mono float32 16 kHz arrays -> [{"text", "tokens", "windows", "segments"}] in input order (plus
        "timed_segments" / "token_times" / "align_ok" when timestamps are asked for). options: see options()."""
        with self.lock:
            t0 = time.perf_counter()
            st = Stats()
            opts = [self.options(i, langs, options) for i in range(len(arrays))]
            if self.mode in STREAM_MODES:
                out = [self._stream_one(x, opts[i], st) for i, x in enumerate(arrays)]
                st.wall_s = time.perf_counter() - t0
                self.stats.add(st)
                self.last = st
                return out
            items, spans, nw = [], {}, []    # items: ((input, window), samples, prompt key, token times); spans: [a, b)
            for i, x in enumerate(arrays):
                pk, tt = self._pkey(opts[i]), opts[i]["timestamps"] == "token"
                cuts = audiolib.windows(x, self.max_seconds, self.min_seconds,
                                        cut_fn=self._ctc_cut if self.cut == "ctc" else None)
                for w, (a, b) in enumerate(cuts):
                    items.append(((i, w), x[a:b], pk, tt))
                    spans[(i, w)] = (a, b)
                nw.append(len(cuts))
                st.audio_seconds += len(x) / audiolib.SR
            aux: dict = {}
            toks = self._decode_items(items, st, aux)
            out = [self._assemble(i, toks, spans, nw[i], opts[i], aux) for i in range(len(arrays))]
            st.wall_s = time.perf_counter() - t0
            self.stats.add(st)
            self.last = st
            return out

    def transcribe_files(self, sources, langs: list | None = None, pool_windows: int = 256, block_seconds=30.0,
                         prefetch_blocks: int = 8, options=None):
        """Files of any length, memory-bounded: each source is decoded in blocks (audio.stream), cut as it
        arrives (Windower), and windows from all sources are decoded in pools of up to `pool_windows` (sorted by
        length inside a pool). Yields one result per source, in input order, as soon as all its windows are done:
        {"index", "text", "tokens", "windows", "segments": [{"start", "end", "text"}], "duration"} (+ the timestamp
        keys when asked for). ctc-stream mode: one stream per source (read whole, at most MAX_STREAM_FRAMES)."""
        if self.mode in STREAM_MODES:
            yield from self._stream_files(sources, langs, options, block_seconds)
            return
        threads, handles = [], [None]
        try:
            with self.lock:
                t0 = time.perf_counter()
                st = Stats()
                opts = {}
                aux: dict = {}
                toks, spans, pool = {}, {}, []
                n_win, dur = {}, {}
                done_reading, next_out = set(), 0

                def run_pool():
                    if pool:
                        toks.update(self._decode_items(list(pool), st, aux))
                        pool.clear()

                def ready():
                    nonlocal next_out
                    while next_out in done_reading and all((next_out, w) in toks for w in range(n_win[next_out])):
                        r = self._assemble(next_out, toks, spans, n_win[next_out], opts[next_out], aux)
                        r.update(index=next_out, duration=round(dur[next_out], 3))
                        for w in range(n_win[next_out]):          # free per-window state of finished inputs
                            toks.pop((next_out, w), None)
                            spans.pop((next_out, w), None)
                            aux.pop((next_out, w), None)
                        next_out += 1
                        yield r

                import queue as _q
                import threading as _th
                q: _q.Queue = _q.Queue(maxsize=max(2, prefetch_blocks))   # decoder thread -> bounded queue -> engine
                stop = _th.Event()                        # set by the consumer on every exit path (also errors)

                def put(item) -> bool:
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.1)
                            return True
                        except _q.Full:
                            continue
                    return False

                def reader():
                    try:
                        for i, src in enumerate(sources):
                            h = audiolib.DecodeHandle()
                            handles[0] = h
                            if stop.is_set():
                                return
                            gen = audiolib.stream(src, block_seconds=block_seconds, handle=h)
                            try:
                                for blk in gen:
                                    if not put(("blk", i, blk)):
                                        return
                            finally:
                                gen.close()                  # file / ffmpeg pipe closed even when cancelled
                            if not put(("end", i, None)):
                                return
                    except BaseException as e:  # noqa: BLE001 (re-raised in the consumer)
                        put(("err", -1, e))
                        return
                    put(("eof", -1, None))
                th = _th.Thread(target=reader, daemon=True)
                threads.append(th)
                th.start()
                try:
                    wd, w, total, pk, tt = None, 0, 0, None, False
                    cut_stats = {"cuts": 0, "fallback": 0, "tail_merge": 0, "tail_recut": 0, "model_cut": 0}
                    while True:
                        kind, i, blk = q.get()
                        if kind == "err":
                            raise blk
                        if kind == "eof":
                            break
                        if wd is None:
                            opts[i] = self.options(i, langs, options)
                            pk, tt = self._pkey(opts[i]), opts[i]["timestamps"] == "token"
                            wd, w, total = self._windower(), 0, 0
                        if kind == "blk":
                            total += len(blk)
                            for a, seg in wd.feed(blk):
                                pool.append(((i, w), seg, pk, tt))
                                spans[(i, w)] = (a, a + len(seg))
                                w += 1
                                if len(pool) >= pool_windows:
                                    run_pool()
                                    yield from ready()
                            continue
                        for a, seg in wd.flush():                    # "end" of source i
                            pool.append(((i, w), seg, pk, tt))
                            spans[(i, w)] = (a, a + len(seg))
                            w += 1
                        n_win[i], dur[i] = w, total / audiolib.SR
                        st.audio_seconds += total / audiolib.SR
                        done_reading.add(i)
                        for k, v in wd.stats.items():
                            cut_stats[k] += v
                        st.extra["cut_stats"] = dict(cut_stats)
                        wd = None
                        if len(pool) >= pool_windows:
                            run_pool()
                        yield from ready()
                    run_pool()
                    yield from ready()
                    st.wall_s = time.perf_counter() - t0
                    self.stats.add(st)
                    self.last = st
                finally:                              # cancel the reader: stop flag, kill a blocked decoder, drain
                    stop.set()
                    if handles[0] is not None:
                        handles[0].cancel()
                    while True:
                        try:
                            q.get_nowait()
                        except _q.Empty:
                            break
        finally:                                      # join the reader outside the engine lock: it must end
            for th in threads:
                th.join(timeout=30)
                if th.is_alive():
                    import sys as _sys
                    print("transcribe_files: decoder thread did not stop within 30 s (its ffmpeg was killed)",
                          file=_sys.stderr, flush=True)

    def _assemble(self, i, toks, spans, n: int, opt: dict | None = None, aux: dict | None = None) -> dict:
        ids, segs = [], []
        ws = range(n)
        ts = (opt or {}).get("timestamps", "none")
        timed, ttimes, align_ok = [], [], True
        fs = 1.0 / self.spec.frames_per_second
        for w in ws:
            tk = toks[(i, w)]
            ids += tk
            a, b = spans[(i, w)]
            seg = {"start": round(a / audiolib.SR, 3), "end": round(b / audiolib.SR, 3), "text": self.detok.decode(tk)}
            if self.mode != "ctc":                # hit the token cap without <|eot|> (plain greedy: likely a loop)
                seg["capped"] = bool(tk) and tk[-1] != self.spec.vocab.eot and len(tk) >= self.max_new
            off = a / audiolib.SR
            if ts == "segment":
                for t in parse_timed(tk, self.spec.ts0, self.spec.vocab.text0, fs, off, self.detok.decode,
                                     (b - a) / audiolib.SR):
                    t["window"] = w
                    timed.append(t)
            elif ts == "token":
                x = (aux or {}).get((i, w), {})
                if self.mode == "ctc":
                    ttimes += [self._tt(v, s * fs + off, e * fs + off, True) for v, s, e in x.get("runs", [])]
                else:
                    tt, ok = self._token_times(tk, x.get("align"), off)
                    seg["align_ok"] = ok
                    align_ok = align_ok and ok
                    ttimes += tt
            segs.append(seg)
        out = {"text": self.detok.decode(ids), "tokens": ids, "windows": len(ws), "segments": segs}
        if ts == "segment":
            out["timed_segments"] = timed
        elif ts == "token":
            out["token_times"] = ttimes
            if self.mode != "ctc":
                out["align_ok"] = align_ok
        return out

    def piece_text(self, v: int) -> str:
        """One token as text: a text piece with its word-start marker as a space, a byte piece / special / timestamp
        by its piece name (Thai characters are several byte-fallback tokens: the window text is the readable form)."""
        p = self.detok.sp.id_to_piece(int(v)) if 0 <= int(v) < self.detok.sp.get_piece_size() else f"<{int(v)}>"
        return p.replace("\u2581", " ") if int(v) >= self.spec.vocab.text0 else p

    def _tt(self, v, start, end, aligned):
        return {"id": int(v), "text": self.piece_text(v), "start": None if start is None else round(start, 3),
                "end": None if end is None else round(end, 3), "aligned": bool(aligned)}

    def _token_times(self, tk, al, off):
        """Decoder tokens + their CTC alignment (first / last frame of each alignable token, ok) -> token_times. A
        token with no CTC row (specials, timestamps, pieces the CTC head lacks) takes the previous aligned token's end
        (or the next one's start) and aligned False; an infeasible alignment gives null times, align_ok False."""
        fs = 1.0 / self.spec.frames_per_second
        if al is None or not al[2]:
            return [self._tt(v, None, None, False) for v in tk], False
        first, last, _ = al
        rows = [self._ctc_row.get(int(v), -1) for v in tk]
        times, j = [None] * len(tk), 0
        for q, r in enumerate(rows):
            if r > 0:
                times[q] = (off + first[j] * fs, off + (last[j] + 1) * fs)
                j += 1
        out = []
        for q, v in enumerate(tk):
            if times[q] is not None:
                out.append(self._tt(v, times[q][0], times[q][1], True))
                continue
            prev = next((times[p][1] for p in range(q - 1, -1, -1) if times[p] is not None and rows[p] > 0), None)
            nxt = next((times[p][0] for p in range(q + 1, len(tk)) if times[p] is not None and rows[p] > 0), None)
            t = prev if prev is not None else nxt
            out.append(self._tt(v, t, t, False))
        return out, True

    @property
    def _ctc_row(self) -> dict:
        if not hasattr(self, "_ctc_row_map"):
            self._ctc_row_map = {int(v): r for r, v in enumerate(self.spec.vocab.ctc_ids)}
        return self._ctc_row_map

    def _align_rows(self, enc_host, elen, toks_rows) -> list:
        """CTC forced alignment of decoded rows: ctc_lp on each row's own frames, then ctc_align_np over the tokens
        that have a CTC row. Returns [(first, last, ok)] per row."""
        if self._ctc_lp is None:
            raise InputError("timestamps=token is not available for this model in decoder mode (no ctc_lp graph)")
        out = []
        for r, tk in enumerate(toks_rows):
            T = int(elen[r])
            tgt = [self._ctc_row[int(v)] for v in tk if self._ctc_row.get(int(v), -1) > 0]
            if not tgt:                          # nothing to align (only <|eot|>, or no CTC pieces): not a failure
                out.append((np.zeros(0, np.int64), np.zeros(0, np.int64), True))
                continue
            lp = self._ctc_lp.run(["ctc_logp"], {"enc": np.ascontiguousarray(enc_host[r: r + 1, :T])})[0][0]
            out.append(ctc_align_np(lp, tgt))
        return out

    def _decode_items(self, items, st: Stats, aux: dict | None = None) -> dict:
        """[(key, samples, prompt key, token times)] -> {key: tokens}: length-sorted batches, decode pooling (v2).
        Batches never mix prompt keys; the token-times flag does not change batching (aux[key]: CTC runs / alignment
        of the rows that asked for token times)."""
        st.windows += len(items)
        aux = {} if aux is None else aux
        items = [it if len(it) == 4 else (it[0], it[1], self._legacy_pk(it[2]), False) for it in items]
        # (lang, rest of the prompt key, -length): with default options the order is the earlier (lang, -length)
        order = sorted(range(len(items)), key=lambda k: (str(items[k][2][0]), str(items[k][2][1:]),
                                                         -len(items[k][1]))) \
            if self.sort == "length" else list(range(len(items)))
        toks: dict = {}
        pool = self.v2 and self.mode != "ctc" and self.decode_batch > 0
        pend = []                             # pooled encoder outputs: (keys, enc (B,T,d) host, elen, clen, pkey, tts)
        for batch in self._batches([(k, None, x, pk, tt) for k, x, pk, tt in (items[o] for o in order)]):
            pk = batch[0][3]
            keys = [b[0] for b in batch]
            tts = [b[4] for b in batch]
            if not pool or pk[4] is not None:     # boosted batches decode on their own (no pooling)
                if pend:
                    self._flush(pend, toks, st, aux)
                    pend = []
                rows_aux = [None] * len(batch)
                for j, (k, t) in enumerate(zip(keys, self._run_batch([b[2] for b in batch], pk, st, tts, rows_aux))):
                    toks[k] = t
                    if rows_aux[j] is not None:
                        aux[k] = rows_aux[j]
            else:
                if pend and (pend[0][4] != pk or sum(len(p[0]) for p in pend) + len(batch) > self.decode_batch):
                    self._flush(pend, toks, st, aux)
                pend.append((keys, *self._encode_host([b[2] for b in batch], st), pk, tts))
            st.batches += 1
        if pend:
            self._flush(pend, toks, st, aux)
        return toks

    def _batches(self, items):
        cur, longest = [], 0
        for it in items:
            n = len(it[2])
            new_long = max(longest, n)
            over_b = len(cur) >= self.batch_size
            over_s = self.max_batch_seconds and cur and (len(cur) + 1) * new_long / audiolib.SR > self.max_batch_seconds
            over_l = cur and it[3] != cur[0][3]
            if cur and (over_b or over_s or over_l):
                yield cur
                cur, new_long = [], n
            cur.append(it)
            longest = new_long
        if cur:
            yield cur

    # ------------------------------------------------------------ one batch
    def _legacy_pk(self, pk):
        """A bare language (callers from before per-request options, e.g. tools/parity.py) -> the prompt key."""
        if not isinstance(pk, tuple):
            pk = (pk, self.itn, self.pnc, self.timestamps == "segment")
        return pk if len(pk) >= 6 else tuple(pk) + (None, 1)[len(pk) - 4:]

    def _run_batch(self, xs: list[np.ndarray], pk, st: Stats, tts=None, rows_aux=None) -> list[list[int]]:
        pk = self._legacy_pk(pk)
        B = len(xs)
        S = max(len(x) for x in xs)
        audio = np.zeros((B, S), np.float32)
        for k, x in enumerate(xs):
            audio[k, : len(x)] = x
        lens = np.array([len(x) for x in xs], np.int64)
        t0 = time.perf_counter()
        io = self.enc.io_binding()
        io.bind_cpu_input("audio", audio)
        io.bind_cpu_input("audio_lens", lens)
        if self.mode == "ctc" and pk[4] is not None:
            return self._ctc_boost_batch(io, B, pk, st, t0, tts, rows_aux)
        if self.mode == "ctc":
            if self.enc_full:                     # a full (shared) encoder graph: its enc output stays on the device
                io.bind_output("enc", self.dev)   # and is dropped
            io.bind_output("enc_lens", "cpu")
            io.bind_output("ctc_ids", "cpu")
            self._enc_run(io)
            outs = io.get_outputs()
            elen, ctc = outs[-2].numpy(), outs[-1].numpy()
            st.encode_s += time.perf_counter() - t0
            for r in range(B):
                if tts and tts[r]:
                    rows_aux[r] = {"runs": _ctc_runs(ctc[r, : int(elen[r])], self.spec.vocab.blank)}
            return [_ctc_collapse(ctc[r, : int(elen[r])], self.spec.vocab.blank) for r in range(B)]
        io.bind_output("enc", self.dev)
        io.bind_output("enc_lens", "cpu")
        io.bind_output("ctc_ids", "cpu")
        self._enc_run(io)
        outs = io.get_outputs()
        enc, elen = outs[0], outs[1].numpy()
        st.encode_s += time.perf_counter() - t0
        t1 = time.perf_counter()
        prompt = self.spec.prompt(*pk[:4])
        if self.v2:
            ctc = outs[2].numpy()
            clen = max(len(_ctc_collapse(ctc[r, : int(elen[r])])) for r in range(B))
            tcap = min(self.Tmax, len(prompt) + 2 * clen + self.kv_margin)
            if pk[4] is not None:                 # word boost: fused beam (beam 1 for a greedy engine)
                res, hit = self._beam2_boost(enc, elen, prompt, st, tcap, pk[4],
                                             self.beam if self.decode_kind == "beam" else 1)
                if not all(hit):                  # rows that wrote no listed phrase: exactly the default output (the
                    plain = (self._beam2 if self.decode_kind == "beam" else self._greedy2)(enc, elen, prompt, st,
                                                                                            tcap)   # whole batch)
                    res = [res[r] if hit[r] else plain[r] for r in range(B)]
            else:
                res = (self._beam2 if self.decode_kind == "beam" else self._greedy2)(enc, elen, prompt, st, tcap)
        elif self.decode_kind == "beam":
            res = self._beam(enc, elen, prompt, st)
        else:
            res = self._greedy(enc, elen, prompt, st)
        st.decode_s += time.perf_counter() - t1
        if tts and any(tts):
            need = [r for r in range(B) if tts[r]]
            eh = enc.numpy()
            for r, al in zip(need, self._align_rows(eh[need], elen[need], [res[r] for r in need])):
                rows_aux[r] = {"align": al}
        return res

    def _encode_host(self, xs, st: Stats):
        """Encoder only; enc to the host (B, T, d), so consecutive encoder batches can form one decode batch."""
        B = len(xs)
        audio = np.zeros((B, max(len(x) for x in xs)), np.float32)
        for k, x in enumerate(xs):
            audio[k, : len(x)] = x
        t0 = time.perf_counter()
        io = self.enc.io_binding()
        io.bind_cpu_input("audio", audio)
        io.bind_cpu_input("audio_lens", np.array([len(x) for x in xs], np.int64))
        io.bind_output("enc", "cpu")
        io.bind_output("enc_lens", "cpu")
        io.bind_output("ctc_ids", "cpu")
        self._enc_run(io)
        enc, elen, ctc = (o.numpy() for o in io.get_outputs())
        clen = np.array([len(_ctc_collapse(ctc[r, : int(elen[r])])) for r in range(B)])
        st.encode_s += time.perf_counter() - t0
        return enc, elen, clen

    def _flush(self, pend, toks, st: Stats, aux: dict | None = None):
        """Decode the pooled rows as one batch: encoder outputs zero-padded to the longest T (the cross-attention
        mask from enc_lens hides the padding, so each row decodes as it would alone)."""
        t1 = time.perf_counter()
        T = max(p[1].shape[1] for p in pend)
        d = pend[0][1].shape[2]
        B = sum(p[1].shape[0] for p in pend)
        enc = np.zeros((B, T, d), pend[0][1].dtype)
        r = 0
        for p in pend:
            enc[r: r + p[1].shape[0], : p[1].shape[1]] = p[1]
            r += p[1].shape[0]
        elen = np.concatenate([p[2] for p in pend]).astype(np.int64)
        clen = int(np.concatenate([p[3] for p in pend]).max())
        keys = [k for p in pend for k in p[0]]
        tts = [t for p in pend for t in p[5]]
        prompt = self.spec.prompt(*pend[0][4][:4])
        tcap = min(self.Tmax, len(prompt) + 2 * clen + self.kv_margin)
        # KV memory budget: 4 buffers (k, v x ping-pong) of L x rows x H x tcap x dh
        es = np.dtype(self.kv_dtype).itemsize
        nb = self.beam if self.decode_kind == "beam" else 1
        per_row = 4 * self.L * self.H * tcap * self.dh * es * nb
        # cross K/V (+ a compacted copy) and the uploaded encoder rows share the same budget
        per_row += (2 * 2 * self.L * self.H * T * self.dh * es) * nb + T * d * enc.itemsize
        step = max(1, min(B, int(self.kv_budget_mb * 2 ** 20 // per_row)))
        if self.gpu_mem_limit_mb:                  # the cross-K/V graph's temporaries live in its 10% arena
            tmp = (3 if self.cross3 else 6) * T * d * es       # per-row projection temporaries (per layer / all)
            step = max(1, min(step, int((0.10 * self.gpu_mem_limit_mb - 64) * 2 ** 20 // tmp)))
        for i in range(0, B, step):
            ov = self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(enc[i:i + step]), self.dev, 0)
            res = (self._beam2 if self.decode_kind == "beam" else self._greedy2)(ov, elen[i:i + step], prompt, st,
                                                                                  tcap)
            for k, t in zip(keys[i:i + step], res):
                toks[k] = t
            need = [r for r in range(i, min(B, i + step)) if tts[r]]
            if need and aux is not None:
                for r, al in zip(need, self._align_rows(enc[need], elen[need], [toks[keys[r]] for r in need])):
                    aux[keys[r]] = {"align": al}
        st.decode_s += time.perf_counter() - t1
        pend.clear()

    def _cross(self, enc):
        io = self.cross.io_binding()
        io.bind_ortvalue_input("enc", enc)
        io.bind_output("cross_k", self.dev)
        io.bind_output("cross_v", self.dev)
        self.cross.run_with_iobinding(io)
        return io.get_outputs()

    def _gather(self, a, b, idx: np.ndarray):
        io = self.gather.io_binding()
        io.bind_ortvalue_input("x", a)
        io.bind_ortvalue_input("x2", b)
        io.bind_cpu_input("idx", idx.astype(np.int64))
        io.bind_output("y", self.dev)
        io.bind_output("y2", self.dev)
        self.gather.run_with_iobinding(io)
        return io.get_outputs()

    def _step(self, io, kv: _KV, tokens, P, ck, cv, elen, out_name):
        io.bind_cpu_input("tokens", tokens)
        io.bind_cpu_input("start", np.array(kv.t, np.int64))
        io.bind_cpu_input("prompt_len", np.array(P, np.int64))
        io.bind_output(out_name, "cpu")          # first: get_outputs() follows the binding order
        kv.bind_step(io, tokens.shape[1])
        io.bind_ortvalue_input("cross_k", ck)
        io.bind_ortvalue_input("cross_v", cv)
        io.bind_cpu_input("enc_lens", elen)
        self.step.run_with_iobinding(io)
        kv.advance(tokens.shape[1])
        return io.get_outputs()[0].numpy()

    def _greedy(self, enc, elen, prompt, st: Stats) -> list[list[int]]:
        B = len(elen)
        eot, pad = self.spec.vocab.eot, self.spec.vocab.pad
        ck, cv = self._cross(enc)
        P = len(prompt)
        kv = _KV(self, B, P + self.max_new)
        io = self.step.io_binding()
        rows = np.arange(B)                          # batch row -> original row
        elen = elen.astype(np.int64)
        nxt = self._step(io, kv, np.array([prompt] * B, np.int64), P, ck, cv, elen, "next_ids")
        st.steps += 1
        out = [[] for _ in range(B)]
        done = np.zeros(B, bool)
        for _ in range(self.max_new):
            for k in np.nonzero(~done)[0]:
                out[rows[k]].append(int(nxt[k]))
            done |= (nxt == eot) | np.array([len(out[r]) >= self.max_new for r in rows])
            if done.all():
                break
            alive = np.nonzero(~done)[0]
            if self.compact and self.gather is not None and len(rows) >= 8 and len(alive) <= len(rows) // 2:
                kv.gather(alive)
                ck, cv = self._gather(ck, cv, alive)
                rows, elen, nxt, done = rows[alive], elen[alive], nxt[alive], done[alive]
            cur = np.where(done, pad, nxt).astype(np.int64)[:, None]
            nxt = self._step(io, kv, cur, P, ck, cv, elen, "next_ids")
            st.steps += 1
        st.tokens += sum(len(o) for o in out)
        return out

    def _beam(self, enc, elen, prompt, st: Stats) -> list[list[int]]:
        """Attention-only beam search: N beams per row, finished hypotheses ranked by score / len ** len_alpha; a row
        ends when it holds N finished hypotheses or reaches max_new_tokens (then its live beams count as finished)."""
        N = self.beam
        B = len(elen)
        eot, pad = self.spec.vocab.eot, self.spec.vocab.pad
        ck, cv = self._cross(enc)
        P = len(prompt)
        kv = _KV(self, B * N, P + self.max_new)
        kv.B = B
        io = self.step.io_binding()
        elen = elen.astype(np.int64)
        logp = self._step(io, kv, np.array([prompt] * B, np.int64), P, ck, cv, elen, "logprobs")
        st.steps += 1
        fin = [[] for _ in range(B)]
        beams = []                                   # per row: list of (score, tokens)
        for r in range(B):
            top = np.argsort(-logp[r])[:N]
            beams.append([(float(logp[r, v]), [int(v)]) for v in top])
        idx = np.repeat(np.arange(B), N)             # expanded rows r * N + j
        kv.gather(idx)
        ck, cv = self._gather(ck, cv, idx)
        elenN = np.repeat(elen, N)
        active = np.ones(B, bool)
        for _ in range(1, self.max_new + 1):
            # finished hypotheses leave the beam; the survivors move down, so their KV rows must move with them
            comp = np.arange(B * N)
            for r in range(B):
                if not active[r]:
                    continue
                keep, slots = [], []
                for k0, (sc, tk) in enumerate(beams[r]):
                    if tk[-1] == eot or len(tk) >= self.max_new:
                        fin[r].append((sc / max(len(tk), 1) ** self.len_alpha, tk))
                    else:
                        keep.append((sc, tk))
                        slots.append(k0)
                if len(fin[r]) >= N or not keep:
                    active[r] = False
                beams[r] = keep
                comp[r * N: r * N + len(slots)] = r * N + np.asarray(slots, np.int64)
            if not active.any():
                break
            if (comp != np.arange(B * N)).any():
                kv.gather(comp)
            cur = np.full((B * N, 1), pad, np.int64)
            for r in range(B):
                for j, (_, tk) in enumerate(beams[r]):
                    cur[r * N + j, 0] = tk[-1]
            logp = self._step(io, kv, cur, P, ck, cv, elenN, "logprobs")
            st.steps += 1
            parent = np.arange(B * N)
            for r in range(B):
                if not active[r]:
                    continue
                nb = len(beams[r])
                sc = np.array([x for x, _ in beams[r]])[:, None] + logp[r * N: r * N + nb]
                k2 = min(N, sc.size)
                flat = np.argpartition(-sc.ravel(), k2 - 1)[:k2]
                flat = flat[np.argsort(-sc.ravel()[flat])]
                new = []
                for f in flat:
                    j, v = divmod(int(f), sc.shape[1])
                    new.append((float(sc[j, v]), beams[r][j][1] + [v], j))
                beams[r] = [(x, tk) for x, tk, _ in new]
                for k, (_, _, j) in enumerate(new):
                    parent[r * N + k] = r * N + j
            kv.gather(parent)
        out = []
        for r in range(B):
            cand = fin[r] or [(x / max(len(tk), 1) ** self.len_alpha, tk) for x, tk in beams[r]]
            out.append(max(cand, key=lambda c: c[0])[1])
        st.tokens += sum(len(o) for o in out)
        return out

    # ------------------------------------------------------------ v2 decoder (per-layer KV, host positions)
    def _cross2(self, enc, elen):
        io = self.cross.io_binding()
        io.bind_ortvalue_input("enc", enc)
        io.bind_cpu_input("enc_lens", elen)
        # outputs into buffers this engine allocates (outside the session arena, counted in the KV/cross budget)
        B, T = enc.shape()[0], enc.shape()[1]
        ov = self.ort.OrtValue.ortvalue_from_shape_and_type
        ck = ov([self.L, B, self.H, T, self.dh], self.kv_dtype, self.dev, 0)
        cv = ov([self.L, B, self.H, T, self.dh], self.kv_dtype, self.dev, 0)
        ca = ov([B, 1, 1, T], np.float32, self.dev, 0)
        if self.cross3:                            # per-layer outputs straight into the slices of ck / cv
            es = np.dtype(self.kv_dtype).itemsize
            n = B * self.H * T * self.dh
            for name, buf in (("k", ck), ("v", cv)):
                for l in range(self.L):
                    io.bind_output(f"cross_{name}{l}", self.dev, 0, self.kv_dtype, [B, self.H, T, self.dh],
                                   buf.data_ptr() + l * n * es)
        else:
            io.bind_ortvalue_output("cross_k", ck)
            io.bind_ortvalue_output("cross_v", cv)
        io.bind_ortvalue_output("cross_add", ca)
        self.cross.run_with_iobinding(io)
        return ck, cv, ca

    def _bind_cross(self, io, ck, cv, cadd, B):
        T = cadd.shape()[3]
        es = np.dtype(self.kv_dtype).itemsize
        n = B * self.H * T * self.dh
        for name, ov in (("k", ck), ("v", cv)):
            base = ov.data_ptr()
            for l in range(self.L):
                io.bind_input(f"cross_{name}{l}", self.dev, 0, self.kv_dtype, [B, self.H, T, self.dh], base + l * n * es)
        io.bind_ortvalue_input("cross_add", cadd)

    def _compact_cross(self, ck, cv, cadd, idx):
        shp = ck.shape()
        ov = self.ort.OrtValue.ortvalue_from_shape_and_type
        k2 = ov([shp[0], len(idx), *shp[2:]], self.kv_dtype, self.dev, 0)
        v2 = ov([shp[0], len(idx), *shp[2:]], self.kv_dtype, self.dev, 0)
        io = self.gather.io_binding()
        io.bind_ortvalue_input("x", ck)
        io.bind_ortvalue_input("x2", cv)
        io.bind_cpu_input("idx", idx.astype(np.int64))
        io.bind_ortvalue_output("y", k2)
        io.bind_ortvalue_output("y2", v2)
        self.gather.run_with_iobinding(io)
        ck, cv = k2, v2
        a = cadd.numpy()[idx]
        return ck, cv, self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a), self.dev, 0)

    def _prefill(self, kv, prompt, B, ck, cv, cadd, out_name):
        io = self.prefill.io_binding()
        names = out_name if isinstance(out_name, (list, tuple)) else [out_name]
        for n in names:
            io.bind_output(n, "cpu")
        io.bind_cpu_input("tokens", np.array([prompt] * B, np.int64))
        self._bind_cross(io, ck, cv, cadd, B)
        kv.bind_prefill(io, len(prompt))
        self.prefill.run_with_iobinding(io)
        kv.advance(len(prompt))
        o = io.get_outputs()
        return o[0].numpy() if len(names) == 1 else [o[i].numpy() for i in range(len(names))]

    def _step1(self, io, kv, tokens, P, ck, cv, cadd, out_name, bind_cross=True):
        names = out_name if isinstance(out_name, (list, tuple)) else [out_name]
        for n in names:
            io.bind_output(n, "cpu")
        io.bind_cpu_input("tokens", tokens)
        pos = min(self.spec.decoder.max_prompt_pos + kv.t - P,
                  self.spec.decoder.max_prompt_pos + self.spec.decoder.max_text_pos - 1)
        io.bind_cpu_input("pos", np.array([pos], np.int64))
        kv.bind_step(io)
        if bind_cross:                           # IOBinding keeps bindings between runs: cross K/V bound once
            self._bind_cross(io, ck, cv, cadd, kv.B)
        self.step.run_with_iobinding(io)
        kv.advance(1)
        o = io.get_outputs()
        return o[0].numpy() if len(names) == 1 else [o[i].numpy() for i in range(len(names))]

    def _greedy2(self, enc, elen, prompt, st: Stats, tcap: int) -> list[list[int]]:
        B = len(elen)
        eot, pad = self.spec.vocab.eot, self.spec.vocab.pad
        elen = elen.astype(np.int64)
        ck, cv, cadd = self._cross2(enc, elen)
        P = len(prompt)
        kv = _KV2(self, B, tcap)
        nxt = self._prefill(kv, prompt, B, ck, cv, cadd, "next_ids")
        st.steps += 1
        io = self.step.io_binding()
        need_cross = True
        rows = np.arange(B)
        out = [[] for _ in range(B)]
        n_out = np.zeros(B, np.int64)
        done = np.zeros(B, bool)
        for _ in range(self.max_new):
            for k in np.nonzero(~done)[0]:
                out[rows[k]].append(int(nxt[k]))
            n_out[~done] += 1
            done |= (nxt == eot) | (n_out >= self.max_new)
            if done.all():
                break
            alive = np.nonzero(~done)[0]
            if self.compact and self.gather is not None and len(rows) >= 8 and len(alive) <= len(rows) // 2:
                kv.gather(alive)
                ck, cv, cadd = self._compact_cross(ck, cv, cadd, alive)
                rows, nxt, done, n_out = rows[alive], nxt[alive], done[alive], n_out[alive]
                io = self.step.io_binding()
                need_cross = True
            cur = np.where(done, pad, nxt).astype(np.int64)[:, None]
            nxt = self._step1(io, kv, cur, P, ck, cv, cadd, "next_ids", need_cross)
            need_cross = False
            st.steps += 1
        st.tokens += sum(len(o) for o in out)
        return out

    def _beam2(self, enc, elen, prompt, st: Stats, tcap: int) -> list[list[int]]:
        N = self.beam
        B = len(elen)
        eot, pad = self.spec.vocab.eot, self.spec.vocab.pad
        elen = elen.astype(np.int64)
        ck, cv, cadd = self._cross2(enc, elen)
        P = len(prompt)
        kv = _KV2(self, B * N, tcap)
        kv.B = B
        outn = ["top_logp", "top_ids"] if self.beam_topk else "logprobs"

        def cand(o):                               # per row: candidate (log-prob, id) arrays, best first
            if self.beam_topk:
                return o[0], o[1]
            idx = np.argsort(-o, axis=1)[:, :N]
            return np.take_along_axis(o, idx, 1), idx
        cv_, ci_ = cand(self._prefill(kv, prompt, B, ck, cv, cadd, outn))
        st.steps += 1
        fin = [[] for _ in range(B)]
        beams = [[(float(cv_[r, j]), [int(ci_[r, j])]) for j in range(N)] for r in range(B)]
        idx = np.repeat(np.arange(B), N)
        kv.gather(idx)
        ck, cv, cadd = self._compact_cross(ck, cv, cadd, idx)
        io = self.step.io_binding()
        need_cross = True
        active = np.ones(B, bool)
        for _ in range(1, self.max_new + 1):
            # finished hypotheses leave the beam; the survivors move down, so their KV rows must move with them
            comp = np.arange(B * N)
            for r in range(B):
                if not active[r]:
                    continue
                keep, slots = [], []
                for k0, (sc, tk) in enumerate(beams[r]):
                    if tk[-1] == eot or len(tk) >= self.max_new:
                        fin[r].append((sc / max(len(tk), 1) ** self.len_alpha, tk))
                    else:
                        keep.append((sc, tk))
                        slots.append(k0)
                if len(fin[r]) >= N or not keep:
                    active[r] = False
                beams[r] = keep
                comp[r * N: r * N + len(slots)] = r * N + np.asarray(slots, np.int64)
            if not active.any():
                break
            if (comp != np.arange(B * N)).any():
                kv.gather(comp)
            cur = np.full((B * N, 1), pad, np.int64)
            for r in range(B):
                for j, (_, tk) in enumerate(beams[r]):
                    cur[r * N + j, 0] = tk[-1]
            cv_, ci_ = cand(self._step1(io, kv, cur, P, ck, cv, cadd, outn, need_cross))
            need_cross = False
            st.steps += 1
            parent = np.arange(B * N)
            for r in range(B):
                if not active[r]:
                    continue
                nb = len(beams[r])
                sc = np.array([x for x, _ in beams[r]])[:, None] + cv_[r * N: r * N + nb, :N]
                k2 = min(N, sc.size)
                flat = np.argpartition(-sc.ravel(), k2 - 1)[:k2]
                flat = flat[np.argsort(-sc.ravel()[flat])]
                new = []
                for f in flat:
                    j, c = divmod(int(f), sc.shape[1])
                    new.append((float(sc[j, c]), beams[r][j][1] + [int(ci_[r * N + j, c])], j))
                beams[r] = [(x, tk) for x, tk, _ in new]
                for k, (_, _, j) in enumerate(new):
                    parent[r * N + k] = r * N + j
            kv.gather(parent)
        out = []
        for r in range(B):
            cand = fin[r] or [(x / max(len(tk), 1) ** self.len_alpha, tk) for x, tk in beams[r]]
            out.append(max(cand, key=lambda c: c[0])[1])
        st.tokens += sum(len(o) for o in out)
        return out


    # ------------------------------------------------------------ word boost (boost.py)
    def _ctc_boost_batch(self, io, B, pk, st: Stats, t0, tts, rows_aux) -> list[list[int]]:
        """CTC mode with a word boost list: the full encoder's output through ctc_lp, then per row the boosted greedy
        re-selection (ctc_beam 1) or the prefix beam search with fusion. Token times (timestamps=token): the
        result's tokens force-aligned to the CTC log-probs."""
        from speech_server import boost as boostlib
        io.bind_output("enc", "cpu")
        io.bind_output("enc_lens", "cpu")
        io.bind_output("ctc_ids", "cpu")
        self._enc_run(io)
        outs = io.get_outputs()
        enc, elen, ctc = outs[0].numpy(), outs[1].numpy(), outs[2].numpy()
        st.encode_s += time.perf_counter() - t0
        t1 = time.perf_counter()
        bo, beam = pk[4], pk[5]
        blank = 0
        res = []
        for r in range(B):
            T = int(elen[r])
            lp = self._ctc_lp.run(["ctc_logp"], {"enc": np.ascontiguousarray(enc[r: r + 1, :T])})[0][0]
            if beam <= 1:
                ids, hit = boostlib.ctc_greedy_boost(lp, self._row_ids, self._ttext, bo, blank)
            else:
                ids, hit = boostlib.ctc_prefix_beam_boost(lp, self._row_ids, self._ttext, bo, beam, blank)
            if not hit:                       # no listed phrase written: exactly the default (greedy) output
                ids = _ctc_collapse(ctc[r, :T], self.spec.vocab.blank)
                if tts and tts[r]:            # ... with the default token times too
                    rows_aux[r] = {"runs": _ctc_runs(ctc[r, :T], self.spec.vocab.blank)}
            elif tts and tts[r]:
                row = {int(v): k for k, v in enumerate(self._row_ids)}
                pairs = [(v, row[v]) for v in ids if row.get(v, 0) > 0]   # tokens and their CTC rows, together
                first, last, ok = ctc_align_np(lp, [c for _, c in pairs]) if pairs else \
                    (np.zeros(0), np.zeros(0), True)
                rows_aux[r] = {"runs": [(v, int(a), int(b) + 1) for (v, _), a, b in zip(pairs, first, last)]
                               if ok else []}
            res.append(ids)
        st.decode_s += time.perf_counter() - t1
        st.tokens += sum(len(x) for x in res)
        return res

    def _beam2_boost(self, enc, elen, prompt, st: Stats, tcap: int, bo, N: int) -> tuple[list[list[int]], list[bool]]:
        """The v2 beam search on the top-k graphs with word-boost shallow fusion; N = 1 is greedy with boost
        re-ranking among each step's top-k tokens. Each hypothesis carries its boost state. Bonuses re-rank real
        text tokens only: <|eot|>, special and timestamp tokens get none. Ending inside a partial match gives that
        partial bonus back at once (in the <|eot|> transition, before pruning); in greedy decoding (N = 1) an <|eot|>
        the model ranked first is lifted to the best fused score. Returns (token rows, whether each row's result completed a listed phrase)."""
        B = len(elen)
        eot, pad = self.spec.vocab.eot, self.spec.vocab.pad
        elen = elen.astype(np.int64)
        ck, cv, cadd = self._cross2(enc, elen)
        P = len(prompt)
        kv = _KV2(self, B * N, tcap)
        kv.B = B
        outn = ["top_logp", "top_ids"]
        text = self._ttext
        prefill, step = self._bprefill, self._bstep

        def fuse(lp_row, id_row, state):
            """[(rank, log-prob, bonus, id, new state)] for one hypothesis' candidates; rank = log-prob + bonus orders
            them within the step. <|eot|> gets no bonus and gives the open partial credit back in its own transition
            (before pruning), so a finished hypothesis never keeps an unearned partial bonus. Greedy (N = 1): an
            <|eot|> that is the step's raw best candidate is lifted to the best rank, so a bonus never outranks an
            ending the model chose (a beam keeps the honest score and lets the hypotheses compete)."""
            items, raw_best = [], max(lp_row.tolist()) if len(lp_row) else -np.inf
            eot_top = False
            closed = (0, 0, 0.0, state[3])        # ending: the open credit is given back now, not after pruning
            for lp, v in zip(lp_row.tolist(), id_row.tolist()):
                if v == eot:
                    g = bo.finish(state)
                    items.append([lp + g, lp, g, v, closed, True])
                    eot_top = eot_top or lp >= raw_best
                    continue
                st2, bonus = bo.step(state, text[int(v)])
                items.append([lp + bonus, lp, bonus, v, st2, False])
            if eot_top and N == 1:
                top = max(it[0] for it in items)
                for it in items:
                    if it[5]:
                        it[0] = max(it[0], top + 1e-6)
            return [tuple(it[:5]) for it in items]

        def final_rank(cands):
            """Finished candidates [(log-prob sum, bonus sum, tokens, hit)] of one row -> the best. Only the log-prob
            is length-normalised; the bonus is divided by one shared length (the row's longest candidate), so a bonus
            can never favour a shorter ending through the normalisation, and with no bonus this is _beam2's rank."""
            a = self.len_alpha
            lref = max(max(len(tk), 1) for _, _, tk, _ in cands) ** a
            return max(cands, key=lambda c: c[0] / max(len(c[2]), 1) ** a + c[1] / lref)
        o = self._prefill_s(prefill, kv, prompt, B, ck, cv, cadd, outn)
        st.steps += 1
        from speech_server.boost import ROOT
        fin = [[] for _ in range(B)]
        beams = []                                   # per row: [(log-prob sum, bonus sum, tokens, boost state)]
        for r in range(B):
            c = sorted(fuse(o[0][r], o[1][r], ROOT), key=lambda x: -x[0])[:N]
            beams.append([(lp, g, [v], s2) for _, lp, g, v, s2 in c])
        idx = np.repeat(np.arange(B), N)
        kv.gather(idx)
        ck, cv, cadd = self._compact_cross(ck, cv, cadd, idx)
        io = step.io_binding()
        need_cross = True
        active = np.ones(B, bool)
        for _ in range(1, self.max_new + 1):
            # finished hypotheses leave the beam; the survivors move down, so their KV rows must move with them
            comp = np.arange(B * N)
            for r in range(B):
                if not active[r]:
                    continue
                keep, slots = [], []
                for k0, (ls, gs, tk, bs) in enumerate(beams[r]):
                    if tk[-1] == eot or len(tk) >= self.max_new:
                        fin[r].append((ls, gs + bo.finish(bs), tk, bool(bs[3])))
                    else:
                        keep.append((ls, gs, tk, bs))
                        slots.append(k0)
                if len(fin[r]) >= N or not keep:
                    active[r] = False
                beams[r] = keep
                comp[r * N: r * N + len(slots)] = r * N + np.asarray(slots, np.int64)
            if not active.any():
                break
            if (comp != np.arange(B * N)).any():
                kv.gather(comp)
            cur = np.full((B * N, 1), pad, np.int64)
            for r in range(B):
                for j, (_, _, tk, _) in enumerate(beams[r]):
                    cur[r * N + j, 0] = tk[-1]
            o = self._step_s(step, io, kv, cur, P, ck, cv, cadd, outn, need_cross)
            need_cross = False
            st.steps += 1
            parent = np.arange(B * N)
            for r in range(B):
                if not active[r]:
                    continue
                cands = []
                for j, (ls, gs, tk, bs) in enumerate(beams[r]):
                    for rank, lp, g, v, ns in fuse(o[0][r * N + j], o[1][r * N + j], bs):
                        # live hypotheses of one row all have the same length: rank them by the plain sum
                        cands.append((ls + gs + rank, j, v, ns, ls + lp, gs + g))
                cands.sort(key=lambda x: -x[0])
                new = cands[:N]
                beams[r] = [(l2, g2, beams[r][j][2] + [v], ns) for _, j, v, ns, l2, g2 in new]
                for k, (_, j, _, _, _, _) in enumerate(new):
                    parent[r * N + k] = r * N + j
            kv.gather(parent)
        out, hit = [], []
        for r in range(B):
            c = fin[r] or [(ls, gs + bo.finish(bs), tk, bool(bs[3])) for ls, gs, tk, bs in beams[r]]
            best = final_rank(c)
            out.append(best[2])
            hit.append(best[3])
        st.tokens += sum(len(x) for x in out)
        return out, hit

    def _prefill_s(self, sess, kv, prompt, B, ck, cv, cadd, out_name):
        """_prefill on another prefill session (the boost path's top-k graph)."""
        keep = self.prefill
        self.prefill = sess
        try:
            return self._prefill(kv, prompt, B, ck, cv, cadd, out_name)
        finally:
            self.prefill = keep

    def _step_s(self, sess, io, kv, tokens, P, ck, cv, cadd, out_name, bind_cross=True):
        """_step1 on another step session (the boost path's top-k graph)."""
        keep = self.step
        self.step = sess
        try:
            return self._step1(io, kv, tokens, P, ck, cv, cadd, out_name, bind_cross)
        finally:
            self.step = keep


def _ctc_collapse(seq: np.ndarray, blank: int = 0) -> list[int]:
    if len(seq) == 0:
        return []
    keep = np.ones(len(seq), bool)
    keep[1:] = seq[1:] != seq[:-1]
    s = seq[keep]
    return [int(v) for v in s[s != blank]]


# ---------------------------------------------------------------- Part 10: timestamps (no training)

TIMESTAMP_MODES = ("none", "segment", "token")


def _check_ts(mode: str, ts: str) -> None:
    if ts not in TIMESTAMP_MODES:
        raise InputError(f"timestamps {ts!r}: one of {list(TIMESTAMP_MODES)}")
    if ts == "segment" and mode != "decoder":
        raise InputError("timestamps=segment needs mode decoder (the decoder writes <|tS|> text <|tE|>); "
                         "use timestamps=token for CTC times")
    if ts != "none" and mode == "dec-stream":
        raise InputError("mode dec-stream always returns token_times (emit times only): timestamps must be none")


def _ctc_runs(seq: np.ndarray, blank: int = 0) -> list[tuple[int, int, int]]:
    """Per-frame argmax ids -> [(id, first frame, last frame + 1)] for each collapsed CTC token (the same tokens as
    _ctc_collapse, with the extent of the argmax run that produced each)."""
    out, prev = [], None
    for t, v in enumerate(np.asarray(seq).tolist()):
        if v != blank and v == prev:
            out[-1] = (out[-1][0], out[-1][1], t + 1)
        elif v != blank:
            out.append((int(v), t, t + 1))
        prev = v
    return out


def ctc_align_np(logp: np.ndarray, tgt) -> tuple[np.ndarray, np.ndarray, bool]:
    """Viterbi CTC forced alignment of one row (fp32 scores, first-index tie breaks). logp (T, R) log-probs over the CTC head rows (row 0 blank), tgt
    (U,) head rows (no blanks). Returns (first (U,), last (U,)) aligned frame of each token and ok (False: empty
    target, no frames or no feasible path; first / last are then -1)."""
    lp = np.asarray(logp, np.float32)
    T = lp.shape[0]
    tgt = np.asarray(tgt, np.int64)
    U = len(tgt)
    S = 2 * U + 1
    ext = np.zeros(S, np.int64)
    ext[1::2] = tgt
    neg = np.float32(-np.inf)
    if T == 0 or U == 0:
        return np.full(U, -1), np.full(U, -1), False
    skip = np.zeros(S, bool)
    if S > 2:
        skip[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    alpha = np.full(S, neg, np.float32)
    alpha[0] = lp[0, ext[0]]
    alpha[1] = lp[0, ext[1]]
    bp = np.zeros((T, S), np.int8)
    em = lp[:, ext]                                   # (T, S) emission per state
    for t in range(1, T):
        step = np.concatenate([[neg], alpha[:-1]])
        jump = np.where(skip, np.concatenate([[neg, neg], alpha[:-2]])[:S], neg)
        cand = np.stack([alpha, step, jump])
        k = cand.argmax(0)
        alpha = (cand[k, np.arange(S)] + em[t]).astype(np.float32)
        bp[t] = k
    a_last, a_tok = alpha[2 * U], alpha[2 * U - 1]
    s = 2 * U - 1 if a_tok > a_last else 2 * U
    ok = bool(np.isfinite(max(a_last, a_tok)))
    if not ok:
        return np.full(U, -1), np.full(U, -1), False
    first, last = np.full(U, T, np.int64), np.full(U, -1, np.int64)
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            j = (s - 1) // 2
            first[j] = min(first[j], t)
            last[j] = max(last[j], t)
        s -= int(bp[t, s])
    return first, last, True


def parse_timed(tokens, ts0: int, text0: int, frame_s: float, offset: float, decode,
                window_s: float | None = None) -> list[dict]:
    """<|tS|> text <|tE|> ... (the training format, 80 ms steps from the window start) -> [{start, end, text}] in
    absolute seconds. Nothing is guessed: text outside a pair, a start without an end (or the reverse), an end before
    its start or a start before the previous end, and a time past the window's end (window_s, one frame of slack)
    keep null / the read times and get a "flag"."""
    segs, cur, prev_end = [], None, None

    def close(end, flag=None):
        nonlocal cur, prev_end
        st, ids = cur
        seg = {"start": None if st is None else round(offset + st * frame_s, 3),
               "end": None if end is None else round(offset + end * frame_s, 3), "text": decode(ids)}
        if flag is None and st is not None and end is not None and end < st:
            flag = "out_of_order"
        if flag is None and st is not None and prev_end is not None and st < prev_end:
            flag = "out_of_order"
        if flag is None and window_s is not None and any(
                t is not None and t * frame_s > window_s + frame_s + 1e-9 for t in (st, end)):
            flag = "past_window"
        if flag:
            seg["flag"] = flag
        if end is not None:
            prev_end = end
        segs.append(seg)
        cur = None

    for v in tokens:
        v = int(v)
        if ts0 <= v < text0:
            k = v - ts0
            if cur is None:
                cur = (k, [])
            elif cur[0] is None:              # text before any start, closed by this time token: unpaired
                close(k, "no_start")
            else:
                close(k)
        else:
            if cur is None:
                cur = (None, [])
            cur[1].append(v)
    if cur is not None:
        if cur[0] is None:
            if decode(cur[1]):
                close(None, "no_timestamps")
        else:
            close(None, "no_end")
    return segs
