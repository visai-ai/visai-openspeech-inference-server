"""Streaming CTC on the chunked encoder (numpy + onnxruntime only).

A streaming model is a ModelSpec with encoder.kind = "fastconformer_chunked", encoder.chunk_sizes (encoder frames per
chunk; look-ahead L streams with chunk L + 1) and encoder.left_context_frames. A streaming package adds

    frontend.fp32.onnx            audio (B, S), audio_lens -> mel (B, T, n_mels), mel_lens   (the offline log-mel)
    encoder_chunk_c<C>.fp32.onnx  mel (1, 8 C, n_mels), n_valid, pos, state.. -> enc (1, C, d), ctc_ids (1, C),
                                  n_out, new_state..

State (fixed shapes, zeros at the start): each stride-2 subsampling stage's last input frame (sub0..2), each layer's
attention keys / values after QK-norm + RoPE for the last K = (left // C) * C frames (k_cache, v_cache, right-aligned,
kv_len valid), each layer's last (kernel - 1) / 2 depthwise-conv inputs (conv). The chunk graph is the model's
chunk-masked streaming encoder.

Front end. MelStream turns audio arriving in pieces into the same log-mel frames as the whole-file front end: frame k
is centred on sample 160 k and needs samples 160 k - 256 .. 160 k + 255, so it is computed once those samples are in
(the window is run through frontend.fp32 with two frames of real left context, whose own first frames are dropped);
the sample tail kept between calls is the last 160 * 2 + 512 samples. The first frames use the graph's zero left
padding, which is exactly the offline centre padding, and finish() runs the last frames with the graph's zero right
padding, which is the offline end; so every frame equals the whole-file mel (float rounding of the conv only, no edge
effect).

CtcStream: mel frames -> chunks of 8 C mel frames -> the chunk graph (state fed back) -> CTC greedy collapse carried
across chunk boundaries (a label repeated over a boundary is one token) -> tokens with first / last frame and the emit
time (end of the chunk holding the token's first frame). Positions are fp32 in the graph: a stream is refused past
MAX_FRAMES encoder frames (no RoPE rebase / endpoint reset here; DecSession below has both).

Streaming decoder (mode dec-stream, streaming-decoder packages):
DecoderGraphs runs dec_stream_cross / dec_stream_prefill / dec_stream_step (the offline graph classes with <wait>
allowed) for one stream. DecStream: per chunk the cross K/V of the new frames are
appended (the cross-attention covers every frame of the segment so far), the prompt is teacher-forced at the first
chunk (one prefill call), then greedy steps until <wait> (format "in": fed back, the position advances; "out": not
fed back, its self-attention K/V dropped, the position retried on the next chunk) or <|eot|>; on the final chunk a
<wait> ends the segment (guard wait_at_end), <|eot|> before it is guard eot_mid; pos_cap at the position table's end.
DecSession: the chunk encoder runs continuously, its CTC ids drive the endpoint
detector (endpoint_frames blank frames after speech) and the segment cap (max_segment_frames); at a reset the decoder
finishes the segment as at the end of the audio, a new one starts re-primed with the prompt, and the encoder's RoPE
positions are rebased past rebase_frames (cached keys re-rotated in fp32), so a session
has no length limit. A full chunk is held back until the next mel frame or finish() arrives, so the last chunk of a
stream is always decoded as the final one: live audio and file mode give the same tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_FRAMES = 9000          # encoder frames (80 ms): 12 min per stream


def stream_frames(n_samples: int, hop: int = 160) -> int:
    """Encoder frames of n samples: mel frames n // hop + 1, then three stride-2 stages (ceil each) = ceil(T / 8)."""
    return -(-(int(n_samples) // hop + 1) // 8)


@dataclass
class StateTensor:
    name: str
    shape: list
    dtype: str = "fp32"
    init: str = "zeros"


@dataclass
class StreamState:
    layout: list = field(default_factory=list)   # [StateTensor]


class StreamLimit(ValueError):
    """The stream passed MAX_FRAMES encoder frames."""


class MelStream:
    """Incremental log-mel through the exported front-end graph (see the module docstring)."""

    def __init__(self, frontend_sess, n_mels: int = 80, hop: int = 160, n_fft: int = 512):
        self.sess, self.hop, self.half, self.n_mels = frontend_sess, hop, n_fft // 2, n_mels
        self.buf = np.zeros(0, np.float32)
        self.buf0 = 0                    # absolute sample index of buf[0]
        self.k = 0                       # next mel frame to emit
        self.n = 0                       # samples received

    def _run(self, a: int, b: int, last: bool) -> np.ndarray:
        """Frames self.k .. b - 1 from samples [a, ...) of the buffer (absolute a)."""
        seg = self.buf[a - self.buf0:]
        if not last:                                   # exactly the samples frames < b need
            seg = seg[: (b - 1) * self.hop + self.half - a]
        mel = self.sess.run(["mel"], {"audio": seg[None, :].astype(np.float32),
                                      "audio_lens": np.array([len(seg)], np.int64)})[0][0]
        drop = (self.k * self.hop - a) // self.hop      # frames of seg before self.k
        out = mel[drop: drop + (b - self.k)]
        assert len(out) == b - self.k, (len(out), b, self.k)
        return out

    def feed(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32).reshape(-1)
        self.buf = np.concatenate([self.buf, x])
        self.n += len(x)
        b = (self.n - self.half) // self.hop + 1 if self.n >= self.half else 0   # frames with all samples in
        b = min(b, self.n // self.hop + 1)
        if b <= self.k:
            return np.zeros((0, self.n_mels), np.float32)
        a = 0 if self.k < 2 else (self.k - 2) * self.hop
        out = self._run(a, b, last=False)
        self.k = b
        keep = max(0, (self.k - 2) * self.hop)          # the next call's window start
        self.buf, self.buf0 = self.buf[keep - self.buf0:], keep
        return out

    def finish(self) -> np.ndarray:
        if self.n == 0:                                 # no audio at all: no frames (never run the graph on 0 samples)
            return np.zeros((0, self.n_mels), np.float32)
        b = self.n // self.hop + 1                      # the offline frame count
        if b <= self.k:
            return np.zeros((0, self.n_mels), np.float32)
        a = 0 if self.k < 2 else (self.k - 2) * self.hop
        out = self._run(a, b, last=True)
        self.k = b
        return out


class StreamingEncoder:
    """The chunk graph with its state: init_state() -> dict; encode_chunk(mel_chunk, n_valid, state, pos) ->
    (enc, ctc_ids, n_out, state)."""

    def __init__(self, sess, chunk: int):
        self.sess, self.chunk = sess, int(chunk)
        self.inputs = {i.name: i for i in sess.get_inputs()}
        self.state_names = [n for n in self.inputs if n not in ("mel", "n_valid", "pos")]
        self.out_names = [o.name for o in sess.get_outputs()]
        self.n_mels = int(self.inputs["mel"].shape[2])

    def init_state(self) -> dict:
        st = {}
        for n in self.state_names:
            i = self.inputs[n]
            st[n] = np.zeros(i.shape, np.int64 if "int64" in i.type else np.float32)
        return st

    def encode_chunk(self, mel: np.ndarray, state: dict, pos: int):
        step = 8 * self.chunk
        nv = mel.shape[0]
        assert 0 < nv <= step, nv
        if nv < step:
            mel = np.concatenate([mel, np.zeros((step - nv, mel.shape[1]), np.float32)])
        out = self.sess.run(self.out_names, {"mel": mel[None].astype(np.float32), "n_valid": np.array(nv, np.int64),
                                             "pos": np.array(pos, np.int64), **state})
        o = dict(zip(self.out_names, out))
        n = int(o["n_out"])
        new = {k: o[f"new_{k}"] for k in self.state_names}
        return o["enc"][0, :n], o["ctc_ids"][0, :n], n, new


MAX_PENDING = 2               # boosted streaming beam: tokens of the best beam left uncommitted after each chunk


class CtcStream:
    """One stream: feed_mel() / feed_audio() -> newly committed tokens [(id, first, last + 1, emit frame)];
    finish() flushes the last (partial) chunk. Tokens are final once out (greedy CTC: a token is committed when its
    first frame's chunk is encoded; its end frame may still grow while the run continues into the next chunk)."""

    def __init__(self, encoder: StreamingEncoder, blank: int = 0, frontend=None, max_frames: int | None = None,
                 fusion: dict | None = None):
        self.e, self.blank = encoder, blank
        self.mel = MelStream(frontend, encoder.n_mels) if frontend is not None else None
        self.state = encoder.init_state()
        self.pending = np.zeros((0, encoder.n_mels), np.float32)
        self.pos, self.prev = 0, blank
        self.max_frames = MAX_FRAMES if max_frames is None else max_frames
        self.tokens: list = []               # [id, first, end, emit] (end grows with the run)
        self.done = False
        # word boost (boost.py): {"boost", "beam", "lp" (ctc_lp session), "rows", "text"}. Beam > 1: prefix beam
        # search with fusion; tokens are committed (never revised) as soon as every beam agrees on them, and after
        # each chunk the best beam's tokens except its last MAX_PENDING are committed anyway (bounded latency; a
        # token's first / end frame is the frame it was committed at). Beam 1: greedy with boost re-selection.
        self.fusion = fusion
        self.keep_greedy = False             # also keep the default (greedy) tokens (uploads fall back to them)
        self.greedy_tokens: list = []
        self.hit = False                     # (fusion) whether the result completed a listed phrase
        self.gprev = blank
        if fusion is not None:
            from speech_server import boost as boostlib
            self._bl = boostlib
            self.beams = {(): (0.0, -np.inf, boostlib.ROOT, 0.0)}
            self.last_row = None
            self.bstate = boostlib.ROOT

    def _run_fused(self, enc: np.ndarray, n: int) -> list:
        f, bl = self.fusion, self._bl
        lp = f["lp"].run(["ctc_logp"], {"enc": np.ascontiguousarray(enc[None, :n])})[0][0]
        new = []
        end = self.pos + n
        for j in range(n):
            t = self.pos + j
            if f["beam"] <= 1:
                r, self.bstate, emitted = bl.ctc_greedy_frame(lp[j], self.prev, self.bstate, f["rows"], f["text"],
                                                              f["boost"], self.blank)
                if r != self.blank and r == self.prev and self.tokens:
                    self.tokens[-1][2] = t + 1
                elif emitted:
                    self.tokens.append([int(f["rows"][r]), t, t + 1, end])
                    new.append(self.tokens[-1])
                self.prev = r
                continue
            self.beams = bl.ctc_beam_frame(self.beams, lp[j], f["rows"], f["text"], f["boost"], f["beam"],
                                           self.blank, last0=self.last_row)
            new += self._commit(bl.common_prefix(self.beams), t, end)
        if f["beam"] > 1:                    # bounded latency: after each chunk, commit the best beam's text but
            best = max(self.beams.items(), key=lambda kv: bl._lse(kv[1][0], kv[1][1]) + kv[1][3])[0]
            if len(best) > MAX_PENDING:      # its last MAX_PENDING tokens; beams that disagree are dropped
                new += self._commit(best[:len(best) - MAX_PENDING], self.pos + n - 1, end, force=True)
        return new

    def _commit(self, rows: tuple, t: int, end: int, force: bool = False) -> list:
        """Commit rows and re-base the beams past them: a prefix every beam shares, or (force) the best beam's
        prefix, dropping the beams that do not start with it."""
        if not rows:
            return []
        if force:
            k = len(rows)
            self.beams = {p: v for p, v in self.beams.items() if p[:k] == rows}
        new = []
        for r in rows:
            self.tokens.append([int(self.fusion["rows"][r]), t, t + 1, end])
            new.append(self.tokens[-1])
        k = len(rows)
        self.beams = {p[k:]: v for p, v in self.beams.items()}
        self.last_row = rows[-1]
        return new

    def _run_chunk(self, m: np.ndarray) -> list:
        n_new = -(-m.shape[0] // 8)                          # encoder frames this chunk gives (stride 8, ceil)
        if self.pos + n_new > self.max_frames:
            raise StreamLimit(f"stream longer than {self.max_frames} encoder frames "
                              f"({self.max_frames * 0.08:.0f} s): RoPE positions are fp32, no rebase in this engine")
        enc, ids, n, self.state = self.e.encode_chunk(m, self.state, self.pos)
        if self.fusion is not None:
            if self.keep_greedy:             # the default greedy collapse of the chunk's own CTC ids, alongside
                end = self.pos + n
                for j, v in enumerate(ids.tolist()):
                    t = self.pos + j
                    if v != self.blank and v == self.gprev:
                        self.greedy_tokens[-1][2] = t + 1
                    elif v != self.blank:
                        self.greedy_tokens.append([int(v), t, t + 1, end])
                    self.gprev = v
            new = self._run_fused(enc, n)
            self.pos += n
            return new
        new = []
        end = self.pos + n
        for j, v in enumerate(ids.tolist()):
            t = self.pos + j
            if v != self.blank and v == self.prev:
                self.tokens[-1][2] = t + 1
            elif v != self.blank:
                self.tokens.append([int(v), t, t + 1, end])
                new.append(self.tokens[-1])
            self.prev = v
        self.pos = end
        return new

    def feed_mel(self, mel: np.ndarray) -> list:
        assert not self.done
        self.pending = np.concatenate([self.pending, np.asarray(mel, np.float32)]) if len(self.pending) else \
            np.asarray(mel, np.float32)
        step = 8 * self.e.chunk
        new = []
        while len(self.pending) >= step:
            m, self.pending = self.pending[:step], self.pending[step:]
            new += self._run_chunk(m)
        return new

    def feed_audio(self, x: np.ndarray) -> list:
        return self.feed_mel(self.mel.feed(x))

    def finish(self) -> list:
        if self.done:
            return []
        new = self.feed_mel(self.mel.finish()) if self.mel is not None else []
        if len(self.pending):
            new += self._run_chunk(self.pending)
            self.pending = self.pending[:0]
        if self.fusion is not None and self.fusion["beam"] > 1:     # the best beam's remaining rows
            best = max(self.beams.items(), key=lambda kv: self._bl._lse(kv[1][0], kv[1][1]) + kv[1][3]
                       + self.fusion["boost"].finish(kv[1][2]))
            self.hit = bool(best[1][2][3])
            new += self._commit(best[0], max(self.pos - 1, 0), self.pos)
        elif self.fusion is not None:
            self.hit = bool(self.bstate[3])
        self.done = True
        return new


# ---------------------------------------------------------------- streaming decoder (mode dec-stream)

class DecoderGraphs:
    """The three dec_stream graphs of one package, batch 1, numpy in and out (CPU; the cache round-trips through the
    host, as the chunk graphs' state does). cross(enc (n, d)) -> (k, v) each (L, 1, H, n, dh); prefill(prompt, k, v)
    -> (next id, past k list, past v list); step(token, pos_id, pk, pv, k, v) -> (next id, present k, present v)."""

    def __init__(self, cross_sess, prefill_sess, step_sess, n_layers: int, max_prompt_pos: int, max_text_pos: int):
        self.c, self.p, self.s, self.L = cross_sess, prefill_sess, step_sess, int(n_layers)
        self.P, self.T = int(max_prompt_pos), int(max_text_pos)
        self.pos_cap = self.P + self.T - 1                 # the position table's end (in format)

    def position(self, idx: int, plen: int) -> int:
        """Decoder positions: prompt tokens 0..plen-1, target tokens P + (idx - plen), clamped."""
        return min(idx if idx < plen else self.P + idx - plen, self.P + self.T - 1)

    def cross(self, enc: np.ndarray):
        n = enc.shape[0]
        k, v, _ = self.c.run(["cross_k", "cross_v", "cross_add"], {"enc": np.ascontiguousarray(enc[None], np.float32),
                                                                   "enc_lens": np.array([n], np.int64)})
        return k, v

    def _feeds(self, k, v):
        L = self.L
        return {**{f"cross_k{i}": k[i] for i in range(L)}, **{f"cross_v{i}": v[i] for i in range(L)},
                "cross_add": np.zeros((1, 1, 1, k.shape[3]), np.float32)}

    def prefill(self, prompt, k, v):
        o = self.p.run(None, {"tokens": np.array([prompt], np.int64), **self._feeds(k, v)})
        return int(o[0][0]), list(o[1:1 + self.L]), list(o[1 + self.L:1 + 2 * self.L])

    def step(self, token: int, pos_id: int, pk, pv, k, v):
        L = self.L
        o = self.s.run(None, {"tokens": np.array([[token]], np.int64), "pos": np.array([pos_id], np.int64),
                              **{f"past_k{i}": pk[i] for i in range(L)}, **{f"past_v{i}": pv[i] for i in range(L)},
                              **self._feeds(k, v)})
        return int(o[0][0]), list(o[1:1 + L]), list(o[1 + L:1 + 2 * L])


class DecStream:
    """One decoder segment on the ONNX graphs. feed(enc_chunk (n, d), final) ->
    token ids emitted now; `out` / `at` (frames of the segment seen when each was emitted) / `guards` / `done`."""

    def __init__(self, graphs: DecoderGraphs, prompt, eot: int, wait: int, fmt: str = "in", max_len: int = 448):
        if fmt not in ("in", "out"):
            raise ValueError(f"streaming decoder format {fmt!r}: in | out")
        self.g, self.prompt, self.eot, self.wait, self.fmt, self.max_len = graphs, list(prompt), int(eot), int(wait), \
            fmt, int(max_len)
        self.out, self.at = [], []
        self.guards = {"eot_mid": 0, "burst_max": 0, "wait_at_end": 0}
        self.pos = 0
        self.hist = list(prompt)
        self.k = self.v = None                   # cross K/V (L, 1, H, frames, dh) of every frame so far
        self.pk = self.pv = None                 # self-attention cache, per layer (1, H, t, dh)
        self.frames = 0
        self.done = False
        self.end = None                          # why the decoder stopped by itself: eot | max_len | pos_cap

    def _drop_last(self):
        self.pk = [x[:, :, :-1] for x in self.pk]
        self.pv = [x[:, :, :-1] for x in self.pv]

    def feed(self, enc: np.ndarray, final: bool = False) -> list:
        if self.done:
            return []
        if enc.shape[0]:
            k, v = self.g.cross(enc)
            self.k = k if self.k is None else np.concatenate([self.k, k], 3)
            self.v = v if self.v is None else np.concatenate([self.v, v], 3)
            self.frames += enc.shape[0]
        if self.frames == 0:                     # nothing to attend yet
            if final:
                self.done = True
            return []
        P = len(self.prompt)
        first = None
        if self.pos == 0:                        # the prompt, teacher-forced at the first chunk: one prefill call
            first, self.pk, self.pv = self.g.prefill(self.prompt, self.k, self.v)   # = its steps 0 .. P - 1
            self.pos = P - 1
        new, burst = [], 0
        while len(self.out) < self.max_len:
            if self.pos >= self.g.pos_cap:
                self.guards["pos_cap"] = self.guards.get("pos_cap", 0) + 1
                self.done, self.end = True, "pos_cap"
                return new
            if first is not None:
                nxt, first = first, None
            else:
                nxt, self.pk, self.pv = self.g.step(self.hist[self.pos], self.g.position(self.pos, P), self.pk,
                                                    self.pv, self.k, self.v)
            if nxt == self.wait:
                if final:
                    self.guards["wait_at_end"] += 1
                    self.done = True
                    return new
                if self.fmt == "out":            # not fed back: drop this attempt's self-attention keys
                    self._drop_last()
                else:
                    self.hist.append(self.wait)
                    self.pos += 1
                break
            if nxt == self.eot:
                if not final:
                    self.guards["eot_mid"] += 1
                self.done, self.end = True, "eot"
                return new
            self.out.append(nxt)
            self.at.append(self.frames)
            new.append(nxt)
            self.hist.append(nxt)
            self.pos += 1
            burst += 1
            self.guards["burst_max"] = max(self.guards["burst_max"], burst)
        if first is not None:                    # max_len 0: the prefill's step at P - 1 was never taken
            self._drop_last()
        if len(self.out) >= self.max_len:
            self.done, self.end = True, "max_len"
        return new


def rope_rotate(k: np.ndarray, offset: float, base: float) -> np.ndarray:
    """apply_rope(k, cos(offset), sin(offset)) in fp32 with the model's RoPE tables (rotate-half): the
    keys' positions shift by `offset` (negative: back)."""
    dh = k.shape[-1]
    inv = (np.float32(1.0) / (np.float32(base) ** (np.arange(0, dh, 2, dtype=np.float32) / np.float32(dh))))
    ang = (np.float32(offset) * inv).astype(np.float32)
    ang = np.concatenate([ang, ang])
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    k = k.astype(np.float32)
    rot = np.concatenate([-k[..., dh // 2:], k[..., :dh // 2]], -1)
    return (k * cos + rot * sin).astype(np.float32)


class DecSession:
    """A whole session (no length limit) through both streams on the ONNX graphs.
    feed_mel() / feed_audio() -> newly committed decoder tokens [(id, emit frame)]; finish() flushes. `tokens` holds
    every committed (id, absolute emit frame), `ctc` the CTC stream's ids, `seg_log` one entry per closed segment
    {start, end (frames), tokens, reason: end | endpoint | max_segment | eot | max_len | pos_cap}, guards() the
    summed guard counts (burst_max: the largest)."""

    def __init__(self, encoder: StreamingEncoder, graphs: DecoderGraphs, prompt, eot: int, wait: int, fmt: str = "in",
                 max_len: int = 448, endpoint_frames: int = 8, max_segment_frames: int = 375,
                 rebase_frames: int = 7500, rope_base: float = 10000.0, blank: int = 0, frontend=None,
                 max_frames: int | None = None):
        self.e, self.g = encoder, graphs
        self.prompt, self.eot, self.wait, self.fmt, self.max_len = list(prompt), eot, wait, fmt, max_len
        self.endpoint_frames, self.max_segment_frames, self.rebase_frames = int(endpoint_frames), \
            int(max_segment_frames), int(rebase_frames)
        self.rope_base, self.blank = float(rope_base), blank
        self.mel = MelStream(frontend, encoder.n_mels) if frontend is not None else None
        self.state = encoder.init_state()
        self.keep = int(self.state["k_cache"].shape[3])     # = (left // chunk) * chunk
        self.max_frames = MAX_FRAMES if max_frames is None else max_frames
        self.pending = np.zeros((0, encoder.n_mels), np.float32)
        self.pos = 0                         # RoPE position of the next encoder frame (rebased)
        self.frames = 0                      # encoder frames so far (absolute)
        self.tokens: list = []               # [id, emit frame]
        self.ctc: list = []
        self.seg_log: list = []
        self.segments = 0
        self.closed: list = []               # finished DecStreams (guards)
        self._new_segment()
        self._prev = blank
        self._blank_run = 0
        self._speech = False
        self.done = False

    def _new_segment(self):
        self.dec = DecStream(self.g, self.prompt, self.eot, self.wait, self.fmt, self.max_len)
        self.seg_frames = 0
        self.seg_start = self.frames
        self.seg_tokens: list = []
        self.segments += 1

    def _take(self, ids) -> list:
        new = [[int(v), self.seg_start + a] for v, a in zip(ids, self.dec.at[len(self.dec.out) - len(ids):])]
        self.tokens += new
        self.seg_tokens += [t[0] for t in new]
        return new

    def _close(self, reason: str):
        self.seg_log.append({"start": self.seg_start, "end": self.frames, "tokens": list(self.seg_tokens),
                             "reason": reason})
        self.closed.append(self.dec)

    def _reset(self):
        self._new_segment()
        self._speech = False
        self._maybe_rebase()

    def _maybe_rebase(self):
        off = self.pos - self.keep
        if self.pos < self.rebase_frames or off <= 0:
            return
        self.state["k_cache"] = rope_rotate(self.state["k_cache"], -float(off), self.rope_base)
        self.pos -= off

    def _run_chunk(self, m: np.ndarray, final: bool) -> list:
        n_new = -(-m.shape[0] // 8)
        if self.pos + n_new > self.max_frames:
            raise StreamLimit(f"stream past {self.max_frames} encoder frames without a segment reset (positions are "
                              "rebased only at resets: lower max_segment_frames)")
        enc, ids, n, self.state = self.e.encode_chunk(m, self.state, self.pos)
        self.pos += n
        self.frames += n
        new = []
        if n:
            for v in ids.tolist():               # the CTC stream and the endpoint detector
                if v != self._prev and v != self.blank:
                    self.ctc.append(int(v))
                self._prev = v
                self._blank_run = self._blank_run + 1 if v == self.blank else 0
                self._speech = self._speech or v != self.blank
            self.seg_frames += n
            endpoint = self._speech and self._blank_run >= self.endpoint_frames
            if final or endpoint or self.seg_frames >= self.max_segment_frames:
                new += self._take(self.dec.feed(enc, final=True))
                self._close("end" if final else "endpoint" if endpoint else "max_segment")
                if not final:
                    self._reset()
            else:
                new += self._take(self.dec.feed(enc))
                if self.dec.done:                # <|eot|>, max_len or pos_cap ended it early: a new segment
                    self._close(self.dec.end or "eot")
                    self._reset()
        return new

    def feed_mel(self, mel: np.ndarray) -> list:
        assert not self.done
        mel = np.asarray(mel, np.float32)
        self.pending = np.concatenate([self.pending, mel]) if len(self.pending) else mel
        step = 8 * self.e.chunk
        new = []
        while len(self.pending) > step:          # strictly: the last full chunk waits to learn if it is the final one
            m, self.pending = self.pending[:step], self.pending[step:]
            new += self._run_chunk(m, False)
        return new

    def feed_audio(self, x: np.ndarray) -> list:
        return self.feed_mel(self.mel.feed(x))

    def finish(self) -> list:
        if self.done:
            return []
        new = self.feed_mel(self.mel.finish()) if self.mel is not None else []
        if len(self.pending):                    # the held-back chunk (always there once any audio came in)
            new += self._run_chunk(self.pending, True)
            self.pending = self.pending[:0]
        self.done = True
        return new

    def guards(self) -> dict:
        ds = self.closed + ([] if self.dec in self.closed else [self.dec])
        g = {"eot_mid": 0, "wait_at_end": 0, "pos_cap": 0, "max_len": 0, "burst_max": 0}
        for d in ds:
            for k in ("eot_mid", "wait_at_end", "pos_cap"):
                g[k] += d.guards.get(k, 0)
            g["max_len"] += int(d.end == "max_len")
            g["burst_max"] = max(g["burst_max"], d.guards["burst_max"])
        return g
