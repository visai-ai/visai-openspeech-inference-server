"""Audio input and long-form windowing (numpy only).

Long audio is cut into windows of at most `max_seconds` (default 30 s, configurable): each cut sits at the centre of
the longest low-energy run (>= 200 ms) between `min_seconds` and `max_seconds` from the window start, or at the
quietest 100 ms if there is no such run. Windows do not overlap; the runtime joins their token ids and detokenises
once, so SentencePiece handles the spacing at the joins.
"""
from __future__ import annotations

import io
import math

import numpy as np

SR = 16000


def load(src, sr: int = SR) -> np.ndarray:
    """A file path, bytes or a file object -> mono float32 at `sr` (soundfile; resampling needs soxr)."""
    import soundfile as sf
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    x, r = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if r != sr:
        try:
            import soxr
        except ImportError as e:
            raise ValueError(f"audio is {r} Hz; install soxr to resample to {sr} Hz") from e
        x = soxr.resample(x, r, sr).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def _frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, np.float32)
    e = (x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)
    return (10 * np.log10(e + 1e-10)).astype(np.float32)


def _cut_offset(seg: np.ndarray, min_run_ms: int = 200, info: dict | None = None) -> int:
    """Frame index (10 ms frames) inside a search region at the centre of its longest low-energy run (frames at or
    below the region's 20th percentile, run >= min_run_ms), else at the quietest 100 ms (counted in info)."""
    thr = np.percentile(seg, 20) if len(seg) else -100.0
    low = seg <= thr
    best_len, best_mid, run = 0, None, 0
    for i, v in enumerate(low):
        run = run + 1 if v else 0
        if run > best_len:
            best_len, best_mid = run, i - run // 2
    if best_mid is None or best_len * 10 < min_run_ms:
        if info is not None:
            info["fallback"] = info.get("fallback", 0) + 1
        k = 10
        sm = np.convolve(seg, np.ones(k) / k, mode="same") if len(seg) >= k else seg
        best_mid = int(np.argmin(sm))
    return int(best_mid)


def windows(x: np.ndarray, max_seconds: float = 30.0, min_seconds: float = 20.0, sr: int = SR,
            min_run_ms: int = 200, cut_fn=None) -> list[tuple[int, int]]:
    """[start, end) sample ranges covering x, each at most max_seconds long (whole-array version of Windower)."""
    w = Windower(max_seconds, min_seconds, sr, min_run_ms, cut_fn=cut_fn)
    out = [(a, a + len(seg)) for a, seg in w.feed(x)]
    out += [(a, a + len(seg)) for a, seg in w.flush()]
    return out


class Windower:
    """Incremental long-form cutter: feed() audio blocks as they are decoded, get (start_sample, window) pairs as soon
    as a cut is decided; flush() at the end. The decision for a window only looks at its own [min, max] search region
    (frames aligned to sample 0), and RAM holds at most two windows plus a block. One decided window is held back so
    the end of the file can be balanced: a tail shorter than `min_tail_seconds` is merged into the previous window when
    the two fit in max_seconds, else the two are re-cut at the quietest point near their middle (no tiny last window).
    `stats`: cuts, fallback (quietest-100 ms cuts), tail merges / re-cuts."""

    def __init__(self, max_seconds: float = 30.0, min_seconds: float = 20.0, sr: int = SR, min_run_ms: int = 200,
                 min_tail_seconds: float = 4.0, cut_fn=None):
        self.W, self.sr, self.hop, self.run_ms = int(max_seconds * sr), sr, sr // 100, min_run_ms
        self.lo = max(1, int(min(min_seconds, max_seconds * 0.67) * 100))
        self.hi = int(max_seconds * 100)
        self.tail = int(min_tail_seconds * sr)
        self.buf = np.zeros(0, np.float32)
        self.start = 0                                 # absolute sample index of buf[0]
        self.pending = None                            # the last decided window, held back for end balancing
        self.stats = {"cuts": 0, "fallback": 0, "tail_merge": 0, "tail_recut": 0, "model_cut": 0}
        self.cut_fn = cut_fn                           # (window audio, lo sample, hi sample) -> cut sample or None

    def feed(self, block: np.ndarray):
        self.buf = np.concatenate([self.buf, np.asarray(block, np.float32)]) if len(self.buf) else \
            np.asarray(block, np.float32)
        while len(self.buf) > self.W:
            cut = None
            if self.cut_fn is not None:
                cut = self.cut_fn(self.buf[: self.W], self.lo * self.hop, self.hi * self.hop)
                if cut is not None:
                    self.stats["model_cut"] += 1
                    cut = int(cut) // self.hop * self.hop
            if cut is None:
                db = _frame_db(self.buf[: (self.hi + 1) * self.hop], self.hop)
                cut = (self.lo + _cut_offset(db[self.lo: self.hi], self.run_ms, self.stats)) * self.hop
            cut = min(max(cut, self.sr), self.W)
            self.stats["cuts"] += 1
            if self.pending is not None:
                yield self.pending
            self.pending = (self.start, self.buf[:cut])
            self.buf = self.buf[cut:]
            self.start += cut

    def flush(self):
        rest = self.buf
        self.buf = np.zeros(0, np.float32)
        if self.pending is None:
            if len(rest):
                yield self.start, rest
            return
        a, prev = self.pending
        self.pending = None
        if 0 < len(rest) < self.tail:
            both = np.concatenate([prev, rest])
            if len(both) <= self.W:
                self.stats["tail_merge"] += 1
                yield a, both
                return
            # re-cut the two at the quietest 100 ms within +-5 s of their middle
            db = _frame_db(both, self.hop)
            mid, half = len(db) // 2, 500
            lo = max(1, mid - half)
            seg = db[lo: mid + half]
            k = 10
            sm = np.convolve(seg, np.ones(k) / k, mode="same") if len(seg) >= k else seg
            cut = (lo + int(np.argmin(sm))) * self.hop
            self.stats["tail_recut"] += 1
            yield a, both[:cut]
            yield a + cut, both[cut:]
            return
        yield a, prev
        if len(rest):
            yield self.start, rest


# ffmpeg inputs: real audio / video containers only. No playlist, concat, image-sequence or other demuxer that can
# open further files or URLs (hls, concat, image2, ...): an upload can only be decoded as itself.
FFMPEG_FORMATS = "wav,w64,flac,mp3,ogg,matroska,webm,mov,mp4,m4a,3gp,3g2,mj2,aac,aiff,caf,amr"
_FF_IN = ["-protocol_whitelist", "file,pipe", "-format_whitelist", FFMPEG_FORMATS]


MAX_CHANNELS = 8                 # uploads with more channels are refused (they are mixed down to mono anyway)
MIN_RATE, MAX_RATE = 4000, 384000    # Hz
BLOCK_BYTES = 32 * 2 ** 20       # decoded float32 bytes per soundfile block, all channels together


def _child_env() -> dict:
    """Environment for the decoder subprocesses: PATH and locale only, so ffmpeg / ffprobe (which parse untrusted
    uploads) never see the server's API key or tokens."""
    import os
    keep = ("PATH", "LANG", "LC_ALL", "TMPDIR", "HOME")
    return {k: os.environ[k] for k in keep if k in os.environ}


class SpawnError(RuntimeError):
    """ffprobe / ffmpeg could not be started (OSError at launch: EAGAIN, ENOMEM, a binary gone): a transient
    server-side failure, not the input's fault (the server answers 503, retry)."""


def _spawn(cmd, **kw):
    import subprocess
    try:
        kw.setdefault("env", _child_env())
        return subprocess.Popen(cmd, **kw)
    except OSError as e:
        raise SpawnError(f"could not start {cmd[0]} ({type(e).__name__}: {e.strerror or e})") from None


class DecodeHandle:
    """Lets another thread stop a decode that is blocked in a read: cancel() kills the current ffprobe / ffmpeg
    process (its pipe then ends, so the reading thread returns). A soundfile decode reads a local file in bounded
    time and stops at the next block once cancelled.

    Outcome, decided once under the lock: state "running" -> "finished" (finish(): the decode completed, or failed on
    its own) or "timed_out" (time_out(): a deadline fired; only that transition kills). Whichever transition happens
    first decides the response; the other then returns False and does nothing."""

    def __init__(self):
        import threading
        self.proc, self.cancelled, self.lock = None, False, threading.Lock()
        self.state = "running"

    def finish(self) -> bool:
        with self.lock:
            if self.state != "running":
                return False
            self.state = "finished"
            return True

    def time_out(self) -> bool:
        with self.lock:
            if self.state != "running":
                return False
            self.state = "timed_out"
            self.cancelled = True
            if self.proc is not None and self.proc.poll() is None:
                self.proc.kill()
            return True

    def set_proc(self, p):
        with self.lock:
            self.proc = p
            if self.cancelled:
                p.kill()

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.proc is not None and self.proc.poll() is None:
                self.proc.kill()


FFPROBE_TIMEOUT = 30.0      # seconds for the container probe (a probe past it is killed: the input is refused)


def _ffprobe_run(entries: list, src, handle: DecodeHandle | None, of: str = "csv=p=0") -> str | None:
    """stdout of ffprobe with the decode's whitelists and the given selection / -show_entries arguments, or None
    without ffprobe; ValueError if ffprobe refuses the file, times out (FFPROBE_TIMEOUT, then killed) or is cancelled
    through `handle` (the probe process is registered on it, so DecodeHandle.cancel() kills it like ffmpeg)."""
    import os
    import shutil
    import subprocess
    if not shutil.which("ffprobe"):
        return None
    p = _spawn(["ffprobe", "-v", "error", *_FF_IN, *entries, "-of", of, "file:" + os.path.abspath(str(src))],
               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if handle is not None:
        handle.set_proc(p)
    try:
        out, _ = p.communicate(timeout=FFPROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise ValueError(f"the container probe took longer than {FFPROBE_TIMEOUT:g} s") from None
    finally:
        if p.poll() is None:                 # an exception while waiting (e.g. KeyboardInterrupt): never leave it
            p.kill()
            p.wait()
    if p.returncode != 0 or not out.strip():
        raise ValueError("not an audio container this server decodes")
    return out


def ffprobe(src, handle: DecodeHandle | None = None) -> tuple[str | None, float | None]:
    """(demuxer name, duration seconds) of a local file through ffprobe (see _ffprobe_run); (None, None) without
    ffprobe."""
    out = _ffprobe_run(["-show_entries", "format=format_name,duration"], src, handle)
    if out is None:
        return None, None
    line = out.strip().splitlines()[0]
    name, _, dur = line.rpartition(",")
    try:
        d = float(dur)
    except ValueError:
        name, d = line, None
    if d is not None and not (math.isfinite(d) and d > 0):   # nan / inf / negative: no usable estimate
        d = None
    return name.strip('"').split(",")[0] or None, d


def check_audio_stream(src, handle: DecodeHandle | None = None) -> None:
    """The first audio stream's channel count and sample rate, through ffprobe, within the same limits as the
    soundfile path (MAX_CHANNELS, MIN_RATE-MAX_RATE): ValueError("unsupported audio: ...") otherwise, or when the file
    has no audio stream or ffprobe does not say."""
    out = _ffprobe_run(["-select_streams", "a:0", "-show_entries", "stream=channels,sample_rate"], src, handle,
                       of="default=noprint_wrappers=1")
    if out is None:
        raise ValueError("unsupported audio: cannot check the stream without ffprobe")
    kv = {}
    for ln in out.splitlines():                  # named fields: "sample_rate=16000", "channels=1"
        k, _, v = ln.partition("=")
        kv.setdefault(k.strip(), v.strip())
    try:
        rate, ch = int(kv["sample_rate"]), int(kv["channels"])
    except (KeyError, ValueError):
        raise ValueError("unsupported audio: no audio stream with a channel count and sample rate") from None
    if not (1 <= ch <= MAX_CHANNELS) or not (MIN_RATE <= rate <= MAX_RATE):
        raise ValueError(f"unsupported audio: {ch} channels at {rate} Hz (at most {MAX_CHANNELS} channels, "
                         f"{MIN_RATE}-{MAX_RATE} Hz)")


def probe_seconds(src, handle: DecodeHandle | None = None) -> float | None:
    """Duration estimate before decoding, for preallocation (soundfile header, else ffprobe); None if unknown or the
    probe fails. An uncompressed (PCM) header cannot claim more frames than the file's bytes hold. SpawnError (the
    probe could not even start) propagates: it is the server's problem, not an unknown duration."""
    import os
    import soundfile as sf
    try:
        info = sf.info(src)
        frames = info.frames
        uncompressed = info.format in ("WAV", "WAVEX", "W64", "RF64", "AIFF", "AU", "RAW") and (
            str(info.subtype).startswith("PCM_") or info.subtype in ("FLOAT", "DOUBLE", "ULAW", "ALAW"))
        if uncompressed:                     # (FLAC also reports PCM_16: compressed, no byte bound)
            width = {"PCM_S8": 1, "PCM_U8": 1, "PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8,
                     "ULAW": 1, "ALAW": 1}.get(info.subtype, 1)
            frames = min(frames, os.path.getsize(src) // max(1, width * info.channels))
        d = frames / info.samplerate
        return d if math.isfinite(d) and d > 0 else None
    except Exception:  # noqa: BLE001
        pass
    try:
        return ffprobe(src, handle)[1]               # on the request's handle: its deadline kills this probe too
    except ValueError:
        return None


def stream(src, sr: int = SR, block_seconds: float = 30.0, handle: DecodeHandle | None = None):
    """Decode a file (path) as mono float32 blocks at `sr`, memory-bounded: soundfile blocks for the formats it reads
    (WAV/FLAC/OGG/MP3, resampled with soxr when the rate differs), else an ffmpeg pipe (m4a, webm, ...: allowed
    containers only, local file only). `handle` lets another thread cancel a blocked decode. Closing the generator
    early kills ffmpeg and raises nothing (the caller's own exception is kept)."""
    import soundfile as sf
    try:
        f = sf.SoundFile(src)
    except Exception:  # noqa: BLE001 (not a soundfile format)
        f = None
    if f is not None:
        with f:
            # header sanity before any allocation: a block is channels x frames float32, so an absurd channel count or
            # sample rate in a small file could otherwise cost many times the upload's size in memory
            if not (1 <= f.channels <= MAX_CHANNELS) or not (MIN_RATE <= f.samplerate <= MAX_RATE):
                raise ValueError(f"unsupported audio: {f.channels} channels at {f.samplerate} Hz (at most "
                                 f"{MAX_CHANNELS} channels, {MIN_RATE}-{MAX_RATE} Hz)")
            rs = None
            if f.samplerate != sr:
                try:
                    import soxr
                    rs = soxr.ResampleStream(f.samplerate, sr, 1, dtype="float32")
                except ImportError:
                    f = None
            if f is not None:
                # frames per block: block_seconds, but never more than BLOCK_BYTES of float32 samples
                n = max(1, min(int(block_seconds * f.samplerate), BLOCK_BYTES // (4 * f.channels)))
                for blk in f.blocks(blocksize=n, dtype="float32", always_2d=True):
                    if handle is not None and handle.cancelled:
                        return
                    x = blk.mean(axis=1) if blk.shape[1] > 1 else blk[:, 0]
                    yield rs.resample_chunk(x) if rs else x
                if rs:
                    yield rs.resample_chunk(np.zeros(0, np.float32), last=True)
                return
    import os
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        raise ValueError("audio format needs ffmpeg (not found) or a 16 kHz WAV/FLAC")
    # local files only: the input is named with the file: protocol, every protocol but file / pipe is refused, and only
    # real containers are demuxed (FFMPEG_FORMATS), with mov's external data references off: a URL as the path, a
    # playlist / concat list or an mp4 dref inside an upload cannot make ffmpeg fetch or open anything else. The
    # container is probed first (ffprobe, same whitelists) so it can be forced with -f and -enable_drefs given to mov
    # only (the option does not exist for other demuxers).
    fmt, _ = ffprobe(src, handle)
    if handle is not None and handle.cancelled:
        return
    check_audio_stream(src, handle)          # the same channel / rate limits as the soundfile path
    if handle is not None and handle.cancelled:
        return
    cmd = ["ffmpeg", "-nostdin", "-v", "error", *_FF_IN]
    if fmt:
        cmd += ["-f", fmt] + (["-enable_drefs", "0"] if fmt == "mov" else [])
    # decode the first audio stream only: the one check_audio_stream checked (no video / subtitle / data streams)
    cmd += ["-i", "file:" + os.path.abspath(str(src)), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    p = _spawn(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if handle is not None:
        handle.set_proc(p)
    n = int(block_seconds * sr) * 4
    done = False
    try:
        while True:
            b = p.stdout.read(n)
            if not b:
                break
            yield np.frombuffer(b[: len(b) // 4 * 4], np.float32).copy()
        done = True
    finally:
        if not done:                  # closed early (caller error, cancel): stop ffmpeg, keep the caller's exception
            p.kill()
        p.stdout.close()
        rc = p.wait()
        if done and rc != 0:
            raise ValueError("ffmpeg could not decode the input")
