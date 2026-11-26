"""Command line: `python -m speech_server serve ...` and `python -m speech_server transcribe ...`.

Engine flags are shared; a JSON `--config` file can set any of them (command-line values win). The common flags
also have an environment variable (ENV_FLAGS below and the README), so a container is configured with `-e`.
The model comes from a local directory (--model / MODEL_DIR) or a Hugging Face repository downloaded on first start
(--model-repo / MODEL_REPO; speech_server/models.py). API keys are never taken on the command line: the server reads
its key from an environment variable (--api-key-env, default SPEECH_API_KEY) or a file only the owner can read
(--api-key-file, mode 0600); clients send it in the `Authorization: Bearer` header.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import stat
import sys
import threading
import time
import urllib.parse
import urllib.request

ENGINE_KEYS = ["model", "device", "provider", "precision", "threads", "decode_threads", "mode", "decode", "beam",
               "batch_size", "max_batch_seconds", "decode_batch", "kv_budget_mb", "lang", "max_new_tokens",
               "max_seconds", "min_seconds", "len_alpha", "graph_opt", "encoder_graph", "itn", "pnc", "gpu_mem_limit_mb",
               "timestamps", "lookahead", "endpoint_frames", "max_segment_frames", "word_boost"]
EXTRA_KEYS = ("timed_segments", "token_times", "align_ok", "lookahead", "guards")   # keys present only when asked for
STREAM_MODES = ("ctc-stream", "dec-stream")      # chunk-encoder modes: fp32, greedy, served from --stream-model


def _num(kind, lo, open_lo: bool = False, hi=None):
    """argparse type: a finite int / float >= lo (> lo with open_lo), <= hi; NaN and inf are refused."""
    import math

    def parse(v):
        try:
            x = kind(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{v!r} is not a{'n integer' if kind is int else ' number'}") from None
        if not math.isfinite(x) or x < lo or (open_lo and x <= lo) or (hi is not None and x > hi):
            raise argparse.ArgumentTypeError(f"{v!r}: must be finite and {'>' if open_lo else '>='} {lo}"
                                             + (f" and <= {hi}" if hi is not None else ""))
        return x
    return parse


def engine_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", help="JSON file with any of the flags below (snake_case keys)")
    ap.add_argument("--model", help="model directory (spec.json, tokenizer, ONNX graphs); env MODEL_DIR")
    ap.add_argument("--model-repo", help="Hugging Face model repository <owner>/<name>, downloaded on first start into "
                    "--model-cache (used when --model is not given); env MODEL_REPO")
    ap.add_argument("--model-revision", help="branch, tag or commit of --model-repo; env MODEL_REVISION")
    ap.add_argument("--model-cache", help="where downloaded repositories are kept (default ~/.cache/speech-server); "
                    "env MODEL_CACHE")
    ap.add_argument("--hf-token-env", default=None, help="environment variable holding a Hugging Face read token for "
                    "private repositories (default HF_TOKEN); the token itself is never a flag")
    ap.add_argument("--device", choices=["cpu", "cuda"])
    ap.add_argument("--provider", choices=["CPU", "CUDA", "TensorRT"])
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--threads", type=int, help="intra-op threads (CPU: cap at the host's share)")
    ap.add_argument("--decode-threads", type=int)
    ap.add_argument("--mode", choices=["decoder", "ctc", "ctc-stream", "dec-stream"], help="ctc: encoder + CTC head "
                    "only, decoder never loaded; ctc-stream: the streaming chunk encoder over each file (a streaming "
                    "package); dec-stream (preview): the streaming decoder on it (a streaming-decoder package)")
    ap.add_argument("--decode", choices=["greedy", "beam"], help="greedy: plain greedy, no loop/length heuristics")
    ap.add_argument("--beam", type=_num(int, 1, hi=16), help="beam width for --decode beam (1-16, default 4)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-batch-seconds", type=float, help="cap on padded audio seconds per encoder batch")
    ap.add_argument("--decode-batch", type=int, help="decoder rows pooled across encoder batches (0: no pooling)")
    ap.add_argument("--kv-budget-mb", type=float, help="cap on decoder KV buffers per decode batch")
    ap.add_argument("--gpu-mem-limit-mb", type=float, help="hard GPU memory cap: per-session arenas + KV buffers "
                    "(a batch that does not fit fails loudly instead of growing into other jobs' memory)")
    ap.add_argument("--encoder-graph", help="auto (per-architecture order) or one encoder graph of the package, e.g. "
                    "encoder5")
    ap.add_argument("--lang", help="language prompt (th, en, zh, ...); default automatic")
    ap.add_argument("--max-new-tokens", type=int, help="generated tokens incl. <|eot|> (default 448; at most the "
                    "model's text-position limit)")
    ap.add_argument("--max-seconds", type=float, help="long-form window length (default 30)")
    ap.add_argument("--min-seconds", type=float, help="earliest cut point in a window (default 20)")
    ap.add_argument("--len-alpha", type=float)
    ap.add_argument("--graph-opt", choices=["all", "extended", "basic", "none"])
    ap.add_argument("--itn", action="store_true", default=None)
    ap.add_argument("--pnc", action="store_true", default=None)
    ap.add_argument("--timestamps", choices=["none", "segment", "token"], help="segment: the decoder's timestamp "
                    "tokens (decoder mode); token: per-token times from the CTC head (all modes)")
    ap.add_argument("--lookahead", type=int, help="ctc-stream / dec-stream: look-ahead L in 80 ms frames (one the "
                    "package has; default the largest)")
    ap.add_argument("--endpoint-frames", type=_num(int, 1), help="dec-stream: CTC-blank frames after speech that end "
                    "a decoder segment (default 8 = 640 ms)")
    ap.add_argument("--max-segment-frames", type=_num(int, 1), help="dec-stream: longest decoder segment in encoder "
                    "frames (default 375 = 30 s)")
    ap.add_argument("--word-boost", action=argparse.BooleanOptionalAction, default=None,
                    help="per-request word boost (word_boost / boost / ctc_beam options; default on): loads what "
                    "boosting needs where the package has it; --no-word-boost refuses word_boost; env WORD_BOOST")
    ap.add_argument("--api-key", help=argparse.SUPPRESS)


DEFAULTS = {"device": "cuda", "precision": "fp16", "mode": "decoder", "decode": "greedy", "beam": 4,
            "batch_size": 64, "max_batch_seconds": 600.0, "decode_batch": 256, "max_seconds": 30.0,
            "min_seconds": 20.0, "len_alpha": 1.0, "graph_opt": "all", "encoder_graph": "auto",
            "max_new_tokens": 448}


# Environment variables for the common flags. They are turned into command-line arguments before parsing (only for
# flags not already on the command line), so they get exactly the flags' validation (choices, types, ranges).
# Precedence: command line, then environment, then the --config file, then the defaults.
#   (subcommands, flag, variable, kind): kind "value" passes the value, "bool" maps true/false to --flag/--no-flag
ENV_FLAGS = [
    (("serve", "transcribe"), "--model", "MODEL_DIR", "value"),
    (("serve", "transcribe"), "--model-repo", "MODEL_REPO", "value"),
    (("serve", "transcribe"), "--model-revision", "MODEL_REVISION", "value"),
    (("serve", "transcribe"), "--model-cache", "MODEL_CACHE", "value"),
    (("serve", "transcribe"), "--device", "DEVICE", "value"),
    (("serve", "transcribe"), "--precision", "PRECISION", "value"),
    (("serve", "transcribe"), "--threads", "THREADS", "value"),
    (("serve", "transcribe"), "--decode", "DECODE", "value"),
    (("serve", "transcribe"), "--beam", "BEAM", "value"),
    (("serve", "transcribe"), "--lang", "LANG_DEFAULT", "value"),
    (("serve", "transcribe"), "--mode", "MODE", "value"),
    (("serve",), "--stream-model", "STREAM_MODEL_DIR", "value"),
    (("serve",), "--stream-model-repo", "STREAM_MODEL_REPO", "value"),
    (("serve",), "--stream-model-revision", "STREAM_MODEL_REVISION", "value"),
    (("serve",), "--modes", "MODES", "value"),
    (("serve",), "--precisions", "PRECISIONS", "value"),
    (("serve",), "--decodes", "DECODES", "value"),
    (("serve",), "--stats", "STATS", "bool"),
    (("serve",), "--host", "HOST", "value"),
    (("serve",), "--port", "PORT", "value"),
    (("serve",), "--max-streams", "MAX_STREAMS", "value"),
    (("serve",), "--model-name", "MODEL_NAME", "value"),
    (("serve",), "--share-encoder", "SHARE_ENCODER", "bool"),
    (("serve", "transcribe"), "--word-boost", "WORD_BOOST", "bool"),
]


def env_argv(argv: list) -> list:
    """argv plus `--flag value` for every ENV_FLAGS variable that is set (non-blank) and whose flag the command line
    does not already give; the subcommand is the first argument that is not an option."""
    pos = next((i for i, x in enumerate(argv) if not x.startswith("-")), None)
    cmd = argv[pos] if pos is not None else None
    extra = []
    for cmds, flag, var, kind in ENV_FLAGS:
        if cmd not in cmds:
            continue
        v = os.environ.get(var)
        if v is None or not v.strip():
            continue
        names = [flag] + (["--no-" + flag[2:]] if kind == "bool" else [])
        names += {"--mode": ["--modes"], "--modes": ["--mode"], "--precision": ["--precisions"],
                  "--precisions": ["--precision"], "--decode": ["--decodes"],
                  "--decodes": ["--decode"]}.get(flag, [])    # one setting, two spellings: the CLI's wins
        if any(x == n or x.startswith(n + "=") for x in argv for n in names):
            continue
        v = v.strip()
        if kind == "bool":
            if v.lower() in ("1", "true", "yes", "on"):
                extra.append(flag)
            elif v.lower() in ("0", "false", "no", "off"):
                extra.append("--no-" + flag[2:])
            else:
                sys.exit(f"${var}={v!r}: expected true or false")
        else:
            extra.append(f"{flag}={v}")
    # right after the subcommand: never after a "--" (where they would become positional inputs); the parsers take
    # no abbreviated flags (allow_abbrev=False), so the full-name check above sees every flag the command line gives
    return list(argv) if pos is None else argv[:pos + 1] + extra + argv[pos + 1:]


def resolve(a) -> dict:
    if getattr(a, "api_key", None):
        sys.exit("refusing --api-key on the command line: use --api-key-env or --api-key-file (mode 0600)")
    cfg = dict(DEFAULTS)
    explicit = set()                              # keys the user set (config file or command line), not defaults
    if a.config:
        with open(a.config) as f:
            c = json.load(f)
        cfg.update(c)
        explicit |= set(c)
    for k in ENGINE_KEYS:
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
            explicit.add(k)
    if getattr(a, "precisions", None):
        explicit.add("precisions")
    cfg["_explicit"] = sorted(explicit)
    if cfg.get("mode") in STREAM_MODES:           # the streaming chunk graphs are fp32 only: an explicit other
        if "precision" in explicit and cfg["precision"] != "fp32":      # precision (any source) is refused; the
            sys.exit(f"--mode {cfg['mode']} runs fp32 chunk graphs only, not {cfg['precision']} (from "  # default fp32
                     f"{'the config file' if 'precision' not in vars(a) or a.precision is None else '--precision'})")
        cfg["precision"] = "fp32"
    if cfg.get("device") == "cpu" and cfg.get("precision") == "fp16" and "precision" not in explicit:
        cfg["precision"] = "fp32"                 # the default (fp16) is a GPU default; CPU defaults to fp32
    if isinstance(cfg.get("lang"), str) and cfg["lang"].strip().lower() in ("", "auto"):
        cfg["lang"] = None                        # automatic language detection
    from speech_server import models
    try:
        pick = lambda k: getattr(a, k, None) or cfg.get(k)    # noqa: E731 (command line / env, then --config)
        cfg["_model_cache"], cfg["_token_env"] = pick("model_cache"), pick("hf_token_env") or "HF_TOKEN"
        cfg["model"] = models.resolve(cfg.get("model"), pick("model_repo"), pick("model_revision"),
                                      cfg["_model_cache"], cfg["_token_env"])
    except models.ModelError as e:
        sys.exit(f"model: {e}")
    if not cfg.get("model"):
        sys.exit("no model: give --model <dir> (env MODEL_DIR) or --model-repo <owner>/<name> (env MODEL_REPO)")
    # the precision default (fp16 on CUDA) only where the package has fp16 graphs: an fp32-only package runs fp32
    # unless a precision was asked for explicitly (then a missing graph is an error at start-up)
    if "precision" not in explicit and "precisions" not in explicit and cfg.get("precision") == "fp16":
        try:
            with open(os.path.join(cfg["model"], "spec.json"), encoding="utf-8") as f:
                gr = json.load(f).get("graphs") or {}
            have = {str(k).rsplit(".", 1)[-1] for k, v in gr.items()   # graphs whose file the package ships
                    if isinstance(v, str) and os.path.isfile(os.path.join(cfg["model"], v))}
        except (OSError, ValueError):
            have = set()
        if "fp16" not in have and "fp32" in have:
            print("model: the package has no fp16 graphs; running fp32 (set --precision / PRECISION to choose)",
                  file=sys.stderr, flush=True)
            cfg["precision"] = "fp32"
    return cfg


def make_engine(cfg: dict, **extra):
    from speech_server.runtime import Engine
    kw = {k: cfg[k] for k in ENGINE_KEYS if k in cfg and k != "model" and cfg[k] is not None}
    return Engine(cfg["model"], **kw, **extra)


def read_key(env: str | None, path: str | None) -> str | None:
    if path:
        st = os.stat(path)
        if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            sys.exit(f"API key file must be mode 0600 (owner only); fix with chmod 600")
        with open(path) as f:
            return f.read().strip() or None
    if env and os.environ.get(env):
        return os.environ[env].strip() or None   # a blank key is no key (never "auth with the empty string")
    return None


# ---------------------------------------------------------------- transcribe

def cmd_transcribe(a) -> None:
    from speech_server import audio as audiolib
    files = list(a.inputs)
    if a.list:
        with open(a.list) as f:
            files += [line.strip() for line in f if line.strip()]
    out = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
    if a.server:
        key = read_key(a.api_key_env, a.api_key_file)
        for p in files:
            qd = {k: v for k, v in (("lang", a.lang), ("mode", a.mode), ("timestamps", a.timestamps),
                                    ("precision", a.precision), ("decode", a.decode)) if v}
            qd.update({k: "true" for k in ("itn", "pnc") if getattr(a, k)})
            if a.lookahead is not None:
                qd["lookahead"] = str(a.lookahead)
            q = ("?" + urllib.parse.urlencode(qd)) if qd else ""
            with open(p, "rb") as fh:
                body = fh.read()
            req = urllib.request.Request(a.server.rstrip("/") + "/v1/transcribe" + q, data=body,
                                         method="POST", headers={"Content-Type": "application/octet-stream",
                                                                 **({"Authorization": f"Bearer {key}"} if key else {})})
            with urllib.request.urlopen(req) as r:
                d = json.loads(r.read())
            out.write(json.dumps({"input": p, "text": d["text"], **{k: d[k] for k in EXTRA_KEYS if k in d}},
                                 ensure_ascii=False) + "\n")
        return
    cfg = resolve(a)
    cfg["word_boost"] = False                     # the local transcribe command takes no boost list: load nothing for it
    eng = make_engine(cfg)
    langs = [a.lang] * len(files) if a.lang else None
    for r in eng.transcribe_files(files, langs, pool_windows=a.pool_windows):
        out.write(json.dumps({"input": files[r["index"]], "text": r["text"], "duration": r["duration"],
                              "windows": r["windows"], "segments": r["segments"] if a.segments else None,
                              **{k: r[k] for k in EXTRA_KEYS if k in r}}, ensure_ascii=False) + "\n")
        out.flush()
    s = eng.stats
    print(json.dumps({"audio_seconds": round(s.audio_seconds, 2), "wall_s": round(s.wall_s, 3),
                      "rtfx": round(s.audio_seconds / max(s.wall_s, 1e-9), 1), "windows": s.windows}), file=sys.stderr)


# ---------------------------------------------------------------- serve

DEFAULT_LANGS = ("th", "en", "zh")


def served_langs(a, model_langs: list, default_lang) -> list:
    """The languages this server offers (--langs, default th,en,zh: the languages the model was trained on; its other
    language tokens were never trained as targets). Every one must be a language token of the model, and the server's
    own --lang default must be among them. No --langs attribute (library callers): every language token."""
    if not hasattr(a, "langs"):
        return list(model_langs)
    raw = a.langs
    if raw is None:                               # the default: the trained languages this model has
        return [x for x in DEFAULT_LANGS if x in model_langs] or list(model_langs)
    want = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    bad = [x for x in want if x not in model_langs]
    if not want or bad:
        raise ValueError(f"--langs takes languages of this model ({', '.join(model_langs)}), not {bad or raw!r}")
    if default_lang and default_lang not in want:
        raise ValueError(f"--lang {default_lang} is not among --langs ({', '.join(want)})")
    return want


def serve_facts(engines: dict, modes: list, cfg: dict, a) -> dict:
    """What this server actually serves, with its real defaults (GET /help, the demo page and request validation).
    No model names, paths or devices: /help is public."""
    from speech_server.spec import V1_LANGS
    from speech_server.stream import MAX_FRAMES
    precs = {}
    for m, p in engines:
        if ":" not in m:                           # "decoder:beam" etc.: extra decode kinds of a mode, not modes
            precs.setdefault(m, []).append(p)
    first = {m: engines[(m, precs[m][0])] for m in modes}
    decodes = ([first["decoder"].decode_kind] + [m.split(":", 1)[1] for m, p in engines
                                                 if m.startswith("decoder:") and p == precs["decoder"][0]]
               if "decoder" in modes else [])
    st = engines.get(("ctc-stream", "fp32"))
    sd = engines.get(("dec-stream", "fp32"))
    sl = st or sd                                  # look-aheads: the streaming engines share one export's chunks
    ids = next(iter(first.values())).spec.vocab.ids
    tbm = {m: ["none", "segment", "token"] if m == "decoder" else ["none"] if m == "dec-stream" else ["none", "token"]
           for m in modes}
    if "decoder" in modes and not all(e.token_timestamps for (m, _), e in engines.items() if m == "decoder"):
        tbm["decoder"] = ["none", "segment"]          # no ctc_lp graph in this export: token times not offered
    e0 = next(iter(first.values()))
    return {"modes": list(modes), "default_mode": modes[0], "precisions": {m: precs[m] for m in modes},
            "default_precision": precs[modes[0]][0], "decode": decodes[0] if decodes else first[modes[0]].decode_kind,
            "decodes": decodes, "default_decode": decodes[0] if decodes else None,
            "lookaheads": list(sl.lookaheads) if sl is not None else [],
            "default_lookahead": sl.lookahead if sl is not None else None,
            # per streaming mode (dec-stream may serve fewer look-aheads than ctc-stream: spec stream_dec.lookaheads)
            "lookaheads_by_mode": {m: list(e.lookaheads) for m, e in (("ctc-stream", st), ("dec-stream", sd))
                                   if e is not None and m in modes},
            "default_lookahead_by_mode": {m: e.lookahead for m, e in (("ctc-stream", st), ("dec-stream", sd))
                                          if e is not None and m in modes},
            "langs": served_langs(a, [lg for lg in V1_LANGS if f"<|{lg}|>" in ids], e0.lang),
            "defaults": {"lang": e0.lang, "itn": bool(e0.itn), "pnc": bool(e0.pnc),
                         "timestamps": {m: first[m].timestamps for m in modes}},
            "timestamps_by_mode": tbm, "live": sl is not None,
            "live_modes": [m for m in modes if m in STREAM_MODES],
            "max_stream_seconds": round(MAX_FRAMES / st.spec.frames_per_second) if st else None,
            "max_body_mb": a.max_body_mb, "max_audio_seconds": a.max_audio_s, "max_streams": a.max_streams,
            "live_idle_seconds": a.ws_idle_s, "live_first_audio_seconds": a.ws_first_audio_s,
            "live_session_seconds": a.ws_session_max_s,
            "word_boost_modes": [m for m in modes if all(getattr(e, "_boost_why", "x") is None
                                                          for (mm, _), e in engines.items()
                                                          if mm.split(":")[0] == m)]}


def build_engines(cfg: dict, modes: list, precs: list, stream_model: str | None, stream_threads: int,
                  stream_precision: str = "fp32", share_encoder: bool = True, decodes: list | None = None) -> dict:
    """{(mode, precision): Engine}. The server's --lang / --itn / --pnc / --timestamps / --lookahead are the engines'
    defaults (segment timestamps only for decoder; lookahead only for ctc-stream). ctc-stream: on --stream-model (or
    --model if it is a streaming package), with its own small intra-op thread count, and fp32 only (the only chunk
    graphs there are): any other precision asked for it is refused here, never silently replaced.
    share_encoder (default): decoder and ctc at one precision use one encoder session (the full graph; one copy of
    the encoder weights), and ctc-stream / dec-stream share their chunk-encoder sessions; False: every engine loads
    its own (ctc then uses the CTC-only encoder graph).
    decodes: the decoder's search kinds served per request (?decode=), the first being cfg's default; each further
    kind is one more decoder engine, keyed ("decoder:<kind>", precision), sharing the default one's encoder."""
    explicit = set(cfg.get("_explicit", []))
    for sm in [m for m in modes if m in STREAM_MODES]:
        if stream_precision != "fp32":
            raise ValueError(f"--stream-precision {stream_precision}: {sm} has fp32 chunk graphs only")
        if not stream_model and explicit & {"precision", "precisions"} and any(p != "fp32" for p in precs):
            raise ValueError(f"{sm} on --model would run fp32, not {','.join(precs)}: serve it with "
                             "--stream-model, or use --precision fp32")
    if "decode" in explicit and cfg.get("decode") != "greedy" and "decoder" not in modes:
        raise ValueError(f"--decode {cfg.get('decode')} applies to the decoder; these modes ({','.join(modes)}) "
                         "decode CTC greedily")
    ts = cfg.get("timestamps") or "none"
    base = {k: v for k, v in cfg.items() if k not in ("timestamps", "lookahead", "endpoint_frames",
                                                      "max_segment_frames")}
    engines = {}
    for m in modes:
        mts = ts if (m == "decoder" or ts != "segment") else "none"
        if m == "dec-stream":                     # token_times always (emit times); no timestamps option
            mts = "none"
        if m in STREAM_MODES:
            kw = {**base, "mode": m, "precision": "fp32", "decode": "greedy", "model": stream_model or cfg["model"],
                  "timestamps": mts, "lookahead": cfg.get("lookahead"), "threads": stream_threads}
            if m == "dec-stream":
                kw.update({k: cfg[k] for k in ("endpoint_frames", "max_segment_frames") if cfg.get(k) is not None})
            # ctc-stream and dec-stream on one package share its front end + chunk graph sessions (one copy of the
            # streaming encoder in memory instead of two)
            share = next((e for (mm, _), e in engines.items() if mm in STREAM_MODES), None) if share_encoder else None
            engines[(m, "fp32")] = make_engine(kw, stream_share=share)
    # offline modes: the decoder engine first, so a ctc engine at the same precision can take its encoder session
    offline = [m for m in ("decoder", "ctc") if m in modes]
    both = share_encoder and len(offline) == 2
    for m in offline:
        for p in precs:
            mts = ts if (m == "decoder" or ts != "segment") else "none"
            kw = {**base, "mode": m, "precision": p, "timestamps": mts}
            extra = {}
            if m == "ctc":                        # CTC decodes greedily; --decode beam is the decoder's
                kw["decode"] = "greedy"
                if both:
                    extra["enc_share"] = engines[("decoder", p)]
            engines[(m, p)] = make_engine(kw, **extra)
            if m == "decoder":
                for d in (decodes or [])[1:]:     # more search kinds: one engine each, sharing the encoder
                    engines[(f"decoder:{d}", p)] = make_engine(
                        {**kw, "decode": d}, **({"enc_share": engines[(m, p)]} if share_encoder else {}))
    # the dict's order is the request-default order of the caller's modes
    return {k: engines[k] for m in modes for k in list(engines) if k[0].split(":")[0] == m}


def _decode_timer(seconds, fn):
    """The decode deadline's timer (a seam for tests: they replace it to fire at a chosen point)."""
    return threading.Timer(seconds, fn)


class Busy(Exception):
    """A transient server-side condition: the in-flight audio budget is used up, or a decoder process could not be
    started. 503, retry later."""


class SpoolFull(Exception):
    """The server-wide upload spool budget (--spool-max-mb) is used up: 503, retry later."""


class ByteBudget:
    """Bytes of request bodies spooled to disk at once, server-wide."""

    def __init__(self, nbytes: int):
        self.cap, self.used, self.lock = int(nbytes), 0, threading.Lock()

    def take(self, n: int) -> bool:
        with self.lock:
            if self.used + n > self.cap:
                return False
            self.used += n
            return True

    def give(self, n: int) -> None:
        with self.lock:
            self.used -= n


class AudioBudget:
    """Samples of decoded audio the server holds at once (decode buffers + queued / running requests), server-wide."""

    def __init__(self, seconds: float, sr: int = 16000):
        self.cap, self.used, self.lock = int(seconds * sr), 0, threading.Lock()

    def take(self, n: int) -> bool:
        with self.lock:
            if self.used + n > self.cap:
                return False
            self.used += n
            return True

    def give(self, n: int) -> None:
        with self.lock:
            self.used -= n


def decode_upload(path: str, mode: str, max_seconds: float, budget: AudioBudget | None = None,
                  timeout: float | None = None):
    """Per request, in its own handler thread, before it joins a micro-batch: the whole upload decoded once into one
    preallocated float32 16 kHz array (sized from the header / ffprobe duration, grown x1.5 if the estimate is
    short, never past max_seconds, trimmed in place at the end), bounded by max_seconds and, for ctc-stream, the
    stream limit. The buffer is reserved from `budget` before it is allocated: TooLong (413) when the input is past
    the limit or would need more than the whole budget, Busy (503) only when it would fit an idle budget but not the
    current one. Growth holds old + new (both are real memory during the copy): at most 2 x max_seconds, which the
    server guarantees fits (inflight >= 2 x max-audio). Returns (array, reserved samples): the caller gives the
    reservation back once the response is sent. Any decode failure is this request's own InputError; the caller's
    InputError is never replaced by a decoder error, and every failure gives its reservation back.
    ctc-stream requests are capped at the stream limit (MAX_FRAMES), not max_seconds: a reservation never exceeds what
    such a request may decode. `timeout`: a wall-clock limit for the decode; past it the decoder is killed (through a
    DecodeHandle) and the request's own InputError says so. A decoder that cannot even be started (SpawnError) is
    Busy (503), logged in one line."""
    import contextlib
    import numpy as np
    from speech_server import audio as audiolib
    from speech_server.runtime import InputError, TooLong
    from speech_server.stream import MAX_FRAMES, stream_frames
    import threading
    sr = audiolib.SR
    limit = int(max_seconds * sr)
    if mode == "ctc-stream":                           # the most samples whose encoder frames fit the stream limit
        limit = min(limit, MAX_FRAMES * 8 * audiolib.SR // 100 - 1)
    reserved = 0

    def reserve(n):
        nonlocal reserved
        if budget is not None:
            if reserved + n > budget.cap:              # more than the whole budget: no retry can help
                raise TooLong("too long for this server (more decoded audio than it holds at once)")
            if not budget.take(n):
                raise Busy()
            reserved += n

    def release(n):
        nonlocal reserved
        if budget is not None:
            budget.give(n)
            reserved -= n
    # the deadline and the decode race for one outcome on the handle (running -> timed_out | finished): the timer
    # kills only if it wins; the decode returns only if it wins; a fired timer always gives the timeout 400
    handle, timer = audiolib.DecodeHandle(), None
    timed_out = InputError(f"decode timed out (more than {timeout:g} s)") if timeout is not None else None
    if timeout is not None:
        timer = _decode_timer(timeout, handle.time_out)
        timer.daemon = True
        timer.start()
    try:
        est = audiolib.probe_seconds(path, handle)     # on the handle: the deadline kills this probe too
        if handle.state == "timed_out":
            raise timed_out
        cap = min(limit, int(est * sr * 1.01) + sr) if est else min(limit, 60 * sr)
        cap = max(cap, min(sr, limit))                 # a 1 s floor, never above the limit
        reserve(cap)
        buf = np.empty(cap, np.float32)
        n = 0
        try:
            with contextlib.closing(audiolib.stream(path, block_seconds=30.0, handle=handle)) as gen:
                for blk in gen:
                    m = len(blk)
                    if mode == "ctc-stream" and stream_frames(n + m) > MAX_FRAMES:
                        raise TooLong(f"ctc-stream takes at most {MAX_FRAMES} encoder frames "
                                      f"({MAX_FRAMES * 0.08:.0f} s)")
                    if n + m > limit:
                        raise TooLong(f"audio longer than this server's limit ({max_seconds:g} s)")
                    if n + m > len(buf):                       # the estimate was short: grow x1.5, capped at the
                        new = min(limit, max(int(len(buf) * 1.5), n + m))   # limit (n + m <= limit here)
                        reserve(new)
                        nb = np.empty(new, np.float32)
                        nb[:n] = buf[:n]
                        old = len(buf)
                        buf = nb
                        release(old)
                    buf[n:n + m] = blk
                    n += m
        except (InputError, Busy):
            raise
        except audiolib.SpawnError:
            raise
        except Exception as e:  # noqa: BLE001 (any decoder failure, early or late in the file)
            if isinstance(e, ValueError) and str(e).startswith("unsupported audio:"):
                raise InputError(str(e)) from None    # our own header check: its message is safe and useful
            raise InputError("the body is not audio this server can decode (WAV, FLAC, OGG, MP3, M4A, WebM)") \
                from None
        if not handle.finish():                        # the deadline won (a cancelled soundfile decode stops quietly)
            raise timed_out
        if n == 0:
            raise InputError("the body decodes to no audio")
        if n < len(buf):                                       # trim in place (realloc), give back the slack
            slack = len(buf) - n
            buf.resize(n, refcheck=False)
            release(slack)
        return buf, reserved
    except audiolib.SpawnError as e:
        if budget is not None and reserved:
            budget.give(reserved)
        if not handle.finish() and handle.state == "timed_out":
            raise timed_out from None
        print(f"serve: {e}; request answered 503", file=sys.stderr, flush=True)
        raise Busy("could not start the decoder") from None
    except BaseException as e:
        if budget is not None and reserved:
            budget.give(reserved)
        # a failure decides the outcome too, unless the deadline already won: then it is the timeout 400
        if e is not timed_out and not handle.finish() and handle.state == "timed_out":
            raise timed_out from None
        raise
    finally:
        if timer is not None:
            timer.cancel()


def cmd_serve(a) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import numpy as np
    from speech_server import usage, ws
    from speech_server.runtime import InputError, TooLong
    key = read_key(a.api_key_env, a.api_key_file)
    if key is None and not a.no_auth:
        sys.exit(f"no API key: set ${a.api_key_env} or --api-key-file (or --no-auth on 127.0.0.1 only)")
    if key is not None and len(key) < 16:
        sys.exit("the API key must be at least 16 characters (e.g. openssl rand -hex 24)")
    if a.no_auth and a.host not in ("127.0.0.1", "localhost", "::1"):
        sys.exit("--no-auth is only allowed on a loopback address")
    if a.spool_max_mb < a.max_body_mb:
        sys.exit(f"--spool-max-mb {a.spool_max_mb:g} must be at least --max-body-mb ({a.max_body_mb:g})")
    if a.inflight_audio_s < 2 * a.max_audio_s:
        sys.exit(f"--inflight-audio-s {a.inflight_audio_s:g} must be at least 2 x --max-audio-s ({2 * a.max_audio_s:g}):"
                 " a maximum-length upload whose duration estimate is short holds old + new buffer while it grows")
    cfg = resolve(a)
    from speech_server import models
    try:
        a.stream_model = models.resolve(a.stream_model, a.stream_model_repo, a.stream_model_revision,
                                        cfg.get("_model_cache"), cfg.get("_token_env") or "HF_TOKEN")
    except models.ModelError as e:
        sys.exit(f"stream model: {e}")
    # downloads are done: the Hugging Face token is not needed any more, so it does not stay in this process's
    # environment (decoder subprocesses never get it either: audio._child_env)
    os.environ.pop(cfg.get("_token_env") or "HF_TOKEN", None)
    if a.share_encoder is None:
        a.share_encoder = True
    modes = list(dict.fromkeys(m.strip() for m in (a.modes if a.modes is not None else cfg["mode"]).split(",")
                               if m.strip()))
    if not modes or any(m not in ("decoder", "ctc") + STREAM_MODES for m in modes):
        sys.exit("--modes takes decoder, ctc, ctc-stream and/or dec-stream")
    if a.stream_model and "ctc-stream" not in modes:
        modes.append("ctc-stream")
    # command line / env first (--decodes, or --decode alone), then --config's decodes, then the config's decode
    raw_dec = a.decodes if a.decodes is not None else (None if getattr(a, "decode", None) is not None
                                                        else cfg.get("decodes"))
    if isinstance(raw_dec, (list, tuple)):
        raw_dec = ",".join(str(x) for x in raw_dec)
    if raw_dec is not None and not isinstance(raw_dec, str):
        sys.exit("decodes must be a comma list (or a JSON list) of greedy / beam")
    decodes = list(dict.fromkeys(d.strip() for d in (cfg["decode"] if raw_dec is None else raw_dec).split(",")
                                 if d.strip()))
    if not decodes or any(d not in ("greedy", "beam") for d in decodes):
        sys.exit("--decodes takes greedy and/or beam")
    if raw_dec is not None and "decoder" not in modes and decodes != ["greedy"]:
        sys.exit("--decodes lists decoder search kinds; serve mode decoder too (--modes)")
    cfg["decode"] = decodes[0]                     # the default; ?decode= picks among them per request
    if len(decodes) > 1:
        cfg["_explicit"] = sorted(set(cfg.get("_explicit", [])) | {"decode"})
    precs = list(dict.fromkeys(p.strip() for p in (a.precisions if a.precisions is not None else cfg["precision"])
                               .split(",") if p.strip()))
    if not precs or any(p not in ("fp32", "fp16", "int8") for p in precs):
        sys.exit("--precisions takes fp32, fp16 and/or int8")
    if cfg["device"] == "cpu" and "fp16" in precs:   # asked for explicitly: served, but say what it costs
        print("serve: fp16 on CPU runs, but slower than fp32 (fp16 is meant for GPUs)", file=sys.stderr, flush=True)
    # one port; mode and precision chosen per request (?mode=, ?precision=); the first of each is the default. The
    # server's --lang / --itn / --pnc / --timestamps / --lookahead are the request defaults (segment timestamps only
    # where the mode has them; lookahead only for ctc-stream).
    try:
        engines = build_engines(cfg, modes, precs, a.stream_model, a.stream_threads, a.stream_precision,
                                share_encoder=a.share_encoder, decodes=decodes)
    except ValueError as e:
        sys.exit(f"serve: {e}")
    try:
        facts = serve_facts(engines, modes, cfg, a)
    except ValueError as e:
        sys.exit(f"serve: {e}")
    live_engs = {m: engines[(m, "fp32")] for m in modes if m in STREAM_MODES}    # live /stream: first is default
    stream_eng = next(iter(live_engs.values()), None)
    queues = {k: queue.Queue() for k in engines}
    run_lock = threading.Lock()                    # one engine runs at a time: the thread budget is per server
    # --stats counters (GET /stats): micro-batches per engine and 503 answers; numbers only, never request content
    counters = {"t0": time.time(), "batches": {}, "http_503": 0, "conn_limit_503": 0}
    counters_lock = threading.Lock()

    def stats_view() -> dict:
        """GET /stats: what each engine did and the limits it runs with (after the memory clamp); numbers only."""
        with counters_lock:
            mb = {k: tuple(v) for k, v in counters["batches"].items()}
            out = {"uptime_s": round(time.time() - counters["t0"], 1), "http_503": counters["http_503"],
                   "conn_limit_503": counters["conn_limit_503"]}
        rows = []
        for (m, p), e in engines.items():
            st, (n, r) = getattr(e, "stats", None), mb.get((m, p), (0, 0))
            rows.append({"mode": m.split(":")[0], "precision": p, "decode": getattr(e, "decode_kind", None),
                         "max_batch_seconds": getattr(e, "max_batch_seconds", None),
                         "kv_budget_mb": getattr(e, "kv_budget_mb", None),
                         "decode_batch": getattr(e, "decode_batch", None),
                         "micro_batches": n, "requests": r, "mean_micro_batch": round(r / n, 2) if n else None,
                         "engine": {k: round(float(getattr(st, k)), 3) for k in
                                    ("audio_seconds", "windows", "encode_s", "decode_s", "steps", "tokens", "wall_s",
                                     "batches")} if st is not None else None})
        out["engines"] = rows
        return out
    streams = threading.BoundedSemaphore(a.max_streams)      # live sessions, server-wide (no per-IP limits: behind
    preauth = {"n": 0, "lock": threading.Lock()}               # the proxy every client IP is the same)
    decode_sem = threading.BoundedSemaphore(a.decode_max)      # concurrent upload decodes (ffmpeg / soundfile)
    budget = AudioBudget(a.inflight_audio_s)                   # decoded audio held at once, server-wide
    spool = ByteBudget(int(a.spool_max_mb * 2 ** 20))         # request bodies on disk at once, server-wide

    def preauth_take() -> bool:
        with preauth["lock"]:
            if preauth["n"] >= a.ws_preauth_max:
                return False
            preauth["n"] += 1
            return True

    def preauth_give():
        with preauth["lock"]:
            preauth["n"] -= 1
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.html"), "rb") as f:
        demo_html = f.read()

    def fail(job, e):
        job[2].clear()
        job[2]["error"] = type(e).__name__
        if isinstance(e, InputError):              # the engine's own message about this input; nothing else leaks
            job[2]["detail"] = str(e)[:300]

    def worker(m):                                 # micro-batching: requests within batch_wait_ms share pools
        eng, jobs = engines[m], queues[m]

        def run(batch):                            # decoded arrays (the handler decoded each upload once)
            with counters_lock:
                c = counters["batches"].setdefault(m, [0, 0])
                c[0] += 1
                c[1] += len(batch)
            with run_lock:
                for j, r in zip(batch, eng.transcribe([j[0] for j in batch], options=[j[1] for j in batch])):
                    j[2].update({k: v for k, v in r.items() if k != "tokens"})
        while True:
            first = jobs.get()
            batch = [first]
            t_end = time.time() + a.batch_wait_ms / 1000
            while len(batch) < cfg["batch_size"]:
                try:
                    batch.append(jobs.get(timeout=max(0.0, t_end - time.time())))
                except queue.Empty:
                    break
            try:
                run(batch)
            except Exception as e:  # noqa: BLE001
                if len(batch) == 1:
                    fail(batch[0], e)
                else:                              # isolate: the ones not done yet run alone, so one bad input
                    for j in batch:                # fails only itself
                        if "text" in j[2]:
                            continue
                        try:
                            run([j])
                        except Exception as e1:  # noqa: BLE001
                            fail(j, e1)
            for j in batch:
                j[0] = None                        # drop the array: its budget is given back with the response
                j[3].set()
            batch = first = None

    for k in engines:
        threading.Thread(target=worker, args=(k,), daemon=True).start()

    class H(BaseHTTPRequestHandler):
        rbufsize = 0                                # unbuffered reads: the WebSocket reads the socket itself, so no
                                                    # request bytes may sit in a reader buffer after the headers
        timeout = a.http_timeout_s                  # per socket operation (request line, headers, each body read):
                                                    # an idle TCP connection cannot pin a handler thread

        def setup(self):
            super().setup()
            self.rfile = _HeadReader(self.connection, a.http_timeout_s)   # + a total deadline for the head

        def handle_one_request(self):
            self.rfile.arm(time.monotonic() + a.http_head_s)
            super().handle_one_request()

        def parse_request(self):
            ok = super().parse_request()            # request line + headers read: the body has per-read timeouts
            self.rfile.disarm()
            return ok
        def log_message(self, fmt, *args):          # no request bodies, keys or texts in the log
            pass

        def _auth(self) -> bool:
            if key is None:
                return True
            h = self.headers.get("Authorization", "")
            got = h[7:] if h.startswith("Bearer ") else self.headers.get("X-API-Key", "")
            return hmac.compare_digest(got.encode(), key.encode())

        def _send(self, code, obj, ctype="application/json; charset=utf-8", extra=None):
            if code == 503:
                with counters_lock:
                    counters["http_503"] += 1
            b = obj if isinstance(obj, bytes) else (obj.encode() if isinstance(obj, str) else
                                                    json.dumps(obj, ensure_ascii=False).encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/health":
                # legacy keys from the engines that run: the default mode's decode / precision and the precisions
                # ?precision= takes for it; `engines` lists every (mode, precision, decode) actually loaded
                return self._send(200, {"ok": True, "modes": modes, "default_mode": modes[0],
                                        "decode": facts["decode"], "precision": facts["default_precision"],
                                        "precisions": facts["precisions"][modes[0]], "device": cfg["device"],
                                        "model": a.model_name or "model",
                                        "share_encoder": bool(a.share_encoder),
                                        "live": facts["live"],
                                        "decodes": facts["decodes"],
                                        "engines": [{"mode": m.split(":")[0], "precision": p, "decode": e.decode_kind}
                                                    for (m, p), e in engines.items()]})
            if u.path == "/stats":                   # opt-in (--stats), behind the API key: serving counters only
                if not a.stats:
                    return self._send(404, {"error": "not found"})
                if not self._auth():
                    return self._send(401, {"error": "unauthorized"})
                return self._send(200, stats_view())
            if u.path == "/help":                    # no auth, as /health: documentation and served facts only
                qs = urllib.parse.parse_qs(u.query)
                fmt = qs.get("format", ["markdown"])[0]
                if fmt == "json":
                    return self._send(200, usage.help_json(facts))
                if fmt in ("markdown", "md"):
                    return self._send(200, usage.help_markdown(facts), "text/markdown; charset=utf-8")
                return self._send(400, {"error": "format must be markdown or json"})
            if u.path == "/demo":
                return self._send(200, demo_html, "text/html; charset=utf-8", {
                    "Cache-Control": "no-store", "X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff",
                    "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src "
                    "'unsafe-inline'; connect-src 'self'; media-src blob:; img-src data:; base-uri 'none'; "
                    "form-action 'none'; frame-ancestors 'none'"})
            if u.path == "/v1/check":                # the demo page's key gate
                if not self._auth():
                    return self._send(401, {"error": "unauthorized"})
                return self._send(200, {"ok": True})
            if u.path == "/stream":
                return self._stream()
            self._send(404, {"error": "not found"})

        # ---------------------------------------------------------------- live streaming (WebSocket)
        def _stream(self):
            if stream_eng is None:
                return self._send(404, {"error": "live streaming unavailable: no streaming model loaded"})
            wkey = self.headers.get("Sec-WebSocket-Key")
            upgrade = self.headers.get("Upgrade", "").lower() == "websocket" and wkey
            if not preauth_take():                   # connections not yet authenticated, server-wide; a plain GET
                return self._send(503, {"error": "busy: too many connections waiting to authenticate, try again"})
            if not upgrade:                          # (the demo's probe after a failed handshake) sees 503 vs 400
                preauth_give()
                return self._send(400, {"error": "expected a WebSocket upgrade"})
            pre, acquired, stop, conn = True, False, threading.Event(), None
            self.close_connection = True
            sock = self.connection
            try:
                sock.settimeout(a.ws_send_timeout_s)
                sock.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                              f"Sec-WebSocket-Accept: {ws.accept_key(wkey)}\r\n\r\n").encode())
                conn = ws.ServerConn(sock, int(a.ws_max_frame_kb * 1024), a.ws_send_timeout_s)
                header_ok = self._auth() if (self.headers.get("Authorization") or self.headers.get("X-API-Key")) \
                    else False
                # ---- before auth: one small unfragmented text message within the auth deadline, nothing else
                try:
                    _, data = conn.recv(deadline=time.monotonic() + a.ws_auth_s, max_payload=START_JSON_MAX,
                                        single_text=True)
                except ws.WSTimeout:
                    conn.send_json({"type": "error", "error": "no start message in time"})
                    return conn.close(1008, "timeout")
                except (ws.WSProtocol, ws.WSTooBig):
                    conn.send_json({"type": "error", "error": "the first message must be one small text frame "
                                    "{\"type\": \"start\", ...}"})
                    return conn.close(1008, "no start")
                msg = _json_small(data, START_JSON_MAX)
                if not isinstance(msg, dict) or msg.get("type") != "start":
                    conn.send_json({"type": "error", "error": "the first message must be {\"type\": \"start\", ...}"})
                    return conn.close(1008, "no start")
                got = msg.get("key")
                got = got if isinstance(got, str) else ""
                if not (key is None or header_ok or hmac.compare_digest(got.encode(), key.encode())):
                    conn.send_json({"type": "error", "error": "unauthorized"})
                    return conn.close(1008, "unauthorized")
                try:                                 # mode (ctc-stream | dec-stream), lookahead, decoder prompt
                    lmode, la, lopt = usage.parse_start(msg, facts)
                    seng = live_engs[lmode]
                    la = seng.lookahead if la is None else seng._check_la(la)
                except (InputError, ValueError) as e:
                    conn.send_json({"type": "error", "error": str(e)})
                    return conn.close(1008, "bad options")
                if not streams.acquire(blocking=False):
                    conn.send_json({"type": "error", "error": "busy", "max_streams": a.max_streams})
                    return conn.close(1013, "busy")
                acquired = True
                try:                                 # with a stream slot held: a word_boost list compiles only for
                    lopt = seng.options(0, None, lopt)    # a session that may run (validates every option)
                except (InputError, ValueError) as e:
                    conn.send_json({"type": "error", "error": str(e)})
                    return conn.close(1008, "bad options")
                preauth_give()
                pre = False
                # ---- authenticated session

                def pinger():                        # proxies close idle connections (nginx: 60 s by default)
                    while not stop.wait(a.ws_ping_s):
                        try:
                            conn.send(ws.OP_PING, b"")
                        except OSError:
                            return
                threading.Thread(target=pinger, daemon=True).start()
                dec = lmode == "dec-stream"
                cs = seng.dec_session(la, live=True, opt=lopt) if dec else seng.ctc_stream(la, live=True, opt=lopt)
                conn.send_json({"type": "ready", "mode": lmode, "lookahead": la, "sample_rate": 16000,
                                "format": "pcm_s16le", "max_seconds": None if dec else facts["max_stream_seconds"],
                                "idle_seconds": a.ws_idle_s, "max_frame_bytes": int(a.ws_max_frame_kb * 1024)})
                t0 = time.monotonic()
                n = 0
                # idle counts against audio actually received: the deadline moves only once 0.1 s of samples has
                # arrived since it last moved (empty or tiny frames do not keep a session); before the first audio a
                # longer grace (the browser's microphone prompt); a hard wall-clock cap per session
                last, fresh, got_audio = t0, 0, False
                end_at = t0 + a.ws_session_max_s

                def times(toks):
                    return seng.dec_token_times(toks) if dec else seng.stream_token_times(toks)

                def final(reason):
                    ids = [t[0] for t in cs.tokens]
                    msg = {"type": "final", "reason": reason, "text": seng.detok.decode(ids), "tokens": ids,
                           "token_times": times(cs.tokens), "audio_seconds": round(n / 16000, 3)}
                    if dec:                           # the decoder's segments and guard counts
                        r = seng.dec_result(cs)
                        msg.update(mode=lmode, segments=r["segments"], guards=r["guards"])
                    conn.send_json(msg)
                    conn.close(1000, reason)

                def engine_error(e):
                    print(f"stream: {type(e).__name__} in the streaming engine; session closed", file=sys.stderr,
                          flush=True)
                    conn.send_json({"type": "error", "error": "internal error in the streaming engine"})
                    conn.close(1011, "internal error")

                from speech_server.stream import StreamLimit
                while True:
                    idle_at = last + (a.ws_idle_s if got_audio else a.ws_first_audio_s)
                    try:
                        op, data = conn.recv(deadline=min(idle_at, end_at))
                    except ws.WSTimeout:              # no audio for the idle time, or the session's wall-clock cap
                        try:
                            cs.finish()
                        except StreamLimit:
                            pass
                        except Exception as e:  # noqa: BLE001
                            return engine_error(e)
                        return final("session_limit" if time.monotonic() >= end_at else "idle")
                    if op == ws.OP_BIN:
                        if len(data) % 2:
                            conn.send_json({"type": "error", "error": "audio must be 16-bit PCM (an even byte count)"})
                            return conn.close(1003, "bad audio")
                        n += len(data) // 2
                        now = time.monotonic()
                        if n / 16000 > a.ws_pace * (now - t0) + a.ws_pace_slack_s:
                            conn.send_json({"type": "error", "error": f"audio arrives faster than {a.ws_pace:g}x real "
                                            "time: this endpoint is for live audio (POST files to /v1/transcribe)"})
                            return conn.close(1008, "too fast")
                        fresh += len(data) // 2
                        if fresh >= 1600:                 # 0.1 s of audio since the deadline last moved
                            last, fresh, got_audio = now, 0, True
                        x = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
                        try:
                            new = cs.feed_audio(x)
                        except StreamLimit as e:
                            conn.send_json({"type": "error", "error": str(e)})
                            return final("limit")
                        except Exception as e:  # noqa: BLE001
                            return engine_error(e)
                        if new:
                            ids = [t[0] for t in cs.tokens]
                            conn.send_json({"type": "partial", "tokens": times(new),
                                            "text": seng.detok.decode(ids), "audio_seconds": round(n / 16000, 3)})
                        continue
                    msg = _json_small(data)
                    if isinstance(msg, dict) and msg.get("type") == "stop":
                        try:
                            cs.finish()
                        except StreamLimit as e:
                            conn.send_json({"type": "error", "error": str(e)})
                            return final("limit")
                        except Exception as e:  # noqa: BLE001
                            return engine_error(e)
                        return final("stop")
                    conn.send_json({"type": "error", "error": "expected binary audio or {\"type\": \"stop\"}"})
            except ws.WSTooBig:
                conn.close(1009, "message too big")
            except ws.WSProtocol:
                conn.close(1002, "protocol error")
            except (ws.WSClosed, OSError):          # client gone, reset, or a send timeout
                pass
            finally:
                stop.set()
                if conn is not None:
                    conn.release()
                if pre:
                    preauth_give()
                if acquired:
                    streams.release()

        def _body_to_file(self, f, held: list) -> int:
            """Request body -> temp file, Content-Length or chunked transfer, never above --max-body-mb, within the
            server-wide spool budget (SpoolFull) and a total deadline (--http-body-s, TimeoutError). `held` collects
            the spool bytes reserved; the caller gives them back."""
            deadline = time.monotonic() + a.http_body_s
            self.rfile.arm(deadline)                 # every read of the body (chunk framing too) has the deadline
            try:
                return self._body_reads(f, held, deadline)
            finally:
                self.rfile.disarm()

        def _body_reads(self, f, held: list, deadline: float) -> int:
            limit, n = int(a.max_body_mb * 2 ** 20), 0

            def put(b):
                if time.monotonic() > deadline:
                    raise TimeoutError("request body deadline")
                if not spool.take(len(b)):
                    raise SpoolFull()
                held[0] += len(b)
                try:
                    f.write(b)
                except OSError:                       # the spool directory is full (ENOSPC) or failing: 503
                    raise SpoolFull() from None
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                while True:
                    raw = self.rfile.readline(1024)
                    if not raw.endswith(b"\n"):      # EOF (or an over-long line) before a chunk-size line: the
                        raise EOFError                # body is truncated, never an implicit last chunk
                    line = raw.split(b";")[0].strip()
                    try:
                        size = int(line, 16)
                    except ValueError:
                        raise EOFError from None
                    if size < 0:                      # "-1" would reach read(-1): read to EOF
                        raise EOFError
                    if size == 0:
                        self.rfile.readline(1024)
                        return n
                    n += size
                    if n > limit:
                        raise OverflowError
                    left = size
                    while left:
                        b = self.rfile.read(min(left, 1 << 20))
                        if not b:
                            raise EOFError
                        put(b)
                        left -= len(b)
                    self.rfile.readline(1024)
            try:
                total = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise EOFError from None
            if total < 0:
                raise EOFError
            if total > limit:
                raise OverflowError
            while n < total:
                b = self.rfile.read(min(total - n, 1 << 20))
                if not b:
                    raise EOFError
                put(b)
                n += len(b)
            return n

        def _drain(self, max_bytes: int = 16 * 2 ** 20, seconds: float = 2.0) -> None:
            """After an early answer to a request whose body was not read: read and drop a bounded amount of it (at
            most max_bytes, for at most `seconds`), so closing the socket does not reset the connection before the
            client has read the answer."""
            try:
                left = min(int(self.headers.get("Content-Length", "0") or 0), max_bytes)
            except ValueError:
                return
            end = time.monotonic() + seconds
            sock = self.connection
            try:
                while left > 0 and time.monotonic() < end:
                    sock.settimeout(max(0.05, end - time.monotonic()))
                    b = sock.recv(min(left, 1 << 16))
                    if not b:
                        break
                    left -= len(b)
            except OSError:
                pass

        def _finish_post(self, ekey, slot, t0):
            if "error" in slot:
                err = {"error": slot["error"], **({"detail": slot["detail"]} if "detail" in slot else {})}
                return self._send(400 if slot["error"] in ("InputError", "ValueError", "LibsndfileError") else 500,
                                  err)
            mode, prec = ekey[0].split(":")[0], ekey[1]
            dec = {"decode": engines[ekey].decode_kind} if mode == "decoder" else {}
            self._send(200, {"mode": mode, "precision": prec, **dec, "text": slot["text"], "windows": slot["windows"],
                             "segments": slot["segments"], "audio_seconds": slot["duration"],
                             "seconds": round(time.time() - t0, 3), **{k: slot[k] for k in EXTRA_KEYS if k in slot}})

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            if u.path != "/v1/transcribe":
                return self._send(404, {"error": "not found"})
            if not self._auth():
                self.close_connection = True
                self._send(401, {"error": "unauthorized"})
                return self._drain()
            # the query is validated before any of the body is read: a bad request costs no upload
            try:                                      # one definition (usage.py) for validation and the docs
                mode, prec, opts = usage.parse_query(urllib.parse.parse_qs(u.query, keep_blank_values=True), facts)
                dec = opts.pop("decode", None)    # the decoder's search kind picks the engine, not an engine option
                ekey = (f"decoder:{dec}" if mode == "decoder" and dec and dec != facts["default_decode"] else mode,
                        prec)
                engines[ekey].options(0, None, opts)
            except (ValueError, InputError) as e:
                self.close_connection = True
                self._send(400, {"error": str(e)} if not isinstance(e, InputError) else
                           {"error": "InputError", "detail": str(e)[:300]})
                return self._drain()
            import tempfile
            tmp = tempfile.NamedTemporaryFile(dir=a.tmp_dir, prefix="speech-upload-", delete=False)
            spooled = [0]
            try:
                try:
                    n = self._body_to_file(tmp, spooled)
                finally:
                    tmp.close()
                if n <= 0:
                    return self._send(400, {"error": "empty body"})
                t0 = time.time()
                try:                                  # the whole upload decoded once, bounded, alone
                    with decode_sem:
                        x, held = decode_upload(tmp.name, mode, a.max_audio_s, budget, a.decode_timeout_s)
                except TooLong as e:
                    return self._send(413, {"error": "too long", "detail": str(e)[:300]})
                except InputError as e:
                    return self._send(400, {"error": "InputError", "detail": str(e)[:300]})
                except Busy as e:
                    return self._send(503, {"error": "busy", "detail": (f"{e.args[0]}; retry shortly" if e.args else
                                            "the server is transcribing as much audio as it holds at once; retry "
                                            "shortly")}, extra={"Retry-After": "5"})
                finally:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass
                    spool.give(spooled[0])            # the spool file is gone: its bytes are free again
                    spooled[0] = 0
                try:
                    slot, ev = {"duration": round(len(x) / 16000, 3)}, threading.Event()
                    queues[ekey].put([x, opts, slot, ev])
                    del x
                    ev.wait()
                    self._finish_post(ekey, slot, t0)
                finally:
                    budget.give(held)                 # after the response is sent (or failed to send)
            except OverflowError:
                self.close_connection = True
                self._send(413, {"error": "too large"})
            except EOFError:
                self.close_connection = True
                self._send(400, {"error": "truncated or malformed body"})
            except SpoolFull:
                self.close_connection = True
                self._send(503, {"error": "busy", "detail": "the server is receiving as many uploads as it holds at "
                                 "once; retry shortly"}, extra={"Retry-After": "5"})
            except TimeoutError:
                self.close_connection = True
                try:
                    self._send(408, {"error": "request body took too long"})
                except OSError:
                    pass
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
                spool.give(spooled[0])

    class Server(ThreadingHTTPServer):
        request_queue_size = 128                   # listen backlog (socketserver's default is 5)
        daemon_threads = True
        active, active_lock = 0, threading.Lock()

        def handle_error(self, request, client_address):
            """One line per failed connection (a client that went away, a reset): no traceback, no client address."""
            e = sys.exc_info()[1]
            print(f"serve: connection ended with {type(e).__name__}", file=sys.stderr, flush=True)

        def process_request(self, request, client_address):
            """At most --http-max-conns handler threads; above that a minimal 503 and close, without a thread."""
            with self.active_lock:
                over = self.active >= a.http_max_conns
                if not over:
                    self.active += 1
            if over:
                with counters_lock:
                    counters["conn_limit_503"] += 1
                try:
                    request.setblocking(False)
                    request.send(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                except OSError:
                    pass
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException as e:  # noqa: BLE001 (e.g. Thread.start: can't start new thread)
                with self.active_lock:             # the thread never ran: its slot comes back now
                    self.active -= 1
                self.shutdown_request(request)
                print(f"serve: could not start a handler thread ({type(e).__name__}); connection closed",
                      file=sys.stderr, flush=True)

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                with self.active_lock:
                    self.active -= 1
    srv = Server((a.host, a.port), H)
    shared = [f"{m} {p}" for (m, p), e in engines.items() if getattr(e, "enc_shared", False)
              or getattr(e, "stream_shared", False)]
    print(f"serving on http://{a.host}:{a.port} (auth {'on' if key else 'off'}, {cfg['device']}; engines "
          f"{', '.join(f'{m} {p} {e.decode_kind}' for (m, p), e in engines.items())}; default {modes[0]}; live "
          f"{'on' if stream_eng else 'off'}; shared encoder: {', '.join(shared) if shared else 'none'})",
          file=sys.stderr, flush=True)
    srv.serve_forever()


CONTROL_JSON_MAX = 4096      # bytes: the WebSocket's JSON control messages (stop), checked before parsing
START_JSON_MAX = 32768       # bytes: the start message (it may carry a word_boost list: 100 phrases x 50 characters)


class _HeadReader:
    """The handler's rfile (unbuffered, on the socket): every read has the per-operation timeout, and while armed
    the request line + headers also have one total deadline, so a client sending a byte just inside each timeout
    cannot hold a thread past it. readline reads byte by byte (as the unbuffered reader does) and checks the deadline
    each time."""

    def __init__(self, sock, op_timeout: float):
        self.sock, self.op, self.deadline = sock, op_timeout, None

    def arm(self, deadline: float):
        self.deadline = deadline

    def disarm(self):
        self.deadline = None
        try:                                     # a read near the deadline left a short timeout on the socket: the
            self.sock.settimeout(self.op)        # response must not inherit it
        except OSError:
            pass

    def _timeout(self) -> float:
        if self.deadline is None:
            return self.op
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("request deadline")
        return min(self.op, left)

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            raise ValueError("unbounded read")
        self.sock.settimeout(self._timeout())
        return self.sock.recv(n)

    def readline(self, limit: int = -1) -> bytes:
        out = bytearray()
        while limit < 0 or len(out) < limit:
            b = self.read(1)
            if not b:
                break
            out += b
            if b == b"\n":
                break
        return bytes(out)

    def close(self):
        pass


def _json_small(data: bytes, limit: int = CONTROL_JSON_MAX):
    """A small JSON control message, or None (too big, not UTF-8 / JSON, or nested past the parser's recursion)."""
    if len(data) > limit:
        return None
    try:
        return json.loads(data)
    except (ValueError, RecursionError):
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m speech_server", allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe", help="transcribe audio files", allow_abbrev=False)
    engine_args(t)
    t.add_argument("inputs", nargs="*")
    t.add_argument("--list", help="file with one audio path per line")
    t.add_argument("--out", help="jsonl output (default stdout)")
    t.add_argument("--pool-windows", type=int, default=256, help="windows (<= 30 s each) decoded together; bounds RAM")
    t.add_argument("--no-segments", dest="segments", action="store_false", help="omit per-window start/end/text")
    t.add_argument("--server", help="send to a running `serve` instead (e.g. http://127.0.0.1:8000)")
    t.add_argument("--api-key-env", default="SPEECH_API_KEY")
    t.add_argument("--api-key-file")
    s = sub.add_parser("serve", allow_abbrev=False, help="HTTP server: POST /v1/transcribe (audio bytes), GET /health, GET /help, "
                       "GET /demo, WebSocket /stream (live, with a streaming model)")
    engine_args(s)
    s.add_argument("--host", default=None, help="listen address (default 127.0.0.1; the Docker images use 0.0.0.0); "
                   "env HOST")
    s.add_argument("--port", type=int, default=None, help="listen port (default 8000); env PORT")
    s.add_argument("--api-key-env", default="SPEECH_API_KEY", help="environment variable holding the API key "
                   "clients must send (default SPEECH_API_KEY)")
    s.add_argument("--api-key-file")
    s.add_argument("--no-auth", action="store_true", help="loopback only")
    s.add_argument("--batch-wait-ms", type=float, default=20.0)
    s.add_argument("--max-body-mb", type=_num(float, 0, open_lo=True), default=1024.0, help="upload limit in MB "
                   "(default 1024; 16 kHz 16-bit mono WAV is ~115 MB per hour, 48 kHz stereo 24-bit ~1 GB per hour)")
    s.add_argument("--modes", help="comma list served on one port, chosen per request with ?mode= (first is the "
                   "default), e.g. decoder,ctc (ctc-stream / dec-stream: on --stream-model, else --model must be a "
                   "streaming package); default: --mode")
    s.add_argument("--decodes", help="decoder search kinds served on one port, chosen per request with ?decode= (first "
                   "is the default), e.g. greedy,beam: Normal and Performance mode side by side (one more decoder "
                   "engine, sharing the encoder); default: --decode; env DECODES")
    s.add_argument("--precisions", help="comma list served on one port, chosen per request with ?precision= (first is "
                   "the default), e.g. fp32,fp16; default: --precision")
    s.add_argument("--tmp-dir", default=None, help="where uploads are spooled while transcribing (default: system tmp)")
    s.add_argument("--stream-model", help="a streaming package served as mode ctc-stream and live /stream, next to "
                   "--model's decoder / ctc (a streaming-decoder package can also serve dec-stream: add it to "
                   "--modes); env STREAM_MODEL_DIR")
    s.add_argument("--stream-model-repo", help="Hugging Face repository of the streaming package, downloaded like "
                   "--model-repo; env STREAM_MODEL_REPO")
    s.add_argument("--stream-model-revision", help="branch, tag or commit of --stream-model-repo; env "
                   "STREAM_MODEL_REVISION")
    s.add_argument("--stats", action=argparse.BooleanOptionalAction, default=False,
                   help="serve GET /stats (needs the API key): per-engine timings, effective batch limits, "
                   "micro-batch sizes and 503 counts, for tuning; default off; env STATS")
    s.add_argument("--share-encoder", action=argparse.BooleanOptionalAction, default=None,
                   help="decoder and ctc share one encoder session (one copy of the encoder weights), and ctc-stream "
                   "/ dec-stream share their chunk encoder (default: on; --no-share-encoder loads one per mode); env "
                   "SHARE_ENCODER=true|false")
    s.add_argument("--model-name", help="name reported by GET /health (default \"model\"); env MODEL_NAME")
    s.add_argument("--max-streams", type=_num(int, 1), default=None, help="concurrent live /stream sessions "
                   "(server-wide, default 4); env MAX_STREAMS")
    s.add_argument("--langs", default=None,
                   help="languages offered per request (comma list; default: th, en and zh, those the model has; "
                        "the languages it was trained on); /help, the demo and ?lang= validation use it")
    s.add_argument("--stream-precision", default="fp32", help="precision of the streaming / live engines: fp32 only "
                   "(the only chunk graphs there are; anything else is refused at startup)")
    s.add_argument("--stream-threads", type=_num(int, 1), default=2, help="intra-op threads of the streaming engine "
                   "(live streams and ctc-stream requests), kept small so streams cannot starve batch requests")
    s.add_argument("--ws-max-frame-kb", type=_num(float, 1), default=256.0, help="largest live WebSocket message")
    s.add_argument("--ws-ping-s", type=_num(float, 1), default=15.0, help="WebSocket ping interval (keeps proxies from "
                   "timing an idle stream out; at least 1 s)")
    s.add_argument("--ws-preauth-max", type=_num(int, 1), default=64, help="WebSocket connections waiting to "
                   "authenticate, server-wide (more: 503 before the upgrade)")
    s.add_argument("--ws-auth-s", type=_num(float, 0, open_lo=True), default=10.0, help="deadline for the start "
                   "message after the upgrade")
    s.add_argument("--ws-first-audio-s", type=_num(float, 0, open_lo=True), default=60.0, help="grace for the first "
                   "0.1 s of audio after ready (a browser's microphone prompt), then --ws-idle-s applies")
    s.add_argument("--ws-idle-s", type=_num(float, 0, open_lo=True), default=30.0, help="a live session that "
                   "receives less than 0.1 s of audio for this long ends (final message, reason idle)")
    s.add_argument("--ws-session-max-s", type=_num(float, 0, open_lo=True), default=900.0, help="hard wall-clock cap "
                   "per live session (final message, reason session_limit)")
    s.add_argument("--ws-send-timeout-s", type=_num(float, 0, open_lo=True), default=10.0, help="a WebSocket write "
                   "that takes longer closes the connection")
    s.add_argument("--ws-pace", type=_num(float, 0, open_lo=True), default=2.0, help="live audio may arrive at most "
                   "this many times faster than real time ...")
    s.add_argument("--ws-pace-slack-s", type=_num(float, 0), default=5.0, help="... plus this many seconds ahead")
    s.add_argument("--http-timeout-s", type=_num(float, 0, open_lo=True), default=60.0, help="HTTP read / write timeout "
                   "per socket operation (request line, headers, each body read): idle connections are dropped")
    s.add_argument("--decode-timeout-s", type=_num(float, 0, open_lo=True), default=300.0, help="wall-clock limit "
                   "for decoding one upload (the decoder is killed past it; the request gets a 400)")
    s.add_argument("--decode-max", type=_num(int, 1), default=4, help="uploads decoded at once (each is decoded once, "
                   "whole, in its request's thread)")
    s.add_argument("--max-audio-s", type=_num(float, 1), default=3600.0, help="longest upload in seconds of audio "
                   "(>= 1; decoded to float32: ~230 MB per hour in memory while it is transcribed); longer: 413")
    s.add_argument("--inflight-audio-s", type=_num(float, 0, open_lo=True), default=7200.0, help="decoded audio the "
                   "server holds at once, all requests together (7200 s ~ 460 MB float32; at least 2 x "
                   "--max-audio-s); over it: 503, retry")
    s.add_argument("--http-body-s", type=_num(float, 0, open_lo=True), default=1800.0, help="total time for one "
                   "upload's body, chunk framing included (default 1800 s); slower uploads get 408")
    s.add_argument("--spool-max-mb", type=_num(float, 0, open_lo=True), default=4096.0, help="request bodies spooled "
                   "to disk at once, all uploads together (default 4096 MB; at least --max-body-mb); over it: 503")
    s.add_argument("--http-head-s", type=_num(float, 0, open_lo=True), default=30.0, help="total time for a request's "
                   "line and headers (on top of the per-read --http-timeout-s)")
    s.add_argument("--http-max-conns", type=_num(int, 1), default=256, help="concurrent HTTP connections (handler "
                   "threads, live sessions included); above it a minimal 503 without a thread")
    a = ap.parse_args(env_argv(sys.argv[1:] if argv is None else list(argv)))
    if a.cmd == "serve":
        a.host = a.host or "127.0.0.1"
        a.port = 8000 if a.port is None else a.port
        a.max_streams = 4 if a.max_streams is None else a.max_streams
        if not (0 < a.port < 65536):
            sys.exit(f"--port {a.port}: must be 1-65535")
        if a.max_streams < 1:
            sys.exit("--max-streams must be at least 1")
    {"transcribe": cmd_transcribe, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    main()
