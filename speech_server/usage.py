"""The server's per-request API as data: one definition of the POST /v1/transcribe query parameters that both the
request validation (parse_query) and the docs (GET /help, GET /help?format=json, the demo page) are built from, so
docs and validation cannot drift. USAGE.md (next to this file) is the prose; the live facts (served modes, precisions, look-aheads,
languages, defaults, limits) are filled in from what the server actually loaded. No model names, paths or devices.

Compatibility with the earlier server: unknown query parameters are ignored and a repeated parameter takes its first
value (except word_boost, where every value counts). A known parameter with a blank or invalid value is a 400.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BOOL_TRUE, BOOL_FALSE = ("true", "1", "yes", "on"), ("false", "0", "no", "off")

# The two strict rules in one place (set False to relax both): an unknown `lang` is a 400 (False: automatic), and a
# parameter that does not apply to the chosen mode (itn / pnc outside decoder / dec-stream, lookahead outside the
# streaming modes) is a 400 (False: ignored). The live /stream start message follows the same two rules.
STRICT_VALUES = True

# name -> definition. "values": allowed values (None = from the live facts, see _values); "modes": the modes the
# parameter applies to (None = all).
QUERY_PARAMS = {
    "mode": {"values": None, "modes": None,
             "doc": "decoder (attention decoder, best accuracy), ctc (CTC head only, fastest), ctc-stream (the "
                    "streaming encoder over the file, chunk by chunk, as a live stream would see it) or dec-stream "
                    "(preview: the streaming decoder on that encoder, a live stream's decoder output)"},
    "decode": {"values": None, "modes": ["decoder"],
               "doc": "the decoder's search, as served: greedy (Normal mode) or beam (Performance mode: more accurate, "
                      "slower)"},
    "precision": {"values": None, "modes": None,
                  "doc": "numeric precision of the graphs; ctc-stream and dec-stream are fp32 only"},
    "lang": {"values": None, "modes": None,
             "doc": "language prompt for the decoder (auto: the model picks; case-insensitive); CTC modes ignore it"},
    "itn": {"values": ["true", "false"], "modes": ["decoder", "dec-stream"],
            "doc": "written form (numbers as digits, ...); false: spoken form"},
    "pnc": {"values": ["true", "false"], "modes": ["decoder", "dec-stream"],
            "doc": "punctuation and case"},
    "timestamps": {"values": None, "modes": None,
                   "doc": "segment (decoder only): the decoder writes <|tS|> text <|tE|> pairs, returned as "
                          "timed_segments; token: per-token times from the CTC head (CTC modes: its own frames; "
                          "decoder: forced alignment of the decoder's tokens, where the export supports it), returned "
                          "as token_times. ctc-stream always returns token_times; dec-stream too (emit times only), "
                          "so it takes none"},
    "lookahead": {"values": None, "modes": ["ctc-stream", "dec-stream"],
                  "doc": "streaming look-ahead L in 80 ms encoder frames (chunk L + 1): more look-ahead, more "
                         "accuracy, more latency"},
    "word_boost": {"values": None, "modes": None,
                   "doc": "a phrase to bias decoding toward (names, organisations, terms); repeat the parameter for "
                          "several, or give one JSON list: [\"phrase\", {\"phrase\": \"...\", \"boost\": 1.5}]. "
                          "Up to 500 phrases of up to 50 characters; per-phrase boost in (0, 2], default 1. "
                          "Not in dec-stream"},
    "boost": {"values": None, "modes": None,
              "doc": "with word_boost: overall boost strength in (0, 2], default 1 (higher: more recall, more false "
                     "alarms)"},
    "ctc_beam": {"values": None, "modes": ["ctc", "ctc-stream"],
                 "doc": "with word_boost in CTC modes: beam width 1-8 of the boosted CTC search (default 4; 1: "
                        "greedy with boost re-selection, cheaper and weaker)"},
}
MULTI_PARAMS = ("word_boost",)               # parameters that may repeat (every value counts)

RESPONSE_KEYS = {
    "mode": "the mode that ran", "precision": "the precision that ran",
    "decode": "decoder mode: the search that ran (greedy | beam)", "text": "the transcript",
    "windows": "number of <= 30 s windows the file was cut into (ctc-stream, dec-stream: 1)",
    "segments": "[{start, end, text}] per window, seconds (decoder: also capped = hit the token cap; dec-stream: one "
                "per decoder segment, with reason = end | endpoint | max_segment | eot | max_len | pos_cap)",
    "audio_seconds": "input duration", "seconds": "server wall time for the request",
    "timed_segments": "timestamps=segment only: [{start, end, text, window}] from the decoder's timestamp tokens; "
                      "flag = no_start | no_end | no_timestamps | out_of_order | past_window where they do not pair up (times "
                      "null, never guessed)",
    "token_times": "timestamps=token or ctc-stream: [{id, text, start, end, aligned}] per token (ctc-stream: also "
                   "emit = when the stream had the token); dec-stream: emit only (start / end null, aligned false)",
    "align_ok": "decoder + timestamps=token: false when a window's CTC alignment was infeasible (its times are null)",
    "lookahead": "ctc-stream, dec-stream: the look-ahead used",
    "guards": "dec-stream: counts over the decoder segments: eot_mid (<|eot|> before the end), wait_at_end (<wait> "
              "on a segment's last chunk), pos_cap (position table full), max_len (token cap), burst_max (most tokens "
              "out of one chunk)",
    "error": "on failure: what went wrong; detail: the engine's message for an input it refused",
}

PUBLIC_FACTS = ("modes", "default_mode", "precisions", "default_precision", "decodes", "default_decode",
                "lookaheads", "default_lookahead",
                "lookaheads_by_mode", "default_lookahead_by_mode",
                "langs", "defaults", "timestamps_by_mode", "decode", "live", "live_modes", "max_stream_seconds",
                "max_body_mb",
                "max_audio_seconds", "max_streams", "live_idle_seconds", "live_first_audio_seconds",
                "live_session_seconds", "word_boost_modes")


def _values(name: str, facts: dict, mode: str | None = None):
    if name == "mode":
        return list(facts["modes"])
    if name == "precision":
        if mode:
            return list(facts["precisions"][mode])
        return sorted({p for ps in facts["precisions"].values() for p in ps})
    if name == "lang":
        return ["auto"] + list(facts.get("langs", []))
    if name == "decode":
        return list(facts.get("decodes") or [])
    if name == "lookahead":
        if mode and mode in (facts.get("lookaheads_by_mode") or {}):
            return [str(x) for x in facts["lookaheads_by_mode"][mode]]
        return [str(x) for x in facts.get("lookaheads", [])]
    if name == "timestamps":
        tbm = facts["timestamps_by_mode"]
        if mode:
            return list(tbm[mode])
        return [t for t in ("none", "segment", "token") if any(t in v for v in tbm.values())]
    return list(QUERY_PARAMS[name]["values"] or [])     # free-form parameters (word_boost, boost, ctc_beam): no list


def _default(name: str, facts: dict):
    d = facts.get("defaults", {})
    if name == "mode":
        return facts["default_mode"]
    if name == "precision":
        return facts["default_precision"]
    if name == "decode":
        return facts.get("default_decode")
    if name == "lookahead":
        return None if facts.get("default_lookahead") is None else str(facts["default_lookahead"])
    if name == "lang":
        return d.get("lang") or "auto"
    if name in ("itn", "pnc"):
        return "true" if d.get(name) else "false"
    if name == "timestamps":
        return d.get("timestamps", {}).get(facts["default_mode"], "none")
    return None


def _bool(name, v):
    s = str(v).strip().lower()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {v!r}")


def parse_query(qs: dict, facts: dict) -> tuple[str, str, dict]:
    """parse_qs(keep_blank_values=True) dict -> (mode, precision, engine options). Unknown parameters are ignored and
    a repeated one takes its first value (as the earlier server; word_boost: every value counts); ValueError (the server's 400) on a blank or invalid
    value of a known parameter. Absent parameters take the engine's (the server's) defaults."""
    q = {k: v[0] for k, v in qs.items() if k in QUERY_PARAMS and k not in MULTI_PARAMS and v}
    wb_raw = [v for v in qs.get("word_boost", [])]
    if wb_raw:
        q["word_boost"] = wb_raw
    for k, v in q.items():
        if any(not x.strip() for x in (v if isinstance(v, list) else [v])):
            raise ValueError(f"{k} is blank")
    mode = q.get("mode", facts["default_mode"])
    if mode not in facts["modes"]:
        raise ValueError(f"mode must be one of {facts['modes']}")
    precs = facts["precisions"][mode]
    prec = q.get("precision", precs[0])
    if prec not in precs:
        raise ValueError(f"precision must be one of {precs} for mode {mode}")
    for k in list(q):
        ms = QUERY_PARAMS[k]["modes"]
        if ms is not None and mode not in ms:
            if STRICT_VALUES:
                raise ValueError(f"{k} applies to mode {' / '.join(ms)} only")
            del q[k]
    opts = {}
    if "lang" in q:
        lang = q["lang"].strip().lower()
        if lang != "auto" and lang not in facts.get("langs", []):
            if STRICT_VALUES:
                raise ValueError(f"lang must be auto or one of {facts.get('langs', [])}")
            lang = "auto"
        opts["lang"] = None if lang == "auto" else lang
    for k in ("itn", "pnc"):
        if k in q:
            opts[k] = _bool(k, q[k])
    if "timestamps" in q:
        ts = q["timestamps"].strip().lower()
        allowed = facts["timestamps_by_mode"][mode]
        if ts not in allowed:
            raise ValueError(f"timestamps must be one of {allowed} for mode {mode}")
        opts["timestamps"] = ts
    if "lookahead" in q:
        if q["lookahead"].strip() not in _values("lookahead", facts, mode):
            raise ValueError(f"lookahead must be one of {_values('lookahead', facts, mode)} for mode {mode}")
        opts["lookahead"] = int(q["lookahead"])
    if "decode" in q:
        dv = q["decode"].strip().lower()
        if dv not in (facts.get("decodes") or []):
            raise ValueError(f"decode must be one of {facts.get('decodes') or []} for mode {mode}")
        opts["decode"] = dv                  # picks the engine (cli); not an engine option
    opts.update(_boost_query(q))
    return mode, prec, opts


def _boost_query(q: dict) -> dict:
    """word_boost (repeated values; a value starting with "[" is a JSON list), boost, ctc_beam -> engine options.
    The engine validates the list itself (boost.parse_list) and refuses what it cannot do."""
    out = {}
    if "word_boost" in q:
        items = []
        for v in q["word_boost"]:
            if v.lstrip().startswith("["):
                try:
                    x = json.loads(v)
                except (ValueError, RecursionError):
                    raise ValueError("word_boost: a value starting with [ must be a JSON list") from None
                if not isinstance(x, list):
                    raise ValueError("word_boost: a value starting with [ must be a JSON list")
                items += x
            else:
                items.append(v)
        out["word_boost"] = items
    if "boost" in q:
        out["boost"] = q["boost"].strip()
    if "ctc_beam" in q:
        v = q["ctc_beam"].strip()
        if not (v.isascii() and v.isdigit() and len(v) <= 2):
            raise ValueError("ctc_beam must be an integer from 1 to 8")
        out["ctc_beam"] = int(v)
    return out


def parse_start(msg: dict, facts: dict) -> tuple[str, int | None, dict]:
    """The live /stream start message -> (mode, lookahead or None = the server's default, decoder options). `mode`:
    one of the served live modes (default the first); `lookahead`: a JSON integer (checked against the export by the
    engine); dec-stream also takes `lang` (auto or a served language), `itn` and `pnc` (JSON booleans). The strict
    rules as for POST: an option for the other mode is refused (STRICT_VALUES). ValueError on a bad value."""
    live = list(facts.get("live_modes") or (["ctc-stream"] if facts.get("live") else []))
    mode = msg.get("mode", live[0] if live else None)
    if not isinstance(mode, str) or mode not in live:
        raise ValueError(f"mode must be one of {live}")
    la = msg.get("lookahead")
    if la is not None and (isinstance(la, bool) or not isinstance(la, int)):
        raise ValueError(f"lookahead must be one of {(facts.get('lookaheads_by_mode') or {}).get(mode, facts.get('lookaheads', []))} (an integer, encoder frames of "
                         "80 ms)")
    opts = {}
    for k in ("lang", "itn", "pnc"):
        if k not in msg:
            continue
        if mode != "dec-stream" and k != "lang":   # lang: ignored by CTC, as on POST
            if STRICT_VALUES:
                raise ValueError(f"{k} applies to mode dec-stream only")
            continue
        v = msg[k]
        if k == "lang":
            if not isinstance(v, str) or (v.strip().lower() != "auto" and v.strip().lower() not in facts.get("langs", [])):
                if STRICT_VALUES or not isinstance(v, str):
                    raise ValueError(f"lang must be auto or one of {facts.get('langs', [])}")
                v = "auto"
            v = v.strip().lower()
            if mode == "dec-stream":
                opts["lang"] = None if v == "auto" else v
        else:
            if not isinstance(v, bool):
                raise ValueError(f"{k} must be true or false (a JSON boolean)")
            opts[k] = v
    # word boost: "word_boost" a JSON list (strings or {"phrase", "boost"}), "boost" a number, "ctc_beam" an integer
    if "word_boost" in msg:
        if not isinstance(msg["word_boost"], list):
            raise ValueError("word_boost must be a JSON list")
        from speech_server.boost import MAX_PHRASES_STREAM
        if len(msg["word_boost"]) > MAX_PHRASES_STREAM:
            raise ValueError(f"word_boost takes at most {MAX_PHRASES_STREAM} phrases on a live stream")
        opts["word_boost"] = msg["word_boost"]
    if "boost" in msg:
        if isinstance(msg["boost"], bool) or not isinstance(msg["boost"], (int, float)):
            raise ValueError("boost must be a number")
        opts["boost"] = msg["boost"]
    if "ctc_beam" in msg:
        if mode != "ctc-stream":
            raise ValueError("ctc_beam applies to mode ctc-stream only")
        if isinstance(msg["ctc_beam"], bool) or not isinstance(msg["ctc_beam"], int):
            raise ValueError("ctc_beam must be an integer from 1 to 8")
        opts["ctc_beam"] = msg["ctc_beam"]
    return mode, la, opts


def help_json(facts: dict) -> dict:
    params = {}
    for k, d in QUERY_PARAMS.items():
        p = {"values": _values(k, facts), "default": _default(k, facts),
             "modes": d["modes"] or list(facts["modes"]), "doc": d["doc"]}
        if k in ("timestamps", "precision"):
            p["values_by_mode"] = {m: _values(k, facts, m) for m in facts["modes"]}
        if k == "lookahead" and facts.get("lookaheads_by_mode"):
            p["values_by_mode"] = {m: _values(k, facts, m) for m in facts["lookaheads_by_mode"]}
            p["default_by_mode"] = {m: str(v) for m, v in (facts.get("default_lookahead_by_mode") or {}).items()}
        if k == "timestamps":
            p["default_by_mode"] = dict(facts.get("defaults", {}).get("timestamps", {}))
        if k in ("word_boost", "boost"):
            p["modes"] = list(facts.get("word_boost_modes") or [])
        if k == "ctc_beam":
            p["modes"] = [m for m in d["modes"] if m in (facts.get("word_boost_modes") or [])]
        params[k] = p
    return {
        "server": {k: facts.get(k) for k in PUBLIC_FACTS},
        "endpoints": {"POST /v1/transcribe": "audio file bytes in the body (WAV / FLAC / OGG / MP3; others need "
                                             "ffmpeg on the server), options as query parameters",
                      "GET /v1/check": "200 if the key is accepted (auth only)",
                      "GET /health": "server status (no auth)", "GET /help": "this documentation (markdown; "
                                                                            "?format=json for this form)",
                      "GET /demo": "browser demo page", "GET /stream": "WebSocket: live streaming (CTC, or the "
                                                                       "streaming decoder preview; see docs)"},
        "query_params": params, "response_keys": dict(RESPONSE_KEYS),
        "auth": "Authorization: Bearer <key> on POST /v1/transcribe, GET /v1/check and the /stream start message",
        "live_start": {"type": "start", "key": "<key, or the Authorization header>", "mode": facts.get("live_modes"),
                       "lookahead": "a look-ahead served for the mode (integer; server.lookaheads_by_mode)",
                       "lang / itn / pnc": "dec-stream only: the decoder prompt (auto or a served language; JSON "
                                           "booleans)",
                       "word_boost / boost / ctc_beam": "ctc-stream (where served for word boost): a JSON list of "
                                                        "phrases (up to 100), a number in (0, 2], an integer 1-8"}
        if facts.get("live") else None,
        "examples": examples(facts),
    }


def examples(facts: dict) -> list[str]:
    ex = ['curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav "https://HOST/v1/transcribe"']
    tbm = facts["timestamps_by_mode"]
    if "decoder" in facts["modes"]:
        ts = "&timestamps=token" if "token" in tbm["decoder"] else ""
        ex.append('curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav '
                  f'"https://HOST/v1/transcribe?mode=decoder&lang=th&itn=true&pnc=true{ts}"')
    if "beam" in (facts.get("decodes") or []) and len(facts["decodes"]) > 1:
        ex.append('curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav '
                  '"https://HOST/v1/transcribe?mode=decoder&decode=beam"')
    if facts.get("word_boost_modes"):
        ex.append('curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav '
                  f'"https://HOST/v1/transcribe?mode={facts["word_boost_modes"][0]}&word_boost=Acme%20Corp&boost=1"')
    if "ctc" in facts["modes"]:
        ex.append('curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav '
                  '"https://HOST/v1/transcribe?mode=ctc&timestamps=token"')
    if "ctc-stream" in facts["modes"] and facts.get("lookaheads"):
        ex.append('curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav '
                  f'"https://HOST/v1/transcribe?mode=ctc-stream&lookahead={facts["lookaheads"][0]}"')
    if "dec-stream" in facts["modes"] and facts.get("lookaheads"):
        dla = (facts.get("default_lookahead_by_mode") or {}).get("dec-stream", facts["lookaheads"][-1])
        ex.append('curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav '
                  f'"https://HOST/v1/transcribe?mode=dec-stream&lang=th&lookahead={dla}"')
    return ex


def help_markdown(facts: dict) -> str:
    with open(os.path.join(HERE, "USAGE.md"), encoding="utf-8") as f:
        doc = f.read()
    j = help_json(facts)
    s = j["server"]
    lines = ["## This server (live)", ""]
    lines.append(f"- modes: {', '.join(s['modes'])} (default {s['default_mode']})")
    for m, ps in s["precisions"].items():
        lines.append(f"- precisions for {m}: {', '.join(ps)}")
    if s.get("decodes"):
        lines.append(f"- decoder search (?decode=): {', '.join(s['decodes'])} (default {s['default_decode']}; greedy is "
                     "Normal mode, beam is Performance mode)")
    for m, ts in s["timestamps_by_mode"].items():
        lines.append(f"- timestamps for {m}: {', '.join(ts)} (default {s['defaults']['timestamps'].get(m, 'none')})")
    if s.get("lookaheads_by_mode"):
        for m, las in s["lookaheads_by_mode"].items():
            lines.append(f"- look-aheads for {m} (POST and live): {', '.join(str(x) for x in las)} "
                         f"(default {s['default_lookahead_by_mode'][m]})")
    elif s.get("lookaheads"):
        lines.append(f"- look-aheads (streaming modes and live): {', '.join(str(x) for x in s['lookaheads'])} "
                     f"(default {s['default_lookahead']})")
    d = s["defaults"]
    lines.append(f"- languages: auto, {', '.join(s.get('langs') or [])} (default {d.get('lang') or 'auto'}); "
                 f"itn default {'true' if d.get('itn') else 'false'}, pnc default {'true' if d.get('pnc') else 'false'}")
    lines.append(f"- live streaming (/stream): {'on' if s.get('live') else 'off (no streaming model loaded)'}"
                 + (f", modes {', '.join(s['live_modes'])} (default {s['live_modes'][0]})"
                    if s.get("live") and s.get("live_modes") else ""))
    if "dec-stream" in s["modes"]:
        lines.append("- dec-stream is a preview: the streaming decoder, segments reset at pauses (640 ms of CTC blank "
                     "after speech) and every 30 s; no stream length limit beyond the upload / session limits")
        las = (s.get("lookaheads_by_mode") or {}).get("dec-stream")
        if las:
            lines.append(f"- dec-stream is served at look-ahead {', '.join(str(x) for x in las)} only (the look-aheads "
                         "the package marks as validated for its streaming decoder)")
    wbm = s.get("word_boost_modes") or []
    lines.append(f"- word boost: {', '.join(wbm) if wbm else 'off'}" + (" (up to 500 phrases per request, 100 on a "
                 "live stream; 50 characters each)" if wbm else ""))
    cap = (f"; ctc-stream / live CTC: at most {s.get('max_stream_seconds')} s per stream"
           if s.get("max_stream_seconds") is not None else "")
    lines.append(f"- upload limits: {s.get('max_body_mb')} MB, {s.get('max_audio_seconds')} s of audio{cap}")
    if s.get("live"):
        lines.append(f"- live sessions: {s.get('max_streams')} at once; idle after {s.get('live_idle_seconds')} s "
                     f"without audio ({s.get('live_first_audio_seconds')} s for the first audio); at most "
                     f"{s.get('live_session_seconds')} s per session")
    lines += ["", "| parameter | values | default | modes |", "|---|---|---|---|"]
    for k, p in j["query_params"].items():
        lines.append(f"| {k} | {', '.join(p['values']) or '-'} | {p['default'] or '-'} | {', '.join(p['modes'])} |")
    lines += ["", "Examples (replace HOST with this server's address and $KEY with your key):", "", "```"]
    lines += j["examples"] + ["```", ""]
    return doc.rstrip() + "\n\n" + "\n".join(lines)
