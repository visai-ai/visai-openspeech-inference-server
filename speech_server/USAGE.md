# Speech-to-text server: usage

This server transcribes audio. Send it a file over HTTPS, or stream microphone audio over a WebSocket. Every API call
needs your key in the `Authorization: Bearer <key>` header (or, for the WebSocket, in its first message). The live
section at the end of this page lists what this particular server serves; `GET /help?format=json` returns the same
content as JSON.

## Transcribe a file

`POST /v1/transcribe` with the audio file's bytes as the request body: WAV, FLAC, OGG / Opus or MP3, and, when the
server has ffmpeg, M4A / MP4 / 3GP, WebM / Matroska, AAC, AIFF, CAF or AMR. Playlists and other files that refer to
further files or URLs are refused. Options go in the query string.

```
curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav "https://HOST/v1/transcribe?mode=decoder&lang=th"
```

Long files are cut into windows of at most 30 s at pauses and decoded together, up to the server's audio limit
(`max_audio_seconds` in the live section; 1 h by default). Longer audio is refused with 413 (`"error": "too
long"`): retrying cannot help, split the file.

### Query parameters

| parameter | values | default | what it does |
|---|---|---|---|
| `mode` | `decoder`, `ctc`, `ctc-stream`, `dec-stream` (as served) | the server's default | `decoder`: attention decoder, best accuracy. `ctc`: CTC head only, fastest. `ctc-stream`: the streaming encoder run over the file chunk by chunk, as a live stream would see it. `dec-stream` (preview): the streaming decoder on that encoder, what a live decoder stream would write (see "Streaming decoder" below) |
| `decode` | as served (`greedy`, `beam`) | the server's default | `decoder` only: the decoder's search; `greedy` is Normal mode, `beam` is Performance mode (more accurate, slower) |
| `precision` | as served (`fp32`, `fp16`, ...) | first served | numeric precision; `ctc-stream` and `dec-stream` are `fp32` only |
| `lang` | `auto` or a language code (`th`, `en`, `zh`, ...; any case) | the server's default (usually `auto`) | language prompt for the decoder; CTC modes ignore it |
| `itn` | `true`, `false` | the server's default (usually `false`) | `decoder` / `dec-stream` only: written form (numbers as digits); `false` gives the spoken form |
| `pnc` | `true`, `false` | the server's default (usually `false`) | `decoder` / `dec-stream` only: punctuation and case |
| `timestamps` | `none`, `segment`, `token` (as served per mode) | the server's default (usually `none`) | see "Timestamps" below |
| `lookahead` | as served per mode (e.g. `ctc-stream`: `0`, `4`, `13`; `dec-stream`: `4`) | the mode's default (see "This server") | `ctc-stream` / `dec-stream` only: look-ahead L in 80 ms frames (chunk L + 1). L = 0 is lowest latency, L = 13 (about 1.1 s) most accurate. A look-ahead the mode does not serve is refused with 400 |

The live section at the end of this page lists the actual values and defaults. A value the server does not serve, a
blank value, or a parameter that does not apply to the chosen mode (`itn` / `pnc` outside `decoder` / `dec-stream`,
`lookahead` outside `ctc-stream` / `dec-stream`) is refused with HTTP 400 and a message (likewise `decode` outside `decoder`). Unknown parameters are ignored, and a repeated parameter
takes its first value (except `word_boost`, where every value counts).

### Request validation (strict)

The server checks every query parameter before it reads your audio, and refuses a request it cannot run exactly as
asked. It never falls back to a default, guesses, or silently drops an option you set. A refused request is HTTP 400
with `{"error": "<reason>"}`, and nothing is transcribed.

| you send | result | `error` |
|---|---|---|
| a parameter the server does not know (`?foo=1`) | ignored, request runs | |
| the same parameter twice (`?lang=th&lang=en`) | the first value is used (`word_boost`: every value is a phrase) | |
| `lang` in any case (`?lang=TH`) | accepted as `th` | |
| a blank value (`?lang=`, `?itn=`) | 400 | `lang is blank` |
| a language the server does not serve (`?lang=ja` on a th / en / zh server) | 400 | `lang must be auto or one of ['th', 'en', 'zh']` (the served list) |
| `itn` or `pnc` with a CTC mode (`?mode=ctc&pnc=true`) | 400 | `pnc applies to mode decoder / dec-stream only` |
| `lookahead` outside the streaming modes (`?mode=decoder&lookahead=4`) | 400 | `lookahead applies to mode ctc-stream / dec-stream only` |
| a mode, precision, timestamps or look-ahead value not served | 400 | e.g. `mode must be one of [...]`, `timestamps must be one of [...] for mode ctc` |
| `itn` / `pnc` as `true`/`false`, `1`/`0`, `yes`/`no` or `on`/`off` (any case) | accepted | |
| `itn` / `pnc` with any other word (`?itn=maybe`) | 400 | `itn must be true or false, got 'maybe'` |

Omit a parameter to get the server's default for it (listed at the end of this page). This strictness is deliberate:
a typo such as `?lang=tha` would otherwise give you automatic language detection without telling you.

### Response (JSON)

| key | when | meaning |
|---|---|---|
| `mode`, `precision` | always | what ran |
| `decode` | `decoder` | the decoder's search that ran (`greedy` or `beam`) |
| `text` | always | the transcript |
| `windows` | always | number of windows (at most 30 s each; `ctc-stream`, `dec-stream`: 1) |
| `segments` | always | `[{start, end, text}]` per window in seconds; decoder rows also have `capped` (the token cap was hit, likely a repetition loop); `dec-stream`: one row per decoder segment, with `reason` (see below) |
| `audio_seconds`, `seconds` | always | input duration and server time |
| `timed_segments` | `timestamps=segment` | `[{start, end, text, window}]` from the decoder's timestamp tokens |
| `token_times` | `timestamps=token`, and always for `ctc-stream` / `dec-stream` | `[{id, text, start, end, aligned}]` per token; `ctc-stream` adds `emit`; `dec-stream` has `emit` only (`start` / `end` null, `aligned` false) |
| `align_ok` | `mode=decoder&timestamps=token` | `false` if a window's alignment was infeasible |
| `lookahead` | `ctc-stream`, `dec-stream` | the look-ahead used |
| `guards` | `dec-stream` | counts over its decoder segments: `eot_mid`, `wait_at_end`, `pos_cap`, `max_len`, and `burst_max` (most tokens out of one chunk) |

Errors: `{"error": ...}` with 400 (bad request or audio; `detail` explains an input the server refused), 401 (key),
414 (the URL is too long: a long `word_boost` list in Thai or Chinese takes about 9 bytes per character once
percent-encoded, so keep lists short; some proxies limit URLs to about 8 KB),
413 (`"too large"` body or `"too long"` audio: retrying cannot help), 503 (`"error": "busy"`: the server is busy with
other requests or connections, or could not start its decoder, right now; retry after a few seconds) or 500. A file
whose decoding takes longer than the server's decode time limit (5 min by default) is refused with 400 ("decode timed
out"). Each upload is decoded in full on its own before it is batched with others (a file that
cannot be decoded, even late in the file, is that request's 400), so one bad file never fails another request.
Idle connections are closed after the server's HTTP timeout (60 s by default) without data, and a request's line
and headers must arrive within 30 s.

### Timestamps

- `segment` (decoder only): the decoder is asked to write `<|tS|> text <|tE|>` pairs (80 ms steps). Caveat: only a
  small share of the training data had this prompt, so segment-time quality is not guaranteed. Pairs that do not match up (text without a start or end time, an end before its start, a time past the
  window's end) are returned with a `flag`, and a missing time is null; nothing is guessed.
- `token`: per-token times from the model's CTC head, no extra training. In `ctc` / `ctc-stream` each token's time is
  the run of 80 ms frames where the CTC head chose it. In `decoder` mode the decoder's tokens are force-aligned to the
  CTC head's probabilities; tokens the CTC head cannot produce (punctuation, some pieces, special tokens) take the
  neighbouring token's time and have `"aligned": false`. If no alignment exists the window's times are null and
  `align_ok` is false (a window with nothing alignable, e.g. only the end token, is not a failure: `align_ok` stays
  true and those tokens have null times). Decoder-mode `token` needs a model package with the CTC log-prob graph; the
  live section shows whether this server offers it. Times have 80 ms resolution; `text` is the token's piece (Thai characters are often several
  byte tokens: use the window text for reading).

### Streaming decoder (`dec-stream`, preview)

The attention decoder run as a stream: after each chunk of audio it writes what it is sure of, and says "wait" for
the rest. A token's `emit` is the end of the chunk it was written on (80 ms frames). The decoder restarts at pauses
(640 ms of CTC blank after speech) and at least every 30 s; each restart is a row of `segments`, whose `reason` says
why it ended: `endpoint` (a pause), `max_segment` (30 s), `end` (the audio ended), or the decoder itself ended it
(`eot`, `max_len`: the token cap, `pos_cap`: its position table). `guards` counts the decoder's own warning signs:
`eot_mid` (it ended a segment before its end), `wait_at_end` (still waiting when the audio ended: the rest of that
segment is lost). There is no stream length limit beyond the upload and session limits. `timestamps` must be `none`
(the decoder has no frame times; `emit` is the time the stream had each token). This mode is a preview. A
streaming-decoder package may mark the look-aheads its decoder was validated at; the server then serves dec-stream at
those only (`ctc-stream` keeps every look-ahead of the package); /help lists the look-aheads per mode.

### Word boost (custom vocabulary)

Pass names, organisations or other terms the transcript should spell correctly, and decoding is biased toward them:

```
curl -s -H "Authorization: Bearer $KEY" --data-binary @audio.wav \
  "https://HOST/v1/transcribe?mode=decoder&word_boost=สมชาย%20ใจดี&word_boost=Acme%20Corp"
```

- `word_boost`: one phrase per parameter (repeat it for several), or one JSON list such as
  `["Acme Corp", {"phrase": "สมชาย ใจดี", "boost": 1.5}]`. Up to 500 phrases per request (100 on a live
  stream), each up to 50 characters. A phrase is matched by its text, however the model's tokenizer splits it, so Thai
  and Chinese need no spaces. Write each phrase in the form the transcript should show (case, spelling); with
  `itn=true`, the written form (digits).
- `boost`: overall strength in (0, 2], default 1; a phrase's own `boost` (also (0, 2], default 1) multiplies it.
  Higher values find more of the phrases and also write them where they were not said: start at the default and
  keep lists short (the words the model gets wrong, not every word you expect).
- `ctc_beam` (`ctc`, `ctc-stream`): with word_boost the CTC modes use a beam search of this width (default 4; 1 is a
  cheaper greedy re-ranking with less effect). Without word_boost nothing changes: decoding is the default.
- How it works: the bonus re-ranks the tokens the model already ranks among its top candidates (8 for CTC, 16 for
  the decoder); it can still write a listed phrase the model gave a modest probability, so keep lists short. The end
  of the transcript and the CTC blank get no bonus. A phrase earns its bonus only from its second token on and at
  most 3 times per 30 s window (or per live stream), so a phrase that is a single token of the model's vocabulary is
  not boosted. A file window in which no listed phrase is written returns exactly the default output.
- A `word_boost` value starting with `[` is read as a JSON list (to pass a phrase that starts with `[`, put it in a
  JSON list). With a beam in `ctc-stream`, a token's start / end times are the frame it was committed at; in `ctc`,
  boosted tokens are force-aligned (empty `token_times` if no alignment exists).
- `dec-stream` does not take word_boost yet. Modes that cannot boost with this server's model refuse it with 400 and
  the reason; the live section lists the modes that can.

## Live streaming (WebSocket)

`GET /stream` upgraded to a WebSocket (wss:// behind HTTPS), when the server has a streaming model (`live: on` below).

1. Within 10 s, send one text message `{"type": "start", "key": "<your key>", "lookahead": 4}` (at most 32 KB, one
   frame; the key may instead be sent as the `Authorization` header of the upgrade request; `lookahead` is an integer
   from the served set, default the server's). Where the server streams the decoder too (`live` modes below), add
   `"mode": "dec-stream"` (default: the first live mode listed) and optionally `"lang"`, `"itn"`, `"pnc"` (JSON
   booleans; `dec-stream` only). `ctc-stream` also takes `"word_boost"` (a JSON list, as above, up to 100 phrases),
   `"boost"` and `"ctc_beam"`; with a beam, partials are never revised: text is sent once every beam agrees on it,
   or after each chunk for all but the last 2 tokens of the best beam (so a list can change the live text slightly
   even where no phrase is said). A bad key, a late or malformed start message closes the connection;
   nothing else is accepted before it.
2. The server answers `{"type": "ready", "mode": M, "lookahead": L, "sample_rate": 16000, "format": "pcm_s16le", ...}`.
3. Send binary messages of 16 kHz mono 16-bit little-endian PCM, any size up to the frame limit (default 256 KiB).
4. The server sends `{"type": "partial", "tokens": [...], "text": "..."}` whenever new tokens are committed (`tokens`
   are the new ones with `start`, `end`, `emit` seconds, `dec-stream`: `emit` only; `text` is everything so far) and
   WebSocket pings every 15 s. `dec-stream` decodes a chunk once the audio after it starts to arrive (about 10 ms
   later), so the live tokens are exactly those of the same audio sent as a file.
5. Send `{"type": "stop"}` (later text messages: at most 4 KB): the server flushes the last chunk and sends `{"type": "final", "text", "tokens",
   "token_times", "audio_seconds"}` (`dec-stream` adds `segments` and `guards`), then closes.

Limits: one stream per connection; a server-wide cap on concurrent streams (a busy server answers
`{"type": "error", "error": "busy"}`) and on connections still authenticating (HTTP 503 before the upgrade: try
again shortly). A stream ends with a `final` message:
- `"reason": "limit"` after 720 s of audio in `ctc-stream` (9,000 frames: positions are 32-bit floats in the graph, and
  the CTC stream does not rebase them; `dec-stream` rebases them at each decoder restart and has no such limit);
- `"reason": "idle"` when less than 0.1 s of audio arrives in 30 s (empty or very short frames do not count), or when no
  audio at all arrives within 60 s of `ready` (time for a browser's microphone prompt);
- `"reason": "session_limit"` after 15 minutes of wall-clock time.
Audio may not arrive faster than twice real time (plus 5 s), or the connection closes (code 1008): send files to
`/v1/transcribe` instead. Oversize messages close the connection (code 1009). The streaming decoder is live only
where the server lists `dec-stream` among its live modes (a preview).

## Limits

- Upload size and length: set by the server (`max_body_mb`, `max_audio_seconds` below).
- `ctc-stream` and live CTC streams: at most 720 s per stream (`dec-stream`: no stream limit of its own).
- Look-ahead values: only the served ones (the model was trained to stream at those).

## Demo

`GET /demo` is a browser page for trying all of the above (file upload, microphone recording, live streaming). The
microphone needs a secure page (HTTPS, or http://localhost).
