"""Model spec: everything the runtime needs to know about one model package, as one JSON file (<model dir>/spec.json).
The runtime reads only this file and the files it names next to it (ONNX graphs, tokenizer, mel statistics).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class FrontEndSpec:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    preemph: float = 0.97
    log_guard: float = 2.0 ** -24
    mel_norm: str = "slaney"
    mel_scale: str = "slaney"
    stats: str = ""                 # path (relative to the model dir) of the {"mean", "std"} JSON; "" = identity


@dataclass
class EncoderSpec:
    kind: str = "fastconformer"     # "fastconformer" (full context) | "fastconformer_chunked" (streaming: also has
                                    # encoder_chunk_c<chunk> graphs, stream.py)
    d_model: int = 768
    n_layers: int = 18
    n_heads: int = 12
    ffn_hidden: int = 2048
    ffn_sub_ln: bool = True
    conv_kernel: int = 9
    sub_channels: int = 256
    subsampling: int = 8            # dw_striding, 3 stride-2 stages
    xscaling: bool = True
    rope_base: float = 10000.0
    qk_norm: bool = True
    # streaming (kind fastconformer_chunked): chunk and left context in encoder frames (80 ms each at 8x)
    chunk_frames: int = 0           # e.g. 6-7 frames ~ 520 ms
    left_context_frames: int = 0
    right_context_frames: int = 0
    # fastconformer_chunked (streaming packages): the chunk sizes in encoder frames; look-ahead L streams with chunk
    # L + 1 (e.g. L in {0, 4, 13} -> chunks 1, 5, 14)
    chunk_sizes: list = field(default_factory=list)


@dataclass
class DecoderSpec:
    n_layers: int = 6
    d_model: int = 768
    n_heads: int = 12
    ffn: int = 3072
    qk_norm: bool = True
    max_prompt_pos: int = 64
    max_text_pos: int = 512
    tie_embeddings: bool = True


@dataclass
class VocabSpec:
    size: int = 21885               # the tokenizer's true size
    padded: int = 21888             # embedding / output rows
    tokenizer: str = "tokenizer.model"   # SentencePiece file, relative to the model dir
    text0: int = 888                # first text-piece id (bytes are 256..511, timestamps 512..text0-1)
    blank: int = 0
    pad: int = 1
    eot: int = 0                    # <|eot|> id (from the tokenizer's ids)
    ids: dict = field(default_factory=dict)   # special-token name -> id
    never: list = field(default_factory=list)  # decoder never-emit ids (blank, pad, padded rows)
    ctc_ids: list = field(default_factory=list)  # full-vocab id of each CTC head row (row 0 = blank)


@dataclass
class ModelSpec:
    name: str = ""
    source_ckpt: str = ""
    weights: str = "ema"
    frontend: FrontEndSpec = field(default_factory=FrontEndSpec)
    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    decoder: DecoderSpec = field(default_factory=DecoderSpec)
    vocab: VocabSpec = field(default_factory=VocabSpec)
    graphs: dict = field(default_factory=dict)  # "<role>.<precision>" -> onnx file name (relative to the model dir)
    # streaming decoder packages: {"format": "in" | "out", "wait_id": <id>, "lookaheads": [...]}; empty otherwise
    # (and then left out of spec.json)
    stream_dec: dict = field(default_factory=dict)

    @property
    def frames_per_second(self) -> float:
        return self.frontend.sample_rate / self.frontend.hop_length / self.encoder.subsampling

    def to_json(self) -> str:
        d = asdict(self)
        if not d["encoder"]["chunk_sizes"]:      # offline specs stay readable by engines older than the field
            del d["encoder"]["chunk_sizes"]
        if not d["stream_dec"]:
            del d["stream_dec"]
        return json.dumps(d, indent=1, ensure_ascii=False)

    def save(self, path: str) -> None:
        import os
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        os.replace(tmp, path)              # atomic: a reader never sees a half-written spec

    @classmethod
    def load(cls, path: str) -> "ModelSpec":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(name=d.get("name", ""), source_ckpt=d.get("source_ckpt", ""), weights=d.get("weights", "ema"),
                   frontend=FrontEndSpec(**d.get("frontend", {})), encoder=EncoderSpec(**d.get("encoder", {})),
                   decoder=DecoderSpec(**d.get("decoder", {})), vocab=VocabSpec(**d.get("vocab", {})),
                   graphs=d.get("graphs", {}), stream_dec=dict(d.get("stream_dec") or {}))

    @property
    def ts0(self) -> int:
        """Id of <|t0|> (timestamp tokens <|t0|>.. <|t375|> = 80 ms steps from the window start, ids ts0 .. text0 - 1)."""
        return int(self.vocab.ids.get("<|t0|>", 512))

    def prompt(self, lang: str | None, itn: bool = False, pnc: bool = False, timestamps: bool = False) -> list[int]:
        """<|sot|> [<|lang|>] <|transcribe|> <|itn|noitn|> <|pnc|nopnc|> <|timestamps|notimestamps|>
        <|nospeakerchange|> (the model's prompt format); lang None or unknown: automatic language detection.
        <|speakerchange|> is never asked for."""
        ids = self.vocab.ids
        p = ["<|sot|>"] + ([f"<|{lang}|>"] if lang and f"<|{lang}|>" in ids else []) + [
            "<|transcribe|>", "<|itn|>" if itn else "<|noitn|>", "<|pnc|>" if pnc else "<|nopnc|>",
            "<|timestamps|>" if timestamps else "<|notimestamps|>", "<|nospeakerchange|>"]
        return [ids[t] for t in p]


# The vocabulary layout's fixed ids: 0 blank, specials 1-17, languages 18-36, 256-511 byte fallback,
# <|t0|>..<|t375|> = 512..887, text pieces from 888.
V1_SPECIALS = ["<|pad|>", "<|sot|>", "<|eot|>", "<|nospeech|>", "<|other|>", "<|timestamps|>", "<|notimestamps|>",
               "<|speakerchange|>", "<|nospeakerchange|>", "<|turn|>", "<|prev|>", "<|transcribe|>", "<|itn|>",
               "<|noitn|>", "<|pnc|>", "<|nopnc|>", "<|std|>"]
V1_LANGS = ["th", "en", "zh", "vi", "id", "km", "ms", "fil", "es", "fr", "pt", "hi", "bn", "ur", "ar", "ja", "ko", "my",
            "lo"]


def v1_ids() -> dict:
    ids = {t: 1 + k for k, t in enumerate(V1_SPECIALS)}
    ids.update({f"<|{lg}|>": 1 + len(V1_SPECIALS) + k for k, lg in enumerate(V1_LANGS)})
    ids.update({f"<|t{k}|>": 512 + k for k in range(376)})
    return ids
