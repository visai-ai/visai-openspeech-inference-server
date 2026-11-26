"""Tokenizer (SentencePiece) decoding for the runtime: the model's own decoding rule: keep byte-fallback and
text-piece ids, drop specials and timestamps, decode the rest in one SentencePiece call."""
from __future__ import annotations

import os

import sentencepiece as spm


class Detokenizer:
    def __init__(self, model_dir: str, spec):
        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, spec.vocab.tokenizer))
        self.text0 = spec.vocab.text0

    def keep(self, ids) -> list[int]:
        return [int(i) for i in ids if int(i) >= 256 and not (512 <= int(i) < self.text0)]

    def decode(self, ids) -> str:
        return self.sp.decode(self.keep(ids))
