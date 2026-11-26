"""Word boost (contextual biasing): per-request phrase lists that bias decoding toward their spelling.

Opt-in and decoding-time only: the model and its graphs are unchanged, and a request without a list never enters
this module (its output is exactly the unbiased decoder's).

How a list biases decoding (shallow fusion):
- The phrases are compiled into a byte-level Aho-Corasick automaton over their UTF-8 text. Tokens are matched by
  the text they write (SentencePiece piece text, "▁" as a space, byte-fallback pieces as single bytes), so a
  phrase matches however the tokenizer splits it, and Thai / Chinese phrases need no word boundaries.
- Each decoding hypothesis carries a match state. A token that continues a phrase earns a bonus proportional to the
  characters it adds (weight x scale x characters); when a partial match breaks, every bonus it earned is taken
  back; a completed phrase keeps its bonus. A hypothesis that ends inside a partial match loses that partial bonus.
- Root-neutral: the token that starts a match earns nothing, so the model itself has to choose a phrase's first
  token, and a list with no hits leaves the output unchanged. (A phrase that is one token is therefore never
  boosted.)
- Each phrase is boosted at most `max_hits` times per hypothesis (a further occurrence earns nothing), so a boost
  can never pay for a phrase repeated over and over.
- Boosts re-rank real text tokens only: blank (CTC), <|eot|> and other special tokens never receive a bonus. In
  greedy decoding an <|eot|> the model ranked first stays first (a beam lets hypotheses compete on fused scores); in
  the CTC beam search a phrase's bonus can win over a blank the model preferred (that is how a boosted phrase gets
  written, and how false insertions happen). Only boosted completions count as hits.
- A search that completes no listed phrase in a window (offline) returns the default output for that window.

Weights are unitless: a phrase's `boost` (default 1, at most 2) times the request's `boost` scale (default 1, at most
2) times the engine's per-character bonus (BONUS_PER_CHAR, nats). The engine values are provisional defaults; tune
them on development data, never on test sets.
"""
from __future__ import annotations

import hashlib
import json
import threading
import unicodedata
from collections import OrderedDict

import numpy as np

BONUS_PER_CHAR = 0.5          # nats of log-probability per matched character at weight 1 (provisional default)
MAX_PHRASES = 500             # per request (files); live streams use MAX_PHRASES_STREAM
MAX_PHRASES_STREAM = 100
MAX_CHARS = 50                # per phrase, after normalisation
MAX_WEIGHT = 2.0
MAX_HITS = 3                  # boosted occurrences of one phrase per hypothesis
CTC_BEAM_DEFAULT = 4
CTC_BEAM_MAX = 8


class BoostError(ValueError):
    """An invalid boost list or option; the message is safe to return to the client."""


def normalise(text: str) -> str:
    """NFC, outer whitespace stripped, inner whitespace runs as one space."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def parse_list(raw, max_phrases: int = MAX_PHRASES) -> list[tuple[str, float]]:
    """A request's word_boost value -> [(phrase, weight)], deduplicated (a repeated phrase keeps its largest weight).
    Accepts a list whose items are strings or {"phrase": str, "boost": number}; BoostError on anything else, an empty
    or over-long phrase, a weight outside (0, MAX_WEIGHT], or more than max_phrases phrases."""
    if not isinstance(raw, list):
        raise BoostError("word_boost must be a list of phrases")
    if len(raw) > max_phrases:
        raise BoostError(f"word_boost takes at most {max_phrases} phrases (got {len(raw)})")
    out: dict = {}
    for k, item in enumerate(raw):
        w = 1.0
        if isinstance(item, dict):
            unknown = set(item) - {"phrase", "boost"}
            if unknown:
                raise BoostError(f"word_boost[{k}]: unknown keys {sorted(unknown)} (phrase, boost)")
            p, w = item.get("phrase"), item.get("boost", 1.0)
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not (0.0 < float(w) <= MAX_WEIGHT) \
                    or w != w:
                raise BoostError(f"word_boost[{k}].boost must be a number in (0, {MAX_WEIGHT:g}]")
            w = float(w)
        else:
            p = item
        if not isinstance(p, str):
            raise BoostError(f"word_boost[{k}] must be a string or {{\"phrase\": ..., \"boost\": ...}}")
        if any(unicodedata.category(c) in ("Cc", "Cs") for c in p):
            raise BoostError(f"word_boost[{k}] contains control characters")
        p = normalise(p)
        if not p:
            raise BoostError(f"word_boost[{k}] is empty")
        if len(p) > MAX_CHARS:
            raise BoostError(f"word_boost[{k}] is longer than {MAX_CHARS} characters")
        out[p] = max(w, out.get(p, 0.0))
    return sorted(out.items())               # canonical order: the same list in any order compiles (and caches) once


def parse_scale(v) -> float:
    """The request's boost scale: a number in (0, MAX_WEIGHT] (strings accepted for query parameters)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise BoostError(f"boost must be a number in (0, {MAX_WEIGHT:g}]") from None
    if isinstance(v, bool) or not (0.0 < x <= MAX_WEIGHT) or x != x:
        raise BoostError(f"boost must be a number in (0, {MAX_WEIGHT:g}]")
    return x


class TokenText:
    """Bytes each vocabulary id writes: text pieces ("▁" -> space), byte-fallback pieces (one byte each);
    None for special / timestamp / unused ids (they write nothing and never take part in a match)."""

    def __init__(self, sp, text0: int, size: int):
        self.table: list = [None] * size
        for i in range(min(size, sp.get_piece_size())):
            if 256 <= i < 512:
                self.table[i] = bytes([i - 256])
            elif i >= text0 and not (sp.is_control(i) or sp.is_unknown(i)):
                piece = sp.id_to_piece(i)
                if sp.is_byte(i):
                    self.table[i] = bytes([int(piece[3:5], 16)])
                else:
                    self.table[i] = piece.replace("▁", " ").encode("utf-8")

    def __getitem__(self, i: int):
        return self.table[i] if 0 <= i < len(self.table) else None


class Automaton:
    """Byte-level Aho-Corasick automaton of one phrase list (immutable once built; shared by every hypothesis)."""

    def __init__(self, phrases: list[tuple[str, float]], bonus_per_char: float):
        self.phrases = [p for p, _ in phrases]
        goto = [dict()]                      # node -> {byte: node}
        chars = [0]                          # characters matched at the node
        weight = [0.0]                       # largest weight of the phrases through the node
        through = [0]                        # bitmask of the phrase ids whose spelling passes through the node
        term = [-1]                          # phrase id ending at the node (-1: none)
        term_w = [0.0]                       # weight of the phrase ending at the node
        for pid, (p, w) in enumerate(phrases):
            node, n = 0, 0
            for b in p.encode("utf-8"):
                if (b & 0xC0) != 0x80:      # a UTF-8 lead byte starts a character
                    n += 1
                nxt = goto[node].get(b)
                if nxt is None:
                    nxt = len(goto)
                    goto.append(dict())
                    chars.append(n)
                    weight.append(0.0)
                    through.append(0)
                    term.append(-1)
                    term_w.append(0.0)
                    goto[node][b] = nxt
                node = nxt
                weight[node] = max(weight[node], w)
                through[node] |= 1 << pid
            term[node] = pid
            term_w[node] = w
        fail = [0] * len(goto)
        order = list(goto[0].values())
        k = 0
        while k < len(order):                # breadth-first: a node's failure link is set before its children
            u = order[k]
            k += 1
            for b, v in goto[u].items():
                f = fail[u]
                while f and b not in goto[f]:
                    f = fail[f]
                fail[v] = goto[f].get(b, 0) if goto[f].get(b, 0) != v else 0
                order.append(v)
        self.goto, self.fail, self.term, self.through = goto, fail, term, through
        self.chars = chars
        self.rate = [bonus_per_char * w for w in weight]   # bonus per character at the node (its heaviest phrase)
        self.term_rate = [bonus_per_char * w for w in term_w]   # ... of the phrase completing at the node
        self.size = len(goto)


ROOT = (0, 0, 0.0, ())       # (node, characters matched when the open match's credit started, open credit,
                             #  ((phrase, hits), ...))


class Boost:
    """One request's compiled list plus its scale: the per-hypothesis transition function and bonus."""

    def __init__(self, auto: Automaton, scale: float, max_hits: int, key: str):
        self.a, self.scale, self.max_hits, self.key = auto, scale, max_hits, key

    def __repr__(self):
        return f"Boost({self.key[:12]})"

    def __hash__(self):
        return hash((self.key, self.scale))

    def __eq__(self, other):
        return isinstance(other, Boost) and (self.key, self.scale) == (other.key, other.scale)

    def _alive(self, node: int, hits: tuple) -> bool:
        """Some phrase through the node can still be boosted (it has not reached max_hits)."""
        thr = self.a.through[node]
        if not hits:
            return thr != 0
        capped = 0
        for p, n in hits:
            if n >= self.max_hits:
                capped |= 1 << p
        return (thr & ~capped) != 0

    def step(self, state: tuple, tb: bytes | None) -> tuple[tuple, float]:
        """(state, bonus) after a token that writes `tb` (None: a special token: it ends an open match and gives its
        credit back; at the root it changes nothing).
        bonus = (credit committed in this token) + (new open credit) - (old open credit), times the scale. The open
        credit is rate(node) x (characters matched since the credit started), never negative; a match that starts in
        this token (from the root, after a broken match or after a completed phrase) earns nothing in this token."""
        if tb is None:                       # a special / timestamp token: it writes no text, so it ends an open
            if state[0] == 0 and state[2] == 0.0:   # match and gives its credit back (a match never spans one,
                return state, 0.0                   # and it gains nothing over the text it displaces)
            return (0, 0, 0.0, state[3]), -self.scale * state[2]
        a = self.a
        node, base, credit, hits = state
        if node == 0:
            tb = tb.lstrip(b" ")             # a word boundary before a phrase is not part of it
        fresh = node == 0                    # the open match started in this token: it earns nothing yet
        committed = 0.0
        open_credit = credit

        def credit_at(n, base_chars):
            if fresh or not self._alive(n, hits):
                return 0.0
            return max(0.0, a.rate[n] * (a.chars[n] - base_chars))
        for b in tb:
            nxt = a.goto[node].get(b)
            if nxt is None:                  # the match breaks: its credit is lost; a suffix may start a new one
                f = node
                while f and b not in a.goto[f]:
                    f = a.fail[f]
                node = a.goto[f].get(b, 0)
                fresh, open_credit = True, 0.0
                base = a.chars[node]
            else:
                node = nxt
                open_credit = credit_at(node, base)
            pid = a.term[node]
            if pid >= 0:                     # a phrase completes here
                n = dict(hits).get(pid, 0)
                gain = a.term_rate[node] * (a.chars[node] - base)
                if n < self.max_hits and not fresh and gain > 0.0:
                    # a boosted completion: it keeps its credit at its own weight (not a longer phrase's), and only
                    # boosted completions (a positive bonus) count as hits (the cap, and the "a listed phrase was
                    # written" flag); counting restarts after it for a longer phrase through the same node
                    committed += gain
                    hits = tuple(sorted({**dict(hits), pid: n + 1}.items()))
                    open_credit, base = 0.0, a.chars[node]
                # otherwise (written unaided, capped, or no gain): the open match toward a longer phrase keeps its
                # credit
                if not a.goto[node]:         # nothing longer goes through here: back to the root, and what
                    node, base, fresh, open_credit = 0, 0, True, 0.0   # follows in this token starts a new match
        if fresh:                            # a match that began in this token: start counting from here
            base = a.chars[node] if node else 0
            open_credit = 0.0
        return (node, base, open_credit, hits), self.scale * (committed + open_credit - credit)

    def finish(self, state: tuple) -> float:
        """Bonus correction when a hypothesis ends: an open (unfinished) match loses its credit."""
        return -self.scale * state[2]


class BoostCache:
    """Compiled automata by list digest (LRU, shared by every engine), bounded by the total number of automaton nodes,
    so a list repeated across requests / streams compiles once and the cache cannot grow past max_nodes."""

    def __init__(self, max_nodes: int = 100_000):
        self.max_nodes, self.lock, self.d, self.nodes = max_nodes, threading.Lock(), OrderedDict(), 0
        self.compile_lock = threading.Lock()

    def get(self, phrases: list[tuple[str, float]], bonus_per_char: float) -> tuple[Automaton, str]:
        key = hashlib.sha256(json.dumps([phrases, bonus_per_char], ensure_ascii=False).encode()).hexdigest()
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key], key
        with self.compile_lock:              # one compilation at a time (no stampede); re-check after waiting, and
            with self.lock:                  # insert before releasing, so a key is compiled once
                if key in self.d:
                    self.d.move_to_end(key)
                    return self.d[key], key
            auto = Automaton(phrases, bonus_per_char)
            with self.lock:
                self.d[key] = auto
                self.nodes += auto.size
                while self.nodes > self.max_nodes and len(self.d) > 1:
                    _, old = self.d.popitem(last=False)
                    self.nodes -= old.size
        return auto, key


CACHE = BoostCache()


# ------------------------------------------------------------------ CTC decoding with a boost list

def ctc_greedy_frame(lp: np.ndarray, prev: int, state: tuple, row_ids, text: TokenText, boost: Boost,
                     blank: int = 0, topk: int = 8):
    """One frame of greedy CTC with boost re-selection -> (row, new state, emitted). The raw best row stands when
    it is blank or a repeat of the previous frame's row (no new token); otherwise the bonus re-ranks the frame's
    top-k non-blank rows (never blank, never the repeat)."""
    r = int(np.argmax(lp))
    if r == blank or r == prev:
        return r, state, False
    k = min(topk, lp.size)
    best, best_s, best_state = r, -np.inf, None
    for c in np.argpartition(-lp, k - 1)[:k]:     # the frame's top-k rows (blank and the repeat skipped)
        c = int(c)
        if c == blank or c == prev:
            continue
        st2, bonus = boost.step(state, text[int(row_ids[c])])
        s = float(lp[c]) + bonus
        if s > best_s:
            best, best_s, best_state = c, s, st2
    if best_state is None:
        best_state, _ = boost.step(state, text[int(row_ids[best])])
    return best, best_state, True


def ctc_greedy_boost(logp: np.ndarray, row_ids, text: TokenText, boost: Boost, blank: int = 0,
                     topk: int = 8) -> tuple[list[int], bool]:
    """Greedy CTC with boost re-selection over a whole utterance (ctc_greedy_frame per frame). logp (T, R) CTC
    log-probs over the head rows (row 0 blank); row_ids[r] = vocabulary id of row r. Returns (vocabulary ids,
    whether a listed phrase was completed)."""
    out, state, prev = [], ROOT, blank
    for t in range(logp.shape[0]):
        r, state, emitted = ctc_greedy_frame(logp[t], prev, state, row_ids, text, boost, blank, topk)
        if emitted:
            out.append(int(row_ids[r]))
        prev = r
    return out, bool(state[3])


def _lse(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = max(a, b)
    return m + float(np.log1p(np.exp(-abs(a - b))))


def ctc_prefix_beam_boost(logp: np.ndarray, row_ids, text: TokenText, boost: Boost, beam: int = CTC_BEAM_DEFAULT,
                          blank: int = 0, topk: int = 8, prune: float = 10.0) -> tuple[list[int], bool]:
    """CTC prefix beam search with boost shallow fusion (offline). A prefix's bonus is earned when the prefix gains a
    token (never on blank or repeat frames). Ranking: logsumexp(p_blank, p_nonblank) + boost credit. Returns
    (vocabulary ids, whether the best prefix completed a listed phrase)."""
    beams = {(): (0.0, -np.inf, ROOT, 0.0)}      # prefix -> (p_b, p_nb, boost state, accumulated bonus)
    for t in range(logp.shape[0]):
        beams = ctc_beam_frame(beams, logp[t], row_ids, text, boost, beam, blank, topk, prune)
    return ctc_beam_best(beams, row_ids, boost)


def ctc_beam_best(beams: dict, row_ids, boost: Boost) -> tuple[list[int], bool]:
    """(vocabulary ids of the best prefix, whether it completed a listed phrase)."""
    best = max(beams.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][3] + boost.finish(kv[1][2]))
    return [int(row_ids[r]) for r in best[0]], bool(best[1][2][3])


def ctc_beam_frame(beams: dict, lp: np.ndarray, row_ids, text: TokenText, boost: Boost, beam: int, blank: int = 0,
                   topk: int = 8, prune: float = 10.0, last0: int | None = None) -> dict:
    """One frame of the prefix beam search (shared by the offline and streaming decoders). Prefixes are tuples of
    CTC head rows; a stream keeps them relative to what it has committed, with last0 the last committed row."""
    top = float(np.max(lp))
    k = min(topk, lp.size)
    cand = [int(c) for c in np.argpartition(-lp, k - 1)[:k] if c != blank and lp[c] >= top - prune]
    nxt: dict = {}

    def add(prefix, pb, pnb, state, acc):
        o = nxt.get(prefix)
        if o is None:
            nxt[prefix] = (pb, pnb, state, acc)
        else:
            nxt[prefix] = (_lse(o[0], pb), _lse(o[1], pnb), o[2], o[3])
    lb = float(lp[blank])
    for prefix, (pb, pnb, state, acc) in beams.items():
        tot = _lse(pb, pnb)
        add(prefix, tot + lb, -np.inf, state, acc)                      # blank
        last = prefix[-1] if prefix else last0
        if last is not None:
            add(prefix, -np.inf, pnb + float(lp[last]), state, acc)     # repeat of the last row, no new token
        for c in cand:
            p_from = pb if c == last else tot          # a repeated row is a new token only after a blank
            if p_from == -np.inf:
                continue
            st2, bonus = boost.step(state, text[int(row_ids[c])])
            add(prefix + (c,), -np.inf, p_from + float(lp[c]), st2, acc + bonus)
    ranked = sorted(nxt.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][3], reverse=True)
    return dict(ranked[:beam])


def common_prefix(beams: dict) -> tuple:
    """The rows every live prefix agrees on (what a stream can commit without ever revising it)."""
    ps = list(beams)
    if not ps:
        return ()
    n = min(len(p) for p in ps)
    k = 0
    while k < n and all(p[k] == ps[0][k] for p in ps):
        k += 1
    return ps[0][:k]
