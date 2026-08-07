#!/usr/bin/env python3
"""Свой byte-level word-BPE (ADR-01): словарь наш, алгоритм открытый (Sennrich 2016).

Дизайн-канон: коммодити-инструмент без чужих зависимостей и чужих словарей.
Слова слоятся по whitespace (пробел/перевод строки — отдельные токены «␣»/«␤»,
слияния только внутри слов): объём работы ∝ словарю уникальных слов, не корпусу —
на стендовых 8–14М символов обучение минутами на CPU.

API:
  tok = train_bpe(texts, vocab_size=16384, min_count=2, verbose=True)
  ids = tok.encode(line)  -> np.ndarray int32
  text = tok.decode(ids)  -> str
  tok.save(path) / BPETokenizer.load(path)
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPACE, NL, UNK = "␣", "␤", "�"
_WORD_RE = re.compile(r"[^\s]+|\s")


def _preseq(word: str) -> tuple[str, ...]:
    """Слово → кортеж символьных байтов (latin-1: каждому байту один chr)."""
    return tuple(word.encode("utf-8").decode("latin-1"))


class BPETokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]],
                 glue: bool = False):
        self.vocab = vocab                       # piece(str) -> id
        self.inv_vocab = {i: p for p, i in vocab.items()}
        self.merges = merges
        self.ranks = {p: r for r, p in enumerate(merges)}
        self.glue = glue                         # EXP-16a: ведущий пробел приклеен к слову

    # ------------------------------------------------------------------ core
    def _apply(self, seq: tuple[str, ...]) -> tuple[str, ...]:
        if len(seq) < 2:
            return seq
        while True:
            best, best_rank = None, len(self.merges) + 1
            for i in range(len(seq) - 1):
                r = self.ranks.get((seq[i], seq[i + 1]))
                if r is not None and r < best_rank:
                    best, best_rank = i, r
            if best is None:
                return seq
            seq = (seq[:best] + (seq[best] + seq[best + 1],) + seq[best + 2:])
            if len(seq) < 2:
                return seq

    def encode(self, text: str) -> np.ndarray:
        out = []
        lead_space = False                     # glue: пробел ждёт своё слово
        for m in _WORD_RE.finditer(text):
            w = m.group(0)
            if w == " ":
                if self.glue:
                    if lead_space:             # второй пробел подряд: первый — сирота
                        out.append(self.vocab[SPACE])
                    lead_space = True
                else:
                    out.append(self.vocab[SPACE])
                continue
            if w == "\n":
                if lead_space:                 # пробел перед переводом строки — сирота
                    out.append(self.vocab[SPACE])
                lead_space = False
                out.append(self.vocab[NL]); continue
            unit = " " + w if (self.glue and lead_space) else w
            lead_space = False
            piece = self._apply(_preseq(unit))
            out.extend(self.vocab.get(x, self.vocab[UNK]) for x in piece)
        if lead_space:
            out.append(self.vocab[SPACE])
        return np.asarray(out, dtype=np.int32)


    def decode(self, ids) -> str:
        joined = "".join(self.inv_vocab[int(i)] for i in ids
                        if int(i) in self.inv_vocab and self.inv_vocab[int(i)] not in (UNK,))
        s = "".join(p for p in joined)
        s = s.replace(SPACE, " ").replace(NL, "\n")
        raw = s.encode("latin-1", errors="replace")
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"vocab": self.vocab, "glue": self.glue,
             "merges": [[a, b] for a, b in self.merges]}, ensure_ascii=False),
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], [tuple(m) for m in d["merges"]],
                   glue=bool(d.get("glue", False)))


def train_bpe(texts: list[str], vocab_size: int = 16384, min_count: int = 2,
              verbose: bool = True, tlog: int = 1000, glue: bool = False) -> BPETokenizer:
    t0 = time.perf_counter()
    words: Counter[tuple[str, ...]] = Counter()
    for ln in texts:
        ln = ln.rstrip("\n")
        lead = False                            # glue: ведущий пробел приклеиваем к слову
        for m in _WORD_RE.finditer(ln):
            w = m.group(0)
            if w in (" ", "\n"):
                lead = glue and w == " "
                continue
            unit = " " + w if lead else w
            lead = False
            if len(unit) > 64:                  # защита от не-слов снаружи
                for i in range(0, len(unit), 64):
                    words[_preseq(unit[i:i + 64])] += 1
            else:
                words[_preseq(unit)] += 1
    if verbose:
        print(f"[bpe] уникальных слов: {len(words)} за {time.perf_counter()-t0:.0f} с",
              flush=True)

    # стартовый словарь: все байты + спецтокены
    vocab: dict[str, int] = {}
    for seq in words:
        for p in seq:
            vocab.setdefault(p, len(vocab))
    for sp in (SPACE, NL, UNK):
        vocab.setdefault(sp, len(vocab))
    merges: list[tuple[str, str]] = []
    pair_index: dict[tuple[str, str], set] = defaultdict(set)  # пара → множество слов-ключей (по id)
    wids = list(words.keys())
    counts = [words[w] for w in wids]
    paircnt: Counter[tuple[str, str]] = Counter()
    for i, seq in enumerate(wids):
        for a, b in zip(seq, seq[1:]):
            paircnt[(a, b)] += counts[i]
            pair_index[(a, b)].add(i)

    target = vocab_size - len(vocab)
    n_merge = 0
    while n_merge < target:
        pair, cnt = (paircnt.most_common(1)[0] if paircnt else (None, 0))
        if pair is None or cnt < min_count:
            break
        a, b = pair
        new = a + b
        affected = list(pair_index.pop(pair, set()))
        for i in affected:
            seq, c = wids[i], counts[i]
            outseq, j = [], 0
            while j < len(seq):
                if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                    outseq.append(new); j += 2
                else:
                    outseq.append(seq[j]); j += 1
            outseq = tuple(outseq)
            if outseq == seq:
                continue
            for x, y in zip(seq, seq[1:]):
                paircnt[(x, y)] -= c
                pair_index[(x, y)].discard(i)
            wids[i] = outseq
            for x, y in zip(outseq, outseq[1:]):
                paircnt[(x, y)] += c
                pair_index[(x, y)].add(i)
        merges.append((a, b))
        vocab[new] = len(vocab)
        paircnt.pop(pair, None)
        n_merge += 1
        if verbose and n_merge % tlog == 0:
            print(f"[bpe] merge {n_merge}/{target} «{a}+{b}» freq={cnt} "
                  f"({time.perf_counter()-t0:.0f} с)", flush=True)
    if verbose:
        print(f"[bpe] словарь {len(vocab)} за {time.perf_counter()-t0:.0f} с", flush=True)
    return BPETokenizer(vocab, merges, glue=glue)
