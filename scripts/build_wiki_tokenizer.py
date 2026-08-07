#!/usr/bin/env python3
"""Токенизатор wiki64 для живого наряда (wikitext2 → алфавит 60 + 4 спец = VOCAB 64).

Зачем: M1-токенизатор (tokenizer_char.json) — кириллический; на wikitext2 он даёт
78.8% <unk> — живой текст превращается в суп из unk, модель учит мусор.
Для живого наряда H-33 (близнец на wikitext2) нужен английский алфавит того же
размера (VOCAB=64 — архитектура зоны и код полки c8 не меняются).

Детерминизм: топ-60 символов train-части по частоте, порядок = сортировка
(совпадение частот разрешается лексикографически, знакоместо не плавает).
Спецтокены фиксированы aira.tokenizer: <pad>=0 <bos>=1 <eos>=2 <unk>=3.

Запуск: .venv/bin/python scripts/build_wiki_tokenizer.py
Выход: data/tokenizer_wiki64.json + отчёт покрытия (train/valid).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aira.tokenizer import CharTokenizer  # noqa: E402

WIKI = ROOT / "corpus_external" / "wikitext2"
OUT = ROOT / "data" / "tokenizer_wiki64.json"
N_CHARS = 60  # + 4 спец = VOCAB 64 (архитектурный замок)


def main() -> None:
    train = (WIKI / "train.txt").read_text(encoding="utf-8")
    valid = (WIKI / "valid.txt").read_text(encoding="utf-8")
    cnt = collections.Counter(train)
    # детерминированный топ-N: частота убыв., при равенстве — символ возр.
    top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:N_CHARS]
    chars = sorted(c for c, _ in top)
    tok = CharTokenizer(chars)
    n_cov = sum(cnt[c] for c in chars)
    oov_tr = 1.0 - n_cov / len(train)
    oov_va = sum(n for c, n in collections.Counter(valid).items()
                 if c not in tok.stoi) / len(valid)
    OUT.parent.mkdir(exist_ok=True)
    tok.save(OUT)
    print(f"wiki64: {len(chars)} символов + 4 спец = vocab {tok.vocab_size}")
    print(f"покрытие train: {1 - oov_tr:.6f}  (OOV {oov_tr:.6f})")
    print(f"покрытие valid: {1 - oov_va:.6f}  (OOV {oov_va:.6f})")
    print(f"алфавит: {''.join(chars)!r}")
    print(f"→ {OUT.relative_to(ROOT)}")
    json.dump({"script": "build_wiki_tokenizer.py", "n_chars": N_CHARS,
               "oov_train": round(oov_tr, 6), "oov_valid": round(oov_va, 6)},
              open(ROOT / "data" / "tokenizer_wiki64_stats.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
