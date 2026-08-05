#!/usr/bin/env python3
"""Сборка корпуса стойки M1: data/{corpus_train,corpus_valid}.txt + data/tokenizer_char.json.

Детерминированно: train и valid генерируются с непересекающимися seed'ами;
пересечение текстов контролируется явно (должно быть 0 — иначе ошибка).

Состав:
  train = истории(seed 1000) + факты(seed 2000)          — обучение языковой модели
  valid = истории(seed 9000) + факты(seed 9100)          — held-out (ppl)
  ситуации (M7-батарея) генерируются отдельным seed'ом по требованию (--situations N)

Запуск: python3 scripts/build_corpus.py [--stories-train 30000] [--stories-valid 3000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aira.corpus import make_facts, make_situations, make_stories  # noqa: E402
from aira.tokenizer import CharTokenizer  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def fact_line(f: dict) -> str:
    return f"{f['fact']} Вопрос: {f['question']} Ответ: {f['answer']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories-train", type=int, default=30000)
    ap.add_argument("--facts-train", type=int, default=6000)
    ap.add_argument("--stories-valid", type=int, default=3000)
    ap.add_argument("--facts-valid", type=int, default=600)
    ap.add_argument("--situations", type=int, default=0, help="сгенерировать M7-батарею (data/situations.jsonl)")
    args = ap.parse_args()

    train = make_stories(args.stories_train, seed=1000)
    train += [fact_line(f) for f in make_facts(args.facts_train, seed=2000)]
    valid = make_stories(args.stories_valid, seed=9000)
    valid += [fact_line(f) for f in make_facts(args.facts_valid, seed=9100)]

    # Точный контроль утечки: valid не должен содержать ни одного текста из train
    # (в малом шаблонном мире коллизии неизбежны — фильтруем, а не надеемся на seed'ы).
    train_set = set(train)
    n_valid_raw = len(valid)
    seen, valid_clean = set(), []
    for t in valid:
        if t not in train_set and t not in seen:
            seen.add(t)
            valid_clean.append(t)
    valid = valid_clean
    n_leak = n_valid_raw - len(valid)

    DATA.mkdir(exist_ok=True)
    (DATA / "corpus_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (DATA / "corpus_valid.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
    tok = CharTokenizer.from_texts(train)
    tok.save(DATA / "tokenizer_char.json")

    stats = {
        "train_texts": len(train), "valid_texts": len(valid),
        "train_chars": sum(len(t) for t in train), "valid_chars": sum(len(t) for t in valid),
        "vocab_chars": tok.vocab_size, "valid_dropped_as_overlap": n_leak,
        "seeds": {"train_stories": 1000, "train_facts": 2000,
                  "valid_stories": 9000, "valid_facts": 9100},
    }
    (DATA / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    if args.situations:
        with (DATA / "situations.jsonl").open("w", encoding="utf-8") as fh:
            for s in make_situations(args.situations, seed=7000):
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
