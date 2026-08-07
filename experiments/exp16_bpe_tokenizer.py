#!/usr/bin/env python3
"""EXP-16 «Вторая стойка: свой BPE-16k» — приёмка ADR-01, фаза 1 (словарь и метрики).

КОНТЕКСТ. ADR-01 принят: свой byte-level word-BPE-16384 (src/aira/bpe.py),
обученный на НАШИХ корпусах. Критерии приёма (ADR-01 §критерии):
  1. компрессия ≥ 3.2 символа/токена на held-out (обе доменные зоны);
  2. UNK < 0.1% токенов на held-out;
  3. roundtrip decode(encode(x)) == x на выборке;
  4. словарь воспроизводим (фиксированные данные, детерминированный алгоритм).
Фаза 2 (после закрытия EXP-14): пара лестницы PC-vs-BP на BPE-ids и рынок
триггер-полки L0 на BPE-ids (≥25% при точности не хуже модельной) — отдельный
запуск; этот скрипт готовит артефакт (словарь) и снимает метрики фазы 1.

Данные обучения: data/corpus_train.txt (M1, ~14 МБ) + corpus_external/wikitext2/
train.txt (~11 МБ). Обучение томить --train-mb при дефиците RAM (стенд 3.9 ГБ).

Запуск: .venv/bin/python experiments/exp16_bpe_tokenizer.py
            [--vocab 16384] [--train-mb 25] [--tag bpe16k]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np  # noqa: F401  (совместимость окружения)

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))

from aira.bpe import BPETokenizer, train_bpe  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def read_mb(path: Path, mb_budget: float) -> list[str]:
    """Читает файл в бюджете мегабайт, по строкам (не режем строку посередине)."""
    out, used = [], 0
    budget = int(mb_budget * 1_048_576)
    with path.open(encoding="utf-8", errors="replace") as f:
        for ln in f:
            if used >= budget:
                break
            out.append(ln.rstrip("\n"))
            used += len(ln.encode("utf-8", errors="replace"))
    return out


def metrics(tok: BPETokenizer, text: str, name: str) -> dict:
    ids = tok.encode(text)
    unk_id = tok.vocab["�"]
    n_ids = len(ids)
    n_chars = len(text.encode("utf-8"))
    n_unk = int(np.count_nonzero(ids == unk_id))
    return {"domain": name, "chars": n_chars, "tokens": n_ids,
            "chars_per_token": round(n_chars / max(n_ids, 1), 3),
            "unk_rate": round(n_unk / max(n_ids, 1), 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--train-mb", type=float, default=25.0,
                    help="суммарный бюджет обучающего текста (M1 первый)")
    ap.add_argument("--tag", default="bpe16k")
    ap.add_argument("--glue", type=int, default=0,
                    help="EXP-16a: ведущий пробел приклеен к слову (SentencePiece-конвенция)")
    args = ap.parse_args()

    m1 = read_mb(ROOT / "data" / "corpus_train.txt", args.train_mb * 14 / 25)
    wiki = read_mb(ROOT / "corpus_external" / "wikitext2" / "train.txt",
                   args.train_mb * 11 / 25)
    print(f"[exp16] обучение: M1 {len(m1)} строк + wiki {len(wiki)} строк, "
          f"словарь {args.vocab}", flush=True)
    tok = train_bpe(m1 + wiki, vocab_size=args.vocab, min_count=2, verbose=True,
                    glue=bool(args.glue))
    out_tok = RESULTS / f"bpe_tokenizer_{args.tag}.json"
    tok.save(out_tok)
    print(f"[exp16] словарь сохранён -> {out_tok} "
          f"({out_tok.stat().st_size / 1e6:.2f} МБ json)", flush=True)

    # --- приёмка ADR-01 --------------------------------------------------
    rec = {"vocab_size": len(tok.vocab), "n_merges": len(tok.merges),
           "train_budget_mb": args.train_mb, "tag": args.tag, "glue": bool(args.glue)}
    probes = {
        "M1_valid": (ROOT / "data" / "corpus_valid.txt", 1.0),
        "wiki_valid": (ROOT / "corpus_external" / "wikitext2" / "valid.txt", 1.0),
        "ruslit": (ROOT / "corpus_external" / "russian_lit" / "voina_i_mir.txt", 0.5),
    }
    rec["metrics"] = []
    for name, (path, mb) in probes.items():
        if not path.exists():
            print(f"[exp16] {name}: файла нет, пропуск", flush=True)
            continue
        text = "\n".join(read_mb(path, mb))
        m = metrics(tok, text, name)
        rec["metrics"].append(m)
        print(f"[exp16] {name}: {m['chars_per_token']} симв/токен, "
              f"UNK {m['unk_rate'] * 100:.4f}%", flush=True)

    # roundtrip
    t0 = time.perf_counter()
    ok, bad = 0, 0
    for ln in (m1[:200] + wiki[:200]):
        back = tok.decode(tok.encode(ln))
        if back == ln:
            ok += 1
        else:
            bad += 1
    rec["roundtrip"] = {"ok": ok, "bad": bad, "s": round(time.perf_counter() - t0, 1)}
    print(f"[exp16] roundtrip: {ok} ок / {bad} битых ({rec['roundtrip']['s']} с)",
          flush=True)

    # верхушка словаря (аудит «смысловых кусков»)
    long_pieces = sorted((p for p in tok.vocab if len(p) >= 6
                          and p not in ("␣", "␤", "�")), key=len, reverse=True)[:30]
    rec["longest_pieces"] = [p.encode("latin-1").decode("utf-8", errors="replace")
                             for p in long_pieces]
    print(f"[exp16] длиннейшие куски: {rec['longest_pieces'][:10]}", flush=True)

    fp = RESULTS / "results_exp16.json"
    out = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    out[f"phase1_{args.tag}"] = rec
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved -> results_exp16.json", flush=True)


if __name__ == "__main__":
    main()
