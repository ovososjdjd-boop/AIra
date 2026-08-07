#!/usr/bin/env python3
"""EXP-15b «Уточнение формы L0» — иерархическая триггерная полка (H-32, Принцип T).

КОНТЕКСТ. EXP-15: плоская полка k=8 на живом тексте (wikitext2) даёт фронтир
29.0%@0.950 / 67.5%@0.875 / 94.1%@0.824 — ворота покрытия ≥50% при строгой
точности НЕ взяты. Гипотеза уточнения формы: цена уверенности не в фиксированной
длине ключа, а в ИЕРАРХИИ уровней — спрашивать сначала длинные ключи (максимум
специфичности), при отсутствии уверенного ответа спускаться к коротким и к
словесным контекстам (семантический уровень, из EXP-06 следы). Короткий ключ
с перекошенной статистикой (счёт большой, распределение острое) не хуже длинного
редкого: уверенность решает, не длина.

СТЕНД. Уровни по умолчанию: char-ключи L∈{32,16,8,4} (6 бит/символ) + word-ключи
W∈{3,2,1} (20 бит/слово, interner хешей слов). Порядок опроса: 32 → w3 → 16 →
w2 → 8 → w1 → 4. Метрики: (а) сетка покрытие/точность по θ на M1-контроле и
wikitext2; (б) доля уровней в отвеченных позициях (кто окупает память);
(в) гибрид L0h→L1 с готовой зоной-96 wikitext2-чекпоинтом EXP-15.

ВОРОТА (приёмка уточнения формы): wikitext2-valid — покрытие ≥ 50% при
точности L0 ≥ 0.93 И ppl_гибрид ≤ ppl_чистой зоны (Δ ≤ 0). СМЕРТЬ: иерархия не
двигает фронтир заметно (покрытие при ≥0.93 растёт < +10 п.п. против плоской 29%).

Запуск:
  .venv/bin/python experiments/exp15b_hier_shelf.py --stage shelf            # M1 контроль
  .venv/bin/python experiments/exp15b_hier_shelf.py --stage wiki --valid-n 600000
  .venv/bin/python experiments/exp15b_hier_shelf.py --stage hybrid           # L0h→L1 (зона из ckpt)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from aira.tokenizer import CharTokenizer  # noqa: E402
from exp12_precond_aa import load_ids  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
V = 64
CHAR_BITS, WORD_BITS = 6, 20

# Тип ключа: ("c", L) — последние L символов; ("w", n) — последние n слов
LEVELS_DEFAULT = [("c", 32), ("w", 3), ("c", 16), ("w", 2), ("c", 8), ("w", 1), ("c", 4)]
THS = (0.30, 0.45, 0.60, 0.75, 0.90, 0.95)


# ------------------------------------------------------------------- полка
def _upd(tab: dict, key: int, sym: int) -> None:
    e = tab.get(key)
    if e is None:
        tab[key] = [sym, 1, -1, 0, 1]
        return
    e[4] += 1
    if sym == e[0]:
        e[1] += 1
    elif sym == e[2]:
        e[3] += 1
        if e[3] > e[1]:
            e[0], e[1], e[2], e[3] = e[2], e[3], e[0], e[1]
    else:
        if e[3] + 1 > e[1]:
            e[0], e[1], e[2], e[3] = sym, e[3] + 1, e[0], e[1]
        else:
            e[2], e[3] = sym, max(e[3], 1)


def _conf(e: list[int]) -> float:
    return (e[1] + 1) / (e[4] + V)


class HierShelf:
    """Иерархия полок char-ключей + word-ключей; онлайн-рост, запрос сверху вниз."""

    def __init__(self, levels: list[tuple[str, int]], m: int = 8) -> None:
        self.levels = levels
        self.m = m
        self.tabs: dict = {lv: {} for lv in levels}
        self.mask: dict = {lv: (1 << (CHAR_BITS * lv[1])) - 1
                           for lv in levels if lv[0] == "c"}
        self.ckeys: dict = {lv: 0 for lv in levels if lv[0] == "c"}
        self.word_ids: dict[tuple, int] = {}      # interner слов (кортеж char-ids -> wid)
        self.wmask = (1 << WORD_BITS) - 1
        self.whist: list[int] = []                 # wids последних слов (старое впереди)
        self.cur_word: list[int] = []

    def _push_word(self) -> None:
        if not self.cur_word:
            return
        t = tuple(self.cur_word)
        wid = self.word_ids.get(t)
        if wid is None:
            if len(self.word_ids) >= (1 << WORD_BITS) - 1:
                wid = hash(t) & self.wmask        # переполнение: грубый хеш (редко)
            else:
                wid = len(self.word_ids) + 1
                self.word_ids[t] = wid
        self.whist.append(wid)
        if len(self.whist) > 4:
            self.whist.pop(0)
        self.cur_word = []

    def _wkey(self, n: int) -> int | None:
        if len(self.whist) < n:
            return None
        k = 0
        for w in self.whist[-n:]:
            k = (k << WORD_BITS) | w
        return k

    def update_level_keys(self, sym: int, is_break: bool) -> None:
        for lv in self.ckeys:
            self.ckeys[lv] = ((self.ckeys[lv] << CHAR_BITS) | sym) & self.mask[lv]
        if is_break:                                 # пробел/перевод строки: слово кончилось
            self._push_word()
        else:
            self.cur_word.append(sym)

    def present(self) -> list[tuple[str, int]]:
        return self.levels

    def reset_stream(self) -> None:
        """Сброс потокового состояния (скользящие ключи/слова) при смене потока;
        ТАБЛИЦЫ и interner слов сохраняются — иначе ключи не сойдутся."""
        for lv in self.ckeys:
            self.ckeys[lv] = 0
        self.whist = []
        self.cur_word = []

    def key_of(self, lv) -> int | None:
        if lv[0] == "c":
            return self.ckeys[lv]
        return self._wkey(lv[1])

    def update(self, y: int) -> None:
        """Записать наблюдение «контекст → y» во ВСЕ уровни (кроме ещё пустых word-ключей)."""
        for lv in self.levels:
            k = self.key_of(lv)
            if k is not None:
                _upd(self.tabs[lv], k, y)

    def route(self, th: float):
        """Первый уровень сверху с уверенностью ≥ θ (и счётом ≥ m)."""
        for lv in self.levels:
            k = self.key_of(lv)
            if k is None:
                continue
            e = self.tabs[lv].get(k)
            if e is None or e[4] < self.m:
                continue
            c = _conf(e)
            if c >= th:
                return lv, e[0], c
        return None

    def stats(self) -> dict:
        out = {}
        for lv in self.levels:
            out[f"{lv[0]}{lv[1]}"] = len(self.tabs[lv])
        out["words"] = len(self.word_ids)
        return out


# ------------------------------------------------------------------- стадии


def build_shelf(sh: HierShelf, ids: np.ndarray, is_break: np.ndarray,
                min_ctx: int) -> None:
    """Онлайн-обучение полки на потоке (контекст к позиции i предсказывает ids[i])."""
    sh.reset_stream()
    for i, s in enumerate(ids):
        s = int(s)
        full = i >= min_ctx          # окна коротких ключей заполнены
        if full:
            sh.update(s)
        sh.update_level_keys(s, bool(is_break[s]))


def scan_shelf(sh: HierShelf, ids: np.ndarray, is_break: np.ndarray,
               min_ctx: int, tag: str) -> dict:
    sh.reset_stream()          # контексты valid начинаются с нуля, таблицы — из train
    t1 = time.perf_counter()
    n = len(ids) - min_ctx
    for i in range(min_ctx):
        sh.update_level_keys(int(ids[i]), bool(is_break[int(ids[i])]))
    lvl_names = [f"{lv[0]}{lv[1]}" for lv in sh.present()]
    grid = {th: {"hits": 0, "cover": 0,
                 "lvl": defaultdict(lambda: [0, 0])} for th in THS}
    for i in range(min_ctx, len(ids)):
        y = int(ids[i])
        for th in THS:
            r = sh.route(th)
            g = grid[th]
            if r is not None:
                lv, sym, _ = r
                g["cover"] += 1
                hit = int(sym == y)
                g["hits"] += hit
                rec = g["lvl"][f"{lv[0]}{lv[1]}"]
                rec[0] += hit
                rec[1] += 1
        # онлайн-рост как в EXP-15 (триггер учится на ходу)
        sh.update(y)
        sh.update_level_keys(y, bool(is_break[y]))
    out = {"n": n, "m": sh.m, "scan_s": round(time.perf_counter() - t1, 1),
           "sizes": sh.stats(), "grid": {}}
    for th in THS:
        g = grid[th]
        cov = g["cover"] / max(n, 1)
        acc = g["hits"] / max(g["cover"], 1)
        shares = {}
        for ln in lvl_names:
            h, c = g["lvl"].get(ln, [0, 0])
            if c:
                shares[ln] = {"share": round(c / g["cover"], 4),
                              "acc": round(h / c, 4)}
        out["grid"][f"th{th}"] = {"cover": round(cov, 4), "acc": round(acc, 4),
                                  "lvl_share": shares}
        print(f"   θ={th:.2f}: покрытие {cov:6.2%} точность {acc:.4f} "
              f"(доли: { {k: v['share'] for k, v in shares.items()} })", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="shelf",
                    choices=["shelf", "wiki", "hybrid", "all"])
    ap.add_argument("--m", type=int, default=8, help="минимальный счёт ключа")
    ap.add_argument("--valid-n", type=int, default=600_000,
                    help="ограничение потока valid (символов) для wiki/hybrid")
    ap.add_argument("--zone-ckpt", default="ckpt_e15_l1_96_wikitext2.npz")
    args = ap.parse_args()

    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    is_break = np.zeros(tok.vocab_size + 8, dtype=bool)
    for ch in (" ", "\n"):
        sid = tok.stoi.get(ch)
        if sid is not None:
            is_break[sid] = True
    max_char_len = max((lv[1] for lv in LEVELS_DEFAULT if lv[0] == "c"))
    min_ctx = max_char_len                      # короткие уровни готовы раньше, ок — m-фильтр сторожит

    out: dict = {}
    if args.stage in ("shelf", "all"):
        train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
        valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
        sh = HierShelf(LEVELS_DEFAULT, m=args.m)
        t0 = time.perf_counter()
        build_shelf(sh, train_ids, is_break, min_ctx)
        print(f"[15b-m1] полка построена за {time.perf_counter()-t0:.0f} с; "
              f"размеры {sh.stats()}", flush=True)
        out["shelf_m1"] = scan_shelf(sh, valid_ids, is_break, min_ctx, "m1")

    if args.stage in ("wiki", "hybrid", "all"):
        wtrain = load_ids(ROOT / "corpus_external" / "wikitext2" / "train.txt", tok)
        wvalid = load_ids(ROOT / "corpus_external" / "wikitext2" / "valid.txt", tok)
        wvalid = wvalid[:args.valid_n]
        if args.stage in ("wiki", "all"):
            shw = HierShelf(LEVELS_DEFAULT, m=args.m)
            t0 = time.perf_counter()
            build_shelf(shw, wtrain, is_break, min_ctx)
            print(f"[15b-wiki] полка построена за {time.perf_counter()-t0:.0f} с; "
                  f"размеры {shw.stats()}", flush=True)
            out["shelf_wiki"] = scan_shelf(shw, wvalid, is_break, min_ctx, "wiki")

        if args.stage in ("hybrid", "all"):
            from aira.zone import CharMLP
            CTX = 32
            assert CTX <= min_ctx
            wfp = RESULTS / args.zone_ckpt
            if not wfp.exists():
                print(f"[15b-hybrid] чекпоинт зоны {wfp.name} не найден — "
                      f"стадия пропущена (обучите её EXP-15 на wikitext2)", flush=True)
            else:
                # СВЕЖАЯ полка из wtrain: не наследуем онлайн-обновления стадии wiki
                shh = HierShelf(LEVELS_DEFAULT, m=args.m)
                build_shelf(shh, wtrain, is_break, min_ctx)
                print(f"[15b-hybrid] свежая полка из wiki-train; размеры {shh.stats()}",
                      flush=True)
                model = CharMLP(vocab=V, ctx=CTX, d_emb=32, d_hid=96, seed=42)
                model.load_arrays({k: v for k, v in np.load(wfp).items()})
                print(f"[15b-hybrid] зона-96 из {wfp.name}; поток {len(wvalid)}",
                      flush=True)
                shh.reset_stream()
                for i in range(min_ctx):
                    shh.update_level_keys(int(wvalid[i]), bool(is_break[int(wvalid[i])]))
                t1 = time.perf_counter()
                n = 0
                ce_sum_pure = 0.0
                hyb = {th: {"n_l0": 0, "hits": 0, "ce": 0.0,
                            "lvl": defaultdict(lambda: [0, 0])} for th in THS}
                for i in range(min_ctx, len(wvalid)):
                    y = int(wvalid[i])
                    x = wvalid[i - CTX:i]
                    logits, _ = model.forward(x[None, :])
                    p = np.exp(logits - logits.max(1, keepdims=True))
                    p /= p.sum(1, keepdims=True)
                    ce = float(-np.log(np.clip(p[0, y], 1e-12, 1)))
                    ce_sum_pure += ce
                    n += 1
                    for th in THS:
                        r = shh.route(th)
                        g = hyb[th]
                        if r is not None:
                            lv, sym, conf = r
                            hit = int(sym == y)
                            g["n_l0"] += 1
                            g["hits"] += hit
                            g["ce"] += (float(-np.log(max(conf, 1e-3))) if hit
                                        else float(np.log(V)))
                            rec = g["lvl"][f"{lv[0]}{lv[1]}"]
                            rec[0] += hit
                            rec[1] += 1
                        else:
                            g["ce"] += ce
                    shh.update(y)
                    shh.update_level_keys(y, bool(is_break[y]))
                ppl_pure = float(np.exp(ce_sum_pure / max(n, 1)))
                out["hybrid_wiki"] = {"n": n, "ppl_model_pure": round(ppl_pure, 4),
                                      "scan_s": round(time.perf_counter() - t1, 1),
                                      "grid": {}}
                print(f"[15b-hybrid] {n} позиций за {time.perf_counter()-t1:.0f} с; "
                      f"чистая зона ppl {ppl_pure:.4f}", flush=True)
                for th in THS:
                    g = hyb[th]
                    cov = g["n_l0"] / max(n, 1)
                    ppl_h = float(np.exp(g["ce"] / max(n, 1)))
                    acc = g["hits"] / max(g["n_l0"], 1)
                    shares = {ln: {"share": round(g["lvl"][ln][1] / max(g["n_l0"], 1), 4),
                                   "acc": round(g["lvl"][ln][0] / max(g["lvl"][ln][1], 1), 4)}
                              for ln in [f"{lv[0]}{lv[1]}" for lv in shh.present()]
                              if g["lvl"][ln][1]}
                    out["hybrid_wiki"]["grid"][f"th{th}"] = {
                        "cover": round(cov, 4), "ppl_hybrid": round(ppl_h, 4),
                        "delta_vs_pure": round(ppl_h / ppl_pure - 1, 4),
                        "l0_acc": round(acc, 4), "compute_saved": round(cov, 4),
                        "lvl_share": shares}
                    print(f"   θ={th:.2f}: покрытие {cov:6.2%} гибрид {ppl_h:.4f} "
                          f"(Δ {ppl_h / ppl_pure - 1:+.2%}) точность L0 {acc:.4f}",
                          flush=True)

    fp = RESULTS / "results_exp15b.json"
    if fp.exists():
        prev = json.loads(fp.read_text(encoding="utf-8"))
        prev.update(out)
        out = prev
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", fp, flush=True)


if __name__ == "__main__":
    main()
