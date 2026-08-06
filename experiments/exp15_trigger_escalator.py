#!/usr/bin/env python3
"""EXP-15 «Триггерный эскалатор» — H-32: диагностика рынка бесплатного ответа (принцип T).

КОНТЕКСТ. Директива 2026-08-06 (принцип T, PROTO_ROADMAP §2 п.7): цена ответа ∝ новизне
для системы, не размеру машины — известное отдаётся триггером почти бесплатно, вычислением
оплачивается только новое. Маршрут по удивлению для чтения памяти доказан (EXP-09, −2.5%
ppl при чтении 4.2% позиций); этот заход измеряет рынок НУЛЕВОГО уровня L0 эскалатора:
сколько позиций корпуса закрывается ассоциативной полкой без раскрутки нейросети вообще
и с какой точностью против обученной зоны на тех же позициях.

СТАДИИ:
  shelf   — n-граммная триггер-полка (k=8, точный 48-бит ключ по 6 бит/символ, V=64):
            сборка по train-потоку, скоринг valid потоково (полка онлайн-растёт — как
            настоящий триггер с полной историей). Профили «покрытие ↔ точность ↔
            поддержка m × порог θ», децили уверенности, оценка экономии.
  hybrid  — эскалатор L0→L1: зона-96 (PC v3.2 step03 @2400, готовая спецификация)
            обучается заново (176 с), затем скоринг valid потоково: триггер отвечает при
            conf ≥ θ, иначе модель. Ворота H-32: покрытие ≥50% при точности L0 ≥ модельной
            на тех же позициях и суммарный CE гибрида ≤ чистой модели +1%.
            Смерть: CE хуже +2% или покрытие <20% ни при каких θ.

Запуск: .venv/bin/python experiments/exp15_trigger_escalator.py [--stage shelf|hybrid|all]
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
K, V = 8, 64           # контекст 8 символов, алфавит 64 — ключ точный (48 бит)
MASK = (1 << (6 * K)) - 1


class TriggerShelf:
    """Ассоциативная полка: контекст k символов → лучший/второй кандидат + счётчики.

    Память: один dict int→list (5 чисел) на ключ; не растёт по длине потока быстрее,
    чем растёт словарь увиденных контекстов (ограничена словарём языка, не KV-позиций)."""

    def __init__(self) -> None:
        self.tab: dict[int, list[int]] = {}

    def update(self, key: int, sym: int) -> None:
        e = self.tab.get(key)
        if e is None:
            self.tab[key] = [sym, 1, -1, 0, 1]
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

    def query(self, key: int):
        e = self.tab.get(key)
        if e is None or e[4] < 2:
            return None
        # уверенность с поправкой Лапласа на редкие ключи
        return e[0], (e[1] + 1) / (e[4] + V), e[4]


def keys_stream(ids: np.ndarray):
    """Генератор (key, y) для позиций с полным контекстом K."""
    key = 0
    for i, s in enumerate(ids):
        key = ((key << 6) | int(s)) & MASK
        if i + 1 < len(ids) and i >= K - 1:
            yield key, int(ids[i + 1]), i + 1


def stage_shelf(train_ids: np.ndarray, valid_ids: np.ndarray) -> dict:
    t0 = time.perf_counter()
    sh = TriggerShelf()
    for key, y, _ in keys_stream(train_ids):
        sh.update(key, y)
    build_s = time.perf_counter() - t0
    print(f"[shelf] ключи: {len(sh.tab)} за {build_s:.0f} с "
          f"(train {len(train_ids)} символов)", flush=True)

    recs = defaultdict(lambda: [0, 0])   # (m, theta) -> [hits, total]
    dec = []                   # (conf, hit) для децильного профиля
    t1 = time.perf_counter()
    ms = (2, 8, 32)
    ths = (0.30, 0.45, 0.60, 0.75, 0.90)
    for key, y, _ in keys_stream(valid_ids):
        q = sh.query(key)          # полка онлайн-растёт — как настоящий триггер
        if q is not None:
            sym, conf, total = q
            hit = int(sym == y)
            dec.append((conf, hit))
            for m in ms:
                if total < m:
                    continue
                for th in ths:
                    if conf >= th:
                        r = recs[(m, th)]
                        r[0] += hit
                        r[1] += 1
        sh.update(key, y)
    n_valid = len(valid_ids) - K
    out = {"k": K, "keys": len(sh.tab), "build_s": round(build_s, 1),
           "scan_s": round(time.perf_counter() - t1, 1), "n_valid": n_valid,
           "grid": {}, "deciles": []}
    for (m, th), (hits, cnt) in sorted(recs.items()):
        out["grid"][f"m{m}_th{th}"] = {"cover": round(cnt / n_valid, 4),
                                       "acc": round(hits / max(cnt, 1), 4)}
        print(f"   m={m:>2} θ={th:.2f}: покрытие {cnt / n_valid:6.2%} "
              f"точность {hits / max(cnt, 1):.4f}", flush=True)
    if dec:
        arr = np.asarray(dec)
        for q in (0.5, 0.7, 0.8, 0.9, 0.95):
            sub = arr[arr[:, 0] >= np.quantile(arr[:, 0], q) - 1e-9]
            out["deciles"].append({"q": q,
                                   "conf_edge": round(float(np.quantile(arr[:, 0], q)), 3),
                                   "acc": round(float(sub[:, 1].mean()), 4)})
    return out


def stage_hybrid(train_ids: np.ndarray, valid_ids: np.ndarray) -> dict:
    """Эскалатор L0→L1: обученная зона-96 (PC v3.2) на остатке после триггера."""
    from aira.zone import CharMLP, AdamW, BusSigmaDelta
    from exp12_precond_aa import LR_MAP, batch

    # --- L1: зона-96, готовая спецификация v3.2 (как lr_96_step03 из EXP-13)
    CTX, D_EMB, B, STEPS = 32, 32, 128, 2400
    model = CharMLP(vocab=V, ctx=CTX, d_emb=D_EMB, d_hid=96, seed=42)
    opt = AdamW(model.arrays(), lr=LR_MAP[96])
    rng = np.random.default_rng(123)
    bus12, bus21 = BusSigmaDelta((B, 96), 0.05), BusSigmaDelta((B, 96), 0.05)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        x, y = batch(train_ids, rng)
        beta = max(0.1, 1.0 + (0.1 - 1.0) * (step / STEPS))
        T_eff = int(round(32 + (64 - 32) * (1.0 - beta) / (1.0 - 0.1)))
        g, _ = model.pc_grads(x, y, beta=beta, method="bb", alpha=1.0, T=T_eff,
                              freeze=3e-3, eps=1e-2, bus12=bus12, bus21=bus21)
        opt.lr = LR_MAP[96] * (0.3 if step > 1600 else 1.0)
        opt.step(g)
        if step % 600 == 0:
            print(f"   L1 зона-96 шаг {step} ({time.perf_counter() - t0:.0f} с)", flush=True)

    # --- потоковый скоринг valid: L0 и L1 на каждой позиции
    sh = TriggerShelf()
    for key, y, _ in keys_stream(train_ids):
        sh.update(key, y)
    t1 = time.perf_counter()
    rows = []          # (q=(sym, conf, total)|None, l1_ce, y)
    for key, y, i in keys_stream(valid_ids):
        x = valid_ids[i - CTX:i]
        logits, _ = model.forward(x[None, :])
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ce = float(-np.log(np.clip(p[0, y], 1e-12, 1)))
        rows.append((sh.query(key), ce, y))
        sh.update(key, y)
    scan_s = time.perf_counter() - t1
    print(f"   поток {len(rows)} позиций за {scan_s:.0f} с", flush=True)

    # CE ответа триггера (консервативно): hit → -log(conf), miss → log(V)
    # (штраф равномерной альтернативы; conf — сглаженная оценка, не калиброванная).
    out = {"l1_spec": "зона-96 PC v3.2 step03@2400", "scan_s": round(scan_s, 1),
           "n": len(rows), "grid": {}}
    ce_model = float(np.mean([r[1] for r in rows]))
    out["ppl_model_pure"] = round(float(np.exp(ce_model)), 4)
    for th in (0.45, 0.60, 0.75, 0.90):
        ces, l0_hits, l1_ce_on_l0 = [], [], []
        for q, r_ce, y in rows:
            if q is not None and q[1] >= th:
                sym, conf, _ = q
                hit = int(sym == y)
                l0_hits.append(hit)
                l1_ce_on_l0.append(r_ce)
                ces.append(float(-np.log(max(conf, 1e-3))) if hit
                           else float(np.log(V)))
            else:
                ces.append(r_ce)
        cov = len(l0_hits) / len(rows)
        ce_hyb = float(np.mean(ces))
        acc_l0 = float(np.mean(l0_hits)) if l0_hits else 0.0
        ppl1_on_l0 = float(np.exp(np.mean(l1_ce_on_l0))) if l1_ce_on_l0 else float("nan")
        out["grid"][f"th{th}"] = {
            "cover": round(cov, 4),
            "ppl_hybrid": round(float(np.exp(ce_hyb)), 4),
            "delta_ppl_vs_pure": round(float(np.exp(ce_hyb) / np.exp(ce_model) - 1), 4),
            "l0_acc": round(acc_l0, 4),
            "l1_ppl_on_l0_pos": round(ppl1_on_l0, 4),
            "compute_saved": round(cov, 4)}
        print(f"   θ={th:.2f}: покрытие {cov:6.2%} ppl_гибр {out['grid'][f'th{th}']['ppl_hybrid']:.4f} "
              f"(чистая {np.exp(ce_model):.4f}, Δ {out['grid'][f'th{th}']['delta_ppl_vs_pure']:+.2%}) "
              f"точность L0 {acc_l0:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["shelf", "hybrid", "all"])
    args = ap.parse_args()
    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    out: dict = {}
    if args.stage in ("shelf", "all"):
        out["shelf"] = stage_shelf(train_ids, valid_ids)
    if args.stage in ("hybrid", "all"):
        out["hybrid"] = stage_hybrid(train_ids, valid_ids)
    fp = RESULTS / "results_exp15.json"
    if fp.exists():
        prev = json.loads(fp.read_text(encoding="utf-8"))
        prev.update(out)
        out = prev
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", fp, flush=True)


if __name__ == "__main__":
    main()
