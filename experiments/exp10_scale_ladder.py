#!/usr/bin/env python3
"""EXP-10 «Лестница масштаба» — ранняя разведка C6: зазор PC↔BP против ёмкости.

МОТИВАЦИЯ (директива заказчика 2026-08-06): система должна быть выстроена так,
чтобы большой масштаб не был проблемой даже на 2 CPU. Прямое следствие: право
на масштабирование надо доказывать НАКЛОНОМ закона роста, измеренным на малой
лестнице, а не верой: 100M на стойке M1 не построить, но производную зазора по
ёмкости — измерить можно. Это самый дешёвый способ заглянуть в M4, не входя в него.

ПРОТОКОЛ. Один и тот же блок ZoneMLP-ctx32 (EXP-09 S1) в трёх ёмкостях
d_hid ∈ {96, 256, 512} (~117k, ~352k, ~800k параметров; 0.83 декады):
  - рука bp: классика, AdamW lr 3e-3, 800 шагов, B=128;
  - рука pcm2_frugal: наш локальный кредит (β=1.0, BB+сторож, T=32, freeze 3e-3,
    eps 1e-2, σ-δ шина θ=0.05) — та самая, что взяла ворота H-26a.
Данные/инициализация/порядок батчей общие внутри каждой ёмкости (seed 42/42/123).
Бюджет данных равный по ёмкостям (800×128 окон ≈ 13.1M символов ≈ 0.16% корпуса
за прогон — без запоминания эпох).

ГИПОТЕЗА H-27 (закон масштаба локального кредита):
  ворота: зазор ppl_PC/ppl_BP − 1 НЕ РАСТЁТ с логарифмом ёмкости: наклон
  (gap_800k − gap_117k) ≤ +5 п.п. [размер эффекта шума протокола ±1–2 п.п.];
  смерть: gap_800k ≥ max(gap_117k × 1.5, gap_117k + 10 п.п.) — зазор растёт с
  ёмкостью уже в малом → локальный кредит в чистом виде к 100M не масштабируется,
  доктрина M4 уходит на редкий BP-якорь (C6) ДО траты большого бюджета.

Дополнительно фиксируем: кривые val ppl каждые 200 шагов (тренд зазора от шага),
работу (work_frac) и сжатие шины по рукам — не меняется ли событийная экономия
с ёмкостью (вторая ось масштаба: экономия обязана ДЕРЖАТЬСЯ при росте размера).

Запуск: .venv/bin/python experiments/exp10_scale_ladder.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from aira.tokenizer import CharTokenizer  # noqa: E402
from aira.zone import AdamW, BusSigmaDelta, CharMLP  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
CTX, D_EMB, B, VOCAB = 32, 32, 128, 64
STEPS = 800
SIZES = [96, 256, 512]


def load_ids(path: Path, tok: CharTokenizer) -> np.ndarray:
    ids = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ids.extend(tok.encode(line.rstrip("\n"), add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.int64)


def batch(data: np.ndarray, rng: np.random.Generator, b: int = B, ctx: int = CTX):
    i = rng.integers(0, len(data) - ctx - 1, size=b)
    return (np.stack([data[k:k + ctx] for k in i]), data[i + ctx])


def val_ppl(model: CharMLP, data: np.ndarray, n: int = 20, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    ls = []
    for _ in range(n):
        x, y = batch(data, rng)
        logits, _ = model.forward(x)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ls.append(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    return float(np.exp(sum(ls) / len(ls)))


def run_arm(kind: str, d_hid: int, train_ids: np.ndarray, valid_ids: np.ndarray,
            steps: int, lr: float = 3e-3, beta_sched: str = "const") -> dict:
    model = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d_hid, seed=42)
    n_par = model.n_params if hasattr(model, "n_params") else sum(
        v.size for v in model.arrays().values())
    opt = AdamW(model.arrays(), lr=lr)
    rng = np.random.default_rng(123)
    bus12 = BusSigmaDelta((B, d_hid), 0.05) if kind == "pc" else None
    bus21 = BusSigmaDelta((B, d_hid), 0.05) if kind == "pc" else None
    log = {"params": int(n_par), "d_hid": d_hid, "kind": kind, "lr": lr,
           "curve": [], "T_mean": None, "work_frac": None, "compress": None}
    T_acc = W_acc = n_acc = 0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = batch(train_ids, rng)
        if kind == "bp":
            g, _ = model.bp_grads(x, y)
        else:
            beta = 1.0 if beta_sched == "const" \
                else max(0.1, 1.0 * (1 - step / steps) + 0.1 * step / steps)
            g, st = model.pc_grads(x, y, beta=beta, method="bb", alpha=1.0, T=32,
                                   freeze=3e-3, eps=1e-2, bus12=bus12, bus21=bus21)
            T_acc += st["T_used"]; W_acc += st["work_frac"]; n_acc += 1
        opt.step(g)
        if step % 200 == 0 or step == steps:
            log["curve"].append({"step": step,
                                 "val_ppl": round(val_ppl(model, valid_ids, n=6), 4)})
    log["wall_s"] = round(time.perf_counter() - t0, 1)
    log["val_ppl_full"] = round(val_ppl(model, valid_ids, n=20), 4)
    if kind == "pc":
        log["T_mean"] = round(T_acc / max(n_acc, 1), 1)
        log["work_frac"] = round(W_acc / max(n_acc, 1), 3)
        dens_bits = log["T_mean"] * steps * 2 * B * d_hid * 16
        log["compress"] = round(dens_bits / max(bus12.traffic_bits()
                                                + bus21.traffic_bits(), 1), 2)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--recipe", default="a", choices=["a", "b", "c"],
                    help="a: замороженный рецепт (lr 3e-3 всем, 800 шагов); "
                         "b: размеро-учитывающий (lr∝1/√d_hid таблица, 1200 шагов) — "
                         "вопрос способности, ответ на артефакт недосходимости руки a; "
                         "c: b + β-затухание 1.0→0.1 (лечение поздней регрессии: "
                         "смещение однофазного оценщика ∝β)")
    args = ap.parse_args()
    steps = 200 if args.quick else (800 if args.recipe == "a" else 1200)
    lr_map = {96: 3e-3, 256: 3e-3, 512: 3e-3} if args.recipe == "a" \
        else {96: 3e-3, 256: 2.1e-3, 512: 1.3e-3}
    anneal_beta = args.recipe == "c"
    sizes = SIZES if args.recipe != "c" else [96, 512]
    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    key = f"ladder_{args.recipe}"
    out: dict = {f"config_{args.recipe}": {"ctx": CTX, "d_emb": D_EMB, "B": B,
                                           "steps": steps, "sizes": sizes,
                                           "beta_anneal": anneal_beta},
                 key: {}}
    for d_hid in sizes:
        lr = lr_map[d_hid]
        print(f"[EXP-10/{args.recipe}] ёмкость d_hid={d_hid} lr={lr}…", flush=True)
        bp = run_arm("bp", d_hid, train_ids, valid_ids, steps, lr=lr)
        print(f"   bp   ppl {bp['val_ppl_full']:.4f} ({bp['wall_s']} с)", flush=True)
        pc = run_arm("pc", d_hid, train_ids, valid_ids, steps, lr=lr,
                     beta_sched="anneal" if anneal_beta else "const")
        gap = pc["val_ppl_full"] / bp["val_ppl_full"] - 1
        pc["gap_vs_bp"] = round(gap, 4)
        print(f"   pc   ppl {pc['val_ppl_full']:.4f} зазор {gap:+.2%} "
              f"работа {pc['work_frac']} T̄ {pc['T_mean']} ×{pc['compress']} "
              f"({pc['wall_s']} с)", flush=True)
        out[key][str(d_hid)] = {"bp": bp, "pc": pc}
    gaps = [out[key][str(d)]["pc"]["gap_vs_bp"] for d in sizes]
    out[f"gap_pp_slope_{args.recipe}"] = round((gaps[-1] - gaps[0]), 4)
    out[f"verdict_{args.recipe}"] = ("✓ наклона нет" if gaps[-1] - gaps[0] <= 0.05
                                     else "✗ зазор растёт — к C6-якорям")
    print(f"[EXP-10/{args.recipe}] зазоры: {[f'{g:+.1%}' for g in gaps]} "
          f"наклон {(gaps[-1] - gaps[0]) * 100:+.1f} п.п. → "
          f"{out[f'verdict_{args.recipe}']}", flush=True)
    fp = RESULTS / "results_exp10.json"
    if fp.exists():
        prev = json.loads(fp.read_text(encoding="utf-8"))
        prev.update(out)
        out = prev
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", fp, flush=True)


if __name__ == "__main__":
    main()
