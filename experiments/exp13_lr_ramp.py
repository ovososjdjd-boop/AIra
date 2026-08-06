#!/usr/bin/env python3
"""EXP-13 «Финал lr» — H-31: затухание lr последней трети против позднего дрейфа 512.

КОНТЕКСТ. EXP-12 S3 (честная асимптота v3.1 @2400): минимум 1.2483@1800 →
дрейф 1.2862@2100 → 1.3455@2400 (+7.8% к минимуму). Фаза малых β ≈ точный
градиент; дрейф — тот же класс норм-спирали, что EXP-08 лечила wd, но на длинной
дистанции wd=0.01 не хватает. Гипотеза H-31: финальная треть lr ×0.3 гасит дрейф
и удерживает зазор ≤ +10% @2400 (знаменатель BP@2400 = 1.1227, измерен EXP-12 bprep).

РУКИ (512, v3.1: β 1.0→0.1, T(β) 32→64, freeze 3e-3, eps 1e-2, шина θ=0.05, @2400):
  const — история EXP-12 S3 (внешний контроль, не перезапускается);
  step03 — lr = lr0·[1, step≤1600; 0.3, step>1600] (резкий, чётко тестируемый);
  cos03  — плавный: lr0 на [0,1600], далее косинус lr0 → 0.3·lr0 к 2400.

ВОРОТА: ppl@2400 ≤ 1.2350 (зазор ≤ +10%) И дрейф хвоста устранён
(ppl@2400 ≤ min(кривой)·1.01). СМЕРТЬ: дрейф сохраняется ИЛИ зазор > +12%.

Запуск: .venv/bin/python experiments/exp13_lr_ramp.py [--sched step03|cos03|all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from aira.tokenizer import CharTokenizer  # noqa: E402
from aira.zone import AdamW, BusSigmaDelta, CharMLP  # noqa: E402
from exp12_precond_aa import LR_MAP, batch, load_ids, val_ppl  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
CTX, D_EMB, B, VOCAB = 32, 32, 128, 64
BP_REF_2400 = {96: 1.1071, 512: 1.1227}  # EXP-12/13 bprep_*_2400
STEPS = 2400


def bp_ref(d_hid: int, steps: int = STEPS) -> float:
    """Знаменатель BP: известные константы или bprep_… из results_exp12.json."""
    if steps == STEPS and d_hid in BP_REF_2400:
        return BP_REF_2400[d_hid]
    fp = RESULTS / "results_exp12.json"
    if fp.exists():
        rec = json.loads(fp.read_text(encoding="utf-8")).get(f"bprep_{d_hid}_{steps}")
        if rec:
            return float(rec["val_ppl_full"])
    raise KeyError(f"нет BP-знаменателя для {d_hid}@{steps} — запустите bprep")


def lr_factor(sched: str, step: int, steps: int = STEPS, at: int = 0) -> float:
    at = at or int(steps * 2 / 3)  # K2: последняя треть (EXP-13)
    if sched == "const" or step <= at:
        return 1.0
    if sched == "step03":
        return 0.3
    if sched == "cos03":
        t = (step - at) / (steps - at)
        return 0.3 + 0.7 * 0.5 * (1.0 + np.cos(np.pi * t))
    raise ValueError(sched)


def save_json(key: str, rec: dict) -> None:
    fp = RESULTS / "results_exp13.json"
    out = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    out[key] = rec
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def run(d_hid: int, sched: str, train_ids: np.ndarray,
        valid_ids: np.ndarray, steps: int = STEPS, lr0: float = 0.0,
        at: int = 0, resume: bool = False, tag: str = "") -> dict:
    lr0 = lr0 or LR_MAP[d_hid]
    at = at or int(steps * 2 / 3)
    tag = tag or f"{d_hid}_{sched}_{steps}"
    model = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d_hid, seed=42)
    opt = AdamW(model.arrays(), lr=lr0)
    rng = np.random.default_rng(123)
    bus12 = BusSigmaDelta((B, d_hid), 0.05)
    bus21 = BusSigmaDelta((B, d_hid), 0.05)
    log = {"d_hid": d_hid, "sched": sched, "curve": [], "lr0": lr0, "at": at}
    start, T_acc, W_acc = 0, 0.0, 0.0
    ckpt = RESULTS / f"ckpt_{tag}.npz"
    if resume and ckpt.exists():  # возобновление после стирания среды
        z = np.load(ckpt, allow_pickle=True)
        model.load_arrays({k[2:]: z[k] for k in z.files if k.startswith("p_")})
        for k in opt.m:
            opt.m[k] = z[f"m_{k}"]; opt.v[k] = z[f"v_{k}"]
        opt.t = int(z["t"]); start = int(z["step"])
        rng = np.random.default_rng()
        rng.bit_generator.state = z["rng_state"].item()
        T_acc, W_acc = float(z["T_acc"]), float(z["W_acc"])
        log = dict(json.loads(str(z["log_json"])))
        print(f"   [resume {tag} @step {start}]", flush=True)
    t0 = time.perf_counter()
    for step in range(start + 1, steps + 1):
        x, y = batch(train_ids, rng)
        beta = max(0.1, 1.0 + (0.1 - 1.0) * (step / steps))
        T_eff = int(round(32 + (64 - 32) * (1.0 - beta) / (1.0 - 0.1)))
        g, st = model.pc_grads(x, y, beta=beta, method="bb", alpha=1.0, T=T_eff,
                               freeze=3e-3, eps=1e-2, bus12=bus12, bus21=bus21)
        T_acc += st["T_used"]; W_acc += st["work_frac"]
        opt.lr = lr0 * lr_factor(sched, step, steps, at)
        opt.step(g)
        if step % 300 == 0 or step == steps:
            log["curve"].append({"step": step,
                                 "val_ppl": round(val_ppl(model, valid_ids, n=6), 4)})
            print(f"      {tag} step {step}: "
                  f"ppl {log['curve'][-1]['val_ppl']}", flush=True)
            np.savez(ckpt, t=opt.t, step=step, T_acc=T_acc, W_acc=W_acc,
                     rng_state=np.asarray(rng.bit_generator.state, dtype=object),
                     log_json=json.dumps(log, ensure_ascii=False),
                     **{f"p_{k}": v for k, v in model.arrays().items()},
                     **{f"m_{k}": v for k, v in opt.m.items()},
                     **{f"v_{k}": v for k, v in opt.v.items()})
            save_json(f"partial_{tag}", log)  # частичный след в json
    n = steps - start
    log["wall_s"] = round(time.perf_counter() - t0, 1)
    log["val_ppl_full"] = round(val_ppl(model, valid_ids, n=20), 4)
    log["T_mean"] = round(T_acc / max(n, 1), 1)
    log["work_frac"] = round(W_acc / max(n, 1), 3)
    dens_bits = log["T_mean"] * n * 2 * B * d_hid * 16
    log["compress"] = round(dens_bits / max(bus12.traffic_bits()
                                            + bus21.traffic_bits(), 1), 2)
    cmin = min(c["val_ppl"] for c in log["curve"])
    log["curve_min"] = cmin
    log["tail_drift"] = round(log["curve"][-1]["val_ppl"] / cmin - 1, 4)
    try:
        log["bp_ref"] = bp_ref(d_hid, steps)
        log["gap_vs_bp"] = round(log["val_ppl_full"] / log["bp_ref"] - 1, 4)
    except KeyError:
        log["bp_ref"] = log["gap_vs_bp"] = None  # знаменатель не посчитан — прогон валиден
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", default="all", choices=["step03", "cos03", "const", "all"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--at", type=int, default=0, help="начало рампы (по умолч. 2/3 дистанции, K2)")
    ap.add_argument("--resume", type=int, default=0, help="продолжить с чекпоинта ckpt_<tag>.npz")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    scheds = ("step03", "cos03") if args.sched == "all" else (args.sched,)
    for sched in scheds:
        ctag = args.tag or f"{args.size}_{sched}_{args.steps}"
        if args.lr:
            ctag += f"_lr{args.lr:g}"
        r = run(args.size, sched, train_ids, valid_ids, args.steps, lr0=args.lr,
                at=args.at, resume=bool(args.resume), tag=ctag)
        print(f"   [{ctag}] ppl {r['val_ppl_full']:.4f} "
              f"зазор {r['gap_vs_bp']:+.2%} (bp {r['bp_ref']}) мин {r['curve_min']:.4f} "
              f"дрейф {r['tail_drift']:+.2%} работа {r['work_frac']} "
              f"({r['wall_s']} с)", flush=True)
        save_json(f"lr_{ctag}", r)
    print("saved -> results_exp13.json", flush=True)


if __name__ == "__main__":
    main()
