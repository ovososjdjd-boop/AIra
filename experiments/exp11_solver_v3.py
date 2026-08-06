#!/usr/bin/env python3
"""EXP-11 «Перенос солвера» — H-28 (µTransfer AIra): снять 512-полку законами, не ручной настройкой.

КОНТЕКСТ (EXP-10): зазор PC↔BP растёт с ёмкостью (+9–12 п.п. на 0.83 декады) при воротах ≤+5;
β-затухание принято в спецификацию; остаток болезни большой зоны — жёсткость релаксации
(T̄ упёрт в 32 на всех размерах) и абсолютный freeze 3e-3, не переносимый по размерам
(работа 0.18 на 117k → 0.42 на 800k).

СТРАТЕГИЯ H-28: вместо повторной ручной подгонки под каждый размер — законы:
  K1  T(d): достаточная глубина релаксации от размера (измерить resid/cos по T на прогретых
      весах 512); если resid не падает с T — проблема не в T, а в предобусловливателе (→M2.1);
  K2  квантильный сон: freeze_q (доля спящих) вместо абсолютного порога — нормирован к размеру
      по построению;
  K3  β(d,t): уже принят закон (затухание 1.0→0.1); вариант пола 0.03 проверяется здесь.

СЦЕНЫ:
  S0  Диагностика полки: прогретые веса 512 (500 шагов BP), resid и min-cos против
      T ∈ {8,16,32,64,96} на 4 батчах (протокол EXP-07 S1).
  S1  Триаж сна на тех же весах, T=32: freeze ∈ {абс 3e-3; q0.5; q0.7; q0.8} → cos против
      работы (фронтир сна; абсолют против квантиля).
  S2  Полные прогоны концов лестницы {96, 512} @1200 шагов, спецификация v3 = β-затухание
      + победители S0/S1 (PC-руки; знаменатели BP из EXP-10/b: 96→1.1166, 512→1.1437 —
      тот же протокол, пересчёт не требуется).
      Ворота H-28a: slope(96→512) ≤ +5 п.п. И работа ≤ 0.35 на обоих концах;
      смерть: slope ≥ +8 п.п. → M2.1 (сильный предобусловливатель) становится блокером M4,
      страховка BP-якорей оформляется.

Запуск: .venv/bin/python experiments/exp11_solver_v3.py [--stage s0|s1|s2|all]
        [--tdeep 64] [--qfreeze 0.7] [--betaend 0.03]
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
from aira.zone import AdamW, BusSigmaDelta, CharMLP, cos_sim  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
CTX, D_EMB, B, VOCAB = 32, 32, 128, 64
LAYER_KEYS = ["W1", "W2", "W3", "emb"]


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


def warm512(train_ids: np.ndarray, steps: int = 500) -> dict[str, np.ndarray]:
    m = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=512, seed=42)
    opt = AdamW(m.arrays(), lr=1.3e-3)
    rng = np.random.default_rng(123)
    for _ in range(steps):
        x, y = batch(train_ids, rng)
        g, _ = m.bp_grads(x, y)
        opt.step(g)
    return {k: v.copy() for k, v in m.arrays().items()}


# ---------------------------------------------------------------- S0/S1 триаж
def s0_depth(warm: dict, valid: np.ndarray) -> dict:
    print("[S0] 512 прогретая: resid/cos по глубине T…", flush=True)
    rng = np.random.default_rng(321)
    out = {"T": []}
    for T in (8, 16, 32, 64, 96):
        coses, minl, resids = [], [], []
        for _ in range(4):
            m = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=512)
            m.load_arrays(warm)
            x, y = batch(valid, rng)
            gb, _ = m.bp_grads(x, y)
            gp, st = m.pc_grads(x, y, beta=1.0, T=T, method="bb", alpha=1.0)
            coses.append(cos_sim(gp, gb))
            minl.append(min(cos_sim(gp, gb, [k]) for k in LAYER_KEYS))
            resids.append(st["resid"])
        rec = {"cos": round(float(np.mean(coses)), 4),
               "minl": round(float(np.mean(minl)), 4),
               "resid": round(float(np.mean(resids)), 5)}
        out["T"].append((T, rec))
        print(f"   T={T:>2} cos={rec['cos']:.4f} min-слой={rec['minl']:.4f} "
              f"resid={rec['resid']:.4f}", flush=True)
    return out


def s1_sleep(warm: dict, valid: np.ndarray) -> dict:
    print("[S1] фронтир сна на 512, T=32…", flush=True)
    rng = np.random.default_rng(654)
    cfgs = {"abs3e-3": {"freeze": 3e-3}, "q0.5": {"freeze_q": 0.5},
            "q0.7": {"freeze_q": 0.7}, "q0.8": {"freeze_q": 0.8}}
    out = {}
    for name, kw in cfgs.items():
        coses, works = [], []
        for _ in range(4):
            m = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=512)
            m.load_arrays(warm)
            x, y = batch(valid, rng)
            gb, _ = m.bp_grads(x, y)
            gp, st = m.pc_grads(x, y, beta=1.0, T=32, method="bb", alpha=1.0, **kw)
            coses.append(cos_sim(gp, gb))
            works.append(st["work_frac"])
        rec = {"cos": round(float(np.mean(coses)), 4),
               "work": round(float(np.mean(works)), 3)}
        out[name] = rec
        print(f"   {name:8s} cos={rec['cos']:.4f} работа={rec['work']:.3f}", flush=True)
    return out


# ---------------------------------------------------------------- S2 полные прогоны
def pc_run(d_hid: int, lr: float, train_ids: np.ndarray, valid_ids: np.ndarray,
           steps: int, T: int, freeze_kw: dict, beta_end: float,
           t_link: int = 0, anchor_every: int = 0) -> dict:
    """anchor_every=K>0: каждый K-й шаг — точный BP-градиент («якорь» доктрины C6;
    бюджет глобальности 1/K публикуется). Остальные шаги — локальный PC."""
    """t_link>0: глубина связана с β — T_eff = T + (t_link−T)·(1−β)/(1−β_end):
    мелко при сильном β (не вбирать смещение), глубоко при малом (точный почти
    несмещённый градиент). Гипотеза-механизм из S2 v3: глубина при β=1 — яд."""
    model = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d_hid, seed=42)
    opt = AdamW(model.arrays(), lr=lr)
    rng = np.random.default_rng(123)
    bus12 = BusSigmaDelta((B, d_hid), 0.05)
    bus21 = BusSigmaDelta((B, d_hid), 0.05)
    T_acc = W_acc = n_acc = 0
    log = {"d_hid": d_hid, "curve": []}
    rms_ema: dict[str, float] | None = None
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = batch(train_ids, rng)
        beta = max(beta_end, 1.0 + (beta_end - 1.0) * (step / steps))
        if anchor_every and step % anchor_every == 0 and step > 50 \
                and rms_ema is not None:
            # якорь как МАСШТАБ-СОГЛАСОВАННАЯ коррекция: BP-градиент нормируется
            # к EMA-уровню PC-градиента по каждому массиву (наивная подмена —
            # пила эффективного lr в общем AdamW: EXP-11, измерено ✗)
            g, _ = model.bp_grads(x, y)
            for k in g:
                r_bp = float(np.sqrt((g[k] ** 2).mean())) + 1e-30
                g[k] = g[k] * (rms_ema[k] / r_bp)
        else:
            T_eff = T if not t_link else int(round(
                T + (t_link - T) * (1.0 - beta) / (1.0 - beta_end)))
            g, st = model.pc_grads(x, y, beta=beta, method="bb", alpha=1.0, T=T_eff,
                                   bus12=bus12, bus21=bus21, **freeze_kw)
            T_acc += st["T_used"]; W_acc += st["work_frac"]; n_acc += 1
            if rms_ema is None:
                rms_ema = {k: float(np.sqrt((g[k] ** 2).mean())) + 1e-30 for k in g}
            else:
                for k in g:
                    rms_ema[k] = 0.95 * rms_ema[k] + \
                        0.05 * float(np.sqrt((g[k] ** 2).mean())) + 1e-30
        opt.step(g)
        if step % 300 == 0 or step == steps:
            log["curve"].append({"step": step,
                                 "val_ppl": round(val_ppl(model, valid_ids, n=6), 4)})
            print(f"      {d_hid} step {step}: ppl {log['curve'][-1]['val_ppl']}",
                  flush=True)
    log["wall_s"] = round(time.perf_counter() - t0, 1)
    log["val_ppl_full"] = round(val_ppl(model, valid_ids, n=20), 4)
    log["T_mean"] = round(T_acc / n_acc, 1)
    log["work_frac"] = round(W_acc / n_acc, 3)
    dens_bits = log["T_mean"] * steps * 2 * B * d_hid * 16
    log["compress"] = round(dens_bits / max(bus12.traffic_bits()
                                            + bus21.traffic_bits(), 1), 2)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["s0", "s1", "s2", "single", "all"])
    ap.add_argument("--tdeep", type=int, default=64)
    ap.add_argument("--tlink", type=int, default=0)
    ap.add_argument("--qfreeze", type=float, default=0.7)
    ap.add_argument("--betaend", type=float, default=0.03)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--anchor", type=int, default=0,
                    help="K>0: каждый K-й шаг — точный BP-якорь (C6, бюджет 1/K)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    out: dict = {}
    if args.stage in ("s0", "s1", "all"):
        warm = warm512(train_ids)
        print("[*] веса 512 прогреты (500 шагов BP)", flush=True)
        if args.stage in ("s0", "all"):
            out["s0_depth"] = s0_depth(warm, valid_ids)
        if args.stage in ("s1", "all"):
            out["s1_sleep"] = s1_sleep(warm, valid_ids)
    if args.stage in ("s2", "all"):
        freeze_kw = {"freeze_q": args.qfreeze} if args.qfreeze else {"freeze": 3e-3}
        spec = {"T": args.tdeep, "beta_end": args.betaend, "freeze": freeze_kw}
        print(f"[S2] спецификация v3: {spec}", flush=True)
        bp_ref = {"96": 1.1166, "512": 1.1437}   # знаменатели EXP-10/b (тот же протокол)
        lr_map = {96: 3e-3, 512: 1.3e-3}
        s2 = {"spec": spec, "bp_ref": bp_ref}
        for d in (96, 512):
            r = pc_run(d, lr_map[d], train_ids, valid_ids, steps=1200,
                       T=args.tdeep, freeze_kw=freeze_kw, beta_end=args.betaend)
            r["gap_vs_bp"] = round(r["val_ppl_full"] / bp_ref[str(d)] - 1, 4)
            print(f"   {str(d):4s}: ppl {r['val_ppl_full']:.4f} зазор "
                  f"{r['gap_vs_bp']:+.2%} работа {r['work_frac']} T̄ {r['T_mean']} "
                  f"×{r['compress']} ({r['wall_s']} с)", flush=True)
            s2[str(d)] = r
        slope = s2["512"]["gap_vs_bp"] - s2["96"]["gap_vs_bp"]
        s2["slope"] = round(slope, 4)
        s2["verdict"] = ("✓ наклон снят" if slope <= 0.05 else
                         "✗ наклон остался → M2.1 блокер M4")
        print(f"[S2] наклон {slope * 100:+.1f} п.п. → {s2['verdict']}", flush=True)
        out["s2_v3"] = s2
    if args.stage in ("single",):
        freeze_kw = {"freeze": 3e-3}
        lr_map = {96: 3e-3, 512: 1.3e-3}
        bp_ref = {"96": 1.1166, "512": 1.1437}
        d = args.size
        r = pc_run(d, lr_map[d], train_ids, valid_ids, steps=1200,
                   T=args.tdeep, t_link=args.tlink, freeze_kw=freeze_kw,
                   beta_end=args.betaend, anchor_every=args.anchor)
        r["gap_vs_bp"] = round(r["val_ppl_full"] / bp_ref[str(d)] - 1, 4)
        tag = args.tag or f"single_{d}_T{args.tdeep}_l{args.tlink}_b{args.betaend}"
        print(f"   [{tag}] ppl {r['val_ppl_full']:.4f} зазор {r['gap_vs_bp']:+.2%} "
              f"работа {r['work_frac']} T̄ {r['T_mean']} ({r['wall_s']} с)", flush=True)
        out[tag] = r
    fp = RESULTS / "results_exp11.json"
    if fp.exists():
        prev = json.loads(fp.read_text(encoding="utf-8"))
        prev.update(out)
        out = prev
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", fp, flush=True)


if __name__ == "__main__":
    main()
