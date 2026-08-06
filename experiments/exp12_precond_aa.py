#!/usr/bin/env python3
"""EXP-12 «Сильный солвер» — M2.1: ускорение Андерсона в релаксации зоны.

КОНТЕКСТ. EXP-11: полка 512 итерационная (resid 0.53@T32 → 0.13@T96, cos растёт),
глубина при β=1 — яд; закон T(β)-связи принят; остаточный наклон +6.2 п.п. — M2.1
(сильный предобусловливатель, resid<0.01) назван блокером M4. Кандидат: ускорение
Андерсона (экстраполяция по истории шагов со сторожем энергии) — классика сжатия
остатка фикс-точечных итераций, ложится на наш сторож без изменения доктрины.

ВОПРОСЫ:
  S1  Диагностика: прогретые веса 512 и 96; resid/cos по T ∈ {16,32,64} × aa ∈ {0,3}.
      Ворота H-30a: aa=3 режет resid при T=32 ≥ ×3 против aa=0 И cos(T=32) ≥ cos(T=64)
      без AA (глубина покупается не итерациями, а алгеброй); смерть: выигрыш < ×1.5
      resid или cos деградирует > 0.02.
  S2  Полный прогон концов лестницы {96, 512} @1200: спецификация v3.2 = v3.1
      (β 1.0→0.1 × T(β) 32→64 связанная) + aa=3 (если S1 прошёл).
      Ворота H-30b: зазор 512 ≤ +9% (против лучшего +11.3%) при работе ≤0.45 и
      T̄ не выросшим > ×1.3; наклон 96→512 ≤ +4 п.п. Смерть: зазор 512 ≥ +13%
      или T̄ ×2+.

Запуск: .venv/bin/python experiments/exp12_precond_aa.py [--stage s1|s2|all]
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


def warm(d_hid: int, lr: float, train_ids: np.ndarray, steps: int = 500):
    m = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d_hid, seed=42)
    opt = AdamW(m.arrays(), lr=lr)
    rng = np.random.default_rng(123)
    for _ in range(steps):
        x, y = batch(train_ids, rng)
        g, _ = m.bp_grads(x, y)
        opt.step(g)
    return {k: v.copy() for k, v in m.arrays().items()}


# ---------------------------------------------------------------- S1 диагностика
def s1_diag(train_ids: np.ndarray, valid_ids: np.ndarray) -> dict:
    out = {}
    for d_hid, lr in ((512, 1.3e-3), (96, 3e-3)):
        w = warm(d_hid, lr, train_ids)
        print(f"[S1] прогрет {d_hid}; resid/cos по β × T × AA…", flush=True)
        rng = np.random.default_rng(321)
        recs = {}
        for beta in (1.0, 0.1):
            for aa in (0, 3):
                for T in (32, 64):
                    coses, minl, resids, hits, dt = [], [], [], [], []
                    for _ in range(4):
                        m = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d_hid)
                        m.load_arrays(w)
                        x, y = batch(valid_ids, rng)
                        gb, _ = m.bp_grads(x, y)
                        t0 = time.perf_counter()
                        gp, st = m.pc_grads(x, y, beta=beta, T=T, method="bb",
                                            alpha=1.0, aa=aa)
                        dt.append(time.perf_counter() - t0)
                        coses.append(cos_sim(gp, gb))
                        minl.append(min(cos_sim(gp, gb, [k]) for k in LAYER_KEYS))
                        resids.append(st["resid"])
                        hits.append(st["aa_hits"])
                    key = f"b{beta}_aa{aa}_T{T}"
                    recs[key] = {"cos": round(float(np.mean(coses)), 4),
                                 "minl": round(float(np.mean(minl)), 4),
                                 "resid": round(float(np.mean(resids)), 5),
                                 "aa_hits": float(np.mean(hits)),
                                 "ms": round(float(np.mean(dt)) * 1000, 1)}
                    r = recs[key]
                    print(f"   {d_hid} β={beta} aa={aa} T={T:>2}: cos {r['cos']:.4f} "
                          f"minl {r['minl']:.4f} resid {r['resid']:.4f} "
                          f"hits {r['aa_hits']:.0f} {r['ms']:.0f}мс", flush=True)
        out[str(d_hid)] = recs
    return out


# ---------------------------------------------------------------- S2 полная лестница
def pc_run(d_hid: int, lr: float, train_ids: np.ndarray, valid_ids: np.ndarray,
           steps: int, aa: int, aa_gate: bool = False) -> dict:
    """Спецификация v3.2: β 1.0→0.1 × T(β): 32→64 связанная + freeze 3e-3 + aa.

    aa_gate=True: AA включается только при β ≤ 0.3 (S1: при сильном β AA — яд,
    при слабом жёсткости уже нет — резерв точности малой фазы)."""
    model = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d_hid, seed=42)
    opt = AdamW(model.arrays(), lr=lr)
    rng = np.random.default_rng(123)
    bus12 = BusSigmaDelta((B, d_hid), 0.05)
    bus21 = BusSigmaDelta((B, d_hid), 0.05)
    T_acc = W_acc = n_acc = H_acc = 0
    log = {"d_hid": d_hid, "aa": aa, "aa_gate": aa_gate, "curve": []}
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = batch(train_ids, rng)
        beta = max(0.1, 1.0 + (0.1 - 1.0) * (step / steps))
        T_eff = int(round(32 + (64 - 32) * (1.0 - beta) / (1.0 - 0.1)))
        aa_eff = aa if (not aa_gate or beta <= 0.3) else 0
        g, st = model.pc_grads(x, y, beta=beta, method="bb", alpha=1.0, T=T_eff,
                               freeze=3e-3, eps=1e-2, aa=aa_eff,
                               bus12=bus12, bus21=bus21)
        T_acc += st["T_used"]; W_acc += st["work_frac"]; H_acc += st["aa_hits"]
        n_acc += 1
        opt.step(g)
        if step % 300 == 0 or step == steps:
            log["curve"].append({"step": step,
                                 "val_ppl": round(val_ppl(model, valid_ids, n=6), 4)})
            print(f"      {d_hid} aa={aa} step {step}: "
                  f"ppl {log['curve'][-1]['val_ppl']}", flush=True)
    log["wall_s"] = round(time.perf_counter() - t0, 1)
    log["val_ppl_full"] = round(val_ppl(model, valid_ids, n=20), 4)
    log["T_mean"] = round(T_acc / n_acc, 1)
    log["work_frac"] = round(W_acc / n_acc, 3)
    log["aa_hits_mean"] = round(H_acc / n_acc, 1)
    dens_bits = log["T_mean"] * steps * 2 * B * d_hid * 16
    log["compress"] = round(dens_bits / max(bus12.traffic_bits()
                                            + bus21.traffic_bits(), 1), 2)
    return log


def s2_ladder(train_ids: np.ndarray, valid_ids: np.ndarray, aa: int) -> dict:
    bp_ref = {"96": 1.1166, "512": 1.1437}
    lr_map = {96: 3e-3, 512: 1.3e-3}
    out = {"aa": aa, "bp_ref": bp_ref, "spec": "v3.2 = β1→0.1 × T(β)32→64 × aa"}
    for d in (96, 512):
        r = pc_run(d, lr_map[d], train_ids, valid_ids, steps=1200, aa=aa,
                   aa_gate=True)
        r["gap_vs_bp"] = round(r["val_ppl_full"] / bp_ref[str(d)] - 1, 4)
        print(f"   {d}: ppl {r['val_ppl_full']:.4f} зазор {r['gap_vs_bp']:+.2%} "
              f"работа {r['work_frac']} T̄ {r['T_mean']} aa-hits {r['aa_hits_mean']} "
              f"({r['wall_s']} с)", flush=True)
        out[str(d)] = r
    slope = out["512"]["gap_vs_bp"] - out["96"]["gap_vs_bp"]
    out["slope"] = round(slope, 4)
    out["verdict"] = ("✓ наклон снят" if slope <= 0.04 else "✗ наклон остался")
    print(f"[S2] наклон {slope * 100:+.1f} п.п. → {out['verdict']}", flush=True)
    return out


BP_REF = {"96": 1.1166, "256": 1.1314, "512": 1.1437}
# честный закон lr ≈ 3e-3·(96/d)^0.55 (EXP-10/13): 96→3.0e-3, 256→1.7e-3,
# 512→1.3e-3, 2048→5.6e-4
LR_MAP = {96: 3e-3, 256: 1.7e-3, 512: 1.3e-3, 2048: 5.6e-4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["s1", "s2", "single", "bprep", "all"])
    ap.add_argument("--aa", type=int, default=3)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--aagate", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    out: dict = {}
    if args.stage in ("s1", "all"):
        out["s1_diag"] = s1_diag(train_ids, valid_ids)
    if args.stage in ("s2", "all"):
        out["s2_v32"] = s2_ladder(train_ids, valid_ids, aa=args.aa)
    if args.stage in ("bprep",):
        d = args.size
        model = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d, seed=42)
        opt = AdamW(model.arrays(), lr=args.lr or LR_MAP[d])
        rng = np.random.default_rng(123)
        for step in range(1, args.steps + 1):
            x, y = batch(train_ids, rng)
            g, _ = model.bp_grads(x, y)
            opt.step(g)
        vp = round(val_ppl(model, valid_ids, n=20), 4)
        print(f"   [bprep_{d}_{args.steps}] bp ppl {vp}", flush=True)
        out[f"bprep_{d}_{args.steps}"] = {"val_ppl_full": vp}
    if args.stage in ("single",):
        d = args.size
        lr = args.lr or LR_MAP[d]
        r = pc_run(d, lr, train_ids, valid_ids, steps=args.steps, aa=args.aa,
                   aa_gate=bool(args.aagate))
        r["gap_vs_bp"] = round(r["val_ppl_full"] / BP_REF[str(d)] - 1, 4)
        tag = args.tag or f"single_{d}_aa{args.aa}_g{args.aagate}_s{args.steps}"
        print(f"   [{tag}] ppl {r['val_ppl_full']:.4f} зазор {r['gap_vs_bp']:+.2%} "
              f"работа {r['work_frac']} T̄ {r['T_mean']} hits {r['aa_hits_mean']} "
              f"({r['wall_s']} с)", flush=True)
        out[tag] = r
    fp = RESULTS / "results_exp12.json"
    if fp.exists():
        prev = json.loads(fp.read_text(encoding="utf-8"))
        prev.update(out)
        out = prev
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", fp, flush=True)


if __name__ == "__main__":
    main()
