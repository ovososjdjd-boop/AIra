#!/usr/bin/env python3
"""Жгут маршрутизатора v3 (ноль обучения, только разбор артефактов).

Контроль бит-в-бит: v3-код (Shelf/code8/load_sufler из exp17_h33_twin) на
чекпоинтах M1 (A_2400 зона+полка, суфлёр B_2400) обязан
воспроизвести канон V1 (research/H33_V1_VALIDATOR.json, rescue):
  полоса кандидатов n = 47 890;
  «B top1 на полосе f>=0.7»: npass=16 208, prec=0.9582922013820335,
                             rec=0.609983112751836
  (rec: верные спасённые / верные всей широкой полосы N≥2, f<0.90).

+ эталон v3 на M1-выборке (hybrid_eval n=20 seed=7, как hybrid_final):
  duty_v2 → duty_v3, acc_v3u, v3_gain_pp — сверка с замороженным +4–6 п.п.

Запуск: .venv/bin/python experiments/h33_v3_burn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from aira.tokenizer import CharTokenizer  # noqa: E402
from exp12_precond_aa import load_ids  # noqa: E402
from exp17_h33_twin import (CTX, NMIN, THETA, Shelf, code8,  # noqa: E402
                            hybrid_eval, load_sufler)

T0 = time.time()
CK = ROOT / "research/ckpts_h33"
J = json.load(open(ROOT / "research/H33_V1_VALIDATOR.json", encoding="utf-8"))

# рука A: зона + полка (та же процедура восстановления, что в раннере)
z = np.load(CK / "ckpt_h33_A_2400.npz", allow_pickle=True)
from aira.zone import CharMLP  # noqa: E402
mA = CharMLP(vocab=64, ctx=CTX, d_emb=32, d_hid=96, seed=42)
mA.load_arrays({k[2:]: z[k] for k in z.files if k.startswith("p_")})
shelf = Shelf()
import collections  # noqa: E402
sd = z["shelf_dump"]
for c, x2, n in zip(sd[:, 0].tolist(), sd[:, 1].tolist(), sd[:, 2].tolist()):
    cc = shelf.cnt.setdefault(int(c), collections.Counter())
    cc[int(x2)] += int(n)
    shelf.seen_pairs += int(n)
mB = load_sufler(CK / "ckpt_h33_B_2400.npz")

tok = CharTokenizer.load(ROOT / "data/tokenizer_char.json")
valid = load_ids(ROOT / "data/corpus_valid.txt", tok)
print(f"артефакты загружены: полка ctx={len(shelf.cnt)}, valid={len(valid)}"
      f"  t={time.time()-T0:.0f}с", flush=True)

# --- полный скан valid (методика V1) --- #
N = len(valid) - CTX - 1
xs = np.stack([valid[i:i + CTX] for i in range(N)])
ys = valid[CTX:N + CTX]
codes = code8(xs)
sel = []
for i in range(N):
    n, f, tx = shelf.query(int(codes[i]))
    if n >= NMIN and f < THETA:
        sel.append((i, n, f, tx))
assert len(sel) == J["band_n"], f"полоса {len(sel)} != канон {J['band_n']}"
idx = np.array([s[0] for s in sel])
f_np = np.array([s[2] for s in sel])
tx_np = np.array([s[3] for s in sel])
ok_np = np.array([int(ys[s[0]]) == s[3] for s in sel])
print(f"полоса {len(sel)} = канону ✓  t={time.time()-T0:.0f}с", flush=True)

# forward суфлёра на полосе, теми же чанками 512, что в V1
probs = []
for s in range(0, len(idx), 512):
    logits, _ = mB.forward(xs[idx[s:s + 512]])
    p = np.exp(logits - logits.max(1, keepdims=True))
    probs.append(p / p.sum(1, keepdims=True))
top1B = np.concatenate(probs).argmax(1)

mask = (top1B == tx_np) & (f_np >= 0.7)
npass = int(mask.sum())
prec = float(ok_np[mask].mean())
rec = float(ok_np[mask].sum() / max(ok_np.sum(), 1))
R = J["rescue"]["B top1 на полосе f>=0.7"]
ok1 = npass == R["npass"]
ok2 = abs(prec - R["prec"]) < 1e-9
ok3 = abs(rec - R["rec"]) < 1e-9
print(f"«B top1 на f≥0.7»: npass {npass} (канон {R['npass']}) {'✓' if ok1 else '✗'}"
      f"  prec {prec:.6f} (канон {R['prec']:.6f}) {'✓' if ok2 else '✗'}"
      f"  rec {rec:.6f} (канон {R['rec']:.6f}) {'✓' if ok3 else '✗'}", flush=True)

# эталон M1-выборки: v2 → v3 (как hybrid_final в раннере)
he = hybrid_eval(mA, shelf, valid, n=20, sufler=mB)
print("\nэталон v3 на M1 (hybrid_eval n=20, seed=7):", flush=True)
for k in ("shelf_cov", "shelf_acc", "duty_v3", "acc_v3u", "v3_gain_pp",
          "ppl_h", "ppl_h_v3", "s6p_prec", "s6p_rec", "s6p_n_pass"):
    print(f"  {k}: {he.get(k)}", flush=True)

verd = "✓ ЖГУТ ПРОЙДЕН" if (ok1 and ok2 and ok3) else "✗ ЖГУТ: РАСХОЖДЕНИЕ С КАНОНОМ"
print(f"\n{verd}  t={time.time()-T0:.0f}с", flush=True)
out = {"burn_npass": npass, "burn_prec": prec, "burn_rec": rec,
       "canon": R, "pass": bool(ok1 and ok2 and ok3),
       "m1_v3_reference": {k: he.get(k) for k in
                           ("shelf_cov", "shelf_acc", "duty_v3", "acc_v3u",
                            "v3_gain_pp", "ppl_h_v3", "s6p_prec", "s6p_rec")}}
(ROOT / "research/H33_V3_BURN.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
