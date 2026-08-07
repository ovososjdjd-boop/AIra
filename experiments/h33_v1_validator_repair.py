#!/usr/bin/env python3
# H-33 / V1 — ремонт валидатора: суфлёр = зона, обученная БЕЗ фильтра (рука B).
# Чистый разбор артефактов EXP-17 (чекпоинты research/ckpts_h33/), обучения НЕТ.
# Контроль: та же решётка на голодной зоне руки A (показывает, что чинит именно суфлёр).
# ===== ПРЕДСКАЗАНИЕ ЗАМОРОЖЕНО ДО ПРОГОНА =====
#  P-V1: суфлёр B на лучшем правиле из решётки: prec ≥ 0.95 при recall ≥ 0.70
#        (ставка S6 к гибриду, CALC-11b). Смерть: prec < 0.95 на всех точках
#        с recall ≥ 0.50. Попутно: эффект чинится именно сменой модели,
#        а не правилом — зона A на той же решётке должна остаться мёртвой.
import json, time, collections, sys
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np  # noqa: E402

from aira.tokenizer import CharTokenizer  # noqa: E402
from aira.zone import CharMLP  # noqa: E402
from exp12_precond_aa import load_ids  # noqa: E402

CTX, D_EMB, VOCAB, KLEN = 32, 32, 64, 8
MOD = VOCAB ** KLEN
THETA, NMIN = 0.90, 2

def load_arm(tag):
    z = np.load(ROOT / f"research/ckpts_h33/ckpt_h33_{tag}.npz", allow_pickle=True)
    log = dict(json.loads(str(z["log_json"])))
    d = log["d_hid"]
    m = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d, seed=42)
    m.load_arrays({k[2:]: z[k] for k in z.files if k.startswith("p_")})
    shelf = collections.defaultdict(collections.Counter)
    if "shelf_dump" in z and z["shelf_dump"].shape[0]:
        sd = z["shelf_dump"]
        for c, x, n in zip(sd[:, 0].tolist(), sd[:, 1].tolist(), sd[:, 2].tolist()):
            shelf[c][x] = n
    return m, shelf, log

mA, shelf, logA = load_arm("A_2400")
mB, _, logB = load_arm("B_2400")
tok = CharTokenizer.load(ROOT / "data/tokenizer_char.json")
valid = load_ids(ROOT / "data/corpus_valid.txt", tok)
print(f"артефакты: зона A (ppl {logA.get('val_ppl_full','n/a')}), суфлёр B (ppl {logB.get('val_ppl_full','n/a')}), "
      f"полка ctx={len(shelf)}, valid {len(valid)}  t={time.time()-T0:.0f}s", flush=True)

# все позиции valid + коды c8; бьём на батчи окон для forward
N = len(valid) - CTX - 1
xs = np.stack([valid[i:i + CTX] for i in range(N)])
ys = valid[CTX:N + CTX]
pows = [VOCAB ** k for k in range(KLEN - 1, -1, -1)]
codes = np.zeros(N, dtype=object)
for k, p in enumerate(pows):
    codes = codes + xs[:, CTX - KLEN + k].astype(object) * p

def shelf_stats(c):
    cnt = shelf.get(int(c))
    if not cnt:
        return 0, 0.0, -1
    n = sum(cnt.values())
    tx, tc = cnt.most_common(1)[0]
    return n, tc / n, tx

# кандидаты полосы: N>=2, f<0.90 (ниже канона), на valid
sel = []
meta = []
for i in range(N):
    n, f, tx = shelf_stats(codes[i])
    if n >= NMIN and f < THETA:
        sel.append(i)
        meta.append((n, f, tx, int(ys[i])))
sel = np.asarray(sel)
print(f"полоса кандидатов на valid: {len(sel)} позиций  t={time.time()-T0:.0f}s", flush=True)

def forward_probs(model, idx_arr, bs=512):
    probs = []
    for s in range(0, len(idx_arr), bs):
        logits, _ = model.forward(xs[idx_arr[s:s + bs]])
        p = np.exp(logits - logits.max(1, keepdims=True))
        probs.append(p / p.sum(1, keepdims=True))
    return np.concatenate(probs)

pB = forward_probs(mB, sel)
pA = forward_probs(mA, sel)
print(f"форварды готовы  t={time.time()-T0:.0f}s", flush=True)

def frontier(P, name):
    tx = np.array([m[2] for m in meta])
    ok = np.array([m[3] == m[2] for m in meta])
    top1 = P.argmax(1)
    ptx = P[np.arange(len(tx)), tx]
    ptop = P.max(1)
    rank_tx = (P > ptx[:, None]).sum(1)  # 0 => аргмакс
    rules = {
        "top1": top1 == tx,
        "top3": rank_tx <= 2,
    }
    grid = {}
    for tau in (0.01, 0.03, 0.05, 0.1, 0.15, 0.2):
        grid[f"p>= {tau}"] = ptx >= tau
    for dl in (0.3, 0.5, 0.7, 0.9):
        grid[f"rel>={dl}"] = ptx >= dl * ptop
    for tau in (0.03, 0.05, 0.1):
        grid[f"top3&p>={tau}"] = rules["top3"] & (ptx >= tau)
    rules.update(grid)
    rows = []
    for rn, passed in rules.items():
        np_ = int(passed.sum())
        if not np_:
            continue
        prec = float(ok[passed].mean())
        rec = float(ok[passed].sum() / max(ok.sum(), 1))
        rows.append(dict(rule=rn, npass=np_, prec=round(prec, 4), rec=round(rec, 4)))
    rows.sort(key=lambda r: -r["prec"])
    print(f"\n[{name}] точность правил на полосе (n={len(sel)}, верных кандидатов {int(ok.sum())}):")
    for r in rows:
        print(f"  {r['rule']:<14} pass={r['npass']:>6} prec={r['prec']:.4f} rec={r['rec']:.4f}", flush=True)
    return rows

rowsB = frontier(pB, "суфлёр B (ремонт)")
rowsA = frontier(pA, "зона A голодная (контроль)")

# --- вариации спасения: консенсус A∩B и сужение полосы (f>=0.7) --- #
extra = {}
tx_np = np.array([m[2] for m in meta]); ok_npb = np.array([m[3] == m[2] for m in meta])
f_np = np.array([m[1] for m in meta])
top1B = pB.argmax(1); top1A = pA.argmax(1)
cons = (top1B == tx_np) & (top1A == tx_np)
for tag, mask in [("консенсус A∩B top1", cons),
                  ("B top1 на полосе f>=0.7", (top1B == tx_np) & (f_np >= 0.7)),
                  ("консенсус на f>=0.7", cons & (f_np >= 0.7))]:
    if mask.sum():
        extra[tag] = dict(npass=int(mask.sum()), prec=float(ok_npb[mask].mean()),
                          rec=float(ok_npb[mask].sum() / max(ok_npb.sum(), 1)))
for tag, r in extra.items():
    print(f"  спасение [{tag}]: pass={r['npass']} prec={r['prec']:.4f} rec={r['rec']:.4f}", flush=True)

# вердикт P-V1: лучшая точка B
best = None
for r in sorted(rowsB, key=lambda r: -r["prec"]):
    if r["rec"] >= 0.70:
        best = r
        break
if best is None:
    best = max(rowsB, key=lambda r: r["prec"] * min(r["rec"] / 0.5, 1))
verd = "✓" if best and best["prec"] >= 0.95 and best["rec"] >= 0.70 else (
    "✗ СМЕРТЬ" if all(r["prec"] < 0.95 or r["rec"] < 0.5 for r in rowsB) else "~ погранично")
print(f"\nP-V1 {verd}: лучшая точка суфлёра B — {best}")

json.dump(dict(band_n=int(len(sel)), rows_B=rowsB, rows_A=rowsA, best_B=best, verdict=verd, rescue=extra,
               armA_ppl=logA.get('val_ppl_full'), armB_ppl=logB.get('val_ppl_full'),
               elapsed_s=time.time() - T0),
          open(ROOT / "research/H33_V1_VALIDATOR.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/H33_V1_VALIDATOR.json  t={time.time()-T0:.0f}s")
