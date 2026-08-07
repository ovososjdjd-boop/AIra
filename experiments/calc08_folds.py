#!/usr/bin/env python3
# CALC-08 (расчёт, НЕ обучение): ступени 1–2 лестницы фолдинга H-34 на стойке ×1.
# ===== ПРЕДСКАЗАНИЯ ЗАМОРОЖЕНЫ ДО ПРОГОНА (эталон B0 = 34.3%@0.9566, B1 каскад = 41.0%@0.9592) =====
#  P-F1 цифро-фолд (все цифры→'0'): duty@(0.90,N≥2) >= +2.0 п.п. над B0 при acc>=0.95.
#       Смерть: прирост < +1.0 п.п.
#  P-F2 wildcard-маска по слоту j∈{1,2,3,4} (окно 8, слот выброшен): макс duty при acc>=0.95
#       >= 38.0% (B0 34.3%). Смерть: все j < 36.0% @0.95.
#  P-F3 каскад c8->c6->c4 + цифро-фолд: макс duty при acc>=0.95 >= 44.0%.
#       Смерть: < 41.0% (т.е. цифро-фолд не добавляет к каскаду).
import json, time, math, collections
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")

def build_ids(txt):
    alpha = {c: i for i, c in enumerate(sorted(set(txt)))}
    return [alpha[c] for c in txt], len(alpha)

def count_ngrams(seq, n, V):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    base = V ** n
    code = 0
    for i, x in enumerate(seq):
        code = (code * V + x) % base
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1
            pair[code * V + seq[i + 1]] += 1
    return ctx, pair

def topmap(pair, V):
    top = {}
    for k, c in pair.items():
        code, x = divmod(k, V)
        cur = top.get(code)
        if cur is None or c > cur[0]:
            top[code] = (c, x)
    return top

def holdout_records(shelf, ho, V, levels):
    mods = {n: V ** n for n in levels}
    codes = {n: 0 for n in levels}
    recs = []
    for i in range(len(ho) - 1):
        x = ho[i]
        for n in levels:
            codes[n] = (codes[n] * V + x) % mods[n]
        if i < max(levels) - 1:
            continue
        nxt = ho[i + 1]
        row = []
        for n in levels:
            ncnt = shelf[n][0].get(codes[n], 0)
            if ncnt == 0:
                row.append((0, 0.0, False))
            else:
                c, tx = shelf[n][2][codes[n]]
                row.append((ncnt, c / ncnt, tx == nxt))
        recs.append(tuple(row))
    return recs

FRONT = [(th, nm) for th, nm in
         [(0.85, 2), (0.85, 5), (0.90, 2), (0.90, 5), (0.95, 2), (0.95, 5)]]

def eval_rule(recs, levels, theta, nmin):
    cov = 0; hit = 0
    for row in recs:
        for n, (cnt, f, ok) in zip(levels, row):
            if cnt >= nmin and f >= theta:
                cov += 1; hit += ok
                break
    return cov / len(recs), (hit / cov if cov else 0.0)

def full_grid(tag, recs, levels):
    out = {}
    for th, nm in FRONT:
        d, a = eval_rule(recs, levels, th, nm)
        out[f"{th},{nm}"] = (round(d, 4), round(a, 4))
    best = max(((v[0], v[1], k) for k, v in out.items() if v[1] >= 0.95), default=None)
    can = out["0.9,2"]
    print(f"  {tag}: канон (0.9,2): duty {can[0]*100:.1f}% acc {can[1]:.4f} | "
          f"лучший@0.95: {best[0]*100:.1f}% (acc {best[1]:.4f}, {best[2]})" if best else
          f"  {tag}: канон (0.9,2): duty {can[0]*100:.1f}% acc {can[1]:.4f} | acc>=0.95 недостижим",
          flush=True)
    return out, best

ids, V = build_ids(text)
n_tr = int(len(ids) * 0.90)
tr, ho = ids[:n_tr], ids[n_tr:]
print(f"стенд ×1: train {len(tr)} holdout {len(ho)} V={V}  t={time.time()-T0:.1f}s", flush=True)
results = {}

# --- контроль B0 (c8) и B1 (каскад) --- #
sh8 = {8: count_ngrams(tr, 8, V)}
sh8[8] = (sh8[8][0], sh8[8][1], topmap(sh8[8][1], V))
recs = holdout_records(sh8, ho, V, (8,))
results["B0"], best_b0 = full_grid("B0 c8 (контроль)", recs, (8,))
del sh8, recs

# --- F1 цифро-фолд --- #
foldd = "".join("0" if "0" <= ch <= "9" else ch for ch in text)
ids_d, Vd = build_ids(foldd)
tr_d, ho_d = ids_d[:n_tr], ids_d[n_tr:]
print(f"цифро-фолд: V {V}→{Vd}  t={time.time()-T0:.1f}s", flush=True)
ctx, pair = count_ngrams(tr_d, 8, Vd)
sh = {8: (ctx, pair, topmap(pair, Vd))}
recs = holdout_records(sh, ho_d, Vd, (8,))
results["F1_digit_c8"], best_f1 = full_grid("F1 цифро-фолд c8", recs, (8,))
del sh, recs, ctx, pair

# --- F2 wildcard-маски по слотам j --- #
def count_masked(seq, n, V, j, ho_eval=None):
    """код окна n с выброшенным слотом j (0=самый старый)."""
    pows = [V ** (n - 1 - t) for t in range(n)]
    psum = sum(pows) - pows[j]
    win = [0] * n
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    for i, x in enumerate(seq):
        win[i % n] = x
        if i >= n - 1 and i + 1 < len(seq):
            code = 0
            for t in range(n):
                if t == j: continue
                code += win[(i - (n - 1) + t) % n] * pows[t]
            ctx[code] += 1
            pair[code * V + seq[i + 1]] += 1
    return ctx, pair

results["F2_wildcard"] = {}
for j in (1, 2, 3, 4):
    ctx, pair = count_masked(tr, 8, V, j)
    top = topmap(pair, V)
    # holdout с той же маской
    pows = [V ** (8 - 1 - t) for t in range(8)]
    win = [0] * 8
    recs_m = []
    for i in range(len(ho) - 1):
        win[i % 8] = ho[i]
        if i < 7: continue
        code = 0
        for t in range(8):
            if t == j: continue
            code += win[(i - 7 + t) % 8] * pows[t]
        ncnt = ctx.get(code, 0)
        if ncnt == 0:
            recs_m.append(((0, 0.0, False),))
        else:
            c, tx = top[code]
            recs_m.append(((ncnt, c / ncnt, tx == ho[i + 1]),))
    results["F2_wildcard"][f"j={j}"], _ = full_grid(f"F2 маска j={j}", recs_m, (None,))
    del ctx, pair, top, recs_m

# --- F3 каскад + цифро-фолд --- #
def build_shelf(sub, levels, Vv):
    shelf = {}
    for n in levels:
        ctx, pair = count_ngrams(sub, n, Vv)
        shelf[n] = (ctx, pair, topmap(pair, Vv))
    return shelf

sh = build_shelf(tr_d, (8, 6, 4), Vd)
recs = holdout_records(sh, ho_d, Vd, (8, 6, 4))
results["F3_cascade_digit"], best_f3 = full_grid("F3 каскад 8>6>4 + цифро-фолд", recs, (8, 6, 4))

# --- вердикты по замороженным прогнозам --- #
b0_can = results["B0"]["0.9,2"]
f1_can = results["F1_digit_c8"]["0.9,2"]
pf1 = "✓" if (f1_can[0] - b0_can[0]) >= 0.02 and f1_can[1] >= 0.95 else (
    "✗ СМЕРТЬ" if (f1_can[0] - b0_can[0]) < 0.01 else "~ погранично")
f2_best = max((v[0] for m in results["F2_wildcard"].values() for k, v in m.items() if v[1] >= 0.95), default=0)
pf2 = "✓" if f2_best >= 0.38 else ("✗ СМЕРТЬ" if f2_best < 0.36 else "~ погранично")
f3_best = max((v[0] for k, v in results["F3_cascade_digit"].items() if v[1] >= 0.95), default=0)
pf3 = "✓" if f3_best >= 0.44 else ("✗ СМЕРТЬ" if f3_best < 0.41 else "~ погранично")
print(f"\nВЕРДИКТЫ: P-F1 {pf1} (канон Δduty={((f1_can[0]-b0_can[0])*100):+.1f} п.п.) | "
      f"P-F2 {pf2} (макс {f2_best*100:.1f}%@0.95) | P-F3 {pf3} (макс {f3_best*100:.1f}%@0.95)")

results.update(verdicts=dict(P_F1=pf1, P_F2=pf2, P_F3=pf3),
               elapsed_s=time.time() - T0)
json.dump(results, open(ROOT / "research/CALC08_FOLDS.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/CALC08_FOLDS.json  t={time.time()-T0:.0f}s")
