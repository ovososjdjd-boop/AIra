#!/usr/bin/env python3
# CALC-03b: исправление слепоты к счёту — маршрутизатор по паре (f, N).
# CALC-03 показал немонотонность P(acc|f) на f≈1 (N=1-синглтоны).
# Здесь: бины f×N, монотонность при N≥2, фронтир правила (f≥θ, N≥Nmin),
# достижимость α=0.95 на живом тексте. Расчёт, обучение НЕ запускается.
import json, time, math, collections

T0 = time.time()
text = open("corpus_external/wikitext2/train.txt", encoding="utf-8").read()
alphabet = {c: i for i, c in enumerate(sorted(set(text)))}
V = len(alphabet)
ids = [alphabet[c] for c in text]
n_tr = int(len(ids) * 0.90)
tr, ho = ids[:n_tr], ids[n_tr:]
uni = collections.Counter(tr); pu = {k: v / len(tr) for k, v in uni.items()}

def count_ngrams(seq, n):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    code = 0; base = V ** n
    for i, x in enumerate(seq):
        code = (code * V + x) % base
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1; pair[code * V + seq[i + 1]] += 1
    return ctx, pair

ctx8, pair8 = count_ngrams(tr, 8)
top8 = {}
for k, c in pair8.items():
    code, x = divmod(k, V)
    cur = top8.get(code)
    if cur is None or c > cur[1]:
        top8[code] = (x, c)
print(f"счётчики готовы  t={time.time()-T0:.1f}s", flush=True)

recs = []
code = 0
for i in range(len(ho) - 1):
    code = (code * V + ho[i]) % (V ** 8)
    if i < 7: continue
    x = ho[i + 1]
    n8c = ctx8.get(code, 0)
    if n8c == 0:
        recs.append((0.0, 0, False)); continue
    tx, tc = top8[code]
    recs.append((tc / n8c, n8c, tx == x))
print(f"recs={len(recs)}  t={time.time()-T0:.1f}s", flush=True)

# --- бины N ---
print("\nacc по классам счёта N (все f):")
for lo, hi, tag in [(1, 1, "N=1 синглтон"), (2, 4, "N=2..4"), (5, 16, "N=5..16"),
                    (17, 10 ** 9, "N≥17")]:
    sel = [r for r in recs if lo <= r[1] <= hi]
    acc = sum(r[2] for r in sel) / max(len(sel), 1)
    print(f"  {tag:<14} n={len(sel):>7}  acc={acc:.4f}")

# --- монотонность P(acc|f) при N≥2 ---
print("\nнадёжность f→acc при N≥2:")
BINS = [(0, .5), (.5, .7), (.7, .85), (.85, .9), (.9, .95), (.95, .99), (.99, 1.01)]
mono_seq = []
for lo, hi in BINS:
    sel = [r for r in recs if r[1] >= 2 and lo <= r[0] < hi]
    acc = sum(r[2] for r in sel) / max(len(sel), 1)
    mono_seq.append((acc, len(sel)))
    print(f"  f∈[{lo:.2f},{hi:.2f}): n={len(sel):>7}  acc={acc:.4f}")
accs = [a for a, n in mono_seq if n > 300]
mono2 = all(a <= b + 0.02 for a, b in zip(accs, accs[1:]))
print(f"монотонность при N≥2 (допуск 2 п.п.): {'ДА' if mono2 else 'НЕТ'}")

# --- фронтир правила (f≥θ, N≥Nmin), цели α ---
print("\nфронтир (f≥θ, N≥Nmin):")
best_map = {}
for nmin in (1, 2, 5):
    for t in (0.8, 0.85, 0.9, 0.92, 0.95, 0.97):
        sel = [r for r in recs if r[0] >= t and r[1] >= nmin]
        if not sel:
            continue
        acc = sum(r[2] for r in sel) / len(sel)
        cov = len(sel) / len(recs)
        best_map[(nmin, t)] = (round(cov, 4), round(acc, 4))
        print(f"  Nmin={nmin} θ={t:>4.2f}: покрытие {cov*100:5.1f}%  acc {acc:.4f}")

print("\nдостижимость целей (макс покрытие при acc≥α):")
for a in (0.90, 0.95, 0.97):
    cands = [(v, k) for k, v in best_map.items() if v[1] >= a]
    if cands:
        (cov, acc), (nmin, t) = max(cands)
        print(f"  α={a}: Nmin={nmin} θ={t} → покрытие {cov*100:.1f}%  acc {acc:.4f}")
    else:
        print(f"  α={a}: недостижима полкой c8 на живом тексте")

json.dump({"N_bins": True, "mono_N2": mono2, "frontier": {f"{k[0]},{k[1]}": v for k, v in best_map.items()},
           "n_positions": len(recs), "elapsed_s": time.time() - T0},
          open("research/CALC03B_ROUTER2D.json", "w"), ensure_ascii=False, indent=1)
print(f"\njson -> research/CALC03B_ROUTER2D.json  t={time.time()-T0:.1f}s")
