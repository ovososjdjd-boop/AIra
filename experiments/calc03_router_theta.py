#!/usr/bin/env python3
# CALC-03 (расчёт, НЕ обучение): аналитический порог маршрутизатора L0/L1.
# Тест теории оптимального отказа (SpecCascade Lemma 4): порог по уверенности
# оптимален ⇔ P(correct | f) монотонна и θ-фронтир == оракул-фронтир.
# Валидация: ручной рабочий θ = 0.90–0.95 (EXP-15b) должен попасть в ±3 п.п.
# от аналитического θ* для целевого качества α ≈ 0.95.
# Полка: канон char-c8 (backoff 8→4→1, λ=2), wikitext2, тот же holdout-срез.
import json, time, math, collections

T0 = time.time()
text = open("corpus_external/wikitext2/train.txt", encoding="utf-8").read()
alphabet = {c: i for i, c in enumerate(sorted(set(text)))}
V = len(alphabet)
ids = [alphabet[c] for c in text]
n_tr = int(len(ids) * 0.90)
tr, ho = ids[:n_tr], ids[n_tr:]
CPT = 1.0
uni = collections.Counter(tr); pu = {k: v / len(tr) for k, v in uni.items()}
print(f"train {len(tr)} holdout {len(ho)}  t={time.time()-T0:.1f}s", flush=True)

def count_ngrams(seq, n):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    code = 0; base = V ** n
    for i, x in enumerate(seq):
        code = (code * V + x) % base
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1; pair[code * V + seq[i + 1]] += 1
    return ctx, pair

ctx8, pair8 = count_ngrams(tr, 8)
ctx4, pair4 = count_ngrams(tr, 4)
# топ-продолжение на контекст (для аргмакс-полки)
top8 = {}
for k, c in pair8.items():
    code, x = divmod(k, V)
    cur = top8.get(code)
    if cur is None or c > cur[1]:
        top8[code] = (x, c)
print(f"n8 pair={len(pair8)} top-ctx={len(top8)} | n4 pair={len(pair4)}  t={time.time()-T0:.1f}s", flush=True)

LAM = 2.0
def p_backoff(code8, x):  # CE-оценка продолжения при отказе полки (fallback-ЕМ)
    c4k = code8 % (V ** 4)
    c4 = pair4.get(c4k * V + x, 0); n4 = ctx4.get(c4k, 0)
    p_u = pu.get(x, 1e-9)
    p4 = (c4 + LAM * p_u) / (n4 + LAM) if n4 else p_u
    c8 = pair8.get(code8 * V + x, 0); n8c = ctx8.get(code8, 0)
    return (c8 + LAM * p4) / (n8c + LAM) if n8c else p4

# --- проход holdout: по каждой позиции f, верность аргмакс, CE-фолбэка ---
recs = []
code = 0
for i in range(len(ho) - 1):
    code = (code * V + ho[i]) % (V ** 8)
    if i < 7: continue
    x = ho[i + 1]
    n8c = ctx8.get(code, 0)
    if n8c == 0:
        recs.append((0.0, False, -math.log2(p_backoff(code, x)), n8c))  # полка молчит
        continue
    tx, tc = top8[code]
    f = tc / n8c
    recs.append((f, tx == x, -math.log2(p_backoff(code, x)), n8c))
print(f"recs={len(recs)}  t={time.time()-T0:.1f}s", flush=True)

# --- надёжность: P(верно | f) по бинам + монотонность ---
BINS = [(0, .5), (.5, .7), (.7, .85), (.85, .9), (.9, .95), (.95, .99), (.99, 1.01)]
bins = []
for lo, hi in BINS:
    sel = [r for r in recs if lo <= r[0] < hi]
    acc = sum(r[1] for r in sel) / max(len(sel), 1)
    bins.append({"lo": lo, "hi": hi, "n": len(sel),
                 "acc": round(acc, 4)})
accs = [b["acc"] for b in bins if b["n"] > 300]
mono = all(a <= b + 0.02 for a, b in zip(accs, accs[1:]))
print("надёжность f→acc:")
for b in bins:
    print(f"  f∈[{b['lo']:.2f},{b['hi']:.2f}): n={b['n']:>7}  acc={b['acc']:.4f}")
print(f"монотонность (допуск 2 п.п.): {'ДА' if mono else 'НЕТ'}")

# --- θ-фронтир и оракул-фронтир (макс. покрытие при целевом acc ≥ α) ---
def frontier_by_threshold(theta):
    sel = [r for r in recs if r[0] >= theta]
    if not sel: return 0.0, 1.0
    acc = sum(r[1] for r in sel) / len(sel)
    return len(sel) / len(recs), acc

oracul = sorted(recs, key=lambda r: -r[0])
def frontier_oracle(k):
    sel = oracul[:k]
    acc = sum(r[1] for r in sel) / len(sel)
    return len(sel) / len(recs), acc

grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99]
print("\nθ-фронтир (полка отвечает при f ≥ θ):")
theta_rows = []
for t in grid:
    cov, acc = frontier_by_threshold(t)
    theta_rows.append({"theta": t, "coverage": round(cov, 4), "acc": round(acc, 4)})
    print(f"  θ={t:>4.2f}  покрытие {cov*100:5.1f}%  acc {acc:.4f}")

targets = [0.90, 0.95, 0.97]
print("\nаналитические θ* при целевом качестве α (θ-фронтир) и оракул-prefix сравнение:")
tstar = {}
for a in targets:
    ok = [r for r in theta_rows if r["acc"] >= a]
    best = max(ok, key=lambda r: r["coverage"], default=None)
    t = best["theta"] if best else None
    # оракул: наибольший prefix с acc ≥ a
    cov_o, k = 0.0, 0
    s, tot = 0, 0
    for kk, r in enumerate(oracul, 1):
        s += r[1]
        if s / kk >= a:
            k = kk
    cov_o = k / len(recs)
    tstar[a] = {"theta": t, "coverage": best["coverage"] if best else 0,
                "oracle_coverage": round(cov_o, 4)}
    print(f"  α={a:.2f}: θ*={t}  покрытие {best['coverage']*100 if best else 0:5.1f}%  | оракул-prefix max {cov_o*100:5.1f}%")

# --- экономика при θ-выборе: гибрид CE полка+фолбэк-ЕМ ---
print("\nгибрид CE (полка при f≥θ с её CE там, фолбэк-ЕМ на остатке):")
for t in [0.85, 0.9, 0.92, 0.95]:
    ce_all, ce_cov = 0.0, 0.0
    n_cov = 0
    for f, ok, cef, _n in recs:
        ce_all += cef
    tot_ce_base = ce_all / len(recs)
    print(f"  (фолбэк-ЕМ везде {tot_ce_base:.3f} бит/поз; точечный гибрид требует CE зоны на позициях — берём из EXP-15b констант при сборке бюджета CALC-04)")

out = {"bins": bins, "monotone_ok": mono, "theta_rows": theta_rows,
       "targets": {str(k): v for k, v in tstar.items()},
       "n_positions": len(recs), "elapsed_s": time.time() - T0}
json.dump(out, open("research/CALC03_ROUTER.json", "w"), ensure_ascii=False, indent=1)
print(f"\njson -> research/CALC03_ROUTER.json  t={time.time()-T0:.1f}s")
