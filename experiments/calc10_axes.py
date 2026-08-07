#!/usr/bin/env python3
# CALC-10 (расчёт, НЕ обучение): две снятые инвариант-оси.
# C-A: третья ось маршрутизатора — энтропия H распределения полки (привратник 3D).
# C-B: локальная тематическая память (кэш документа) поверх глобальной полки.
# ===== ПРЕДСКАЗАНИЯ ЗАМОРОЖЕНЫ ДО ПРОГОНА =====
#  P-R1 (3D-привратник, c8): макс duty при acc>=0.95 по оптимальной
#      (f,N,H)-сортировке ячеек >= 35.8% (2D-эталон 34.3%, +1.5 п.п.).
#      Смерть: < 34.8% (+0.5 п.п.).
#  P-L1 (кэш поверх ГЛУБОКОГО каскада 42.4%@0.9579, правило (0.90,N>=5)):
#      союз (глобальный каскад ИЛИ кэш c4 с N_l>=2 в окне 4096, f_l>=θ_l)
#      даёт duty >= 45.4% при acc >= 0.95. Смерть: прирост < +1.0 п.п.
import json, time, math, collections
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")
alpha = {c: i for i, c in enumerate(sorted(set(text)))}
V = len(alpha)
ids = [alpha[c] for c in text]
n_tr = int(len(ids) * 0.90)
tr, ho = ids[:n_tr], ids[n_tr:]

# ---------- C-A: 3D-привратник на c8 ---------- #
print("=== C-A: (f, N, H) привратник, c8 ===", flush=True)
base = V ** 8
target = set()
code = 0
for i in range(len(ho) - 1):
    code = (code * V + ho[i]) % base
    if i >= 7:
        target.add(code)
ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
code = 0
for pos, x in enumerate(tr):
    code = (code * V + x) % base
    if pos >= 7 and pos + 1 < len(tr) and code in target:
        ctx[code] += 1
        pair[code * V + tr[pos + 1]] += 1
dist = collections.defaultdict(list)
for k, c in pair.items():
    cd, x = divmod(k, V)
    dist[cd].append((x, c))
del pair
recs = []  # (f, N, H, ok)
code = 0
for i in range(len(ho) - 1):
    code = (code * V + ho[i]) % base
    if i < 7: continue
    n = ctx.get(code, 0)
    if n < 2:
        recs.append(None); continue
    d = dist[code]
    d.sort(key=lambda t: -t[1])
    tx, tc = d[0]
    h = -sum((c / n) * math.log2(c / n) for _, c in d)
    recs.append((tc / n, n, h, tx == ho[i + 1]))
print(f"записей с полкой: {sum(r is not None for r in recs)} из {len(recs)}  t={time.time()-T0:.0f}s", flush=True)

# фронтир по ячейкам f×N×H, жадная сортировка по acc
def frontier(cells, alpha=0.95):
    tot_ok = tot_n = 0
    cells_sorted = sorted(cells, key=lambda c: -c[2])
    best = (0.0, 0.0)
    for n_c, n_ok, acc in cells_sorted:
        tot_ok += n_ok; tot_n += n_c
        cum_acc = tot_ok / tot_n
        duty = tot_n / len(recs)
        if cum_acc >= alpha and duty > best[0]:
            best = (duty, cum_acc)
    return best

# 2D-эталон этой же метрикой (без H)
F_B = [(0, .9), (.9, .95), (.95, .99), (.99, 1.01)]
N_B = [(2, 4), (5, 16), (17, 10 ** 9)]
H_B = [(0, .5), (.5, 1.0), (1.0, 2.0), (2.0, 99)]
cells2 = {}
cells3 = {}
for r in recs:
    if r is None: continue
    f, n, h, ok = r
    fi = next(i for i, (lo, hi) in enumerate(F_B) if lo <= f < hi)
    ni = next(i for i, (lo, hi) in enumerate(N_B) if lo <= n <= hi)
    hi_ = next(i for i, (lo, hi) in enumerate(H_B) if lo <= h < hi)
    for tab, key in ((cells2, (fi, ni)), (cells3, (fi, ni, hi_))):
        a, b = tab.get(key, (0, 0))
        tab[key] = (a + 1, b + ok)
c2 = [(v[0], v[1], v[1] / v[0]) for v in cells2.values() if v[0] >= 30]
c3 = [(v[0], v[1], v[1] / v[0]) for v in cells3.values() if v[0] >= 30]
d2, a2 = frontier(c2)
d3, a3 = frontier(c3)
pr1 = "✓" if d3 >= 0.358 and a3 >= 0.95 else ("✗ СМЕРТЬ" if d3 < 0.348 else "~ погранично")
print(f"  2D-фронтир: duty {d2*100:.1f}% acc {a2:.4f}")
print(f"  3D-фронтир: duty {d3*100:.1f}% acc {a3:.4f}  → P-R1 {pr1}  t={time.time()-T0:.0f}s", flush=True)
del ctx, dist

# ---------- C-B: локальный кэш поверх глубокого каскада ---------- #
print("\n=== C-B: кэш документа (c4, окно 4096) поверх каскада 16>12>8>6>4 ===", flush=True)
LEVELS = (16, 12, 8, 6, 4)
mods = {n: V ** n for n in LEVELS}
targets = {n: set() for n in LEVELS}
codes_ho = {n: 0 for n in LEVELS}
ho_rows = []
for i in range(len(ho) - 1):
    for n in LEVELS:
        codes_ho[n] = (codes_ho[n] * V + ho[i]) % mods[n]
    if i < max(LEVELS) - 1: continue
    ho_rows.append(({n: codes_ho[n] for n in LEVELS}, ho[i + 1]))
    for n in LEVELS:
        targets[n].add(codes_ho[n])
print(f"проба {len(ho_rows)} позиций  t={time.time()-T0:.0f}s", flush=True)

gctx = {n: collections.defaultdict(int) for n in LEVELS}
gpair = {n: collections.defaultdict(int) for n in LEVELS}
codes = {n: 0 for n in LEVELS}
for pos, x in enumerate(tr):
    for n in LEVELS:
        codes[n] = (codes[n] * V + x) % mods[n]
    if pos + 1 >= len(tr): continue
    nxt = tr[pos + 1]
    for n in LEVELS:
        if pos >= n - 1 and codes[n] in targets[n]:
            c = codes[n]
            gctx[n][c] += 1
            gpair[n][c * V + nxt] += 1
gtop = {n: {} for n in LEVELS}
for n in LEVELS:
    for k, cnt in gpair[n].items():
        cd, xx = divmod(k, V)
        cur = gtop[n].get(cd)
        if cur is None or cnt > cur[0]:
            gtop[n][cd] = (cnt, xx)
del gpair
print(f"глобальный прицельный счёт готов  t={time.time()-T0:.0f}s", flush=True)

def global_fire(row, theta=0.90, nmin=5):
    for n in LEVELS:
        cnt = gctx[n].get(row[n], 0)
        if cnt >= nmin:
            tc, tx = gtop[n][row[n]]
            if tc / cnt >= theta:
                return True, tx
    return False, None

# кэш c4: локальные счётчики в окне W
W = 4096
FS = [1.0]  # f_l: при N_l собранном в окне f почти всегда 1 → калибруем по N_l
NMIN_L = [2, 3, 4]
cache = {}          # code -> [Counter, last_pos]
c4len = V ** 4
code = 0
local_recs = []     # (N_l, ok) для позиций, НЕ закрытых глобальным каскадом
glob_cov = glob_hit = 0
for i, (row, nxt) in enumerate(ho_rows):
    pos = i + 15  # реальный индекс символа в ho
    # обновляем кэш текущим символом ho[pos] (контекст = пред. 4)
    if pos >= 4:
        c4 = 0
        for t in range(1, 5):
            c4 = c4 * V + ho[pos - 4 + (t - 1)]
        if c4 in cache:
            cache[c4][0][ho[pos]] += 1
            cache[c4][1] = pos
        else:
            cache[c4] = [collections.Counter({ho[pos]: 1}), pos]
    g_fire, g_tx = global_fire(row)
    if g_fire:
        glob_cov += 1; glob_hit += (g_tx == nxt)
        local_recs.append(None)
    else:
        if pos >= 3:
            c4 = 0
            for t in range(4):
                c4 = c4 * V + ho[pos - 3 + t]
            ent = cache.get(c4)
            if ent and pos - ent[1] < W:
                cnt = ent[0]
                nl = sum(cnt.values())
                tx, tc = cnt.most_common(1)[0]
                local_recs.append((nl, tx == nxt))
            else:
                local_recs.append((0, False))
        else:
            local_recs.append((0, False))
    # периодическая чистка устаревших
    if i % 100_000 == 99_999:
        dead = [k for k, v in cache.items() if pos - v[1] >= W]
        for k in dead:
            del cache[k]
print(f"кэш-прогон готов, глобальное покрытие {glob_cov/len(ho_rows)*100:.1f}% acc {glob_hit/glob_cov:.4f}"
      f"  t={time.time()-T0:.0f}s", flush=True)

res_rows = []
base_duty = glob_cov / len(ho_rows)
best = (0.0, 0.0, None)
for nm in NMIN_L:
    add_cov = add_hit = 0
    tot = 0
    for r in local_recs:
        if r is None: continue
        tot += 1
        if r[0] >= nm:
            add_cov += 1; add_hit += r[1]
    duty = base_duty + add_cov / len(ho_rows)
    acc = (glob_hit + add_hit) / (glob_cov + add_cov) if (glob_cov + add_cov) else 0.0
    res_rows.append(dict(nmin_local=nm, add_pct=round(add_cov / len(ho_rows), 4),
                         add_acc=round(add_hit / max(add_cov, 1), 4),
                         duty=round(duty, 4), acc=round(acc, 4)))
    print(f"  N_l>={nm}: +{add_cov/len(ho_rows)*100:5.2f} п.п. (acc добавки {add_hit/max(add_cov,1):.4f})"
          f" → duty {duty*100:5.1f}% acc {acc:.4f}", flush=True)
    if acc >= 0.95 and duty > best[0]:
        best = (duty, acc, nm)

pl1 = "✓" if best[0] >= 0.454 else ("✗ СМЕТЬ" if best[0] < base_duty + 0.01 else "~ погранично")
pl1 = pl1.replace("СМЕТЬ", "СМЕРТЬ")
print(f"P-L1 {pl1}: лучший союз {best[0]*100:.1f}%@{best[1]:.4f} (N_l={best[2]}) при базе {base_duty*100:.1f}%")

json.dump(dict(router3d=dict(d2=[round(d2, 4), round(a2, 4)], d3=[round(d3, 4), round(a3, 4)], verdict=pr1),
               cache=dict(base_duty=round(base_duty, 4), rows=res_rows, best=best, verdict=pl1),
               elapsed_s=time.time() - T0),
          open(ROOT / "research/CALC10_AXES.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/CALC10_AXES.json  t={time.time()-T0:.0f}s")
