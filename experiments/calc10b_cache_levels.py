#!/usr/bin/env python3
# CALC-10b (расчёт, НЕ обучение): кэш по длинным ключам — решающий конфиг.
# ===== ПРЕДСКАЗАНИЯ ЗАМОРОЖЕНЫ ДО ПРОГОНА =====
#  P-L1b: c8-повторы в окне 4096 (N_l>=1, f_l≡1): добавка к каскаду 42.4%
#     >= +0.5 п.п. при add-acc >= 0.90 (достаточно для 0.95-гейта: нужно ~0.94).
#     Смерть семейства «кэш»: add-acc < 0.85 при c8/N_l>=1
#     (тогда локальная повторность в принципе не дотягивает до 0.95-гейта).
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

# --- глобальный глубокий каскад (прицельный, как в CALC-09/10a) --- #
LEVELS = (16, 12, 8, 6, 4)
mods = {n: V ** n for n in LEVELS}
targets = {n: set() for n in LEVELS}
codes_ho = {n: 0 for n in LEVELS}
ho_rows = []
for i in range(len(ho) - 1):
    for n in LEVELS:
        codes_ho[n] = (codes_ho[n] * V + ho[i]) % mods[n]
    if i < max(LEVELS) - 1: continue
    ho_rows.append(({n: codes_ho[n] for n in LEVELS}, ho[i + 1], i))
    for n in LEVELS:
        targets[n].add(codes_ho[n])
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
print(f"глобальный прицель готов  t={time.time()-T0:.0f}s", flush=True)

def global_fire(row, theta=0.90, nmin=5):
    for n in LEVELS:
        cnt = gctx[n].get(row[n], 0)
        if cnt >= nmin:
            tc, tx = gtop[n][row[n]]
            if tc / cnt >= theta:
                return True, tx
    return False, None

# --- локальный кэш с уровнями: c8 N_l>=1, c6 N_l>=2, c4 N_l>=4 --- #
W = 4096
CFGS = [("c8N1", 8, 1), ("c6N2", 6, 2), ("c8N2", 8, 2), ("c6N4", 6, 4)]
result = {}
for tag, L, nl_min in CFGS:
    cache = {}
    ml = V ** L
    code = 0
    base_cov = base_hit = add_cov = add_hit = 0
    for row, nxt, pos in ho_rows:
        # запись текущего символа ho[pos] под ключом контекста ho[pos-L..pos-1]
        if pos >= L:
            key = 0
            for t in range(L):
                key = key * V + ho[pos - L + t]
            if key in cache:
                cache[key][0][ho[pos]] += 1
                cache[key][1] = pos
            else:
                cache[key] = [collections.Counter({ho[pos]: 1}), pos]
        g_fire, g_tx = global_fire(row)
        if g_fire:
            base_cov += 1; base_hit += (g_tx == nxt)
            continue
        # чтение для предсказания ho[pos+1]: ключ ho[pos-L+1..pos]
        if pos >= L - 1:
            key = 0
            for t in range(L):
                key = key * V + ho[pos - L + 1 + t]
            ent = cache.get(key)
            if ent and pos - ent[1] < W:
                nl = sum(ent[0].values())
                if nl >= nl_min:
                    tx, tc = ent[0].most_common(1)[0]
                    add_cov += 1; add_hit += (tx == nxt)
        if (pos + 1) % 200_000 == 0:
            dead = [k for k, v in cache.items() if pos - v[1] >= W]
            for k in dead:
                del cache[k]
    duty = (base_cov + add_cov) / len(ho_rows)
    acc = (base_hit + add_hit) / max(base_cov + add_cov, 1)
    add_acc = add_hit / max(add_cov, 1)
    result[tag] = dict(add_pp=round(add_cov / len(ho_rows), 4), add_acc=round(add_acc, 4),
                       union_duty=round(duty, 4), union_acc=round(acc, 4))
    print(f"  {tag}: +{add_cov/len(ho_rows)*100:5.2f} п.п. | add-acc {add_acc:.4f} | "
          f"союз duty {duty*100:5.1f}% acc {acc:.4f}  t={time.time()-T0:.0f}s", flush=True)

c8 = result["c8N1"]
verd = "✓" if c8["add_pp"] >= 0.005 and c8["add_acc"] >= 0.90 else (
    "✗ СМЕРТЬ СЕМЕЙСТВА «КЭШ»" if c8["add_acc"] < 0.85 else "~ погранично")
print(f"\nP-L1b {verd} (c8N1: +{c8['add_pp']*100:.2f} п.п. @ {c8['add_acc']:.4f})")

json.dump(dict(results=result, verdict=verd, elapsed_s=time.time() - T0),
          open(ROOT / "research/CALC10B_CACHE.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/CALC10B_CACHE.json  t={time.time()-T0:.0f}s")
