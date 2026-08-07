#!/usr/bin/env python3
# CALC-05 опорные числа (расчёт, НЕ обучение).
# 1) распределение c8-пар: доля синглтонов (c==1), их вклад в покрытие;
# 2) закон роста числа пар полки с объёмом корпуса (Heaps-экспонент) char-c8;
# 3) смета хранения полки в байтах (сырая vs фильтр N≥2 vs проекция ×30);
# 4) траектория novelty-duty по потоку обучения (online-полка, среднее за эпох).
import json, time, math, collections

T0 = time.time()
text = open("corpus_external/wikitext2/train.txt", encoding="utf-8").read()
alphabet = {c: i for i, c in enumerate(sorted(set(text)))}
V = len(alphabet)
ids = [alphabet[c] for c in text]
n_tr = int(len(ids) * 0.90)
tr = ids[:n_tr]

def count_pairs(seq, n=8):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    code = 0; base = V ** n
    for i, x in enumerate(seq):
        code = (code * V + x) % base
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1; pair[code * V + seq[i + 1]] += 1
    return ctx, pair

# 1) полная статистика c8
ctx, pair = count_pairs(tr)
tot_hits = sum(pair.values())
c1_pairs = sum(1 for c in pair.values() if c == 1)
c1_hits = sum(c for c in pair.values() if c == 1)
c2p_pairs = sum(1 for c in pair.values() if c >= 2)
c2p_hits = sum(c for c in pair.values() if c >= 2)
print(f"пар c8: {len(pair)} | синглтонов: {c1_pairs} ({c1_pairs/len(pair)*100:.1f}%) — обращений {c1_hits/tot_hits*100:.1f}%")
print(f"пар c>=2: {c2p_pairs} ({c2p_pairs/len(pair)*100:.1f}%) — обращений {c2p_hits/tot_hits*100:.1f}%")

# 2) Heaps: число пар по подвыборкам
heaps = []
for fr in (1/32, 1/16, 1/8, 1/4, 1/2, 1.0):
    n = int(len(tr) * fr)
    _, p = count_pairs(tr[:n])
    heaps.append((n, len(p)))
    print(f"frac={fr:>5.3f} N={n:>8}  уникальных пар={len(p)}  t={time.time()-T0:.1f}s", flush=True)
xs = [math.log(n) for n, _ in heaps]; ys = [math.log(p) for _, p in heaps]
mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
bH = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
aH = my - bH * mx
print(f"закон heaps: пар(N) = exp({aH:.3f})·N^{bH:.3f}")

# 3) смета хранения (компактная запись: ctx-hash 4Б + next 1Б + cnt 2Б = 7 Б/пара; + 25% оверхед хеш-таблицы ≈ 9 Б)
B_REC = 9
def mb(x_pairs): return x_pairs * B_REC / 1e6
N30 = heaps[-1][0] * 30
p30 = math.exp(aH) * N30 ** bH
print(f"\nсмета полки (запись {B_REC} Б/пара):")
print(f"  сейчас: все пары {mb(len(pair)):.0f} МБ | N>=2 {mb(c2p_pairs):.0f} МБ")
print(f"  ×30 (Kaggle): все {mb(p30):.0f} МБ | N>=2 (оценка ×0.6) {mb(p30*0.6):.0f} МБ")

# 4) траектория duty: online-полка растёт вместе с обучением
# s(K) = s_end·√(K/N_end) — качественно-ограниченная из CALC-03b: s_end=0.343
s_end = 0.343
first_epoch_mean_cov = (2/3) * s_end          # integral √x dx на 0..1
print(f"\nnovelty-duty (s_end=0.343 качественно-ограниченная из CALC-03b):")
print(f"  средняя покрытие за 1-й проход потока: {first_epoch_mean_cov*100:.1f}% → duty 1-го прохода {100-first_epoch_mean_cov*100:.1f}%")
print(f"  устойчивый duty (полка собрана): {100-s_end*100:.1f}%")

json.dump({
    "pairs_all": len(pair), "pairs_c1": c1_pairs, "hits_c1_share": c1_hits/tot_hits,
    "pairs_c2p": c2p_pairs, "hits_c2p_share": c2p_hits/tot_hits,
    "heaps": {"a": aH, "b": bH, "points": heaps},
    "shelf_mb_now_all": mb(len(pair)), "shelf_mb_now_n2": mb(c2p_pairs),
    "shelf_mb_x30_all": mb(p30), "shelf_mb_x30_n2": mb(p30 * 0.6),
    "duty_first_epoch": 1 - first_epoch_mean_cov, "duty_steady": 1 - s_end,
    "elapsed_s": time.time() - T0,
}, open("research/CALC05_NUMBERS.json", "w"), ensure_ascii=False, indent=1)
print(f"\njson -> research/CALC05_NUMBERS.json  t={time.time()-T0:.1f}s")
