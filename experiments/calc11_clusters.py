#!/usr/bin/env python3
# CALC-11 (расчёт, НЕ обучение): дистрибутивные смысловые классы (Браун-идея)
# как ключи памяти — последняя бумажная инварианта.
# Метод: k-means (numpy) по дистрибутивному профилю «кто идёт за словом»
# на топ-1500 слов → K=100 классов; ключи (класс,класс) → топ-СЛОВО.
# Никаких градиентов — чистые счётчики и линейная алгебра.
# ===== ПРЕДСКАЗАНИЯ ЗАМОРОЖЕНЫ ДО ПРОГОНА =====
#  P-K1: кластерный w2-орган: duty@(0.90,N>=2) >= 6% (×10 к поверхностному
#        w2-word 0.6%) при acc >= 0.55. Фронтир: duty >= 2% при acc>=0.95.
#        Смерть: duty@канон < 2.5% ИЛИ acc@канон < 0.40 (классы не несут сигнала).
import json, time, math, collections, re
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")
n_tr = int(len(text) * 0.90)
tr_words = text[:n_tr].split()
ho_words = text[n_tr:].split()
print(f"train {len(tr_words)} слов | holdout {len(ho_words)}  t={time.time()-T0:.1f}s", flush=True)

freq = collections.Counter(tr_words)
TOPW = 1500
TOPF = 400
vocab = [w for w, _ in freq.most_common(TOPW)]
w2i = {w: i for i, w in enumerate(vocab)}
feats = [w for w, _ in freq.most_common(TOPF)]
f2i = {w: i for i, w in enumerate(feats)}

import numpy as np
X = np.zeros((TOPW, TOPF), dtype=np.float64)
prev = tr_words[0]
for w in tr_words[1:]:
    i = w2i.get(prev)
    if i is not None:
        j = f2i.get(w)
        if j is not None:
            X[i, j] += 1
    prev = w
Xn = X / np.maximum(X.sum(1, keepdims=True), 1e-9)

# k-means (евклид по нормированным профилям), детерминированный отбор центров:
# kmeans++-лайт: первый центр — самый частотный вектор, дальше — макс-дальние
K = 100
rng = np.random.RandomState(20260807)
C = np.empty((K, TOPF))
C[0] = Xn[0]
dmin = ((Xn - C[0]) ** 2).sum(1)
for k in range(1, K):
    C[k] = Xn[int(np.argmax(dmin))]
    dmin = np.minimum(dmin, ((Xn - C[k]) ** 2).sum(1))
cid = np.zeros(TOPW, dtype=np.int64)
for it in range(20):
    d2 = ((Xn[:, None, :] - C[None, :, :]) ** 2).sum(2)
    cid = d2.argmin(1)
    newC = np.zeros_like(C)
    np.add.at(newC, cid, Xn)
    cnts = np.bincount(cid, minlength=K).clip(1)
    C = newC / cnts[:, None]
print(f"k-means готов (K={K}, топ-слов {TOPW})  t={time.time()-T0:.0f}s", flush=True)

def cl(w):
    i = w2i.get(w)
    return int(cid[i]) + 1 if i is not None else 0  # 0 = вне топ-1500

# кластерная полка w2: ключ (c_{i-2}, c_{i-1}) → топ-слово; и w1: (c_{i-1})
TRc = [cl(w) for w in tr_words]
HOc = [cl(w) for w in ho_words]

def build_and_eval(key_len):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    for i in range(key_len, len(tr_words) - 1):
        key = tuple(TRc[i - key_len + 1: i + 1])
        ctx[key] += 1
        pair[(key, tr_words[i + 1])] += 1
    top = {}
    for (k, x), c in pair.items():
        cur = top.get(k)
        if cur is None or c > cur[0]:
            top[k] = (c, x)
    out = {}
    for th, nm in [(0.85, 2), (0.90, 2), (0.95, 2), (0.90, 5), (0.95, 5), (0.98, 5)]:
        cov = hit = tot = 0
        for i in range(key_len, len(ho_words) - 1):
            tot += 1
            key = tuple(HOc[i - key_len + 1: i + 1])
            n = ctx.get(key, 0)
            if n >= nm:
                tc, tx = top[key]
                if tc / n >= th:
                    cov += 1; hit += (tx == ho_words[i + 1])
        out[f"{th},{nm}"] = (round(cov / tot, 4), round(hit / max(cov, 1), 4))
    return out

res = {}
print("длина ключа 1 (w1):", flush=True)
res["c-w1"] = build_and_eval(1)
print("длина ключа 2 (w2):", flush=True)
res["c-w2"] = build_and_eval(2)
for tag in ("c-w1", "c-w2"):
    d, a = res[tag]["0.9,2"]
    best95 = max(((v[0], v[1], k) for k, v in res[tag].items() if v[1] >= 0.95), default=None)
    print(f"  {tag}: канон duty {d*100:5.2f}% acc {a:.4f} | @0.95: "
          + (f"{best95[0]*100:.2f}% (acc {best95[1]:.4f} {best95[2]})" if best95 else "нет"), flush=True)

d, a = res["c-w2"]["0.9,2"]
b95 = max((v[0] for v in res["c-w2"].values() if v[1] >= 0.95), default=0)
verd = "✓" if d >= 0.06 and a >= 0.55 and b95 >= 0.02 else (
    "✗ СМЕРТЬ" if d < 0.025 or a < 0.40 else "~ погранично")
print(f"P-K1 {verd}")

# примеры классов для отчёта (интерпретируемость)
examples = {}
for k in range(K):
    ws = [vocab[i] for i in range(TOPW) if cid[i] == k][:8]
    if ws:
        examples[k] = ws
json.dump(dict(clusters=res, verdict=verd, cluster_examples={str(k): v for k, v in list(examples.items())[:12]},
               elapsed_s=time.time() - T0),
          open(ROOT / "research/CALC11_CLUSTERS.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/CALC11_CLUSTERS.json  t={time.time()-T0:.0f}s")
