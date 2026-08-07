#!/usr/bin/env python3
# CALC-08b/09r (расчёт, НЕ обучение): морфо-фолд на словесном уровне + доменный штраф смесей.
# ===== ПРЕДСКАЗАНИЯ ЗАМОРОЖЕНЫ ДО ПРОГОНА =====
#  P-F4 морфо-фолд (w2-word орган, ключи из основ слов): duty@(0.90,N≥2) растёт
#     КРАТНО против поверхностного w2-word (относительный множитель >= ×1.5).
#     Смерть: множитель < ×1.15.
#  P-D1 доменный штраф (проба = wikitext2-holdout, счётчики = смесь
#     wikitext2+gsm8k+opus46+russian_lit, всего ~33М симв = ×3.4):
#     duty@(0.90,N≥2) на полной смеси ∈ [34.5%, 38%] — внедоменный текст даёт
#     малую добавку (чистый фит дал бы ~40%). Смерть: duty >= 40%
#     (смеси бесплатны — хвост-закон доменно-слеп, тогда Kaggle-рецепт проще).
import json, time, math, collections, re
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")
n_tr = int(len(text) * 0.90)
tr_txt, ho_txt = text[:n_tr], text[n_tr:]

# ---------- часть A: морфо-фолд на словесном уровне ---------- #
print("=== A. морфо-фолд w2-word орган ===", flush=True)
tr_words = tr_txt.split()
ho_words = ho_txt.split()
print(f"words: train {len(tr_words)} holdout {len(ho_words)}", flush=True)

SUFF = ("'s", "ing", "es", "ed", "ly", "s")
def stem(w):
    w = w.lower()
    for s in SUFF:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[: -len(s)]
    return w

def count_word_shelf(words, key_fn):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    k1 = k2 = None
    for i, w in enumerate(words):
        k = key_fn(w)
        if i >= 2 and i + 1 < len(words):
            ctx[(k2, k1)] += 1
            pair[((k2, k1), key_fn(words[i + 1]) if False else words[i + 1])] += 1
        k2, k1 = k1, k
    return ctx, pair

def count_w2(words):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    for i in range(2, len(words) - 1):
        c = (words[i - 2], words[i - 1])
        ctx[c] += 1
        pair[(c, words[i + 1])] += 1
    return ctx, pair

def count_w2_stem(words):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    st = [stem(w) for w in words]
    for i in range(2, len(words) - 1):
        c = (st[i - 2], st[i - 1])
        ctx[c] += 1
        pair[(c, words[i + 1])] += 1  # предсказываем ПОВЕРХНОСТНУЮ форму
    return ctx, pair

def topmap_w(pair):
    top = {}
    for (c, x), cnt in pair.items():
        cur = top.get(c)
        if cur is None or cnt > cur[0]:
            top[c] = (cnt, x)
    return top

def eval_words(ctx, top, words_ctx, words_raw):
    out = {}
    for th, nm in [(0.85, 2), (0.90, 2), (0.90, 5), (0.95, 2), (0.95, 5)]:
        cov = hit = tot = 0
        for i in range(2, len(words_raw) - 1):
            tot += 1
            c = (words_ctx[i - 2], words_ctx[i - 1])
            n = ctx.get(c, 0)
            if n >= nm:
                cnt, tx = top[c]
                if cnt / n >= th:
                    cov += 1; hit += (tx == words_raw[i + 1])
        out[f"{th},{nm}"] = (round(cov / tot, 4), round(hit / max(cov, 1), 4))
    return out

# эталон: поверхностный w2-word
ctx, pair = count_w2(tr_words)
res_surf = eval_words(ctx, topmap_w(pair), ho_words, ho_words)
d, a = res_surf["0.9,2"]
print(f"  w2-word поверхностный: канон duty {d*100:.1f}% acc {a:.4f}", flush=True)
del ctx, pair

ctx, pair = count_w2_stem(tr_words)
top = topmap_w(pair)
ho_stems = [stem(w) for w in ho_words]
res_stem = eval_words(ctx, top, ho_stems, ho_words)
d2, a2 = res_stem["0.9,2"]
print(f"  w2-word морфо: канон duty {d2*100:.1f}% acc {a2:.4f}", flush=True)
mult = (d2 / max(d, 1e-9))
pf4 = "✓" if mult >= 1.5 and a2 >= a - 0.01 else ("✗ СМЕРТЬ" if mult < 1.15 else "~ погранично")
print(f"  множитель морфо duty ×{mult:.2f} → P-F4 {pf4}  t={time.time()-T0:.0f}s", flush=True)
del ctx, pair, top

# ---------- часть B: доменный штраф смеси (probe = wikitext2-holdout) ---------- #
print("\n=== B. смесь доменов ~33М симв, прицельный счёт по пробе wikitext2-ho ===", flush=True)
gsm = []
for line in (ROOT / "corpus_external/gsm8k/train.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    gsm.append(r["question"] + "\n" + r["answer"])
gsm_txt = "\n".join(gsm)
opus = []
for line in (ROOT / "corpus_external/opus46/opus46_final.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    opus.append("\n".join(str(m.get("content") or "") for m in r["messages"]))
opus_txt = "\n".join(opus)
rus_txt = ""
for f in sorted((ROOT / "corpus_external/russian_lit").glob("*.txt")):
    rus_txt += f.read_text(encoding="utf-8") + "\n"
stages = [("+wikitext2", tr_txt), ("+gsm8k", gsm_txt), ("+opus46", opus_txt), ("+russian_lit", rus_txt)]
mix_txt = "".join(t for _, t in stages)
bounds = []
acc_len = 0
for name, t in stages:
    acc_len += len(t)
    bounds.append((name, acc_len))
print("длины смеси:", {n: l for n, l in bounds}, flush=True)

alpha = {c: i for i, c in enumerate(sorted(set(mix_txt + ho_txt)))}
V = len(alpha)
print(f"V смеси = {V}  t={time.time()-T0:.0f}s", flush=True)
ho_ids = [alpha[c] for c in ho_txt]

# целевые c8-контексты пробы
target = set()
code = 0; base = V ** 8
ho_records = []
for i in range(len(ho_ids) - 1):
    code = (code * V + ho_ids[i]) % base
    if i < 7: continue
    target.add(code)
    ho_records.append(code)
print(f"целевых контекстов: {len(target)} из {len(ho_records)} позиций  t={time.time()-T0:.0f}s", flush=True)

# единый прицельный проход по смеси со снимками на границах
ctx_c = collections.defaultdict(int); pair_c = collections.defaultdict(int)
code = 0
snap_rows = []
b_i = 0
mix_ids = (alpha[c] for c in mix_txt)
pos = 0
for x in mix_ids:
    pos += 1
    code_next = (code * V + x) % base
    # контекст = предыдущие 8 символов = code; продолжение = x
    if pos > 8 and code in target:
        ctx_c[code] += 1
        pair_c[code * V + x] += 1
    code = code_next
    if b_i < len(bounds) and pos == bounds[b_i][1]:
        # снимок: оценка пробы
        top = {}
        for k, c in pair_c.items():
            cd, xx = divmod(k, V)
            cur = top.get(cd)
            if cur is None or c > cur[0]:
                top[cd] = (c, xx)
        cov = hit = 0
        j = 0
        for i, cd in enumerate(ho_records):
            n = ctx_c.get(cd, 0)
            if n >= 2:
                cnt, tx = top[cd]
                if cnt / n >= 0.90:
                    cov += 1
                    hit += (tx == ho_ids[i + 8])
        duty = cov / len(ho_records); accv = hit / max(cov, 1)
        snap_rows.append(dict(stage=bounds[b_i][0], chars=bounds[b_i][1],
                              duty=round(duty, 4), acc=round(accv, 4),
                              ctx_seen=len(ctx_c)))
        print(f"  снимок {bounds[b_i][0]:<14} симв={bounds[b_i][1]:>10}: duty {duty*100:5.1f}% acc {accv:.4f}"
              f"  t={time.time()-T0:.0f}s", flush=True)
        b_i += 1

# вердикт P-D1
d_end = snap_rows[-1]["duty"]; a_end = snap_rows[-1]["acc"]
pd1 = "✓" if 0.345 <= d_end < 0.38 else ("✗ СМЕРТЬ" if d_end >= 0.40 else "~ вне вилки")
print(f"\nP-D1: duty на полной смеси {d_end*100:.1f}% (прогноз [34.5,38)) → {pd1}")

json.dump(dict(morph=dict(surface=res_surf, stem=res_stem, mult=mult, verdict=pf4),
               domain=dict(snaps=snap_rows, verdict=pd1),
               elapsed_s=time.time() - T0),
          open(ROOT / "research/CALC08B_MORPH_DOMAIN.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/CALC08B_MORPH_DOMAIN.json  t={time.time()-T0:.0f}s")
