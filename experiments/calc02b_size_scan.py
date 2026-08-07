#!/usr/bin/env python3
# CALC-02b (расчёт, НЕ обучение): закон «масса фактов vs объём корпуса».
# Подвыборки train {1/32, 1/8, 1/2, 1} → переобучаем СЧЁТЧИКИ, та же holdout.
# Вопрос: при каком масштабе факт-носимая масса CE (F2-F5) пересекает планку,
# легитимизирующую токенную полку; проекция на Kaggle-объёмы (×30, ×100).
import json, time, math, collections, sys, re
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aira.bpe import BPETokenizer

WORD_RE = re.compile(r" |\n|[^\s]+")
tok = BPETokenizer.load(ROOT / "experiments/results/bpe_tokenizer_bpe16k_glue.json")
V = len(tok.vocab)

cache: dict[str, list[int]] = {}
def encode_cached(text: str) -> list[int]:
    out, lead = [], False
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        if w == " ":
            if lead: out.append(tok.vocab["␣"])
            lead = True; continue
        if w == "\n":
            if lead: out.append(tok.vocab["␣"])
            lead = False; out.append(tok.vocab["␤"]); continue
        unit = " " + w if lead else w; lead = False
        ids = cache.get(unit)
        if ids is None:
            ids = [tok.vocab.get(p, tok.vocab["�"]) for p in tok._apply(tuple(unit))]
            cache[unit] = ids
        out.extend(ids)
    if lead: out.append(tok.vocab["␣"])
    return out

text = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")
n_tr = int(len(text) * 0.90)
tr = encode_cached(text[:n_tr]); ho = encode_cached(text[n_tr:])
CPT = len(text[n_tr:]) / len(ho)
print(f"train {len(tr)} | holdout {len(ho)} ({CPT:.2f} c/t)  t={time.time()-T0:.1f}s", flush=True)

def count_ngrams(seq, n):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    code = 0
    for i, x in enumerate(seq):
        code = (code * V + x) % (V ** n)
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1; pair[(code, seq[i + 1])] += 1
    return ctx, pair

LAM = 2.0
def eval_holdout(ctx2, pair2, ctx3, pair3, pu):
    # вычисляем и CE (через w2-контрольную форму) и бакеты с «виданностью» на уровне w2/w3
    tot2 = tot3 = 0.0
    seen2 = seen3 = 0; m2 = m3 = 0.0
    code = 0; L = 0
    for i in range(len(ho) - 1):
        code = (code * V + ho[i]) % (V ** 3)
        if i < 2: continue
        L += 1
        x = ho[i + 1]
        c2k = code % (V * V)
        c2 = pair2.get((c2k, x), 0); n2 = ctx2.get(c2k, 0)
        c3 = pair3.get((code, x), 0); n3 = ctx3.get(code, 0)
        p_u = pu.get(x, 1e-9)
        p2 = (c2 + LAM * p_u) / (n2 + LAM) if n2 else p_u
        p3 = (c3 + LAM * p2) / (n3 + LAM) if n3 else p2
        b2, b3 = -math.log2(p2), -math.log2(p3)
        tot2 += b2; tot3 += b3
        if c2 > 0: seen2 += 1; m2 += b2
        if c3 > 0: seen3 += 1; m3 += b3
    return {"bpc_w2": tot2 / L / CPT, "bpc_w3": tot3 / L / CPT,
            "seen_share_w2": seen2 / L, "seen_mass_w2": m2 / tot2,
            "seen_share_w3": seen3 / L, "seen_mass_w3": m3 / tot3}

rows = []
for fr in (1 / 32, 1 / 8, 1 / 2, 1.0):
    sub = tr[: int(len(tr) * fr)]
    ctx2, pair2 = count_ngrams(sub, 2)
    ctx3, pair3 = count_ngrams(sub, 3)
    pu = {k: v / len(sub) for k, v in collections.Counter(sub).items()}
    r = eval_holdout(ctx2, pair2, ctx3, pair3, pu)
    r["frac"] = fr; r["train_tokens"] = len(sub)
    rows.append(r)
    print(f"frac={fr:>5.3f} tok={len(sub):>8} | w2: {r['bpc_w2']:.3f} бит/симв, видано {r['seen_share_w2']*100:5.1f}% поз / {r['seen_mass_w2']*100:5.1f}% CE"
          f" | w3: видано {r['seen_share_w3']*100:5.1f}% поз  t={time.time()-T0:.1f}s", flush=True)

# экстраполяция log-log: seen_mass_w2(fr) ≈ a * fr^b  (грубый тренд из 4 точек)
import statistics
xs = [math.log(r["frac"]) for r in rows]; ys = [math.log(max(r["seen_mass_w2"], 1e-9)) for r in rows]
b = sum((x - statistics.mean(xs)) * (y - statistics.mean(ys)) for x, y in zip(xs, ys)) / sum((x - statistics.mean(xs)) ** 2 for x in xs)
a = statistics.mean(ys) - b * statistics.mean(xs)
print(f"\nтренд seen_mass_w2 ≈ exp({a:.3f})·frac^{b:.3f} → ×30 корпуса: {math.exp(a)*(30)**b*100:.1f}% | ×100: {math.exp(a)*(100)**b*100:.1f}% (масса CE виданных, w2)")
json.dump({"rows": rows, "trend": {"a": a, "b": b}, "CPT": CPT, "elapsed_s": time.time() - T0},
          open(ROOT / "research/CALC02B_SIZE.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/CALC02B_SIZE.json  t={time.time()-T0:.1f}s")
