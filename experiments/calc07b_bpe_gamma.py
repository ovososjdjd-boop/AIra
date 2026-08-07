#!/usr/bin/env python3
# CALC-07b (расчёт, НЕ обучение): экспонента γ покрытия BPE-органа (w2).
# Гипотеза после CALC-07: «уровень абстракции крутит γ» —
#   char-c8 показал γ≈0.073 (мёртвая дорога: duty 0.90 лишь при ~2.5e17 ток);
#   если факты хранить на уровне слов/токенов (BPE w2), хвост должен быть круче.
# ===== ПРЕДСКАЗАНИЯ ЗАМОРОЖЕНЫ ДО ПРОГОНА =====
#  P1: γ_w2 > 0.073, ожидание 0.10–0.30 (референс: seen-МАССА w2 росла как 0.493·√-клон
#      в CALC-02b, но duty@acc≥0.95 мягче массы → не жду 0.49).
#  P2: пересечение с char-c8 органом при N* ≤ 3e9 токенов (иначе к Opus-объёму 1.3e13
#      BPE-орган не успевает стать основным).
#  P3 критерий смерти гипотезы: γ_w2 ≤ 0.073 → лестница абстракций ✗.
# Стенд идентичен CALC-02b (токенизатор bpe16k_glue, тот же сплит 90/10).
import json, time, math, collections, re, sys
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
            if lead: out.append(tok.vocab["␣"]); lead = True
            continue
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
print(f"bpe-стенд: train {len(tr)} | holdout {len(ho)} | V={V}  t={time.time()-T0:.1f}s", flush=True)

def count_ngrams(seq, n):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    base = V ** n
    code = 0
    for i, x in enumerate(seq):
        code = (code * V + x) % base
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1
            pair[(code, seq[i + 1])] += 1
    return ctx, pair

def topmap(pair):
    top = {}
    for (code, x), c in pair.items():
        cur = top.get(code)
        if cur is None or c > cur[0]:
            top[code] = (c, x)
    return top

scan = []
print("\n=== duty(N) BPE w2, правило (θ=0.90, Nmin=2) ===", flush=True)
for fr in (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0):
    sub = tr[: int(len(tr) * fr)]
    ctx, pair = count_ngrams(sub, 2)
    top = topmap(pair)
    mod = V * V
    code = 0; cov = 0; hit = 0; tot = 0
    for i in range(len(ho) - 1):
        code = (code * V + ho[i]) % mod
        if i < 1: continue
        tot += 1
        n = ctx.get(code, 0)
        if n >= 2:
            c, tx = top[code]
            if c / n >= 0.90:
                cov += 1; hit += (tx == ho[i + 1])
    duty = cov / tot; acc = hit / max(cov, 1)
    scan.append(dict(fr=fr, train_tok=len(sub), duty=duty, acc=acc))
    print(f"  fr={fr:>6.4f} tok={len(sub):>9} | duty={duty*100:5.2f}% acc={acc:.4f}  t={time.time()-T0:.0f}s", flush=True)
    del ctx, pair, top

# фит 1-duty = c·N^-γ
pts = [(r["train_tok"], r["duty"]) for r in scan if r["duty"] > 0.005]
xs = [math.log(n) for n, _ in pts]; ys = [math.log(1 - d) for _, d in pts]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
g = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
c = math.exp(my + g * mx)
print(f"\nγ_w2 = {g:.4f}  (1-duty = {c:.3f}·N^-{g:.4f})")
print(f"  P1 (γ_w2 > 0.0737): {'ПОДТВЕРЖДЕНО' if g > 0.0737 else 'ОПРОВЕРГНУТО — покрытие w2 плоское в нашем окне'}")

# acc(N): w2-орган растёт НАДЁЖНОСТЬЮ, не покрытием → где acc=0.95?
za = [r["acc"] for r in scan]
b_acc = sum((x - mx) * (y - sum(za) / len(za)) for x, y in zip(xs, za)) / sum((x - mx) ** 2 for x in xs)
a_acc = sum(za) / len(za) - b_acc * mx
N95_acc = math.exp((0.95 - a_acc) / b_acc) if b_acc > 0 else float("inf")
print(f"  acc(N) ≈ {a_acc:.3f} + {b_acc:.4f}·ln N  → acc=0.95 при N ≈ {N95_acc:.2e} ток (duty там ~{scan[-1]['duty']*100:.1f}%)")

c_ch, g_ch = 1.915, 0.0737
Ns = None
if g > 0.005:
    lo, hi = 5e4, 1e18  # домен левее 5e4 для char-закона бессмыслен
    n = lo
    while n < hi:
        d_ch = 1 - c_ch * n ** (-g_ch)
        d_bp = 1 - c * n ** (-g)
        if d_bp > d_ch:
            Ns = n; break
        n *= 1.05
    if Ns:
        print(f"  N пересечения с char-c8: {Ns:.2e} ток  (P2 ≤3e9: {'OK' if Ns <= 3e9 else 'МИМО'})")
    else:
        print("  пересечения с char-c8 в домене [5e4, 1e18] НЕТ — char-орган доминирует всюду в окне")
    for tgt in (0.90, 0.95):
        Nt = (c / (1 - tgt)) ** (1 / g)
        print(f"  N(duty={tgt:.2f}) по этой экспоненте: {Nt:.2e} ток")
else:
    print("  γ≈0 → экстраполяция покрытия бессмысленна; рост органа идёт через acc(N)")
    print("  P2: пересечения с char-c8 нет в измеренном окне (char 34.3% vs w2 7.4% @2–4M ток)")

json.dump(dict(scan=scan, gamma=g, c=c, N_cross=Ns, acc_fit=dict(a=a_acc, b=b_acc, N_at_095=N95_acc),
               predictions_frozen={"P1": "gamma>0.0737", "P2": "N*<=3e9", "P3_death": "gamma<=0.0737"},
               verdict_P1="ОПРОВЕРГНУТО" if g <= 0.0737 else "ПОДТВЕРЖДЕНО",
               elapsed_s=time.time() - T0),
          open(ROOT / "research/CALC07B_BPE_GAMMA.json", "w"), ensure_ascii=False, indent=1)
print(f"\njson -> research/CALC07B_BPE_GAMMA.json  t={time.time()-T0:.0f}s")
