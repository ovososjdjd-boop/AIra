#!/usr/bin/env python3
# CALC-02 (расчёт, НЕ обучение): разложение CE живого текста на факты/умение
# на BPE-ключах (w3 каскад w3→w2→w1; контроль w2), словарь bpe16k-glue.
# Прогноз (зафиксирован ДО счёта в RESEARCH_PLAN фронт D2):
#  P-a полка-каскад ≤1.6 бит/симв (<1.970 char-c8);  P-b на сопоставимой
#  глубине ключа (w2≈9симв) масса виданных продолжений ≥50% (vs 41.3%);
#  P-c бакет F1 <7% позиций (<25% массы). Выход: stdout + research/CALC02_SPLIT.json
import json, time, math, collections, sys, re
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aira.bpe import BPETokenizer

WORD_RE = re.compile(r" |\n|[^\s]+")
VOCAB = ROOT / "experiments/results/bpe_tokenizer_bpe16k_glue.json"
SRC = ROOT / "corpus_external/wikitext2/train.txt"

tok = BPETokenizer.load(VOCAB)
print(f"словарь: {len(tok.vocab)} id, glue={tok.glue}  t={time.time()-T0:.1f}s", flush=True)

# --- encode с кэшем по unit (энкодер BPE не имеет мемоизации) ---
cache: dict[str, list[int]] = {}
def encode_cached(text: str) -> list[int]:
    out = []
    lead = False
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        if w == " ":
            if lead:
                out.append(tok.vocab["␣"])
            lead = True
            continue
        if w == "\n":
            if lead:
                out.append(tok.vocab["␣"])
            lead = False
            out.append(tok.vocab["␤"]); continue
        unit = " " + w if lead else w
        lead = False
        ids = cache.get(unit)
        if ids is None:
            ids = [tok.vocab.get(p, tok.vocab["�"]) for p in tok._apply(tuple(unit))]
            cache[unit] = ids
        out.extend(ids)
    if lead:
        out.append(tok.vocab["␣"])
    return out

text = SRC.read_text(encoding="utf-8")
n_tr = int(len(text) * 0.90)
tr_text, ho_text = text[:n_tr], text[n_tr:]
tr = encode_cached(tr_text)
ho = encode_cached(ho_text)
print(f"train {len(tr)} ids | holdout {len(ho)} ids ({len(ho_text)/len(ho):.2f} c/t) "
      f"| кэш {len(cache)} unit-ов  t={time.time()-T0:.1f}s", flush=True)

# --- n-грамм счётчики ---
def count_ngrams(seq, n):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    code = 0
    for i, x in enumerate(seq):
        code = (code * V + x) % (V ** n)   # модуль ДО счёта: окно строго n токенов
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1                 # (v1 баг: модуль был после — контекст n+1)
            pair[(code, seq[i + 1])] += 1
    return ctx, pair

V = len(tok.vocab)
ctx3, pair3 = count_ngrams(tr, 3)
ctx2, pair2 = count_ngrams(tr, 2)
uni = collections.Counter(tr)
pu = {k: v / len(tr) for k, v in uni.items()}
print(f"n3 ctx={len(ctx3)} pair={len(pair3)} | n2 ctx={len(ctx2)} pair={len(pair2)}  t={time.time()-T0:.1f}s", flush=True)

# --- backoff w3→w2→w1 ---
LAM = 2.0
def p_est(code3, x):
    c3 = pair3.get((code3, x), 0); n3 = ctx3.get(code3, 0)
    c2 = pair2.get((code3 % (V * V), x), 0); n2 = ctx2.get(code3 % (V * V), 0)
    p_u = pu.get(x, 1e-9)
    p_21 = (c2 + LAM * p_u) / (n2 + LAM) if n2 else p_u
    p = (c3 + LAM * p_21) / (n3 + LAM) if n3 else p_21
    return p, c3, n3

# --- проход по holdout w3 (основной) ---
def run_holdout(seq, n_key, ctx, pair, pu_):
    BUCKETS = [
        ("F0 ключ-контекст не видан (новизна)", lambda c, n: n == 0),
        ("F1 контекст видан, продолжение НЕТ", lambda c, n: n > 0 and c == 0),
        ("F2 one/two-shot факт (c=1..2)", lambda c, n: 1 <= c <= 2),
        ("F3 малый паттерн (3..16)", lambda c, n: 3 <= c <= 16),
        ("F4 рабочий паттерн (17..256)", lambda c, n: 17 <= c <= 256),
        ("F5 шаблон (257+)", lambda c, n: c >= 257),
    ]
    cnt = {b[0]: 0 for b in BUCKETS}; ce = {b[0]: 0.0 for b in BUCKETS}
    tot = 0.0; code = 0
    for i in range(len(seq) - 1):
        code = (code * V + seq[i]) % (V ** 3)   # держим w3-код всегда
        if i < 2:
            continue
        if n_key == 3:
            p, c_pair, c_ctx = p_est(code, seq[i + 1])
        else:  # w2-контроль: backoff 2→1
            c2k = code % (V * V)
            c_p = pair2.get((c2k, seq[i + 1]), 0); n2c = ctx2.get(c2k, 0)
            p_u = pu_.get(seq[i + 1], 1e-9)
            p = (c_p + LAM * p_u) / (n2c + LAM) if n2c else p_u
            c_pair, c_ctx = c_p, n2c
        b = -math.log2(p); tot += b
        for name, f in BUCKETS:
            if f(c_pair, c_ctx):
                cnt[name] += 1; ce[name] += b; break
    L = sum(cnt.values())
    return {"n_key": n_key, "positions": L,
            "bits_per_token": tot / L,
            "buckets": [{"name": n_, "share": cnt[n_] / L,
                         "bits_per_tok": ce[n_] / max(cnt[n_], 1),
                         "ce_mass_share": ce[n_] / tot} for n_, _ in BUCKETS]}

res3 = run_holdout_seq = None
res3 = run_holdout(ho, 3, ctx3, pair3, pu)
res2 = run_holdout(ho, 2, ctx2, pair2, pu)

CPT = len(ho_text) / len(ho)  # символов на токен в holdout
def conv(r):  # бит/симв и ppl
    bpc = r["bits_per_token"] / CPT
    return bpc, 2 ** bpc

for r, tag in ((res3, "w3 каскад w3→w2→w1"), (res2, "w2 контроль w2→w1")):
    bpc, ppl = conv(r)
    print(f"\n=== CALC-02 {tag} ===  модель-полка: {bpc:.3f} бит/симв (ppl {ppl:.2f})")
    for row in r["buckets"]:
        print(f"{row['name']:<40} позиций {row['share']*100:5.1f}%  бит/поз {row['bits_per_tok']:5.2f}  МАССА CE {row['ce_mass_share']*100:5.1f}%")

out = {
    "source": str(SRC), "vocab": str(VOCAB), "holdout_frac": 0.10,
    "train_tokens": len(tr), "holdout_tokens": len(ho), "chars_per_token_holdout": CPT,
    "model_bits_per_char": {f"n{r['n_key']}": conv(r)[0] for r in (res3, res2)},
    "model_ppl": {f"n{r['n_key']}": conv(r)[1] for r in (res3, res2)},
    "w3": res3["buckets"], "w2": res2["buckets"], "elapsed_s": time.time() - T0,
}
(ROOT / "research/CALC02_SPLIT.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\njson -> research/CALC02_SPLIT.json  t={time.time()-T0:.1f}s")
