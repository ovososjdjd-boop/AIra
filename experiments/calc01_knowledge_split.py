#!/usr/bin/env python3
# Расчёт (не эксперимент): разложение cross-entropy живого текста на
# "фактовую" (редкие/one-shot ключи, полка) и "умение-вую" (частые паттерны,
# веса) массы. Источник истины — голый корпус, обучение НЕ запускается.
# Выход: stdout-таблица + research/CALC01_SPLIT.json
import json, time, collections

T0 = time.time()
SRC = "corpus_external/wikitext2/train.txt"
N_LIST = [8]          # глубина ключа (как c8 в EXP-15b)
HOLDOUT_FR = 0.10     # хвост корпуса — честный live-срез (как valid в EXP-15b)

text = open(SRC, encoding="utf-8").read()
alphabet = {c: i for i, c in enumerate(sorted(set(text)))}
V = len(alphabet)
ids = [alphabet[c] for c in text]
n_tr = int(len(ids) * (1 - HOLDOUT_FR))
tr, ho = ids[:n_tr], ids[n_tr:]
print(f"chars={len(text)} V={V} train={len(tr)} holdout={len(ho)} t={time.time()-T0:.1f}s")

# --- подсчёт n-грамм частот (упаковка контекста в int, память-бережно) ---
def count_ngrams(seq, n):
    ctx = collections.defaultdict(int)      # код контекста -> всего
    pair = collections.defaultdict(int)     # код (ctx,next) -> счёт
    base = V ** n
    code = 0
    for i, x in enumerate(seq):
        code = (code * V + x) % base
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1
            pair[code * V + seq[i + 1]] += 1
    return ctx, pair

uni = collections.Counter(tr)
p_uni = {k: v / sum(uni.values()) for k, v in uni.items()}
N = 8
ctx8, pair8 = count_ngrams(tr, N)
ctx4, pair4 = count_ngrams(tr, 4)
print(f"ngrams8: ctx={len(ctx8)} pair={len(pair8)}  ngrams4: ctx={len(ctx4)} pair={len(pair4)}  t={time.time()-T0:.1f}s")

# --- backoff-вероятность 8 -> 4 -> unigram (Katz-стиль, одна лестница) ---
LAM = 2.0
def p_est(code_n, x, n):
    # возвращает p (сглаженная) и сырой счёт пары на глубине n
    if n == 0:
        return p_uni.get(x, 1e-9), 0, 0
    c_pair = (pair8 if n == 8 else pair4).get(code_n * V + x, 0)
    c_ctx = (ctx8 if n == 8 else ctx4).get(code_n, 0)
    if n == 8:
        p_back, _, _ = p_est(code_n % (V ** 4), x, 4)
    else:
        p_back = p_uni.get(x, 1e-9)
    if c_ctx == 0:
        return p_back, c_pair, c_ctx
    return (c_pair + LAM * p_back) / (c_ctx + LAM), c_pair, c_ctx

# --- проход по holdout: бакеты и CE-массы ---
BUCKETS = [
    ("F0 ключ-контекст не видан (новизна)", lambda c, n: n == 0),
    ("F1 контекст видан, продолжение НЕТ (полуновизна)", lambda c, n: n > 0 and c == 0),
    ("F2 one/two-shot факт (c=1..2)", lambda c, n: 1 <= c <= 2),
    ("F3 малый паттерн (3..16)", lambda c, n: 3 <= c <= 16),
    ("F4 рабочий паттерн (17..256)", lambda c, n: 17 <= c <= 256),
    ("F5 шаблон (257+)", lambda c, n: c >= 257),
]
cnt = {b[0]: 0 for b in BUCKETS}
ce = {b[0]: 0.0 for b in BUCKETS}
tot_ce = 0.0
base8 = V ** 8
code = 0
# Исправление выравнивания (баг v1): контекст = окно, ЗАКАНЧИВАЮЩЕЕСЯ на i,
# предсказываем ho[i+1] ДО того, как он попадёт в код (v1 смотрел на 1 вперёд
# и "предсказывал" символ, видя его самого в ключе — все вероятности мусор).
for i in range(len(ho) - 1):
    code = (code * V + ho[i]) % base8
    if i < N - 1:
        continue
    x = ho[i + 1]
    p, c_pair, c_ctx = p_est(code, x, N)
    tot_ce += -__import__("math").log2(p)
    for name, f in BUCKETS:
        if f(c_pair, c_ctx):
            cnt[name] += 1
            ce[name] += -__import__("math").log2(p)
            break

L = sum(cnt.values())
out = {
    "source": SRC, "holdout_frac": HOLDOUT_FR, "n_key": N, "V": V,
    "positions": L,
    "ngram_model_bits_per_char": tot_ce / L,
    "ngram_model_ppl": 2 ** (tot_ce / L),
    "buckets": [
        {"name": n_, "share": cnt[n_] / L, "bits_per_char": ce[n_] / max(cnt[n_], 1),
         "ce_mass_share": ce[n_] / tot_ce}
        for n_, _ in BUCKETS],
    "elapsed_s": time.time() - T0,
}
print("\n=== РАЗЛОЖЕНИЕ CE ЖИВОГО ТЕКСТА (wikitext2, ключ c8, backoff 8→4→1, v2 fixed-align) ===")
print(f"модель-полка в одиночку: {out['ngram_model_bits_per_char']:.3f} bits/char  (ppl {out['ngram_model_ppl']:.2f})")
for row in out["buckets"]:
    print(f"{row['name']:<46} доля позиций {row['share']*100:5.1f}%  бит/поз {row['bits_per_char']:5.2f}  МАССА CE {row['ce_mass_share']*100:5.1f}%")
json.dump(out, open("research/CALC01_SPLIT.json", "w"), ensure_ascii=False, indent=1)
print(f"\njson -> research/CALC01_SPLIT.json  t={time.time()-T0:.1f}s")
