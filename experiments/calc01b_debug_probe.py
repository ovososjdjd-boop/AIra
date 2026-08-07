#!/usr/bin/env python3
# Диагностика оценщика вероятностей из calc01 (баг-охота, без обучения).
import collections, time, math
T0=time.time()
text = open("corpus_external/wikitext2/train.txt", encoding="utf-8").read()
alphabet = {c: i for i, c in enumerate(sorted(set(text)))}
V = len(alphabet)
ids = [alphabet[c] for c in text]
n_tr = int(len(ids)*0.90); tr, ho = ids[:n_tr], ids[n_tr:]
uni = collections.Counter(tr); tot=sum(uni.values())
p_uni = {k: v/tot for k,v in uni.items()}

def count_ngrams(seq, n):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    base = V**n; code = 0
    for i, x in enumerate(seq):
        code = (code * V + x) % base
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1; pair[code * V + seq[i+1]] += 1
    return ctx, pair

ctx8, pair8 = count_ngrams(tr, 8)
ctx4, pair4 = count_ngrams(tr, 4)
inv = {i: c for c, i in alphabet.items()}
print(f"built {time.time()-T0:.1f}s  V={V}")

LAM = 2.0
def p4(code4, x):
    c = pair4.get(code4*V+x, 0); n4 = ctx4.get(code4, 0)
    return (c + LAM*p_uni.get(x,1e-9))/(n4 + LAM) if n4 else p_uni.get(x,1e-9)
def p8(code8, x):
    c = pair8.get(code8*V+x, 0); n8 = ctx8.get(code8, 0)
    pb = p4(code8 % (V**4), x)
    return ((c + LAM*pb)/(n8 + LAM) if n8 else pb), c, n8

# 1) точечные пробы распространённых контекстов
for probe in ["the ", "ation ", " of t", "the cat"]:
    x = ord(" ") if probe!="the cat" else alphabet.get(" ",0)
    seq = [alphabet.get(ch,0) for ch in probe]
    code8=0
    for ch in seq: code8=(code8*V+ch)%(V**8)
    nxt = " " 
    p,c,n = p8(code8, alphabet.get(nxt,0))
    print(f"probe ctx={probe!r} next=' ' -> p8={p:.4f} (pair8c={c}, ctx8n={n}) p4={p4(code8%(V**4),alphabet.get(nxt,0)):.4f} pu={p_uni.get(alphabet.get(nxt,0),0):.4f}")

# 2) CE на подвыборке holdout: чистый 4-грам vs интерполяция 8
import random
random.seed(0)
L=60000; off=5000
ce4=ce8=c0s=c1_2=c3p=0
cc={}
for i in range(off, off+L):
    code=0
    for j in range(i-8, i+1):
        code=(code*V+ho[j])%(V**8)
    x=ho[i+1] if i+1<len(ho) else 0
    p,c,n = p8(code, x)
    ce8 += -math.log2(max(p,1e-12))
    ce4 += -math.log2(max(p4(code%(V**4), x),1e-12))
    if n==0: c0s+=1
    elif 1<=c<=2: c1_2+=1
    else: c3p+=1
print(f"sample {L}: CE4={ce4/L:.3f} bits/char  CE8interp={ce8/L:.3f} bits/char")
print(f"ctx-unseen share={c0s/L:.3f}  pair c=1..2 share={c1_2/L:.3f}  c>=3 share={c3p/L:.3f}")
# 3) разбивка CE8 по классам
