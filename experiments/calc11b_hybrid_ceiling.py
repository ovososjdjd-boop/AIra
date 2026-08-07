#!/usr/bin/env python3
# CALC-11b (расчёт, НЕ обучение): потолок гибридной двери H-33.
# Вопрос: сколько покрытия добавил бы ИДЕАЛЬНЫЙ валидатор (умение-машина,
# проверяющая кандидатов полки), при сохранении acc>=0.95 на пройденных?
# Меряем на полке c8, N>=2: бины по f; потолок = Σ(позиций × acc_бина) в бинах
# ниже канона, минус реализм-дисконт валидатора p.
# ===== ПРЕДСКАЗАНИЕ ЗАМОРОЖЕНО ДО ПРОГОНА =====
#  P-H1: идеальный валидатор (p=1.0) добавляет >= +12 п.п. к duty 34.3%
#        (c8, канон). Смерть: < +6 п.п. (дверь гибрида мала и не стоит упора).
import json, time, collections
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")
alpha = {c: i for i, c in enumerate(sorted(set(text)))}
V = len(alpha)
ids = [alpha[c] for c in text]
n_tr = int(len(ids) * 0.90)
tr, ho = ids[:n_tr], ids[n_tr:]
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
top = {}
for k, c in pair.items():
    cd, x = divmod(k, V)
    cur = top.get(cd)
    if cur is None or c > cur[0]:
        top[cd] = (c, x)
del pair

F_B = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.90), (0.90, 1.01)]
band_cov = [0] * len(F_B); band_ok = [0] * len(F_B)
tot = 0
code = 0
for i in range(len(ho) - 1):
    code = (code * V + ho[i]) % base
    if i < 7: continue
    tot += 1
    n = ctx.get(code, 0)
    if n < 2: continue
    tc, tx = top[code]
    f = tc / n
    bi = next(b for b, (lo, hi) in enumerate(F_B) if lo <= f < hi)
    band_cov[bi] += 1
    band_ok[bi] += (tx == ho[i + 1])

print(f"позиций {tot}; полка N>=2: {sum(band_cov)}  t={time.time()-T0:.0f}s", flush=True)
canon_cov = canon_ok = 0
rows = []
for b, (lo, hi) in enumerate(F_B):
    acc = band_ok[b] / max(band_cov[b], 1)
    rows.append(dict(f=f"[{lo},{hi})", cov=round(band_cov[b] / tot, 4),
                     acc=round(acc, 4), mass=round(band_ok[b] / tot, 4)))
    print(f"  f∈[{lo:>4.2f},{hi:>4.2f}): доля {band_cov[b]/tot*100:5.1f}% | acc {acc:.4f} | "
          f"верная масса {band_ok[b]/tot*100:5.1f}%", flush=True)

# канон = f>=0.90: это band 5; потолок гибрида = канон + верная масса бинов ниже
canon_cov = band_cov[5]; canon_ok = band_ok[5]
print(f"\nканон c8 (f>=0.9,N>=2): duty {canon_cov/tot*100:.1f}% acc {canon_ok/canon_cov:.4f}")
ceiling = {}
for p in (1.0, 0.95, 0.90, 0.8):
    add = p * sum(band_ok[:5]) / tot
    add_cov = p * sum(band_cov[:5]) / tot
    # валидатор с точностью p пропускает p·ok верных + p_second·(1-ok) ложных;
    # оптимистично: пропускает только верных-отобранных → acc добавки ≈ p·ok/(p·cov)=ok/cov=band acc…
    # честная форма: валидатор пропускает верных с recall p и ложных с impurity q=p·(1-band_acc)*?
    # простая модель: пропущенные = p·ok (верные) + p·(1-beta)·nok, beta=разборчивость=0.9
    nok = sum(band_cov[:5]) - sum(band_ok[:5])
    passed_ok = p * sum(band_ok[:5])
    passed_bad = p * 0.1 * nok
    duty = (canon_cov + passed_ok + passed_bad) / tot
    acc = (canon_ok + passed_ok) / (canon_cov + passed_ok + passed_bad)
    ceiling[p] = dict(duty=round(duty, 4), acc=round(acc, 4), add_pp=round((passed_ok + passed_bad) / tot, 4))
    print(f"  p={p:.2f}: союз duty {duty*100:5.1f}% acc {acc:.4f} (+{(passed_ok + passed_bad)/tot*100:.1f} п.п.)")

c = ceiling[1.0]
verd = "✓" if c["duty"] >= 0.463 and c["acc"] >= 0.95 else ("✗ СМЕРТЬ" if c["add_pp"] < 0.06 else "~ погранично")
print(f"\nP-H1 {verd} (идеальный p=1.0: {c['duty']*100:.1f}%@{c['acc']:.4f}, добавка +{c['add_pp']*100:.1f} п.п.)")

json.dump(dict(bands=rows, canon=[round(canon_cov/tot, 4), round(canon_ok/canon_cov, 4)],
               ceiling=ceiling, verdict=verd, elapsed_s=time.time() - T0),
          open(ROOT / "research/CALC11B_HYBRID_CEILING.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/CALC11B_HYBRID_CEILING.json  t={time.time()-T0:.0f}s")
