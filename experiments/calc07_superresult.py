#!/usr/bin/env python3
# CALC-07 (расчёт, НЕ обучение): «Дорога к сверхрезультату».
# Вопрос заказчика: оптимист CALC-06 (duty 0.90) — и СВЕРХ него (0.95/0.98).
# Часть A: duty(N) ИЗМЕРЯЕМ, а не гадаем — та же стенд-схема CALC-03b
#   (char c8, правило 2D f>=θ, N>=Nmin), но счётчики обучены на подвыборках
#   train {1/64..1} → 7 точек закона покрытия vs объём данных.
# Часть B: рычаги сверх масштаба, измеримые сегодня на том же тексте:
#   L1 многоуровневая полка (8->6->4 backoff-union под тем же 2D-правилом);
#   L2 встроенное обобщение (casefold: сливалка регистра — «факт» перестаёт
#      зависеть от капитализации);
#   L1+L2 совместно.
# Часть C: два конкурирующих фита хвоста (power-in-unseen, power-in-seen) →
#   БРЕКЕТ объёма данных под duty 0.90/0.95 (экстраполяция честно помечена).
# Часть D: пересчёт денег/времени CALC-06 при duty 0.90/0.95/0.98.
import json, time, math, collections
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")
CPT = 4.683  # измерено (CALC-02b): символов на BPE-токен wikitext2

def build_ids(txt):
    alpha = {c: i for i, c in enumerate(sorted(set(txt)))}
    return [alpha[c] for c in txt], len(alpha)

def count_ngrams(seq, n, V):
    ctx = collections.defaultdict(int); pair = collections.defaultdict(int)
    base = V ** n
    code = 0
    for i, x in enumerate(seq):
        code = (code * V + x) % base
        if i >= n - 1 and i + 1 < len(seq):
            ctx[code] += 1
            pair[(code, seq[i + 1])] += 1
    return ctx, pair

def topmap(pair, V):
    top = {}
    for (code, x), c in pair.items():
        cur = top.get(code)
        if cur is None or c > cur[0]:
            top[code] = (c, x)
    return top

ids, V = build_ids(text)
n_tr = int(len(ids) * 0.90)
tr, ho = ids[:n_tr], ids[n_tr:]
print(f"char-стенд: train {len(tr)} | holdout {len(ho)} | V={V}  t={time.time()-T0:.1f}s", flush=True)

# ---------- стенд: полка уровней {8,6,4} на заданном train ---------- #
def build_shelf(sub, levels=(8, 6, 4), Vv=None):
    Vv = V if Vv is None else Vv
    shelf = {}
    for n in levels:
        ctx, pair = count_ngrams(sub, n, Vv)
        shelf[n] = (ctx, pair, topmap(pair, Vv))
    return shelf

def holdout_records(shelf, V, levels=(8, 6, 4)):
    """fused-записи: на позицию — (n,f,ok) каждого уровня."""
    mods = {n: V ** n for n in levels}
    codes = {n: 0 for n in levels}
    recs = []
    for i in range(len(ho) - 1):
        x = ho[i]
        for n in levels:
            codes[n] = (codes[n] * V + x) % mods[n]
        if i < max(levels) - 1:
            continue
        nxt = ho[i + 1]
        row = []
        for n in levels:
            ncnt = shelf[n][0].get(codes[n], 0)
            if ncnt == 0:
                row.append((0, 0.0, False))
            else:
                c, tx = shelf[n][2][codes[n]]
                row.append((ncnt, c / ncnt, tx == nxt))
        recs.append(tuple(row))
    return recs

def eval_rule(recs, levels, theta, nmin):
    """покрытие и acc: самый длинный уровень, прошедший (f>=θ, N>=Nmin)."""
    cov = 0; hit = 0
    for row in recs:
        for n, (cnt, f, ok) in zip(levels, row):
            if cnt >= nmin and f >= theta:
                cov += 1; hit += ok
                break
    return cov / len(recs), (hit / cov if cov else 0.0), cov

# ---------- Часть A: duty(N) — 7 опор ---------- #
print("\n=== A. duty(N) измеренный, char c8, правило (θ=0.90, Nmin=2) ===", flush=True)
scan = []
for fr in (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0):
    sub = tr[: int(len(tr) * fr)]
    shelf = build_shelf(sub, levels=(8,))
    recs = holdout_records(shelf, V, levels=(8,))
    duty, acc, cov_abs = eval_rule(recs, (8,), 0.90, 2)
    # фронтир по мини-сетке: макс покрытие при acc>=0.95
    best = (0.0, 0.0, None)
    for th in (0.85, 0.90, 0.95):
        for nm in (2, 5):
            d, a, _ = eval_rule(recs, (8,), th, nm)
            if a >= 0.95 and d > best[0]:
                best = (d, a, (th, nm))
    ctx8, pair8 = shelf[8][0], shelf[8][1]
    shelf_n2 = sum(1 for v in ctx8.values() if v >= 2)
    scan.append(dict(fr=fr, train_chars=len(sub), train_tok=len(sub) / CPT,
                     duty=duty, acc=acc, duty95=best[0], acc95=best[1],
                     rule95=best[2], shelf_ctx_n2=shelf_n2))
    del shelf, recs, ctx8, pair8
    r = scan[-1]
    print(f"  fr={fr:>6.4f} tok={r['train_tok']:>10.0f} | duty(канон)={duty*100:5.2f}% acc={acc:.4f}"
          f" | лучший@0.95: {best[0]*100:5.2f}% (acc {best[1]:.4f}, {best[2]})"
          f" | полка N>=2: {shelf_n2} контекстов  t={time.time()-T0:.0f}s", flush=True)

# ---------- Часть B: рычаги на полном train ---------- #
print("\n=== B. рычаги (полный train, 9.7M симв) ===", flush=True)
FRONT = [(th, nm) for th in (0.85, 0.90, 0.95) for nm in (2, 5)]
levers = {}

# B0 эталон: только c8 (повтор на полном train = строка fr=1 выше, канон)
shelf8 = build_shelf(tr, levels=(8,))
recs8 = holdout_records(shelf8, V, levels=(8,))
levers["B0_c8"] = {f"{th},{nm}": eval_rule(recs8, (8,), th, nm)[:2] for th, nm in FRONT}
d, a = levers["B0_c8"]["0.9,2"]
print(f"  B0 эталон c8: duty {d*100:.1f}% acc {a:.4f}  t={time.time()-T0:.0f}s", flush=True)

# B1 многоуровневая полка 8->6->4
shelf864 = build_shelf(tr, levels=(8, 6, 4))
recs864 = holdout_records(shelf864, V, levels=(8, 6, 4))
levers["B1_c864"] = {f"{th},{nm}": eval_rule(recs864, (8, 6, 4), th, nm)[:2] for th, nm in FRONT}
d, a = levers["B1_c864"]["0.9,2"]
print(f"  B1 union c8->c6->c4: duty {d*100:.1f}% acc {a:.4f}  t={time.time()-T0:.0f}s", flush=True)
del shelf8, recs8, shelf864, recs864

# B2 casefold-обобщение (c8) и B3 (8->6->4)
fold = "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in text)
ids_f, Vf = build_ids(fold)
tr_f, ho_f = ids_f[:n_tr], ids_f[n_tr:]
print(f"  casefold: V {V}→{Vf}  t={time.time()-T0:.0f}s", flush=True)

def holdout_records_f(shelf, levels):
    mods = {n: Vf ** n for n in levels}
    codes = {n: 0 for n in levels}
    recs = []
    for i in range(len(ho_f) - 1):
        x = ho_f[i]
        for n in levels:
            codes[n] = (codes[n] * Vf + x) % mods[n]
        if i < max(levels) - 1:
            continue
        nxt = ho_f[i + 1]
        row = []
        for n in levels:
            ncnt = shelf[n][0].get(codes[n], 0)
            if ncnt == 0:
                row.append((0, 0.0, False))
            else:
                c, tx = shelf[n][2][codes[n]]
                row.append((ncnt, c / ncnt, tx == nxt))
        recs.append(tuple(row))
    return recs

shelf8f = build_shelf(tr_f, levels=(8,), Vv=Vf)
recs8f = holdout_records_f(shelf8f, (8,))
levers["B2_c8_fold"] = {f"{th},{nm}": eval_rule(recs8f, (8,), th, nm)[:2] for th, nm in FRONT}
d, a = levers["B2_c8_fold"]["0.9,2"]
print(f"  B2 casefold c8: duty {d*100:.1f}% acc {a:.4f}  t={time.time()-T0:.0f}s", flush=True)
del shelf8f, recs8f

shelf864f = build_shelf(tr_f, levels=(8, 6, 4), Vv=Vf)
recs864f = holdout_records_f(shelf864f, (8, 6, 4))
levers["B3_c864_fold"] = {f"{th},{nm}": eval_rule(recs864f, (8, 6, 4), th, nm)[:2] for th, nm in FRONT}
d, a = levers["B3_c864_fold"]["0.9,2"]
print(f"  B3 union+casefold: duty {d*100:.1f}% acc {a:.4f}  t={time.time()-T0:.0f}s", flush=True)
del shelf864f, recs864f

# ---------- Часть C: хвост закона, брекет ---------- #
print("\n=== C. фиты хвоста duty(N) → брекет объёма под цели ===", flush=True)
pts = [(r["train_tok"], r["duty"]) for r in scan if r["duty"] > 0.01]

def fit_unseen(pts):
    xs = [math.log(n) for n, _ in pts]; ys = [math.log(1 - d) for _, d in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    g = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    c = math.exp(my + g * mx)
    return c, g  # 1-duty = c*N^-g

def N_for(c, g, target):
    return (c / (1 - target)) ** (1 / g)

fits = {}
for tag, pp in (("все точки", pts), ("последние 4", pts[-4:]), ("последние 3", pts[-3:])):
    c, g = fit_unseen(pp)
    fits[tag] = dict(c=c, gamma=g, N90=N_for(c, g, 0.90), N95=N_for(c, g, 0.95))
    print(f"  unseen-power [{tag}]: 1-duty = {c:.3f}·N^-{g:.4f}"
          f"  → N(duty=0.90) = {fits[tag]['N90']:.2e} ток | N(0.95) = {fits[tag]['N95']:.2e} ток", flush=True)

# reference: power-in-seen (как CALC-02b)
xs = [math.log(n) for n, _ in pts]; ys = [math.log(d) for _, d in pts]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
a0 = math.exp(my - b * mx)
print(f"  seen-power (референс, насыщается >1!): duty ≈ {a0:.4f}·N^{b:.3f}"
      f"  → duty=1.0 формально при N = {(1/a0)**(1/b):.2e} ток (признак излома закона)", flush=True)

# «сколько данных стоит рычаг»: duty B3 при наших 2.07M ток — какой объём дал бы B0
full_pts = dict((r["fr"], r["duty"]) for r in scan)
duty_b3 = levers["B3_c864_fold"]["0.9,2"][0]
duty_b1 = levers["B1_c864"]["0.9,2"][0]
c, g = fit_unseen(pts)
for tag, dl in (("B1 union", duty_b1), ("B3 union+fold", duty_b3)):
    if dl < 1:
        N_eq = N_for(c, g, dl)
        print(f"  рычаг {tag}: duty {dl*100:.1f}% уже сейчас = эталонная полка при ~{N_eq/2.07e6:.1f}× данных", flush=True)

# ---------- Часть D: деньги при сверх-duty (формулы CALC-06) ---------- #
print("\n=== D. пересчёт класса Opus (N=1e12, D=1.3e13 допущениями) при сверх-duty ===", flush=True)
B_TOK = 4096
q = dict(mac=0.5, hbm=25.0, sram=0.8, a=0.30, w=0.17, al=25)  # оптимист-фурнитура CALC-06
money = []
for s in (0.90, 0.95, 0.98):
    duty = 1 - s
    N = 1e12; Pz = N / q["al"]
    bp_tr = 6 * N * q["mac"] + (16 * N) * q["hbm"] / B_TOK
    ai_tr = duty * q["w"] * (6 * Pz * q["mac"] + 3 * Pz * 2 * q["sram"])
    bp_inf = 2 * N * q["hbm"] + 2 * N * q["mac"]
    ai_inf = duty * q["a"] * (2 * Pz * q["sram"] + 2 * Pz * q["mac"])
    r_fl = (6 * N) / (duty * q["w"] * 6 * Pz)
    tr_ratio, inf_ratio = bp_tr / ai_tr, bp_inf / ai_inf
    re_, rt = 0.25, 0.35
    e = (30 / (tr_ratio * re_), 100 / (tr_ratio * re_))
    cost = (100 / (tr_ratio * re_), 300 / (tr_ratio * re_))
    days = (60 / (r_fl * rt), 120 / (r_fl * rt))
    money.append(dict(s=s, tr_ratio=tr_ratio, inf_ratio=inf_ratio, flops=r_fl,
                      GWh=(round(min(e), 3), round(max(e), 3)),
                      MUSD=(round(min(cost), 3), round(max(cost), 3)),
                      hours=(round(min(days) * 24, 2), round(max(days) * 24, 2))))
    print(f"  duty={s:.2f}: обучение ×{tr_ratio:,.0f} пол (реалист. {min(e):.3f}–{max(e):.3f} ГВт·ч,"
          f" ${min(cost):.2f}–{max(cost):.2f}M, {min(days)*24:.1f}–{max(days)*24:.1f} ч) |"
          f" вывод ×{inf_ratio:,.0f} пол (реалист. ×{inf_ratio*0.1:,.0f}–{inf_ratio*0.5:,.0f})", flush=True)

json.dump(dict(scan=scan, levers={k: {kk: [round(vv[0], 4), round(vv[1], 4)] for kk, vv in v.items()}
                                 for k, v in levers.items()},
               fits=fits, seen_power=dict(a=a0, b=b), money=money, CPT=CPT,
               elapsed_s=time.time() - T0),
          open(ROOT / "research/CALC07_SUPER.json", "w"), ensure_ascii=False, indent=1)
print(f"\njson -> research/CALC07_SUPER.json  t={time.time()-T0:.0f}s")
