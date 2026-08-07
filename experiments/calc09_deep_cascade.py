#!/usr/bin/env python3
# CALC-09 (расчёт, НЕ обучение): экспонента γ ГЛУБОКОГО каскада c16->c12->c8->c6->c4.
# Идея-двигатель дороги: с ростом N в работу вступают всё более ДЛИННЫЕ ключи
# (на ×30 данных c16 так же плотен, как c8 на ×1) → покрытие каскада растёт
# быстрее, чем у любого плоского уровня. Прицельный счёт (только holdout-ключи)
# — память ~1 ГБ при 3 ГБ RAM (полный счёт невозможен, и это честно отмечаем).
# ===== ПРЕДСКАЗАНИЯ ЗАМОРОЖЕНЫ ДО ПРОГОНА =====
#  P-C1 γ_каскада(16,12,8,6,4) > γ_c8 (0.0737); ожидание 0.10–0.18.
#  P-C2 на ×1 (полный train) глубокий каскад НЕ хуже мелкого B1: duty >= 41.0%@0.95
#       (реализация каскада корректна, глубина не вредит на малых N).
#  Смерть P-C1: γ_каскада <= 0.0737 → «глубина с масштабом» мертва; дорога к
#  сверхрезультату должна идти другими ступенями (семантика).
import json, time, math, collections
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")
alpha = {c: i for i, c in enumerate(sorted(set(text)))}
V = len(alpha)
ids = [alpha[c] for c in text]
n_tr = int(len(ids) * 0.90)
tr, ho = ids[:n_tr], ids[n_tr:]
LEVELS = (16, 12, 8, 6, 4)
print(f"стенд ×1: train {len(tr)} holdout {len(ho)} V={V} уровни {LEVELS}  t={time.time()-T0:.1f}s", flush=True)

# --- целевые контексты: все коды уровней по holdout + эталонная позиция --- #
targets = {n: set() for n in LEVELS}
codes_ho = {n: 0 for n in LEVELS}
mods = {n: V ** n for n in LEVELS}
ho_rows = []  # (idx -> {level: code}), next char отдельно
for i in range(len(ho) - 1):
    x = ho[i]
    for n in LEVELS:
        codes_ho[n] = (codes_ho[n] * V + x) % mods[n]
    if i < max(LEVELS) - 1:
        if i == max(LEVELS) - 2:
            pass
        continue
    row = {n: codes_ho[n] for n in LEVELS}
    ho_rows.append((row, ho[i + 1]))
    for n in LEVELS:
        targets[n].add(codes_ho[n])
print(f"позиций пробы {len(ho_rows)}; целевых кодов: "
      + ", ".join(f"c{n}:{len(targets[n])}" for n in LEVELS) + f"  t={time.time()-T0:.0f}s", flush=True)

RULES = [(0.90, 2), (0.90, 5), (0.95, 2), (0.95, 5)]

def targeted_pass(sub_len):
    ctx = {n: collections.defaultdict(int) for n in LEVELS}
    pair = {n: collections.defaultdict(int) for n in LEVELS}
    codes = {n: 0 for n in LEVELS}
    limit = int(len(tr) * sub_len)
    for pos in range(limit):
        x = tr[pos]
        nxt = tr[pos + 1] if pos + 1 < len(tr) else None
        for n in LEVELS:
            codes[n] = (codes[n] * V + x) % mods[n]
        if nxt is None:
            continue
        for n in LEVELS:
            c = codes[n]
            if c in targets[n]:
                ctx[n][c] += 1
                pair[n][c * V + nxt] += 1
    top = {n: {} for n in LEVELS}
    for n in LEVELS:
        for k, cnt in pair[n].items():
            cd, xx = divmod(k, V)
            cur = top[n].get(cd)
            if cur is None or cnt > cur[0]:
                top[n][cd] = (cnt, xx)
    out = {}
    for th, nm in RULES:
        cov = hit = 0
        for row, nxt in ho_rows:
            for n in LEVELS:
                cnt = ctx[n].get(row[n], 0)
                if cnt >= nm:
                    tc, tx = top[n][row[n]]
                    if tc / cnt >= th:
                        cov += 1; hit += (tx == nxt)
                        break
        out[f"{th},{nm}"] = (cov / len(ho_rows), hit / max(cov, 1))
    return out

rows = []
for fr in (1 / 4, 1 / 2, 1.0):
    res = targeted_pass(fr)
    row = dict(fr=fr, chars=int(len(tr) * fr), rules={k: [round(v[0], 4), round(v[1], 4)] for k, v in res.items()})
    rows.append(row)
    for k, (d, a) in res.items():
        print(f"  fr={fr:>4.2f} | правило ({k}): duty={d*100:5.1f}% acc={a:.4f}  t={time.time()-T0:.0f}s", flush=True)

# фиты γ по каждому правилу + калиброванный (acc>=0.95) фит
fits = {}
for rule in RULES:
    key = f"{rule[0]},{rule[1]}"
    pts = [(r["chars"], r["rules"][key][0]) for r in rows if r["rules"][key][0] > 0.01]
    xs = [math.log(n) for n, _ in pts]; ys = [math.log(1 - d) for _, d in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    gn = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    fits[key] = gn
    print(f"  γ({key}) = {gn:.4f}  (duty: " + " -> ".join(f"{r['rules'][key][0]*100:.1f}%" for r in rows) + ")")
# калиброванный ряд: на каждой fr берём правило с макс duty при acc>=0.95 (по ближайшему)
cal_pts = []
for r in rows:
    cands = [(v[0], k) for k, v in r["rules"].items() if v[1] >= 0.95]
    cal_pts.append((r["chars"], max(cands)[0] if cands else None))
print("  калиброванный ряд (макс duty при acc>=0.95):",
      [(n, f"{d*100:.1f}%" if d is not None else "н/д") for n, d in cal_pts])
cal_ok = [(n, d) for n, d in cal_pts if d is not None]
if len(cal_ok) >= 2:
    xs = [math.log(n) for n, _ in cal_ok]; ys = [math.log(1 - d) for _, d in cal_ok]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    g = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
else:
    g = fits["0.9,2"]
print(f"\nγ_каскада (канон 0.9,2) = {fits['0.9,2']:.4f} | калиброванный γ = {g:.4f} (γ_c8 = 0.0737)")

if g > 0.01 and len(cal_ok) >= 2:
    xs_l = [math.log(n) for n, _ in cal_ok]; ys_l = [math.log(1 - d) for _, d in cal_ok]
    cc = math.exp(my + g * mx)
    for tgt in (0.90, 0.95):
        Nt = (cc / (1 - tgt)) ** (1 / g) / 4.683
        print(f"  N(s={tgt:.2f}) по калиброванному каскаду: {Nt:.2e} ток (Opus D=1.3e13: {'ДОСТИЖИМО' if Nt < 1.3e13 else f'x{Nt/1.3e13:.0f} сверх'})")

verdict_c1 = "✓ ПОДТВЕРЖДЕНО" if g > 0.0737 else "✗ СМЕРТЬ (γ не круче плоского)"
fr1 = rows[-1]["rules"]
best95 = max(((v[0], v[1], k) for k, v in fr1.items() if v[1] >= 0.95), default=None)
verdict_c2 = "✓" if best95 and best95[0] >= 0.41 else ("~ погранично" if best95 and best95[0] >= 0.38 else "✗")
print(f"P-C1 {verdict_c1} | P-C2 {verdict_c2} (×1 лучший@0.95: "
      + (f"{best95[0]*100:.1f}%@{best95[1]:.4f} {best95[2]}" if best95 else "нет") + ")")

json.dump(dict(rows=rows, gamma_by_rule=fits, gamma_calibrated=g, gamma_c8=0.0737,
               calibrated_points=[(n, d) for n, d in cal_pts],
               verdicts=dict(P_C1=verdict_c1, P_C2=verdict_c2),
               levels=LEVELS, elapsed_s=time.time() - T0),
          open(ROOT / "research/CALC09_DEEP_CASCADE.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/CALC09_DEEP_CASCADE.json  t={time.time()-T0:.0f}s")
