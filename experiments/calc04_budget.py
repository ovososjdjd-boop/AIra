#!/usr/bin/env python3
# CALC-04 (расчёт, НЕ обучение): энергобюджет двухконтурной машины.
# Все опоры — измеренные (наш стенд) или литературные (помечены):
#  s  — доля позиций, закрываемых полкой при acc≥0.95: 0.343 (CALC-03b, Nmin=2, θ=0.90)
#       потолок 0.413 (CALC-01 масса CE), консерв. 0.279 (CALC-03b, acc≥0.975)
#  a  — активная доля весов L1 (сон/freeze): 0.42 (EXP-13, макс)
#  w  — работа обучения внутри L1 относительно плотного: 0.17..0.42 (EXP-13)
#  α  — уменьшение числа весов при выносе фактов: 10..25 (RETRO, литература!)
#  retr — цена чтения полки: ×1e-3..1e-4 от шага зоны (EXP-05: ×10³..10⁴ против KV)
#  √N — рост s с объёмом корпуса: s(N) = 0.139·√(N/2.12M) (CALC-02b, BPE-w2)
# Энергомодель (прокси v2): E = MAC×pJ_mac + bytes×pJ_by(tier)
#  pJ_mac fp16 ≈ 0.5..4 | pJ_HBM ≈ 4..25 пДж/Б | pJ_SRAM(zone) ≈ 0.6..6 пДж/Б
# Пессимист/база/оптимист — диапазоны рынка 45нм..7нм.
import json

P = 124e6          # GPT-2 small (эталон DoD)
BYTES_INF = 2.0    # fp16 вывод
BYTES_TR = 16.0    # fp32: w4 + grad4 + adam8 (per step); движение на шаг
B_TOK = 4096       # токенов на шаг BP-бейзлайна

SCEN = {
    #          pJ_mac pJ_hbm pJ_sram   a    w    s_inf s_tr  alpha
    "пессимист": dict(mac=4.0, hbm=4.0, sram=6.0, a=0.42, w=0.42, si=0.279, st=0.279, al=10),
    "база": dict(mac=1.0, hbm=12.0, sram=3.0, a=0.42, w=0.30, si=0.343, st=0.343, al=17),
    "оптимист": dict(mac=0.5, hbm=25.0, sram=0.8, a=0.30, w=0.17, si=0.413, st=0.413, al=25),
    "kaggle-база": dict(mac=1.0, hbm=12.0, sram=3.0, a=0.42, w=0.30,
                    si=None, st=None, al=25),  # s по закону √N при ×30
}
# закон √N для kaggle-сценария: N0=2.12M токенов, s0=0.139 (w2-виданная масса)
s_k = min(0.139 * (30 ** 0.5) * 0.9, 0.85)  # ×30 объёма, 10% скидка на изгиб насыщения

def budget(q):
    si = q["si"] if q["si"] is not None else s_k
    st = q["st"] if q["st"] is not None else s_k
    Pz = P / q["al"]
    out = {}
    # ---- вывод, batch-1, на токен ----
    bp_inf = P * BYTES_INF * q["hbm"] + 2 * P * q["mac"]            # пДж
    ai_inf = (1 - si) * q["a"] * (Pz * BYTES_INF * q["sram"] + 2 * Pz * q["mac"]) \
        + si * 1e-3 * (Pz * BYTES_INF * q["sram"])                   # полка бесплатно
    out["inference"] = dict(bp_uJ=bp_inf / 1e6, ai_uJ=ai_inf / 1e6,
                            bp_mJ=bp_inf / 1e9, ai_mJ=ai_inf / 1e9,
                            ratio=bp_inf / ai_inf)
    # ---- обучение, на токен ----
    bp_tr = 6 * P * q["mac"] + P * BYTES_TR * q["hbm"] / B_TOK
    ai_tr = (1 - st) * q["w"] * (6 * Pz * q["mac"] + 3 * Pz * 2 * q["sram"]) \
        + st * 1e-3 * (Pz * 2 * q["sram"])
    out["train"] = dict(bp_uJ=bp_tr / 1e6, ai_uJ=ai_tr / 1e6,
                        bp_mJ=bp_tr / 1e9, ai_mJ=ai_tr / 1e9, ratio=bp_tr / ai_tr)
    out["s_inf"], out["s_tr"] = si, st
    return out

print("=== CALC-04 энергобюджет (мкДж-мДж/токен (помечено), арифметический пол) ===")
res = {}
for name, q in SCEN.items():
    r = budget(q)
    res[name] = r
    print(f"\n[{name}]  s_inf={r['s_inf']:.2f} s_tr={r['s_tr']:.2f}"
          f"{' (√N×30)' if q['si'] is None else ''}")
    print(f"  вывод:    BP {r['inference']['bp_mJ']:.3f} мДж | AIra {r['inference']['ai_uJ']:.2f} мкДж | ×{r['inference']['ratio']:.0f}")
    print(f"  обучение: BP {r['train']['bp_mJ']:.3f} мДж | AIra {r['train']['ai_uJ']:.2f} мкДж | ×{r['train']['ratio']:.0f}")

json.dump(res, open("research/CALC04_BUDGET.json", "w"), ensure_ascii=False, indent=1)
print("\njson -> research/CALC04_BUDGET.json")
