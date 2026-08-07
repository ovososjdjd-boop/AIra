#!/usr/bin/env python3
# CALC-06 (расчёт, НЕ обучение): проекция энергобюджета на класс Claude-Opus.
# Формулы — CALC-04. Опоры — измеренные на стойке (s,a,w) + √N-закон (CALC-02b)
# + единственная литературная α (RETRO 10-25). Класс-рамка Opus: N≈5e11..2e12
# параметров, D≈1.3e13 токенов (публичных данных нет — аналогия GPT-4/Llama3,
# помечается как допущение). Расчёт — сценарии ратио, не абсолютная правда:
# открытые риски C6/ватты/железо — в отчёте.
import json, math

N_BAND = [5e11, 1e12, 2e12]   # класс параметров (допущение!)
D = 1.3e13                    # объём обучения, токенов (допущение, аналогия)
B_TOK = 4096                  # батч бейзлайна

# обучение/вывод: как в CALC-04
# s на Opus-объёме: √N-экстраполяция 0.139·√(N/2.12e6) с потолком и скидкой δ kaggle/Opus-
def s_at(n_tok, cap_disc=1.0):
    raw = 0.139 * math.sqrt(n_tok / 2.12e6)
    return min(raw, 0.95) * cap_disc

SCEN = {
    #          pJ_mac pJ_hbm pJ_sram   a    w   alpha   s фикс (duty = 1-s)
    "пессимист": dict(mac=4.0, hbm=4.0, sram=6.0, a=0.42, w=0.42, al=10, s=0.50),
    "база": dict(mac=1.0, hbm=12.0, sram=3.0, a=0.42, w=0.30, al=17, s=0.75),
    "оптимист": dict(mac=0.5, hbm=25.0, sram=0.8, a=0.30, w=0.17, al=25, s=0.90),
}
print(f"контроль √N-экстраполяции: s({D:.0e} ток) = {s_at(D):.3f} (потолок) — в сценариях использованы скидки 0.50/0.75/0.90 как duty-разумные\n")

rows = {}
for name, q in SCEN.items():
    duty = 1 - q["s"]
    for N in N_BAND:
        Pz = N / q["al"]
        # обучение, на токен (назад-совместимо с CALC-04)
        bp_tr = 6 * N * q["mac"] + (16 * N) * q["hbm"] / B_TOK          # пДж
        ai_tr = duty * q["w"] * (6 * Pz * q["mac"] + 3 * Pz * 2 * q["sram"])
        # вывод, batch-1, на токен
        bp_inf = 2 * N * q["hbm"] + 2 * N * q["mac"]
        ai_inf = duty * q["a"] * (2 * Pz * q["sram"] + 2 * Pz * q["mac"])
        # скорость обучения: чистые FLOPs
        r_flops = (6 * N) / (duty * q["w"] * 6 * Pz)
        rows.setdefault(name, []).append(dict(
            N=N, duty=duty,
            tr_bp_mJ=bp_tr / 1e9, tr_ai_uJ=ai_tr / 1e6, tr_ratio=bp_tr / ai_tr,
            inf_bp_mJ=bp_inf / 1e9, inf_ai_uJ=ai_inf / 1e6, inf_ratio=bp_inf / ai_inf,
            flops_ratio=r_flops))

for name, band in rows.items():
    print(f"=== {name} ===")
    for r in band:
        print(f"  N={r['N']:.1e}: обучение {r['tr_bp_mJ']:.1f} мДж→{r['tr_ai_uJ']:.0f} мкДж (×{r['tr_ratio']:.0f}) | "
              f"вывод {r['inf_bp_mJ']:.1f} мДж→{r['inf_ai_uJ']:.0f} мкДж (×{r['inf_ratio']:.0f}) | "
              f"FLOP-ускорение ×{r['flops_ratio']:.0f}")

# итоги прогона целиком (N=1e12 центр класса): бенч Opus-прогона ~$100-300M, 30-100 ГВт·ч, 60-120 дней @30к H100
print("\n=== итог прогона (класс-рамка $100-300M / 30-100 ГВт·ч / 60-120 дней) ===")
for name, band in rows.items():
    r = band[1]  # N=1e12
    real_e = 0.25  # реализм-фактор CALC-04 (0.1-0.5)
    real_t = 0.35  # аппаратно-фактор скорости (плотные GPU не любят события) ×0.35 (0.1..0.6)
    e_lo, e_hi = 30 / (r['tr_ratio'] * real_e), 100 / (r['tr_ratio'] * real_e)
    c_lo, c_hi = 100 / (r['tr_ratio'] * real_e), 300 / (r['tr_ratio'] * real_e)
    t_lo, t_hi = 60 / (r['flops_ratio'] * real_t), 120 / (r['flops_ratio'] * real_t)
    print(f"[{name}] N=1e12: энергия {min(e_lo,e_hi):.1f}–{max(e_lo,e_hi):.1f} ГВт·ч | "
          f"цена ${min(c_lo,c_hi):.1f}–{max(c_lo,c_hi):.1f}M | время {min(t_lo,t_hi):.1f}–{max(t_lo,t_hi):.1f} дн")
    print(f"   вывод-энергия реалист.: ×{r['inf_ratio']*0.1:.0f}–{r['inf_ratio']*0.5:.0f}")

json.dump(rows, open("research/CALC06_OPUS.json", "w"), ensure_ascii=False, indent=1)
print("\njson -> research/CALC06_OPUS.json")
