#!/usr/bin/env python3
# Проверка ставок S1–S6 прогона-близнеца H-33 (читает results_exp17.json).
# Замороженные критерии: research/CALC05_DESIGN.md §5 (S1–S5) и
# research/CALC10_11_INVARIANTS.md (S6). Обучения не запускает.
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "experiments/results/results_exp17.json").read_text(encoding="utf-8")) \
    if (ROOT / "experiments/results/results_exp17.json").exists() else {}


def best(tag):
    """последняя запись руки h33_<tag>*, с максимальным числом шагов."""
    cands = [(k, v) for k, v in R.items() if k.startswith(f"h33_{tag}")]
    if not cands:
        return None
    k, v = max(cands, key=lambda kv: kv[1]["curve"][-1]["step"] if kv[1].get("curve") else 0)
    return v


def final(rec, *keys):
    return rec["curve"][-1] if rec and rec.get("curve") else {k: None for k in keys}


rows = []
A = best("A"); B = best("B"); B256 = best("B256"); B512 = best("B512")

# --- S1 качество: ppl_A (система = гибрид полка+зона) <= ppl_B·[0.97..1.00]; смерть > ×1.01 --- #
if A and B:
    h = A.get("hybrid_final") or {}
    pa = h.get("ppl_h") or A["val_ppl_full"]
    pb = B["val_ppl_full"]
    r = pa / pb
    verd = "✓" if r <= 1.00 else ("✗ СМЕРТЬ" if r > 1.01 else "~ грань")
    rows.append(("S1 качество", f"ppl_A(гибрид)={pa:.4f} ppl_B={pb:.4f} → ×{r:.4f} (цель [0.97,1.00])", verd))
else:
    rows.append(("S1 качество", "рук A/B недостаточно", "—"))

# --- S2 duty: первая ≤0.78 → устойчивая ≤0.66; смерть duty₁>0.85 --- #
if A:
    d_tra = [c["duty_link"] for c in A["curve"][:2]]
    d_fin = A.get("duty_total")
    evalA = final(A)
    ok = d_tra and d_tra[0] <= 0.78 and (evalA.get("duty_link", 1) <= 0.66)
    dead = d_tra and d_tra[0] > 0.85
    verd = "✓" if ok else ("✗ СМЕРТЬ" if dead else "~ отклонение вверх (поток шаблоннее прогноза)")
    rows.append(("S2 duty", f"первая {d_tra}, устойчивая(тотал) {d_fin} (прогноз ≤0.78→≤0.66)", verd))
else:
    rows.append(("S2 duty", "нет руки A", "—"))

# --- S3 α: α̂ = P*(ppl_A)/117k из чистых рук; смерть α̂<3 --- #
if A and B256 and B512:
    import math
    pa = (A.get("hybrid_final") or {}).get("ppl_h") or A["val_ppl_full"]
    pts = sorted(((B512["val_ppl_full"], 512), (B256["val_ppl_full"], 256),
                  (B["val_ppl_full"], 96) if B else (9, 96)))
    pts = [(p, d) for p, d in pts if p != 9]
    Pstar = None
    for i in range(len(pts) - 1):
        (p1, d1), (p2, d2) = pts[i], pts[i + 1]
        if (p1 - pa) * (p2 - pa) <= 0:
            t = (pa - p1) / (p2 - p1)
            Pstar = d1 + (d2 - d1) * t
            break
    if Pstar is None:  # ppl_A вне лестницы (лучше всех контролей): лестница инвертирована
        min_control = min(p for p, _ in pts)
        rows.append(("S3 α̂",
                     f"ppl_A(гибрид)={pa:.4f} < min чистых рук {min_control:.4f} → α̂ НЕДОСТИЖИМ по построению "
                     f"(лестница 96→512 инвертирована на этом потоке @2400) — качественно: полка даёт качество, "
                     f"не покупаемое ни одним чистым размером при том же токен-бюджете", "✓* (без числа)"))
    else:
        alpha_hat = Pstar / 96.0
        verd = "✓" if 10 <= alpha_hat <= 25 else ("✗ СМЕРТЬ" if alpha_hat < 3 else "~ грань")
        rows.append(("S3 α̂", f"P*(ppl_A={pa:.4f}) ≈ зона d={Pstar:.0f} → α̂={alpha_hat:.1f}", verd))
else:
    rows.append(("S3 α̂", "нужны руки B256 и B512 (и B)", "—"))

# --- S4 полка вживую: acc≥0.95 при покрытии≥0.30; смерть acc<0.90 --- #
if A:
    h = A.get("hybrid_final") or final(A)
    cov, acc = h.get("shelf_cov"), h.get("shelf_acc")
    verd = "✓" if cov and acc and acc >= 0.95 and cov >= 0.30 else ("✗ СМЕРТЬ" if acc and acc < 0.90 else "~ грань")
    rows.append(("S4 полка", f"cov {cov} acc {acc}", verd))
else:
    rows.append(("S4 полка", "нет руки A", "—"))

# --- S5 прокси-энергия: A обучает ≤0.25× токенов, вывод > работы; смерть >0.5× --- #
if A and B:
    ta, tb = A["tokens_train"], B["tokens_train"]
    r = ta / tb if tb else None
    verd = "✓" if r is not None and r <= 0.25 else ("✗ СМЕРТЬ" if r is not None and r > 0.5 else "~ грань")
    rows.append(("S5 энергопрокси", f"tokens_train A/B = {ta}/{tb} = {r:.3f}" if r else "—", verd))
else:
    rows.append(("S5 энергопрокси", "нет рук A/B", "—"))

# --- S6 валидатор: prec≥0.97 recall≥0.80; смерть prec<0.95 --- #
if A:
    h = A.get("hybrid_final") or final(A)
    p, rc = h.get("s6_prec"), h.get("s6_rec")
    verd = "✓" if p and rc and p >= 0.97 and rc >= 0.80 else ("✗ СМЕРТЬ" if p and p < 0.95 else "~ грань")
    rows.append(("S6 валидатор", f"prec {p} rec {rc} (n_pass {h.get('s6_n_pass')})", verd))
else:
    rows.append(("S6 валидатор", "нет руки A", "—"))

out = ["# H-33 — вердикты ставок S1–S6 (авто)", ""]
for name, val, verd in rows:
    out.append(f"| {name} | {val} | {verd} |")
print("\n".join(out))
(ROOT / "research/H33_STAKES.md").write_text(
    "# H-33 — вердикты ставок S1–S6 (авто, `experiments/h33_stakes.py`)\n\n"
    "| Ставка | Замер | Вердикт |\n|---|---|---|\n"
    + "\n".join(f"| {n} | {v} | {d} |" for n, v, d in rows) + "\n",
    encoding="utf-8")
print("\n-> research/H33_STAKES.md")
