"""EXP-01 — Трилемма локального кредита: PC-релаксация vs BP (THEORY §11, FF-11..FF-13).

Серии:
  S1  FF-11 (валидация): равновесный апдейт PC == BP на равновесных состояниях (точно);
      «объект-смузинг»: зазор к BP@ff квадратичен масштабу остатка σ.
  S2  FF-12 (челюсть): WU до равновесия (‖∇E‖≤tol) vs глубина L для транспорта α=1.0
      и α=0.9 + измеренное обусловление κ(L).
  S3  H-14: Jacobi-предобуславливатель против Richardson (ускорение).
  S4  H-18 (флагман): Multigrid-иерархия по глубине против плоской релаксации.
  S5  §12 Т1: зонная релаксация (точное решение внутри зоны) — цена по размеру зоны.
  S6  Санити нелинейности (tanh, бэктрекинг-GD): картина сохраняется.

Первичная метрика времени: WU до ‖∇E‖ ≤ 1e−4 (монотонна, не зависит от шага проб).
cos-выравнивание PC-апдейтов замеряется у равновесия (S1) — там оно структурно точное.
Запуск: .venv/bin/python experiments/exp01_pc_trilemma.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aira import pc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

BETA = 100.0
TOL = 1e-4
D = 8


def bp_at_states(Ws, s_full, y, beta, kind="linear"):
    """BP-рекурсия, вычисленная на произвольных состояниях (виртуальный прогон)."""
    L = len(Ws)
    delta = beta * (s_full[L] - y)
    grads = [None] * L
    for l in range(L, 0, -1):
        grads[l - 1] = np.outer(delta, pc.f_act(s_full[l - 1], kind))
        if l > 1:
            delta = pc.f_act_deriv(s_full[l - 1], kind) * (Ws[l - 1].T @ delta)
    return grads


def fit_slope_tail(xs, ys, n=4):
    lx, ly = np.log(xs[-n:]), np.log(ys[-n:])
    return float(np.polyfit(lx, ly, 1)[0])


results = {"config": {"beta": BETA, "tol": TOL, "d": D}, "series": {}}
t_start = time.time()

# ---------------------------------------------------------------- S1: FF-11
print("S1: FF-11 — точность равновесного правила и объект-смузинг")
s1, s1b = [], []
for L in (8, 16):
    Ws = pc.make_chain(L, D, 1.0, seed=0)
    x, y = pc.make_io(Ws, seed=0)
    A, b = pc.ab_matrices(Ws, x, y, BETA)
    s_eq = pc.exact_solve(A, b)
    s_full = [x] + [s_eq[(l - 1) * D:l * D] for l in range(1, L + 1)]
    u = pc.pc_update(s_eq, Ws, x, y, BETA)
    c_eq = pc.cos_align(u, bp_at_states(Ws, s_full, y, BETA))
    c_ff = pc.cos_align(u, pc.bp_gradient(Ws, x, y, BETA))
    s1.append({"L": L, "cos_min_vs_BP_eq": c_eq[0], "cos_glob_vs_BP_eq": c_eq[1],
               "cos_glob_vs_BP_ff": c_ff[1]})
    print(f"  L={L}: vs BP@eq cos_min={c_eq[0]:.6f} | vs BP@ff cos_glob={c_ff[1]:.4f}")
Ws1 = pc.make_chain(8, D, 1.0, seed=0)
for sigma in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
    x, y = pc.make_io(Ws1, seed=0, sigma=sigma)
    A, b = pc.ab_matrices(Ws1, x, y, BETA)
    s_eq = pc.exact_solve(A, b)
    u = pc.pc_update(s_eq, Ws1, x, y, BETA)
    cf = pc.cos_align(u, pc.bp_gradient(Ws1, x, y, BETA))[1]
    s1b.append({"sigma": sigma, "cos_glob_vs_BP_ff": cf})
    print(f"  sigma={sigma:4.2f}: 1-cos = {1-cf:.2e}")
results["series"]["S1_exact_rule"] = s1
results["series"]["S1b_bias_vs_sigma"] = s1b

fig, ax = plt.subplots(figsize=(4.5, 3))
sig = [r["sigma"] for r in s1b]
ax.loglog(sig, [1 - r["cos_glob_vs_BP_ff"] for r in s1b], "o-", label="измерено")
ax.loglog(sig, [0.2 * s ** 2 for s in sig], "--", label="0.2·σ²")
ax.set_xlabel("σ (масштаб остатка цели)"); ax.set_ylabel("1 − cos")
ax.set_title("S1: объект-смузинг PC ~ σ²"); ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "s1_bias_vs_sigma.png", dpi=110); plt.close(fig)

# ------------------------------------------- S2+S3: челюсть + предобуславливатель
print("S2/S3: WU до равновесия v глубина L (α=1.0 / α=0.9) + Jacobi")
s2 = []
for alpha in (1.0, 0.9):
    for L in [4, 8, 12, 16, 24, 32, 48, 64]:
        Ws = pc.make_chain(L, D, alpha, seed=0)
        x, y = pc.make_io(Ws, seed=0)
        A, b = pc.ab_matrices(Ws, x, y, BETA)
        lmin, lmax = pc.eig_bounds(A)
        s0 = pc.ff_states_vec(Ws, x)
        t_max = 2_000_000 if (alpha == 1.0 and L >= 64) else 600_000
        _, wu_r, _, ok_r = pc.solve_richardson(A, b, s0, t_max, TOL)
        omega_j = 0.7
        _, wu_j, _, ok_j = pc.solve_jacobi(A, b, s0, t_max, TOL, omega=omega_j)
        row = {"alpha": alpha, "L": L, "kappa": lmax / lmin,
               "wu_richardson": wu_r if ok_r else None,
               "wu_jacobi": wu_j if ok_j else None}
        s2.append(row)
        print(f"  α={alpha} L={L:3d}: κ={row['kappa']:.0f}  R={row['wu_richardson']}  J={row['wu_jacobi']}")
results["series"]["S2_jaw"] = s2

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
ax = axes[0]
for alpha, m in ((1.0, "o"), (0.9, "s")):
    pts = [(r["L"], r["wu_richardson"]) for r in s2 if r["alpha"] == alpha and r["wu_richardson"]]
    xs, ys = zip(*pts)
    k = fit_slope_tail(xs, ys)
    ax.loglog(xs, ys, m + "-", label=f"α={alpha} (наклон хвоста≈{k:.2f})")
pts = [(r["L"], r["wu_jacobi"]) for r in s2 if r["alpha"] == 1.0 and r["wu_jacobi"]]
xs, ys = zip(*pts)
k = fit_slope_tail(xs, ys)
ax.loglog(xs, ys, "d--", label=f"α=1.0 + Jacobi (≈{k:.2f})")
ax.set_xlabel("глубина L"); ax.set_ylabel("WU до ‖∇E‖ ≤ 1e−4")
ax.set_title("S2/S3: челюсть трилеммы и предобуславливатель")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
ax = axes[1]
for alpha, m in ((1.0, "o"), (0.9, "s")):
    pts = [(r["L"], r["kappa"]) for r in s2 if r["alpha"] == alpha]
    xs, ys = zip(*pts)
    k = fit_slope_tail(xs, ys)
    ax.loglog(xs, ys, m + "-", label=f"α={alpha} (κ, наклон≈{k:.2f})")
ax.set_xlabel("глубина L"); ax.set_ylabel("κ = λ_max/λ_min")
ax.set_title("S2: обусловление релаксационной системы")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "s2_jaw_kappa.png", dpi=110); plt.close(fig)

# ------------------------------------------------- S4: Multigrid (H-18, флагман)
print("S4: H-18 — Multigrid V-циклы по глубине")
s4 = []
for L in [8, 16, 32, 64, 128]:
    Ws = pc.make_chain(L, D, 1.0, seed=0)
    x, y = pc.make_io(Ws, seed=0)
    A, b = pc.ab_matrices(Ws, x, y, BETA)
    s0 = pc.ff_states_vec(Ws, x)
    s_mg, wu_mg, _, ok_mg = pc.solve_multigrid(A, b, s0, L, D, 300, TOL)
    s4.append({"L": L, "wu_mg": wu_mg if ok_mg else None})
    print(f"  L={L:3d}: MG WU={wu_mg:.0f} ok={ok_mg}")
results["series"]["S4_multigrid"] = s4

fig, ax = plt.subplots(figsize=(5, 3.5))
pts = [(r["L"], r["wu_richardson"]) for r in s2 if r["alpha"] == 1.0 and r["wu_richardson"]]
xs, ys = zip(*pts)
k_r = fit_slope_tail(xs, ys)
ax.loglog(xs, ys, "o-", label=f"плоская релаксация (≈{k_r:.2f})")
xs, ys = zip(*[(r["L"], r["wu_mg"]) for r in s4 if r["wu_mg"]])
k_m = fit_slope_tail(xs, ys)
ax.loglog(xs, ys, "^-", label=f"Multigrid по глубине (≈{k_m:.2f})")
ax.set_xlabel("глубина L"); ax.set_ylabel("WU до ‖∇E‖ ≤ 1e−4")
ax.set_title("S4: H-18 — иерархия шкал снимает диффузию кредита")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "s4_multigrid_vs_flat.png", dpi=110); plt.close(fig)

# ------------------------------------------------- S2b: асимптотическая скорость
print("S2b: измеренная скорость сходимости (возмущённый старт возбуждает медленные моды)")
s2b = []
for L in (16, 32, 64):
    Ws = pc.make_chain(L, D, 1.0, seed=0)
    x, y = pc.make_io(Ws, seed=0)
    A, b = pc.ab_matrices(Ws, x, y, BETA)
    lmin, lmax = pc.eig_bounds(A)
    Dv = np.diag(A)
    rng = np.random.default_rng(5)
    s_p0 = pc.ff_states_vec(Ws, x) + 0.1 * rng.standard_normal(L * D)

    eta = 1.5 / lmax
    s = s_p0.copy()
    rn = np.empty(60_000)
    for t in range(len(rn)):
        r = b - A @ s
        rn[t] = np.linalg.norm(r)
        s = s + eta * r
    def tchar(rn):
        floor = rn[0] * 1e-12
        idx = np.arange(len(rn)) >= len(rn) // 2
        mask = idx & (rn > floor)
        if mask.sum() < 100:
            mask = rn > floor          # быстрая сходимость: фит по всему окну
        if mask.sum() < 100:
            return float("nan")
        rate = float(np.polyfit(np.arange(len(rn))[mask], np.log(rn[mask]), 1)[0])
        return -1.0 / rate if rate < 0 else float("inf")
    T_r = tchar(rn)
    s = s_p0.copy()
    rn = np.empty(60_000)
    for t in range(len(rn)):
        r = b - A @ s
        rn[t] = np.linalg.norm(r)
        s = s + 0.7 * r / Dv
    T_j = tchar(rn)
    s2b.append({"L": L, "kappa": lmax / lmin, "T_char_richardson": T_r,
                "T_char_jacobi": T_j, "T_char_theory_k_div_1p5": (lmax / lmin) / 1.5})
    print(f"  L={L:3d}: T_char GD={T_r:.0f} (теор κ/1.5={lmax/lmin/1.5:.0f}), Jacobi={T_j:.0f}")
results["series"]["S2b_rate"] = s2b

# ------------------------------------------------- S5: зонная релаксация
print("S5: §12 Т1 — цена зонной релаксации по размеру зоны")
s5 = []
for L in (16, 32, 64, 128):
    Ws = pc.make_chain(L, D, 1.0, seed=0)
    x, y = pc.make_io(Ws, seed=0)
    A, b = pc.ab_matrices(Ws, x, y, BETA)
    s0 = pc.ff_states_vec(Ws, x)
    zs = (2, 4, 8) if L <= 32 else (2, 4)
    for z in zs:
        _, wu, ok = pc.solve_block_gauss_seidel(A, b, s0, z, 2000, TOL, L, D)
        s5.append({"L": L, "zone": z, "wu": wu, "ok": ok})
        print(f"  L={L} z={z:2d}: WU={wu:.1f} ok={ok}")
# глобальный случай z=L для малых L (дорого при больших)
for L in (16, 32):
    Ws = pc.make_chain(L, D, 1.0, seed=0)
    x, y = pc.make_io(Ws, seed=0)
    A, b = pc.ab_matrices(Ws, x, y, BETA)
    s0 = pc.ff_states_vec(Ws, x)
    _, wu, ok = pc.solve_block_gauss_seidel(A, b, s0, L, 5, TOL, L, D)
    s5.append({"L": L, "zone": L, "wu": wu, "ok": ok})
    print(f"  L={L} z={L:2d} (глоб.): WU={wu:.1f} ok={ok}")
results["series"]["S5_zones"] = s5

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
ax = axes[0]
for L, m in ((16, "o"), (32, "s")):
    pts = [(r["zone"], r["wu"]) for r in s5 if r["L"] == L and r["ok"] and r["zone"] <= L]
    if pts:
        pts.sort()
        ax.plot(*zip(*pts), m + "-", label=f"L={L}")
ax.set_xlabel("размер зоны z (слоёв)"); ax.set_ylabel("WU до tol")
ax.set_yscale("log")
ax.set_title("S5: цена зоны (z=L = глобальная)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
# флагманский свод: все пути к равновесию на одной карте
pts = [(r["L"], r["wu_richardson"]) for r in s2 if r["alpha"] == 1.0 and r["wu_richardson"]]; ax.loglog(*zip(*pts), "o-", label=f"плоский GD (≈{fit_slope_tail(*zip(*pts)):.2f})")
pts = [(r["L"], r["wu_jacobi"]) for r in s2 if r["alpha"] == 1.0 and r["wu_jacobi"]]; ax.loglog(*zip(*pts), "d--", label=f"+Jacobi (≈{fit_slope_tail(*zip(*pts)):.2f})")
pts = sorted({(r["L"], r["wu"]) for r in s5 if r["zone"] == 2 and r["ok"]}); ax.loglog(*zip(*pts), "s-", label=f"зоны z=2 (≈{fit_slope_tail(*zip(*pts)):.2f})")
pts = [(r["L"], r["wu_mg"]) for r in s4 if r["wu_mg"]]; ax.loglog(*zip(*pts), "^-", label=f"Multigrid (≈{fit_slope_tail(*zip(*pts), 3):.2f})")
ax.set_xlabel("глубина L"); ax.set_ylabel("WU до ‖∇E‖ ≤ 1e−4")
ax.set_title("Флагман: пути релаксации (наклоны log-log)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "s5_flagship_relaxation_paths.png", dpi=110); plt.close(fig)

# ------------------------------------------------- S6: tanh-санити
print("S6: tanh-цепочка, бэктрекинг-GD")
s6 = []
for kind in ("linear", "tanh"):
    L = 8
    Ws = pc.make_chain(L, D, 0.9, seed=0)
    x, y = pc.make_io(Ws, seed=0, kind=kind)

    def E_of(s):
        s_full = [x] + [s[(l - 1) * D:l * D] for l in range(1, L + 1)]
        E = 0.0
        for l in range(1, L + 1):
            e = s_full[l] - Ws[l - 1] @ pc.f_act(s_full[l - 1], kind)
            E += 0.5 * float(e @ e)
        return E + 0.5 * BETA * float((s_full[L] - y) @ (s_full[L] - y))

    def grad_of(s):
        s_full = [x] + [s[(l - 1) * D:l * D] for l in range(1, L + 1)]
        g = np.zeros_like(s)
        epss = [None] * L
        for l in range(1, L + 1):
            epss[l - 1] = s_full[l] - Ws[l - 1] @ pc.f_act(s_full[l - 1], kind)
            g[(l - 1) * D:l * D] += epss[l - 1]
        for l in range(2, L + 1):
            g[(l - 2) * D:(l - 1) * D] -= pc.f_act_deriv(s_full[l - 1], kind) * (Ws[l - 1].T @ epss[l - 1])
        g[(L - 1) * D:L * D] += BETA * (s_full[L] - y)
        return g

    g_ff = pc.bp_gradient(Ws, x, y, BETA, kind)
    s = pc.ff_states_vec(Ws, x, kind)
    eta, hist = 0.05, []
    g0n = np.linalg.norm(grad_of(s))
    for t in range(400_001):
        g = grad_of(s)
        gn = np.linalg.norm(g)
        s_new = s - eta * g
        if E_of(s_new) > E_of(s):
            eta = max(eta * 0.5, 1e-4)
            continue
        eta = min(eta * 1.1, 0.2)
        s = s_new
        if t % 2000 == 0:
            hist.append((t, pc.cos_align(pc.pc_update(s, Ws, x, y, BETA, kind), g_ff)[0]))
        if gn / g0n < 1e-6:
            break
    iters_done = t
    def grab(sv):
        svf = [x] + [sv[(l - 1) * D:l * D] for l in range(1, L + 1)]
        u = pc.pc_update(sv, Ws, x, y, BETA, kind)
        return (pc.cos_align(u, pc.bp_gradient(Ws, x, y, BETA, kind)),
                pc.cos_align(u, bp_at_states(Ws, svf, y, BETA, kind)))
    c_ff, c_self = grab(s)
    s6.append({"kind": kind, "iters": iters_done,
               "cos_glob_vs_BP_ff": c_ff[1], "cos_glob_vs_BP_eq": c_self[1],
               "cos_min_vs_BP_eq": c_self[0], "hist": hist})
    print(f"  {kind}: ит={iters_done}  vs BP@eq cos_min={c_self[0]:.4f}  vs BP@ff cos_glob={c_ff[1]:.4f}")
results["series"]["S6_tanh"] = s6

fig, ax = plt.subplots(figsize=(4.5, 3))
for r in s6:
    ax.plot([h[0] for h in r["hist"]], [h[1] for h in r["hist"]], label=r["kind"])
ax.set_xlabel("итерации GD по состояниям"); ax.set_ylabel("cos_min к BP@ff")
ax.set_title("S6: PC-релаксация — линейная и tanh-цепочки")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "s6_tanh.png", dpi=110); plt.close(fig)

results["wall_time_s"] = round(time.time() - t_start, 1)
with open(OUT / "results_exp01.json", "w") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=1)
print(f"\nГотово за {results['wall_time_s']} c → {OUT / 'results_exp01.json'}")
