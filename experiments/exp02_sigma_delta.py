"""EXP-02 — Сигма-дельта событийное кодирование (THEORY §1–§3, FF-1..FF-3).

Проверяем:
  A  FF-1 (граница): n_events ≤ V_T/θ + 1 и max|s − ŝ| ≤ θ — по построению.
  B  Масштабирование: n_events ∝ 1/θ, а n_events·θ → V_T (коэффициент порядка 1).
  C  FF-3 (дрейф): пороговое обновление сжимающей динамики h = λh + x на σ-δ-входе
     дрейфует от плотной версии не более чем на θ_eff/(1−λ) — измеряем наклон.
  D  Энергия (FF-2 + ENERGY_MODEL): бюджет экономии событийного канала против плотного
     в битах как функция статистики сигнала (вариация/динамический диапазон).

Запуск: .venv/bin/python experiments/exp02_sigma_delta.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aira import events

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
results = {"series": {}}

# ---------------------------------------------------------------- сигналы
def ou_process(T, tau=100.0, sigma=0.1, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = x[t - 1] - x[t - 1] / tau + sigma * np.sqrt(2 / tau) * r.standard_normal()
    return x

def random_walk(T, step=0.05, seed=1):
    r = np.random.default_rng(seed)
    return np.cumsum(r.standard_normal(T) * step)

T = 20_000
x_ou = ou_process(T)
x_rw = random_walk(T)

# ---------------------------------------------------------------- A+B
print("A/B: граница событий и масштабирование по θ")
rows = []
for name, x in (("OU", x_ou), ("RW", x_rw)):
    V = events.total_variation(x)
    for theta in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        ev, rec = events.sigma_delta_encode(x, theta)
        bound = V / theta + 1
        maxerr = float(np.max(np.abs(x - rec)))
        rows.append({"signal": name, "theta": theta, "n_events": len(ev),
                     "bound": bound, "ratio_events_to_bound": len(ev) / bound,
                     "V_over_T": V, "max_err": maxerr})
        assert maxerr <= theta * (1 + 1e-9), "нарушена граница ошибки θ!"
        assert len(ev) <= bound + 1, "нарушена граница событий FF-1!"
        print(f"  {name} θ={theta:5.3f}: n={len(ev):6d} ≤ V/θ+1={bound:8.0f}  "
              f"(n·θ/V={len(ev)*theta/V:.3f})  maxErr={maxerr:.4f}")
results["series"]["AB_events_bound"] = rows

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
ax = axes[0]
sel = x_ou[:2000]
ev, rec = events.sigma_delta_encode(sel, 0.02)
ax.plot(sel, lw=0.8, label="сигнал (OU)")
ax.plot(rec, lw=0.8, label="σ-δ реконструкция θ=0.02")
ax.plot(ev, sel[ev], "|", ms=7, label="события")
ax.legend(fontsize=8, loc="upper right"); ax.set_title("A: событийный кодек")
ax.grid(alpha=0.3)
ax = axes[1]
for name, m in (("OU", "o"), ("RW", "s")):
    pts = [(1 / r["theta"], r["n_events"]) for r in rows if r["signal"] == name]
    ax.loglog(*zip(*pts), m + "-", label=name)
ax.set_xlabel("1/θ"); ax.set_ylabel("число событий")
ax.set_title("B: n_events ∝ 1/θ (вариационный бюджет)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "exp02_events.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- C: дрейф FF-3
print("C: FF-3 — дрейф сжимающей динамики на σ-δ-входе")
rows_c = []
x = x_ou
h_dense_ref = events.threshold_dynamics(x, 0.0)  # не используется; для красоты
for lam in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
    h_dense = events.threshold_dynamics(x, lam)
    for theta in (0.02, 0.05):
        ev, rec = events.sigma_delta_encode(x, theta)
        h_lazy = events.threshold_dynamics(rec, lam)
        drift = float(np.max(np.abs(h_dense - h_lazy)))
        theory = theta / (1 - lam)
        rows_c.append({"lambda": lam, "theta": theta, "drift": drift,
                       "theory": theory, "frac": drift / theory})
        print(f"  λ={lam:4.2f} θ={theta:4.2f}: drift={drift:.4f} ≤ θ/(1−λ)={theory:6.2f} "
              f"(доля {drift/theory:.3f})")
results["series"]["C_drift"] = rows_c

fig, ax = plt.subplots(figsize=(4.5, 3.4))
for theta, m in ((0.02, "o"), (0.05, "s")):
    pts = sorted({(1 / (1 - r["lambda"]), r["drift"] / r["theta"]) for r in rows_c if r["theta"] == theta})
    ax.plot(*zip(*pts), m + "-", label=f"измерено, θ={theta}")
gg = np.linspace(1, 100, 50)
ax.plot(gg, gg, "k--", lw=1, label="теория: drift/θ = 1/(1−λ)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("1/(1−λ)"); ax.set_ylabel("drift / θ")
ax.set_title("C: дрейф подчиняется θ/(1−λ) (ниже — лучше запас)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "exp02_drift.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- D: энергия/биты
print("D: бюджет экономии событийного канала (d=16 каналов, payload 8 бит)")
rows_d = []
dch = 16
X = np.stack([ou_process(T, sigma=s, seed=10 + i) for i in range(dch) for s in [0.05]], axis=0)
addr_bits = int(np.ceil(np.log2(dch)))
b_payload = 8
dense_bits = T * dch * b_payload
for theta in (0.01, 0.02, 0.05, 0.1, 0.2):
    n_ev = sum(len(events.sigma_delta_encode(X[i], theta)[0]) for i in range(dch))
    event_bits = n_ev * (addr_bits + b_payload)
    savings = dense_bits / event_bits
    rows_d.append({"theta": theta, "n_events_total": int(n_ev),
                   "dense_bits": dense_bits, "event_bits": event_bits,
                   "savings_x": savings})
    print(f"  θ={theta:4.2f}: событий {n_ev:7d}, экономия ×{savings:6.1f}")
results["series"]["D_energy"] = rows_d

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.semilogx([r["theta"] for r in rows_d], [r["savings_x"] for r in rows_d], "o-")
ax.set_xlabel("θ"); ax.set_ylabel("экономия бит ×")
ax.set_title("D: экономия событийного канала (OU, d=16)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "exp02_energy.png", dpi=110); plt.close(fig)

with open(OUT / "results_exp02.json", "w") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=1)
print(f"\nГотово → {OUT / 'results_exp02.json'}")
