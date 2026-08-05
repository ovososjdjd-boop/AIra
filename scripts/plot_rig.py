#!/usr/bin/env python3
"""Графики стойки M1 по experiments/results/results_exp03_*.json → benchmarks/rig_m1.png.

Две панели: (a) кривые val ppl по шагам; (b) «цена качества»: итоговые Дж-прокси vs ppl.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
TAGS = [("d96l3-ctx96-baseline", "0,35M (d96·L3·ctx96)"),
        ("d64l2-ctx64-small", "0,11M (d64·L2·ctx64)")]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for tag, label in TAGS:
        obj = json.loads((RES / f"results_exp03_{tag}.json").read_text(encoding="utf-8"))
        h = obj["history"]
        ax1.plot([p["step"] for p in h], [p["val_ppl"] for p in h], marker="o", label=label)
        r = obj["run"]
        ax2.scatter(r["val_ppl_final"], r["energy_j_total_train"], s=90, label=f"{label}")
        ax2.annotate(f"  {r['val_ppl_final']:.3f} / {r['energy_j_total_train']:.1f} Дж",
                     (r["val_ppl_final"], r["energy_j_total_train"]), fontsize=9)
    ax1.set_xlabel("шаг обучения")
    ax1.set_ylabel("val ppl (held-out)")
    ax1.set_title("GPT-micro на генеративном корпусе M1")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set_xlabel("итоговая val ppl")
    ax2.set_ylabel("энергия обучения, Дж (прокси v2)")
    ax2.set_title("«Цена качества» стойки")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "rig_m1.png"
    fig.savefig(out, dpi=130)
    print("ok:", out)


if __name__ == "__main__":
    main()
