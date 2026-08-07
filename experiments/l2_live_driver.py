#!/usr/bin/env python3
"""Драйвер акта L2 живого наряда H-33 (виза «проверяй модель пробуй» 08.08).

Регламент (тот же, что руками): один тяжёлый процесс в момент, звенья
--max-minutes 12, чекпоинт каждые 1500 шагов (+зеркало в research/ckpts_h33),
авто-резюме по чекпоинту, прогресс в логе после каждого звена.
Порядок: B (эталон/суфлёр) 0→76k, затем A (полка+v2-фильтр) 0→76k с
--sufler = финальный B. Прогнозы L1/L2 заморожены в research/H33_STAKES.md —
посреди прогона ставки не меняются. Падение раннера → стоп с кодом 2.

Запуск: .venv/bin/python experiments/l2_live_driver.py   (или start_process)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
RUNNER = ROOT / "experiments" / "exp17_h33_twin.py"
RES = ROOT / "experiments" / "results" / "results_exp17.json"
CKDIR = ROOT / "research" / "ckpts_h33"
STEPS = 76_000
SEG_MIN = 12

ARMS = [
    dict(arm="B", tag="liveB_76k", sufler=""),
    dict(arm="A", tag="liveA_76k", sufler=str(CKDIR / "ckpt_h33_liveB_76k.npz")),
]


def step_done(tag: str) -> int:
    """последний сохранённый шаг руки (из зеркала чекпоинта в git-папке)."""
    ck = CKDIR / f"ckpt_h33_{tag}.npz"
    if ck.exists():
        try:
            with np.load(ck, allow_pickle=True) as z:
                return int(z["step"])
        except Exception:
            pass
    try:
        d = json.loads(RES.read_text(encoding="utf-8"))
        rec = d.get(f"h33_{tag}")
        if rec and rec.get("curve"):
            return int(rec["curve"][-1]["step"])
    except Exception:
        pass
    return 0


def main() -> int:
    t_all = time.time()
    for spec in ARMS:
        tag = spec["tag"]
        while True:
            done = step_done(tag)
            if done >= STEPS:
                print(f"[L2] {tag} завершена (step {done})  t={(time.time()-t_all)/60:.1f} мин",
                      flush=True)
                break
            cmd = [str(PY), str(RUNNER), "--arm", spec["arm"], "--src", "live",
                   "--steps", str(STEPS), "--tag", tag, "--max-minutes", str(SEG_MIN),
                   "--report-every", "1500", "--resume", "1"]
            if spec["sufler"]:
                cmd += ["--sufler", spec["sufler"]]
            print(f"[L2] звено {tag} с шага {done} →", flush=True)
            t0 = time.time()
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            tail = "\n".join(p.stdout.strip().splitlines()[-3:])
            print(f"[L2] звено {tag} кончилось rc={p.returncode} "
                  f"({(time.time()-t0)/60:.1f} мин)\n{tail}", flush=True)
            if p.returncode != 0:
                print(f"[L2] СТОП: раннер упал rc={p.returncode}\n{p.stderr[-2000:]}",
                      flush=True)
                return 2
    print(f"[L2] наряд исполнен целиком ({(time.time()-t_all)/60:.1f} мин стены)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
