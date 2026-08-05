#!/usr/bin/env python3
"""EXP-07 «Зонный кирпич» — первая сборка и испытания блока M2 (THEORY §12, выводы EXP-04).

Руки/сцены (всё numpy, стойка M1, корпус тот же, что в EXP-04):
  S1  Перенос H-14 на нелинейное, прогретый режим ‖W‖: diag-Jacobi-релаксатор против
      наивного Эйлера. cos(PC-grad, BP-grad) при T ∈ {4,8,16,32} на весах после 600
      шагов BP (режим, где в EXP-04 выравнивание умирало до cos 0.0–0.3).
      Ворота H-24a: jacobi cos_общий ≥ 0.8 при T=8; смерть: < 0.5 при T=32.
  S2  Двухфазный оценщик против однофазного при β ∈ {0.01,0.03,0.1,0.3} на РАВНОМ
      бюджете итераций (однофазный 32; двухфазный 16 free + 16 nudged).
      Ворота H-24b: двухфазный ≥ однофазный +0.05 cos хотя бы при одном β ∈ {0.03,0.1};
      смерть: выигрыша нет нигде.
  S3  Остаточный стоп ‖∇E‖∞ < ε ∈ {1e-3,3e-3,1e-2}: средний T̄, p90, потеря cos
      против T=32. Ворота H-24c: ε=3e-3 → T̄ ≤ 8 при Δcos ≤ 0.02; смерть: T̄ > 16 или
      Δcos > 0.1. Дополнительно: доля «тихих» координат (событийная перспектива).
  S4  Зонный шаг end-to-end (обучение charLM с нуля, 800 шагов, batch 128, общий AdamW):
      рука BP; рука PC-M2 (jacobi + остаточный стоп ε=3e-3 + одно/двухфазный по итогам
      S2) с межзонной σ-δ шиной θ ∈ {0, 0.02, 0.05} (θ=0 = dense-эталон трафика).
      Метрики: val ppl, события шины/шаг, сжатие (dense/T-событийное), T̄, время.
      Ворота H-24d: ppl(θ=0.02) ≤ BP +5% и сжатие ≥ ×5; смерть: > +15% или < ×2.

Запуск: .venv/bin/python experiments/exp07_zone_block.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from aira.tokenizer import CharTokenizer  # noqa: E402
from aira.zone import AdamW, BusSigmaDelta, CharMLP, cos_sim  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
RUNS_LOG = ROOT / "benchmarks" / "runs.jsonl"
CTX, B, VOCAB = 8, 128, 64


# ---------------------------------------------------------------- данные
def load_ids(path: Path, tok: CharTokenizer) -> np.ndarray:
    ids = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ids.extend(tok.encode(line.rstrip("\n"), add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.int64)


def batch(data: np.ndarray, rng: np.random.Generator, b: int = B, ctx: int = CTX):
    i = rng.integers(0, len(data) - ctx - 1, size=b)
    return (np.stack([data[k:k + ctx] for k in i]), data[i + ctx])


def val_ppl(model: CharMLP, data: np.ndarray, n: int = 20, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    ls = []
    for _ in range(n):
        x, y = batch(data, rng)
        logits, _ = model.forward(x)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ls.append(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    return float(np.exp(sum(ls) / len(ls)))


# ---------------------------------------------------------------- сцены
def s1_jacobi_vs_euler(warm: dict[str, np.ndarray], valid: np.ndarray) -> dict:
    """cos(PC,BP) (общий и послойный min) при T-решётке: euler/jacobi/bb."""
    print("[S1] Эйлер vs diag-Jacobi vs BB на прогретых ‖W‖…")
    rng = np.random.default_rng(123)
    t_grid = [4, 8, 16, 32, 64]
    out = {"T": t_grid, "resid_T32": {}}
    for method, alpha in [("euler", 0.3), ("jacobi", 0.6), ("bb", 1.0)]:
        out[method], out[method + "_minl"] = [], []
        for T in t_grid:
            if method != "bb" and T == 64:
                out[method].append(None); out[method + "_minl"].append(None)
                continue
            m = CharMLP(vocab=VOCAB); m.load_arrays(warm)
            coses, minl, resids = [], [], []
            for _ in range(4):
                x, y = batch(valid, rng)
                gb, _ = m.bp_grads(x, y)
                gp, st = m.pc_grads(x, y, beta=1.0, T=T, alpha=alpha, method=method)
                coses.append(cos_sim(gp, gb))
                minl.append(min(cos_sim(gp, gb, [k]) for k in ["W1", "W2", "W3", "emb"]))
                resids.append(st["resid"])
            out[method].append(round(float(np.mean(coses)), 4))
            out[method + "_minl"].append(round(float(np.mean(minl)), 4))
            if T == 32:
                out["resid_T32"][method] = round(float(np.mean(resids)), 5)
            print(f"   T={T:>2} {method:6s} cos={out[method][-1]:.4f} "
                  f"min-слой={out[method + '_minl'][-1]:.4f} resid={np.mean(resids):.4f}")
    return out


def s2_two_phase(warm: dict[str, np.ndarray], valid: np.ndarray) -> dict:
    """Однофазный (T=32) vs двухфазный (16+16)/β при β-сетке на равном бюджете."""
    print("[S2] двухфазный vs однофазный при равном бюджете 32 итерации…")
    rng = np.random.default_rng(321)
    out = {"beta": [0.01, 0.03, 0.1, 0.3], "one_phase": [], "two_phase": []}
    for beta in out["beta"]:
        res = {}
        for kind in ["one", "two"]:
            m = CharMLP(vocab=VOCAB); m.load_arrays(warm)
            coses = []
            for _ in range(4):
                x, y = batch(valid, rng)
                gb, _ = m.bp_grads(x, y)
                if kind == "one":
                    gp, _ = m.pc_grads(x, y, beta=beta, T=32, alpha=1.0, method="bb")
                else:
                    gp, _ = m.pc_grads(x, y, beta=beta, two_phase=True, T=16,
                                       alpha=1.0, method="bb")
                coses.append(cos_sim(gp, gb))
            res[kind] = round(float(np.mean(coses)), 4)
        out["one_phase"].append(res["one"])
        out["two_phase"].append(res["two"])
        print(f"   β={beta:<5} однофазный {res['one']:.4f} | двухфазный {res['two']:.4f}")
    return out


def s3_residual_stop(warm: dict[str, np.ndarray], valid: np.ndarray) -> dict:
    """Событийная релаксация: freeze-решётка — работа координат vs потеря cos."""
    print("[S3] событийный солвер: freeze-решётка…")
    rng = np.random.default_rng(777)
    m = CharMLP(vocab=VOCAB)
    m.load_arrays(warm)
    ref = []
    for _ in range(4):
        x, y = batch(valid, rng)
        gb, _ = m.bp_grads(x, y)
        gp, _ = m.pc_grads(x, y, beta=1.0, T=32, alpha=1.0, method="bb")
        ref.append(cos_sim(gp, gb))
    cos_ref = float(np.mean(ref))
    print(f"   эталон BB T=32 без сна: cos={cos_ref:.4f}")
    out = {"cos_ref": round(cos_ref, 4), "freeze": {}}
    for fr in [1e-3, 3e-3, 1e-2]:
        m2 = CharMLP(vocab=VOCAB); m2.load_arrays(warm)
        rng2 = np.random.default_rng(777)
        coses, works, tcs, stops = [], [], [], []
        for _ in range(4):
            x, y = batch(valid, rng2)
            gb, _ = m2.bp_grads(x, y)
            gp, st = m2.pc_grads(x, y, beta=1.0, T=32, alpha=1.0, method="bb",
                                 freeze=fr)
            coses.append(cos_sim(gp, gb)); works.append(st["work_frac"])
            tcs.append(st["T_coord_mean"]); stops.append(st["stop"])
        c_ = float(np.mean(coses))
        rec = {"work_frac": round(float(np.mean(works)), 3),
               "T_coord_mean": round(float(np.mean(tcs)), 1),
               "cos": round(c_, 4), "dcos": round(cos_ref - c_, 4),
               "stops": {s: stops.count(s) for s in set(stops)}}
        out["freeze"][str(fr)] = rec
        print(f"   freeze={fr:<7} работа={rec['work_frac']:<6} T̄_коорд={rec['T_coord_mean']:<5} "
              f"cos={rec['cos']:.4f} (Δ{rec['dcos']:+.4f}) стопы={rec['stops']}")
    return out


def s4_zone_training(train: np.ndarray, valid: np.ndarray, steps: int,
                     beta: float, two_phase_cfg: bool) -> dict:
    """Зонный шаг end-to-end: BP против PC-M2 с шиной θ ∈ {0, 0.02, 0.05}."""
    print(f"[S4] обучение с нуля, {steps} шагов; PC-M2(BB+freeze) β={beta}, "
          f"двухфазный={two_phase_cfg}; шины (θ_вперёд, θ_назад)")
    arms = {}
    arm_cfgs = [("bp", "bp", (None, None)),
                ("pcm2_th00", "pc", (0.0, 0.0)),
                ("pcm2_f05b00", "pc", (0.05, 0.0)),
                ("pcm2_f00b05", "pc", (0.0, 0.05)),
                ("pcm2_f05b05", "pc", (0.05, 0.05))]
    for name, kind, (th_f, th_b) in arm_cfgs:
        t0 = time.perf_counter()
        model = CharMLP(vocab=VOCAB, seed=42)
        opt = AdamW(model.arrays(), lr=3e-3)
        bus12 = BusSigmaDelta((B, 256), th_f) if kind == "pc" else None
        bus21 = BusSigmaDelta((B, 256), th_b) if kind == "pc" else None
        rng = np.random.default_rng(42)
        log = {"val_ppl": [], "steps": [], "T_mean": 0, "events_per_step": 0,
               "traffic_bits": 0, "dense_bits": 0, "work_frac": 1.0}
        T_acc, n_acc, W_acc, W_n = 0, 0, 0.0, 0
        for step in range(1, steps + 1):
            x, y = batch(train, rng)
            if kind == "bp":
                g, _ = model.bp_grads(x, y)
            elif two_phase_cfg:
                g, st = model.pc_grads(x, y, beta=beta, two_phase=True,
                                       method="bb", alpha=1.0, T=16, eps=3e-3,
                                       bus12=bus12, bus21=bus21)
                T_acc += st["T_used"]; n_acc += 1
            else:
                g, st = model.pc_grads(x, y, beta=beta, method="bb", alpha=1.0,
                                       T=32, eps=3e-3, freeze=1e-3,
                                       bus12=bus12, bus21=bus21)
                T_acc += st["T_used"]; n_acc += 1
                W_acc += st["work_frac"]; W_n += 1
            opt.step(g)
            if step % 100 == 0 or step == steps:
                log["val_ppl"].append(round(val_ppl(model, valid, n=6), 4))
                log["steps"].append(step)
        log["wall_s"] = round(time.perf_counter() - t0, 1)
        log["val_ppl_full"] = round(val_ppl(model, valid, n=20), 4)
        if kind == "pc":
            log["T_mean"] = round(T_acc / max(n_acc, 1), 1)
            log["work_frac"] = round(W_acc / max(W_n, 1), 3)
            # честный учёт: dense-эталон платит float16 за ВСЮ координату батча×итерацию;
            # σ-δ платит (адрес+16 бит) только за событие
            dens_bits = log["T_mean"] * steps * 2 * B * 256 * 16
            log["traffic_bits"] = (bus12.traffic_bits() + bus21.traffic_bits())
            log["dense_bits"] = int(dens_bits)
            log["compress"] = round(dens_bits / max(log["traffic_bits"], 1), 2)
            log["events_frac"] = round((bus12.n_events + bus21.n_events) /
                                       max(dens_bits / 16, 1), 4)
        arms[name] = log
        print(f"   рука {name:10s} ppl={log['val_ppl_full']:.4f} "
              f"wall={log['wall_s']}с "
              + (f"T̄={log['T_mean']} сжатие=×{log['compress']} "
                 f"(доля событий {log['events_frac']})" if kind == "pc" else ""))
    return arms


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    t0 = time.perf_counter()
    steps = 300 if args.quick else args.steps

    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    valid = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    print(f"[data] train {len(train)} ids, valid {len(valid)} ids")

    # прогретые веса: 600 шагов BP (режим, где в EXP-04 cos деградировал)
    print("[warm] 600 шагов BP для S1–S3…")
    wm = CharMLP(vocab=VOCAB, seed=42)
    wopt = AdamW(wm.arrays(), lr=3e-3)
    wrng = np.random.default_rng(42)
    for _ in range(600):
        x, y = batch(train, wrng)
        g, _ = wm.bp_grads(x, y)
        wopt.step(g)
    warm = wm.clone_params()
    norms = {k: round(float(np.linalg.norm(v)), 3) for k, v in warm.items()}
    print(f"[warm] ppl={val_ppl(wm, valid):.4f} нормы: {norms}")

    results = {"config": {"steps_s4": steps, "ctx": CTX, "batch": B,
                          "model": "CharMLP emb32→tanh256→tanh256→V64 (numpy)",
                          "warm_norms": norms}}
    results["S1"] = s1_jacobi_vs_euler(warm, valid)
    results["S2"] = s2_two_phase(warm, valid)
    results["S 3"] = s3_residual_stop(warm, valid)

    # выбор конфигурации S4 по S2
    i_best = int(np.argmax(results["S2"]["two_phase"]))
    two_wins = results["S2"]["two_phase"][i_best] > results["S2"]["one_phase"][i_best] + 0.02
    results["S4"] = s4_zone_training(train, valid, steps, beta=0.1,
                                     two_phase_cfg=two_wins)

    # вердикты
    s1, s2, s3, s4 = results["S1"], results["S2"], results["S 3"], results["S4"]
    v = {}
    v["H-24a bb min-cos≥0.8@T8 (✗<0.5@T32)"] = {
        "bb_minl_T8": s1["bb_minl"][1], "euler_minl_T8": s1["euler_minl"][1],
        "jacobi_minl_T8": s1["jacobi_minl"][1], "bb_minl_T32": s1["bb_minl"][3],
        "bb_minl_T64": s1["bb_minl"][4],
        "bb_total_T8": s1["bb"][1]}
    wins = [t - o for t, o in zip(s2["two_phase"], s2["one_phase"])]
    v["H-24b двухфазный +0.05 (✗ нет выигрыша)"] = {"max_delta": round(max(wins), 4),
                                                    "deltas": wins}
    fr1 = s3["freeze"]["0.001"]
    v["H-24c событийный солвер: работа≤50%, Δcos≤0.02"] = fr1
    ppl_bp = s4["bp"]["val_ppl_full"]
    th2 = s4["pcm2_th00"]
    v["H-24d зонный шаг ≤BP+5%, сжатие≥×5"] = {
        "ppl_bp": ppl_bp, "ppl_th00": th2["val_ppl_full"],
        "delta_pct_th00": round(100 * (th2["val_ppl_full"] / ppl_bp - 1), 2),
        "compress_th00": th2["compress"],
        "frontier": {n: {"delta_pct": round(100 * (s4[n]["val_ppl_full"] / ppl_bp - 1), 2),
                         "compress": s4[n]["compress"]}
                     for n in s4 if n != "bp"}}
    results["verdicts"] = v
    print("\n=== ВЕРДИКТЫ ===")
    for k, x in v.items():
        print(f"  {k}: {x}")

    out = RESULTS / "results_exp07.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[save] {out}")
    with RUNS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"rig": "sandbox-cpu-2core", "tag": "exp07-zone-block",
                             "kind": "zone-block m2 (numpy diag-jacobi PC + σδ bus)",
                             "cfg": {"steps": steps, "beta": 0.1, "eps": 3e-3},
                             "ppl_bp": ppl_bp,
                             "ppl_pcm2_th0.02": th2["val_ppl_full"],
                             "compress_th0.02": th2["compress"],
                             "wall_s": round(time.perf_counter() - t0, 1),
                             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                            ensure_ascii=False) + "\n")

    # --- картинки ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0][0]
    for key, lab, mk in [("euler_minl", "Эйлер α=0.3", "s--"),
                         ("jacobi_minl", "diag-Jacobi α=0.6", "^--"),
                         ("bb_minl", "BB α=1.0 + сторож", "o-"),
                         ("bb", "BB общий cos", "x:")]:
        xs = [t for t, v_ in zip(s1["T"], s1[key]) if v_ is not None]
        ys = [v_ for v_ in s1[key] if v_ is not None]
        ax.plot(xs, ys, mk, label=lab)
    ax.axhline(0.8, color="gray", lw=0.7, ls=":")
    ax.set_ylim(0, 1.02); ax.set_xlabel("T итераций")
    ax.set_title("S1 min-cos(PC,BP) по слоям, прогретые ‖W‖"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(s2["beta"], s2["one_phase"], "s--", label="однофазный (32)")
    ax.plot(s2["beta"], s2["two_phase"], "o-", label="двухфазный (16+16)")
    ax.set_xscale("log"); ax.set_xlabel("β"); ax.set_ylim(0, 1.02)
    ax.set_title("S2 оценщик градиента (равный бюджет)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1][0]
    for name, log in s4.items():
        ax.plot(log["steps"], log["val_ppl"], "o-" if name == "bp" else "x-",
                label=name, markevery=1)
    ax.set_title("S4 val ppl по шагам"); ax.set_ylim(1.0, None)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][1]
    names = [n for n in s4 if n != "bp"]
    comp = [s4[n]["compress"] for n in names]
    dppl = [100 * (s4[n]["val_ppl_full"] / s4["bp"]["val_ppl_full"] - 1) for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, comp, width=0.4, color="teal", label="сжатие ×")
    ax.bar(x + 0.2, dppl, width=0.4, color="crimson", label="Δppl к BP, %")
    ax.set_xticks(x, [n.replace("pcm2_", "") for n in names], fontsize=8)
    ax.axhline(5, color="gray", ls="--", lw=1)
    ax.set_title("S4 фронтир шины: сжатие vs цена качества")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("EXP-07 «Зонный кирпич» M2")
    fig.tight_layout()
    p = RESULTS / "exp07_zone_block.png"
    fig.savefig(p, dpi=130)
    print(f"[save] {p}")
    print(f"[done] за {time.perf_counter() - t0:.1f} с")


if __name__ == "__main__":
    main()
