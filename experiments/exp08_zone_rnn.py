#!/usr/bin/env python3
"""EXP-08 «Рекуррентный носитель» — M3 v0: обучение ZoneRNN локальным кредитом.

СИТУАЦИЯ НА ВХОДЕ (диагностика предыдущего витка, зафиксированная в runs):
  - градиенты корректны: ∇E релаксации против float64 FD cos = 1.000000; BPTT против
    полного FD relL2 ≤ 8% (усечение tanh-пути — BPTT принят «правдой»);
  - однофазное EP-смещение ∝β подтверждено и на рекурренте (β=0.1: cos 0.82 /
    Wh +0.38; β=0.03: cos 0.79 / Wh +0.81 — «лучшего» β подо все слои нет);
  - обучение: BP-BPTT ppl@300 = 1.727; PC при β=0.05, T=32, lr=1.5e-3, клип 1.0 —
    улучшается до 4.106@200, ЗАТЕМ РАЗНОС: 6.19@400, resid растёт 0.02→0.065.
  Гипотеза-механизм разноса: по мере роста ‖Wh‖,‖W3‖ жёсткость энергии растёт,
  релаксация за фиксированный T=32 перестаёт сходиться → градиент мусорный →
  веса растут ещё → спираль. Та же болезнь, что в EXP-07, но по петле времени.

ЛЕКАРСТВА (принципиальные, перенос находок M2 на ось времени):
  L1  адаптивная релаксация: крутим до resid < 0.01 или T_max=96 (честная цена — T̄);
  L2  resid-гейт: обновление весов ПРОПУСКАЕТСЯ, если финальный resid > 0.02
      (доверительная область качества релаксации; мусорный градиент не применяем);
  L3  wd=0.01 на все веса (AdamW) — держим петлю контрактивной, как в M2.

ГИПОТЕЗА H-25 (ZoneRNN обучаем локальным кредитом):
  ворота: PC-лечение (S3) ppl@400 ≤ BP-BPTT (S1) + 25% И разноса нет (‖grad‖ стабилен,
  skip-фракция ≤ 30%);
  смерть: разнос (ppl@400 > ppl@200 и растёт), или ppl@400 > BP + 50%,
  или T̄ > 64 (цена нечестная — эквивалент полного BPTT).

РУКИ:
  S0  FD-санити этой сборки (float64, микромодель): bptt vs FD relL2 < 1e-3;
      ∇E релаксации vs FD cos > 0.9999.
  S1  BP-BPTT эталон: AdamW lr=3e-3, клип 1.0, wd=0.01, 400 шагов.
  S2  PC «как было» (воспроизведение смерти): β=0.05, T=32, eps=0, lr=1.5e-3.
  S3  PC-лечение L1+L2+L3: β=0.05, T_max=96, eps=0.01, гейт 0.02, lr=1.5e-3.

Данные/инициализация общие: стойка M1, char-токенизатор V=64, seed 42 во всех руках.
Запуск: .venv/bin/python experiments/exp08_zone_rnn.py [--arm s0|s1|s2|s3|all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from aira.seq import ZoneRNN  # noqa: E402
from aira.tokenizer import CharTokenizer  # noqa: E402
from aira.zone import AdamW  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
VOCAB, D_EMB, D_H, L, B = 64, 32, 96, 24, 64
STEPS = 400
VAL_EVERY = 100


# ---------------------------------------------------------------- данные
def load_ids(path: Path, tok: CharTokenizer) -> np.ndarray:
    ids = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ids.extend(tok.encode(line.rstrip("\n"), add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.int64)


def batch_stream(data: np.ndarray, rng: np.random.Generator,
                 b: int = B, l: int = L) -> tuple[np.ndarray, np.ndarray]:
    i = rng.integers(0, len(data) - l - 1, size=b)
    x = np.stack([data[k:k + l] for k in i])
    y = np.stack([data[k + 1:k + l + 1] for k in i])
    return x, y


def val_ppl(model: ZoneRNN, val_ids: np.ndarray, l: int = L, blocks: int = 160) -> float:
    x = val_ids[: blocks * l].reshape(blocks, l)
    y = val_ids[1: blocks * l + 1].reshape(blocks, l)
    return model.ppl(x, y)


# ---------------------------------------------------------------- утилиты
def arrays_f64(model: ZoneRNN) -> dict[str, np.ndarray]:
    return {k: v.astype(np.float64).copy() for k, v in model.arrays().items()}


def grad_norm(g: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(float((v * v).sum()) for v in g.values())))


def clip_(g: dict[str, np.ndarray], c: float = 1.0) -> dict[str, np.ndarray]:
    n = grad_norm(g)
    if n > c:
        for k in g:
            g[k] *= c / n
    return g


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------- S0 FD-санити
def s0_fd_check(h: float = 1e-4) -> dict:
    """BPTT против центральных FD. FD считаем по независимому float64-форварду

    (стандартный forward льёт состояния в float32 — это убило бы точность
    конечных разностей, поэтому здесь свой 6-строчный float64-проход; заодно
    он независимо перепроверяет сам forward).
    """
    m = ZoneRNN(vocab=8, d_emb=4, d_h=8, seed=1)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 8, size=(2, 6))
    y = rng.integers(0, 8, size=(2, 6))

    def loss_f64(params: dict[str, np.ndarray]) -> float:
        p64 = {k: v.astype(np.float64) for k, v in params.items()}
        B_, L_ = x.shape
        xe = p64["emb"][x]
        hh = np.zeros((B_, 8))
        ce = 0.0
        for t in range(L_):
            hh = np.tanh(xe[:, t] @ p64["Wx"].T + hh @ p64["Wh"].T + p64["b"])
            z = hh @ p64["W3"].T + p64["b3"]
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z); pr = e / e.sum(axis=1, keepdims=True)
            ce += -np.log(pr[np.arange(B_), y[:, t]]).sum()
        return float(ce / (B_ * L_))

    def loss_with(params: dict[str, np.ndarray]) -> float:
        return loss_f64(params)


    g_bp, _ = m.bptt_grads(x, y)
    out = {}
    for name in ("W3", "Wh", "Wx", "b", "emb"):
        base = m.arrays()[name]
        fd = np.zeros_like(base, dtype=np.float64)
        it = np.nditer(base, flags=["multi_index"], op_flags=["readwrite"])
        count = 0
        while not it.finished:
            idxm = it.multi_index
            if count % max(1, base.size // 60) == 0:  # до ~60 точек на массив
                orig = float(base[idxm])
                pp = arrays_f64(m); pp[name][idxm] = orig + 1e-3
                lp = loss_with(pp)
                mm_ = arrays_f64(m); mm_[name][idxm] = orig - 1e-3
                lm = loss_with(mm_)
                fd[idxm] = (lp - lm) / 2e-3
            count += 1
            it.iternext()
        mask = fd != 0
        r = rel_l2(fd[mask], np.asarray(g_bp[name], dtype=np.float64)[mask])
        out[f"bptt_vs_fd::{name}"] = r
    return out


# ---------------------------------------------------------------- обучение
def train(arm: str, grads_fn, lr: float, seed: int = 42,
          steps: int = STEPS, val_every: int = VAL_EVERY) -> dict:
    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    val_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    model = ZoneRNN(vocab=VOCAB, d_emb=D_EMB, d_h=D_H, seed=seed)
    opt = AdamW(model.arrays(), lr=lr, wd=0.01)
    rng = np.random.default_rng(123)
    log = {"arm": arm, "curve": [], "resid": [], "T": [], "skips": 0,
           "wh_fro": [], "time_s": 0.0}
    t0 = time.time()
    ppl_run, n_run = 0.0, 0
    for step in range(1, steps + 1):
        x, y = batch_stream(train_ids, rng)
        g, info = grads_fn(model, x, y)
        if info.get("skip"):
            log["skips"] += 1
        else:
            opt.step(clip_(g, 1.0))
        ppl_run += float(np.exp(min(info["loss"], 20.0))); n_run += 1
        if step % 25 == 0:
            print(f"[{arm}] step {step:4d}  train_ppl {ppl_run / n_run:7.3f}  "
                  f"resid {info.get('resid', float('nan')):.4f}  T {info.get('T', 0):3d}  "
                  f"skip {log['skips']}", flush=True)
            ppl_run, n_run = 0.0, 0
        log["resid"].append(info.get("resid", 0.0))
        log["T"].append(info.get("T", 0))
        if step % val_every == 0:
            vp = val_ppl(model, val_ids)
            wh = float(np.linalg.norm(model.Wh))
            log["curve"].append({"step": step, "val_ppl": vp})
            log["wh_fro"].append(wh)
            print(f"[{arm}] == step {step}: val_ppl {vp:.4f}  ‖Wh‖_F {wh:.2f}", flush=True)
    log["time_s"] = round(time.time() - t0, 1)
    log["T_mean"] = float(np.mean(log["T"]))
    log["resid_tail"] = float(np.mean(log["resid"][-50:]))
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["s0", "s1", "s2", "s3", "s4", "all"])
    args = ap.parse_args()
    out: dict = {"config": {"V": VOCAB, "d_emb": D_EMB, "d_h": D_H, "L": L, "B": B,
                            "steps": STEPS}}

    if args.arm in ("s0", "all"):
        t0 = time.time()
        out["s0_fd"] = s0_fd_check()
        print(f"[s0] FD-санити за {time.time() - t0:.0f} с: {out['s0_fd']}", flush=True)

    if args.arm in ("s1", "all", "s2", "s3"):
        pass  # данные грузим внутри train; руки ниже

    if args.arm in ("s1", "all"):
        def bp_fn(m: ZoneRNN, x, y):
            g, loss = m.bptt_grads(x, y)
            return g, {"loss": loss, "resid": 0.0, "T": 0}
        out["s1_bp"] = train("s1_bp", bp_fn, lr=3e-3)

    if args.arm in ("s2", "all"):
        def pc_old(m: ZoneRNN, x, y):
            g, st = m.pc_grads(x, y, beta=0.05, T=32, freeze=3e-3)
            # восстановим loss для лога: посчитаем на прогоне forward
            logits, _, _ = m.forward(x)
            from aira.zone import softmax
            p = softmax(logits.reshape(-1, m.vocab))
            yy = y.reshape(-1)
            loss = float(-np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1)).mean())
            return g, {"loss": loss, "resid": st["resid"], "T": st["T_used"]}
        out["s2_pc_staraya"] = train("s2_pc_old", pc_old, lr=1.5e-3)

    if args.arm in ("s3", "all"):
        GATE = 0.02
        def pc_cure(m: ZoneRNN, x, y):
            g, st = m.pc_grads(x, y, beta=0.05, T=96, eps=0.01, freeze=3e-3)
            logits, _, _ = m.forward(x)
            from aira.zone import softmax
            p = softmax(logits.reshape(-1, m.vocab))
            yy = y.reshape(-1)
            loss = float(-np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1)).mean())
            skip = st["resid"] > GATE
            return g, {"loss": loss, "resid": st["resid"], "T": st["T_used"],
                       "skip": skip}
        out["s3_pc_cure"] = train("s3_pc_cure", pc_cure, lr=1.5e-3)

    if args.arm in ("s4",):
        # Длинный контроль (H-25, асимптота зазора): BP и PC-cure, 1500 шагов.
        LONG = 1500
        def bp_fn2(m: ZoneRNN, x, y):
            g, loss = m.bptt_grads(x, y)
            return g, {"loss": loss, "resid": 0.0, "T": 0}
        out["s4_bp_long"] = train("s4_bp_long", bp_fn2, lr=3e-3,
                                  steps=LONG, val_every=250)
        GATE = 0.02
        def pc_cure2(m: ZoneRNN, x, y):
            g, st = m.pc_grads(x, y, beta=0.05, T=96, eps=0.01, freeze=3e-3)
            logits, _, _ = m.forward(x)
            from aira.zone import softmax
            p = softmax(logits.reshape(-1, m.vocab))
            yy = y.reshape(-1)
            loss = float(-np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1)).mean())
            skip = st["resid"] > GATE
            return g, {"loss": loss, "resid": st["resid"], "T": st["T_used"],
                       "skip": skip}
        out["s4_pc_long"] = train("s4_pc_long", pc_cure2, lr=1.5e-3,
                                  steps=LONG, val_every=250)

    RESULTS.mkdir(exist_ok=True, parents=True)
    fp = RESULTS / "results_exp08.json"
    if fp.exists():  # руки запускаются по отдельности — сливаем, а не затираем
        prev = json.loads(fp.read_text(encoding="utf-8"))
        prev.update(out)
        out = prev
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", RESULTS / "results_exp08.json", flush=True)


if __name__ == "__main__":
    main()
