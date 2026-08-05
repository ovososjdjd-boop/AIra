#!/usr/bin/env python3
"""EXP-04 «Мост обучения»: равновесный PC против BP на посимвольной языковой задаче.

Критический тест аудита (AUDIT_2026-08-05 §3.1): впервые наш локальный способ обучения
соревнуется с backprop не на игрушке, а на живом тексте (генеративный корпус M1).

Модель — CharMLP (без внимания): окно из 8 символов → emb(32) → tanh(256) → tanh(256) →
логиты 64, кросс-энтропия. Две руки СТРОГО на стойке: одинаковый init/батчи/LR-расписание/
оптимизатор (AdamW), различается только способ получения градиентов:
  - BP: autograd;
  - PC: равновесная релаксация состояний (T итераций), градиенты весов = локальное
    произведение ошибка×вход (ε_l ⊙ f' · s_{l-1}); слой выхода — точный CE-градиент.
Метрики: val ppl обеих рук по шагам; cos(PC-grad, BP-grad) послойно в контролях; время;
учёт цены шага в fwd-эквивалентах (честно: на обычном кремнии PC-шаг дороже по MAC —
экономика FF-13 про транспорт/локальность, не про MAC).

Запуск: .venv/bin/python experiments/exp04_pc_bridge.py [--smoke] [--steps 1500]
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from aira.tokenizer import CharTokenizer  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
RUNS_LOG = ROOT / "benchmarks" / "runs.jsonl"


# ---------------------------------------------------------------- модель
class CharMLP(nn.Module):
    def __init__(self, vocab: int = 64, ctx: int = 8, d_emb: int = 32, d_hid: int = 256):
        super().__init__()
        self.ctx, self.d_emb = ctx, d_emb
        self.emb = nn.Embedding(vocab, d_emb)
        self.W1 = nn.Linear(ctx * d_emb, d_hid)
        self.W2 = nn.Linear(d_hid, d_hid)
        self.W3 = nn.Linear(d_hid, vocab)
        self._init()

    def _init(self) -> None:
        nn.init.normal_(self.emb.weight, 0, 0.05)
        for lin in (self.W1, self.W2, self.W3):
            nn.init.normal_(lin.weight, 0, 1.0 / math.sqrt(lin.in_features))
            nn.init.zeros_(lin.bias)

    def forward(self, idx):                     # idx (B, ctx) → логиты (B, V)
        x = self.emb(idx).reshape(idx.size(0), -1)
        h1 = torch.tanh(self.W1(x))
        h2 = torch.tanh(self.W2(h1))
        return self.W3(h2)


# ---------------------------------------------------------------- PC-рука
@torch.no_grad()
def pc_states_and_grads(model: CharMLP, idx: torch.Tensor, y: torch.Tensor,
                        t_relax: int, alpha: float):
    """Релаксация состояний (s1,s2) к равновесию энергии E, затем локальные градиенты.

    E = ½||s1−tanh(W1x)||² + ½||s2−tanh(W2 s1)||² + CE(softmax(W3 s2), y).
    Обновления спуском по ds = −∂E/∂s; градиенты весов — произведения ошибка×вход.
    """
    x = model.emb(idx).reshape(idx.size(0), -1)
    B = idx.size(0)
    # свободный проход: инициализация состояний предсказаниями
    s1 = torch.tanh(model.W1(x))
    s2 = torch.tanh(model.W2(s1))
    y_oh = F.one_hot(y, model.W3.out_features).float()

    for _ in range(t_relax):
        f1 = torch.tanh(model.W1(x))
        f2 = torch.tanh(model.W2(s1))
        e1 = s1 - f1                                  # (B, 256)
        e2 = s2 - f2
        logits = model.W3(s2)
        d_ce = torch.softmax(logits, dim=1) - y_oh    # ∂CE/∂логиты (B, V)
        g_s1 = e1 - (e2 * (1 - f2.pow(2))) @ model.W2.weight   # ∂E/∂s1 = e1 − W₂ᵀ(e₂⊙f₂′)
        g_s2 = e2 + d_ce @ model.W3.weight
        s1 = s1 - alpha * g_s1
        s2 = s2 - alpha * g_s2

    # локальные градиенты весов при равновесных состояниях
    f1 = torch.tanh(model.W1(x))
    f2 = torch.tanh(model.W2(s1))
    e1 = s1 - f1
    e2 = s2 - f2
    logits = model.W3(s2)
    d_ce = torch.softmax(logits, dim=1) - y_oh
    t1 = e1 * (1 - f1.pow(2))                       # e⊙f′ слоя 1
    t2 = e2 * (1 - f2.pow(2))                       # e⊙f′ слоя 2
    grads = {
        "W1.weight": -(t1.T @ x) / B,               # ∂E/∂W = −(e⊙f′)·вход
        "W1.bias": -t1.mean(0),
        "W2.weight": -(t2.T @ s1) / B,
        "W2.bias": -t2.mean(0),
        "W3.weight": d_ce.T @ s2 / B,
        "W3.bias": d_ce.mean(0),
    }
    # входной градиент ∂E/∂x = −W1ᵀ(e1⊙f1′) → разметка по эмбеддингам окна
    g_emb = torch.zeros_like(model.emb.weight)
    flat = -(t1 @ model.W1.weight)                      # (B, ctx*d)
    flat = flat.reshape(B, model.ctx, model.d_emb)
    g_emb.index_add_(0, idx.reshape(-1), flat.reshape(-1, model.d_emb))
    grads["emb.weight"] = g_emb / B
    loss = F.cross_entropy(model.W3(s2), y)
    return grads, float(loss)


# ---------------------------------------------------------------- данные
def load_ids(path: Path, tok: CharTokenizer) -> np.ndarray:
    ids = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ids.extend(tok.encode(line.rstrip("\n"), add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.int64)


def batch(data: np.ndarray, rng: np.random.Generator, b: int, ctx: int):
    i = rng.integers(0, len(data) - ctx - 1, size=b)
    x = torch.from_numpy(np.stack([data[k:k + ctx] for k in i]))
    y = torch.from_numpy(data[i + ctx])
    return x, y


@torch.no_grad()
def val_ppl(model: CharMLP, data: np.ndarray, ctx: int, b: int, n: int, seed: int) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    ls = []
    for _ in range(n):
        x, y = batch(data, rng, b, ctx)
        ls.append(F.cross_entropy(model(x), y).item())
    model.train()
    return math.exp(sum(ls) / len(ls))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.flatten() @ b.flatten()) /
                 (a.norm() * b.norm() + 1e-12))


# ---------------------------------------------------------------- главный цикл
def run_arm(kind: str, args, model: CharMLP, train_ids, valid_ids, tok) -> dict:
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    history, cos_log = [], []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(train_ids, rng, args.batch, model.ctx)

        if kind == "bp":
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step % args.cos_every == 0 or step == 1:
                bp_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        else:  # pc
            grads, loss_t = pc_states_and_grads(model, x, y, args.t_relax, args.alpha)
            loss_v = loss_t
            if step % args.cos_every == 0 or step == 1:
                # эталон BP-градиента на ТОМ ЖЕ батче весов для cos (без применения)
                with torch.enable_grad():
                    logits = model(x)
                    lb = F.cross_entropy(logits, y)
                    ref = torch.autograd.grad(lb, list(model.parameters()))
                bp_grads = {n: g for (n, _), g in zip(model.named_parameters(), ref)}
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.grad = grads[n].clone()
            if step % args.cos_every == 0 or step == 1:
                cos_log.append({"step": step, **{
                    f"cos_{n.split('.')[0]}": cos(bp_grads[n], grads[n])
                    for n in bp_grads}})
            loss = torch.tensor(loss_v)

        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == args.steps:
            ppl = val_ppl(model, valid_ids, model.ctx, args.batch, 12, args.seed)
            history.append({"step": step, "val_ppl": round(ppl, 3),
                            "wall_s": round(time.perf_counter() - t0, 1)})
            print(f"  [{kind} step {step:>5}] val_ppl={ppl:8.3f}")
    wall = time.perf_counter() - t0
    return {"kind": kind, "history": history, "cos_log": cos_log,
            "final_ppl": history[-1]["val_ppl"], "wall_s": round(wall, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-relax", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--cos-every", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.cos_every = 40, 20, 20

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, len(__import__("os").sched_getaffinity(0))))
    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    print(f"[data] train={train_ids.size:,} valid={valid_ids.size:,}")

    # одинаковый init обеих рук: фиксируем начальное состояние
    torch.manual_seed(args.seed)
    proto = CharMLP(ctx=args.ctx)
    init_state = {k: v.clone() for k, v in proto.state_dict().items()}

    out = {"cfg": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                   "seed": args.seed, "t_relax": args.t_relax, "alpha": args.alpha,
                   "ctx": args.ctx, "model": "CharMLP emb32→tanh256→tanh256→V64"},
           "note_cost": "PC-шаг ≈ (3+2·T) fwd-эквивалентов MAC против 3 у BP "
                        "(T=%d): на потоковом кремнии PC дороже по MAC — экономика "
                        "в транспорте/локальности (FF-13), не в MAC" % args.t_relax,
           "arms": {}}
    for kind in ("bp", "pc"):
        torch.manual_seed(args.seed)
        model = CharMLP(ctx=args.ctx)
        model.load_state_dict(init_state)
        print(f"[arm {kind}] старт (одинаковый init/данные/LR)")
        out["arms"][kind] = run_arm(kind, args, model, train_ids, valid_ids, tok)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results_exp04.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # лог стойки
    for kind, r in out["arms"].items():
        run = {"rig": "sandbox-cpu-2core", "tag": f"exp04-{kind}-charmlp",
               "kind": f"{kind}-char-mlp (ctx8, tanh, {out['cfg']['t_relax']} relax iters)"
               if kind == "pc" else "bp-char-mlp (ctx8, tanh)",
               "cfg": out["cfg"], "params": sum(p.numel() for p in proto.parameters()),
               "tokens_seen": args.steps * args.batch,
               "val_ppl_final": r["final_ppl"], "wall_s": r["wall_s"],
               "tok_per_s": round(args.steps * args.batch / r["wall_s"], 1),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with RUNS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(run, ensure_ascii=False) + "\n")
    print(f"[итог] BP ppl={out['arms']['bp']['final_ppl']:.3f} за {out['arms']['bp']['wall_s']}с | "
          f"PC ppl={out['arms']['pc']['final_ppl']:.3f} за {out['arms']['pc']['wall_s']}с")
    if out["arms"]["pc"]["cos_log"]:
        last = out["arms"]["pc"]["cos_log"][-1]
        print("[cos PC↔BP на финале]:", {k: round(v, 4) for k, v in last.items() if k != "step"})


if __name__ == "__main__":
    main()
