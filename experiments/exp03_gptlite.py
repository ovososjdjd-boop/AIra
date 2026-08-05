#!/usr/bin/env python3
"""EXP-03 / M1: бейзлайн GPT-micro (char-level) на генеративном корпусе — первая строка стойки.

Собственная реализация в духе философии nanoGPT (минимализм, pre-LN, tied-эмбеддинги),
без копирования чужого кода. Цель — не SOTA, а честная точка отсчёта:
  {конфиг, val ppl, Дж-прокси/токен, время, ток/с} → benchmarks/runs.jsonl (формат RIG).

Честность стойки (ROADMAP §4): фиксированные seed'ы, held-out без утечек,
прокси-энергия с явными допущениями (src/aira/energy.py, «прокси v2»).

Запуск:
  .venv/bin/python experiments/exp03_gptlite.py --smoke          # проверка работоспособности
  .venv/bin/python experiments/exp03_gptlite.py --steps 1500 --tag baseline-d96l3
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from aira import energy  # noqa: E402
from aira.tokenizer import CharTokenizer  # noqa: E402

RUNS_LOG = ROOT / "benchmarks" / "runs.jsonl"
RESULTS = ROOT / "experiments" / "results"


# ---------------------------------------------------------------- модель
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, ctx: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx, dtype=torch.bool)),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, c // self.heads).transpose(1, 2)
        k = k.view(b, t, self.heads, c // self.heads).transpose(1, 2)
        v = v.view(b, t, self.heads, c // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        att = att.masked_fill(~self.mask[:t, :t], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return self.proj(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ctx: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, ctx)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=False), nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPTLite(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, ctx: int):
        super().__init__()
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(ctx, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, ctx) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        cfg = dict(vocab=vocab, dim=dim, layers=layers, heads=heads, ctx=ctx)
        self.cfg = cfg
        # инициализация σ=0.02 (GPT-стиль): старт близок к равномерному (loss≈ln V),
        # без взрыва логитов от стандартного N(0,1) эмбеддинга
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T            # tied unembedding
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- данные
def load_corpus(path: Path, tok: CharTokenizer) -> np.ndarray:
    ids = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ids.extend(tok.encode(line.rstrip("\n"), add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.int64)


def sample_batch(data: np.ndarray, rng: np.random.Generator, batch: int, ctx: int):
    ix = rng.integers(0, len(data) - ctx - 1, size=batch)
    x = np.stack([data[i:i + ctx] for i in ix])
    y = np.stack([data[i + 1:i + ctx + 1] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def val_ppl(model: GPTLite, data: np.ndarray, batch: int, ctx: int, n_batches: int,
            seed: int) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(data, rng, batch, ctx)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return math.exp(sum(losses) / len(losses))


# ---------------------------------------------------------------- запуск
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--smoke", action="store_true", help="10 шагов, проверка работоспособности")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.eval_batches = 10, 5, 2

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, len(__import__("os").sched_getaffinity(0))))
    rng = np.random.default_rng(args.seed)

    data_dir = ROOT / "data"
    for f in ("corpus_train.txt", "corpus_valid.txt", "tokenizer_char.json"):
        if not (data_dir / f).exists():
            sys.exit(f"нет {data_dir / f} — сначала: python3 scripts/build_corpus.py")
    tok = CharTokenizer.load(data_dir / "tokenizer_char.json")
    vocab = tok.vocab_size
    if vocab % 8:  # кратность 8 — чище для будущего int8/тайлинга
        vocab += 8 - vocab % 8
    train_ids = load_corpus(data_dir / "corpus_train.txt", tok)
    valid_ids = load_corpus(data_dir / "corpus_valid.txt", tok)
    print(f"[data] train={train_ids.size:,} tok, valid={valid_ids.size:,} tok, vocab={vocab}")

    model = GPTLite(vocab, args.dim, args.layers, args.heads, args.ctx)
    n_params = model.count_params()
    print(f"[model] пар-ов: {n_params:,} ({n_params / 1e6:.2f}M), cfg={model.cfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    # --- энергопрокси (v2): стоимость токена обучения при текущем конфиге ---
    tokens_per_step = args.batch * args.ctx
    pj_per_token = energy.transformer_train_token_pj(
        n_params=n_params, n_layers=args.layers, dim=args.dim, ctx=args.ctx,
        tokens_per_step=tokens_per_step, weight_bits=16)
    print(f"[energy] прокси v2: {pj_per_token:,.0f} пДж/токен обучения "
          f"= {pj_per_token / 1e6:,.3f} мкДж/токен")

    history, t0, tokens_seen = [], time.perf_counter(), 0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = sample_batch(train_ids, rng, args.batch, args.ctx)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += tokens_per_step
        if step % args.eval_every == 0 or step == args.steps:
            ppl = val_ppl(model, valid_ids, args.batch, args.ctx, args.eval_batches, args.seed)
            wall = time.perf_counter() - t0
            tps = tokens_seen / wall
            history.append({"step": step, "train_loss": round(loss.item(), 4),
                            "val_ppl": round(ppl, 3), "wall_s": round(wall, 1),
                            "tok_per_s": round(tps, 1)})
            print(f"[step {step:>5}] loss={loss.item():.3f} val_ppl={ppl:7.2f} "
                  f"tok/s={tps:,.0f}")

    wall = time.perf_counter() - t0
    final = history[-1]
    run = {
        "rig": "sandbox-cpu-2core", "tag": args.tag, "kind": "gpt-lite-char (dense bp)",
        "cfg": model.cfg | {"batch": args.batch, "steps": args.steps, "lr": args.lr,
                            "seed": args.seed},
        "params": n_params, "tokens_seen": tokens_seen,
        "val_ppl_final": final["val_ppl"],
        "energy_pj_per_token_train": round(pj_per_token, 1),
        "energy_j_total_train": round(pj_per_token * tokens_seen / 1e12, 4),
        "wall_s": round(wall, 1), "tok_per_s": round(tokens_seen / wall, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"results_exp03_{args.tag}.json").write_text(
        json.dumps({"run": run, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    RUNS_LOG.parent.mkdir(exist_ok=True)
    with RUNS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run, ensure_ascii=False) + "\n")
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
