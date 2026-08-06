#!/usr/bin/env python3
"""EXP-09 «Окно + амбразура памяти» — M3, вторая нога носителя.

КОНТЕКСТ. Дорожная карта M3: «зонно-рекуррентный блок памяти И честный гибрид
внимания: ограниченное окно W (bounded KV) + редкий событийный доступ за окно».
Первая нога (ZoneRNN, EXP-08) — в статусе ⏸: PC-кредит учится (стабильно после
лечения wd+клип+гейт), но зазор к BP на 400 шагах +102% (ворота +25% не взяты).
Здесь строим вторую ногу: окно W=32 + эпизодический HDC-стор (EXP-05/06) как
редкий доступ за окно — «кора + гиппокамп». Стор НЕ обучается градиентами:
пишется и читается онлайн по ходу потока, ограничен кольцом (анти-KV-рост).

S1  НОСИТЕЛЬ. CharMLP ctx=32, d_hid=256 (~352 тыс. параметров — ёмкость бейзлайна
    d96l3 из стойки M1). Руки: bp; pcm2 (BB+freeze 1e-3, T=32, eps=3e-3, β=1.0,
    σ-δ шина θ=(0.05,0.05)) — рецепт EXP-07 S4 без изменений, 800 шагов, B=128.
    H-26a ворота: ppl(PC) ≤ ppl(BP)+10% при работе (work_frac·T̄/T) ≤ 35%;
    смерть: > +25% или работа > 60%.

S2  АМБРАЗУРА. Потоковая оценка на первых 128k ids валкорпуса: p смесь
    p = (1−λ)·p_модель + λ·p_стор, λ ∈ {0,0.1,0.2,0.3,0.5}.
    Стор: HDC DIM=1024, ключ 1-бит (128 Б/эпизод), payload — гистограмма
    следующего символа (64 счётчика fp16 ≈ 128 Б); кольцо 16384 эпизода (FIFO);
    ключ контекста — бандл π-перестановок последних K=16 символов;
    гейт чтения cos_top1 ≥ 0.25, слияние записи при cos ≥ 0.5 (иначе новый эпизод).
    Контроль специфичности: тот же протокол, но стор кормят ЧУЖИМ текстом
    (wikitext2) — выигрыш обязан исчезнуть.
    H-26b ворота: выигрыш ppl ≥ 3% при λ* на своём потоке И ≤ 1% на чужом;
    смерть: < 1% выигрыша на своём или > 3% на чужом (стор неразборчив/бесполезен).

Запуск: .venv/bin/python experiments/exp09_windowed_memory.py [--arm s1|s2|all]
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

from aira.tokenizer import CharTokenizer  # noqa: E402
from aira.zone import AdamW, BusSigmaDelta, CharMLP  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
CTX, D_HID, B, VOCAB = 32, 256, 128, 64
STEPS = 800


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


# ---------------------------------------------------------------- S1 носитель
def s1_carrier(train_ids: np.ndarray, valid_ids: np.ndarray, steps: int = STEPS) -> dict:
    print(f"[S1] носитель ctx={CTX}: BP против PC-M2 (рецепт EXP-07 S4), {steps} шагов",
          flush=True)
    arms = {}
    cfgs = [("bp", "bp", {"freeze": 1e-3, "eps": 3e-3}),
            ("pcm2_f05b05", "pc", {"freeze": 1e-3, "eps": 3e-3}),
            ("pcm2_frugal", "pc", {"freeze": 3e-3, "eps": 1e-2})]
    for name, kind, relax_p in cfgs:
        t0 = time.perf_counter()
        model = CharMLP(vocab=VOCAB, ctx=CTX, d_hid=D_HID, seed=42)
        opt = AdamW(model.arrays(), lr=3e-3)
        bus12 = BusSigmaDelta((B, D_HID), 0.05) if kind == "pc" else None
        bus21 = BusSigmaDelta((B, D_HID), 0.05) if kind == "pc" else None
        rng = np.random.default_rng(42)
        log = {"val_ppl": [], "steps": []}
        T_acc = W_acc = n_acc = 0
        for step in range(1, steps + 1):
            x, y = batch(train_ids, rng)
            if kind == "bp":
                g, _ = model.bp_grads(x, y)
            else:
                g, st = model.pc_grads(x, y, beta=1.0, method="bb", alpha=1.0,
                                       T=32, bus12=bus12, bus21=bus21, **relax_p)
                T_acc += st["T_used"]; W_acc += st["work_frac"]; n_acc += 1
            opt.step(g)
            if step % 200 == 0 or step == steps:
                log["val_ppl"].append(round(val_ppl(model, valid_ids, n=6), 4))
                log["steps"].append(step)
                print(f"   {name} step {step}: ppl {log['val_ppl'][-1]:.4f}", flush=True)
        log["wall_s"] = round(time.perf_counter() - t0, 1)
        log["val_ppl_full"] = round(val_ppl(model, valid_ids, n=20), 4)
        if kind == "pc":
            log["T_mean"] = round(T_acc / max(n_acc, 1), 1)
            log["work_frac"] = round(W_acc / max(n_acc, 1), 3)
            dens_bits = log["T_mean"] * steps * 2 * B * D_HID * 16
            log["traffic_bits"] = bus12.traffic_bits() + bus21.traffic_bits()
            log["compress"] = round(dens_bits / max(log["traffic_bits"], 1), 2)
            print(f"   {name}: итог ppl {log['val_ppl_full']:.4f} "
                  f"T̄={log['T_mean']} работа={log['work_frac']} "
                  f"сжатие шины ×{log['compress']}", flush=True)
        else:
            log["final_model"] = model  # для S2
            print(f"   {name}: итог ppl {log['val_ppl_full']:.4f}", flush=True)
        arms[name] = log
    return arms


# ---------------------------------------------------------------- HDC-кэш
class HdcCache:
    """Онлайн-кэш «контекст → следующий символ». 1-бит ключи, fp16-пейлоады,
    кольцо FIFO (память ограничена — анти-KV-рост)."""

    def __init__(self, dim: int = 1024, ring: int = 16384, vocab: int = VOCAB,
                 k_ctx: int = 16, seed: int = 0):
        self.dim, self.ring, self.k = dim, ring, k_ctx
        rng = np.random.default_rng(seed)
        self.cb = rng.normal(0, 1, (vocab, dim)).astype(np.float32)
        self.keys = np.zeros((ring, dim), np.float32)      # знаковые ключи ±1
        self.pay = np.zeros((ring, vocab), np.float32)     # счётчики следующего
        self.n = 0
        self.ptr = 0

    def ctx_vec(self, ctx: np.ndarray) -> np.ndarray:
        """ctx (Nc,K) → presum (Nc,dim): бандл π^i(cb[c_{t-1-i}])."""
        out = np.zeros((ctx.shape[0], self.dim), np.float32)
        for i in range(self.k):
            out += np.roll(self.cb[ctx[:, self.k - 1 - i]], (i + 1) * 37, axis=1)
        return out

    def query(self, ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """→ (p_store (Nc,V), cos_top1 (Nc,)); пустой стор → нули."""
        q = self.ctx_vec(ctx)
        if self.n == 0:
            return np.zeros((len(ctx), self.pay.shape[1]), np.float32), \
                np.zeros(len(ctx), np.float32)
        keys = self.keys[:self.n]
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-30)
        sims = (qn @ keys.T) / np.sqrt(self.dim)             # cos со знаковым ключом
        top = sims.argmax(1)
        cos_top = sims[np.arange(len(ctx)), top]
        p = np.zeros((len(ctx), self.pay.shape[1]), np.float32)
        cnt = self.pay[top]
        ssum = cnt.sum(1, keepdims=True)
        p = cnt / np.clip(ssum, 1e-30, None)
        return p, cos_top

    def update(self, ctx: np.ndarray, nxt: np.ndarray, cos_top: np.ndarray,
               top_idx: np.ndarray | None = None, merge: float = 0.5) -> dict:
        q = self.ctx_vec(ctx)
        sg = np.sign(q); sg[sg == 0] = 1
        merged = int((cos_top >= merge).sum())
        # слияние с top-1 (переиспользуем query-результат: заново считать топ)
        if self.n > 0 and merged:
            keys = self.keys[:self.n]
            qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-30)
            sims = (qn @ keys.T) / np.sqrt(self.dim)
            ti = sims.argmax(1)
            for r in np.nonzero(cos_top >= merge)[0]:
                self.pay[ti[r], nxt[r]] += 1.0
        wrote = 0
        for r in np.nonzero(cos_top < merge)[0]:
            self.keys[self.ptr] = sg[r]
            self.pay[self.ptr] = 0.0
            self.pay[self.ptr, nxt[r]] = 1.0
            self.ptr = (self.ptr + 1) % self.ring
            self.n = min(self.n + 1, self.ring)
            wrote += 1
        return {"merged": merged, "wrote": wrote}


def stream_eval(model: CharMLP, ids: np.ndarray, feeder_ids: np.ndarray,
                lam_grid: list[float], n_eval: int = 131072, chunk: int = 2048,
                gate: float = 0.25) -> dict:
    """Один проход потока: p_модель и p_стор по каждой позиции (свой или чужой корм)."""
    cache = HdcCache()
    n_eval = min(n_eval, len(ids) - CTX - 1, len(feeder_ids) - CTX - 1)
    p_mod_true = np.zeros(n_eval, np.float32)
    p_sto_true = np.zeros(n_eval, np.float32)
    hits = wrote = merged = 0
    t0 = time.perf_counter()
    for st in range(0, n_eval, chunk):
        en = min(st + chunk, n_eval)
        # окна модели (ctx) и контексты ключа (K — хвост окна)
        pos = np.arange(st, en)
        x = np.stack([ids[p:p + CTX] for p in pos])
        y = ids[pos + CTX]
        logits, _ = model.forward(x)
        pr = np.exp(logits - logits.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        ctxk = x[:, -cache.k:]
        p_store, cos_top = cache.query(ctxk)
        ok = cos_top >= gate
        hits += int(ok.sum())
        rr = np.arange(len(y))
        p_mod_true[st:en] = pr[rr, y]
        # промах гейта ⇒ локальный λ=0 (фолбэк на модель, как в реальной системе)
        p_sto_true[st:en] = np.where(ok, p_store[rr, y], pr[rr, y])
        # стор кормит feeder: контексты свои (форма), но содержимое — feeder-поток
        upd = cache.update(ctxk, feeder_ids[pos + CTX], cos_top)
        wrote += upd["wrote"]; merged += upd["merged"]
        if (st // chunk) % 16 == 0:
            print(f"      …{en}/{n_eval} hit {hits} wrote {wrote} merged {merged}",
                  flush=True)
    out = {"n": n_eval, "hit_frac": round(hits / n_eval, 4),
           "wrote": wrote, "merged": merged, "occupancy": cache.n,
           "wall_s": round(time.perf_counter() - t0, 1), "lam": {}, "ppl": {}}
    for lam in lam_grid:
        if lam == 0.0:
            p = p_mod_true
        else:
            p = (1 - lam) * p_mod_true + lam * p_sto_true
        ce = -np.log(np.clip(p, 1e-12, 1)).mean()
        out["lam"][lam] = None
        out["ppl"][str(lam)] = round(float(np.exp(ce)), 4)
    return out


def s2_ambrazura(model: CharMLP, valid_ids: np.ndarray, tok: CharTokenizer,
                 n_eval: int = 131072) -> dict:
    lam_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    print("[S2] амбразура: свой поток (стор кормится тем же валпотоком, каузально)",
          flush=True)
    own = stream_eval(model, valid_ids, valid_ids, lam_grid, n_eval=n_eval)
    alien_txt = (ROOT / "corpus_external" / "wikitext2" / "train.txt") \
        .read_text(encoding="utf-8")[:600000]
    alien_ids = np.asarray(tok.encode(alien_txt), dtype=np.int64)
    print("[S2] контроль: стор кормится ЧУЖИМ текстом (wikitext2)", flush=True)
    alien = stream_eval(model, valid_ids, alien_ids, lam_grid, n_eval=n_eval)
    best_lam = min((l for l in lam_grid if l > 0),
                   key=lambda l: own["ppl"][str(l)])
    win_own = 1 - own["ppl"][str(best_lam)] / own["ppl"]["0.0"]
    win_alien = 1 - alien["ppl"][str(best_lam)] / alien["ppl"]["0.0"]
    out = {"own": own, "alien": alien, "best_lam": best_lam,
           "win_own_frac": round(float(win_own), 4),
           "win_alien_frac": round(float(win_alien), 4)}
    print(f"[S2] λ*={best_lam}: свой выигрыш {win_own:+.3%} "
          f"(ppl {own['ppl']['0.0']} → {own['ppl'][str(best_lam)]}); "
          f"чужой {win_alien:+.3%}; hit own {own['hit_frac']} vs alien {alien['hit_frac']}",
          flush=True)
    return out


# ---------------------------------------------------------------- S3 удивление
def stream_probe(model: CharMLP, ids: np.ndarray, feeder_ids: np.ndarray,
                 n_eval: int = 131072, chunk: int = 2048, gate: float = 0.25):
    """Проход с сохранением попозиционных вероятностей (свой корм)."""
    cache = HdcCache()
    n_eval = min(n_eval, len(ids) - CTX - 1, len(feeder_ids) - CTX - 1)
    p_mod = np.zeros(n_eval, np.float32)
    p_sto = np.zeros(n_eval, np.float32)
    hit = np.zeros(n_eval, bool)
    for st in range(0, n_eval, chunk):
        en = min(st + chunk, n_eval)
        pos = np.arange(st, en)
        x = np.stack([ids[p:p + CTX] for p in pos])
        y = ids[pos + CTX]
        logits, _ = model.forward(x)
        pr = np.exp(logits - logits.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        ctxk = x[:, -cache.k:]
        p_store, cos_top = cache.query(ctxk)
        ok = cos_top >= gate
        rr = np.arange(len(y))
        p_mod[st:en] = pr[rr, y]
        p_sto[st:en] = p_store[rr, y]
        hit[st:en] = ok
        cache.update(ctxk, feeder_ids[pos + CTX], cos_top)
    return p_mod, p_sto, hit


def s3_surprise(model: CharMLP, valid_ids: np.ndarray) -> dict:
    """Амбразура по удивлению: стор читаем только на позициях, где модель неуверенна.

    Метрики: (а) выигрыш внутри маски p_mod<τ при λ=0.5; (б) глобальный ppl при
    триггерном чтении; (в) профиль выигрыша по децилям уверенности модели.
    """
    print("[S3] проход-зонд (свой корм)…", flush=True)
    t0 = time.perf_counter()
    p_mod, p_sto, hit = stream_probe(model, valid_ids, valid_ids)
    p_eff = np.where(hit, p_sto, p_mod)          # фолбэк гейта как в S2
    base_ce = -np.log(np.clip(p_mod, 1e-12, 1))
    out = {"n": len(p_mod), "deciles": [], "trigger": {}}
    qs = np.quantile(p_mod, np.linspace(0, 1, 11))
    lam = 0.5
    for d in range(10):
        m = (p_mod >= qs[d]) & (p_mod <= qs[d + 1] if d == 9 else p_mod < qs[d + 1])
        if m.sum() == 0:
            continue
        mix = (1 - lam) * p_mod + lam * p_eff
        ce0 = float(base_ce[m].mean())
        ce1 = float((-np.log(np.clip(mix[m], 1e-12, 1))).mean())
        out["deciles"].append({"d": d, "q": [round(float(qs[d]), 3),
                                             round(float(qs[d + 1]), 3)],
                               "frac": round(float(m.mean()), 4),
                               "hit": round(float(hit[m].mean()), 3),
                               "ce_model": round(ce0, 4), "ce_mix": round(ce1, 4),
                               "win_frac": round((ce0 - ce1) / max(ce0, 1e-9), 4)})
    for tau in (0.3, 0.5, 0.7):
        m = p_mod < tau
        mix = np.where(m, (1 - lam) * p_mod + lam * p_eff, p_mod)
        ce_t = float((-np.log(np.clip(mix, 1e-12, 1))).mean())
        ce_b = float(base_ce.mean())
        out["trigger"][str(tau)] = {
            "coverage": round(float(m.mean()), 4),
            "hit_in_mask": round(float(hit[m].mean()) if m.any() else 0.0, 3),
            "ppl": round(float(np.exp(ce_t)), 4),
            "win_global_frac": round((ce_b - ce_t) / ce_b, 4),
            "win_in_mask_frac": round(
                float((base_ce[m].mean()
                       - (-np.log(np.clip(
                           (1 - lam) * p_mod[m] + lam * p_eff[m], 1e-12, 1))).mean())
                      / max(base_ce[m].mean(), 1e-9)), 4) if m.any() else 0.0}
    out.update({"wall_s": round(time.perf_counter() - t0, 1),
                "ppl_base_stream": round(float(np.exp(base_ce.mean())), 4)})
    for r in out["deciles"]:
        print(f"   d{r['d']} p∈{r['q']} доля {r['frac']:6.3f} hit {r['hit']:5.3f} "
              f"ce {r['ce_model']:.3f}→{r['ce_mix']:.3f} ({r['win_frac']:+.1%})", flush=True)
    for tau, r in out["trigger"].items():
        print(f"   τ={tau}: покрытие {r['coverage']:.3f}, ppl {r['ppl']} "
              f"(глоб {r['win_global_frac']:+.2%}, в маске {r['win_in_mask_frac']:+.1%})",
              flush=True)
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["s1", "s2", "s3", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
    train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
    valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    out: dict = {"config": {"ctx": CTX, "d_hid": D_HID, "B": B, "steps": STEPS,
                            "hdc": {"dim": 1024, "ring": 16384, "k": 16}}}
    model = None
    if args.arm in ("s1", "all"):
        arms = s1_carrier(train_ids, valid_ids, steps=60 if args.quick else STEPS)
        model = arms["bp"].pop("final_model")
        out["s1"] = arms
    if args.arm in ("s2", "all"):
        if model is None:  # s2 без s1 — переобучим BP-носитель с нуля
            arms = s1_carrier(train_ids, valid_ids, steps=60 if args.quick else STEPS)
            model = arms["bp"].pop("final_model")
            out["s1"] = arms
        out["s2"] = s2_ambrazura(model, valid_ids, tok,
                                 n_eval=8192 if args.quick else 131072)
    if args.arm in ("s3", "all"):
        if model is None:
            arms = s1_carrier(train_ids, valid_ids, steps=60 if args.quick else STEPS)
            model = arms["bp"].pop("final_model")
            out.setdefault("s1", arms)
        out["s3"] = s3_surprise(model, valid_ids)
    RESULTS.mkdir(exist_ok=True, parents=True)
    fp = RESULTS / "results_exp09.json"
    if fp.exists():
        prev = json.loads(fp.read_text(encoding="utf-8"))
        prev.update(out)
        out = prev
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", fp, flush=True)


if __name__ == "__main__":
    main()
