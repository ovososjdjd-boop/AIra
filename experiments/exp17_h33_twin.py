#!/usr/bin/env python3
"""EXP-17 «Близнец H-33» — прогон-двойник: зона-умение ± полка 2D.

ПАКЕТ СМОНТИРОВАН 07.08 по design-freeze CALC-05; ставки S1–S6 заморожены
(research/CALC05_DESIGN.md §5 + CALC11B: S6-валидатор). Запуск ТОЛЬКО по визе.

  рука A: полка c8 (запись всегда, фильтр-чтение N>=2, θ=0.90) + duty-filtered
          обучение зоны-96 (позиции, закрытые полкой, сняты с градиента);
  рука B: чистая зона-96 v3.2 (эталон, = exp13 step03 96@2400);
  контроли α: руки B256/B512 (чистые, той же химии).

Каждое звено: --max-minutes (щадящая остановка с чекпоинтом), --resume,
прогресс каждые 300 шагов, один процесс. Чекпоинт включает параметры, AdamW,
генератор, накопители duty и ПОЛКУ.

Запуск: .venv/bin/python experiments/exp17_h33_twin.py --arm A [--max-minutes 12]

ЖИВОЙ НАРЯД (смонтирован 08.08, виза ждёт «пускай»):
  --src live   поток wikitext2 (corpus_external/wikitext2/{train,valid}.txt)
               через токенизатор wiki64 (data/tokenizer_wiki64.json, OOV ~1%);
               химия v3.2 и код полки c8 неизменны (VOCAB=64).
  --sufler X   npz-чекпоинт суфлёра (рука B) → в ОЦЕНКЕ руки A включается
               маршрутизатор v3 (канон V1): covered_v3 = {N≥2, f≥0.90}
               ∪ {N≥2, f∈[0.70,0.90), top1_суфлёр == кандидат}.
               Обучение остаётся v2-фильтром (θ=0.90) — дизайн заморожен;
               v3 — режим применения/оценки. Метрики: duty_v3, acc_v3u,
               ppl_h_v3, S6' (s6p_*: суфлёр-кандидат на полосе [0.7,0.9)).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from aira.tokenizer import CharTokenizer  # noqa: E402
from aira.zone import AdamW, BusSigmaDelta, CharMLP  # noqa: E402
from exp12_precond_aa import LR_MAP, batch, load_ids  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
CTX, D_EMB, B, VOCAB = 32, 32, 128, 64
KLEN = 8
THETA, NMIN = 0.90, 2
MOD = VOCAB ** KLEN


def code8(xs: np.ndarray) -> np.ndarray:
    """код последних 8 символов окна (батч-строки x[:, -8:])."""
    w = xs[:, -KLEN:].astype(object)
    c = np.zeros(len(xs), dtype=object)
    p = 1
    for k in range(KLEN - 1, -1, -1):
        c = c + w[:, k] * p
        p *= VOCAB
    return c


class Shelf:
    """полка L0: ctx-code -> Counter(next). Запись всегда (роутер-фильтр по N>=2)."""

    def __init__(self):
        self.cnt: dict[int, collections.Counter] = {}
        self.seen_pairs = 0

    def query(self, code) -> tuple[int, float, int]:
        """(N, f_top, top_next) для кода."""
        c = self.cnt.get(code)
        if not c:
            return 0, 0.0, -1
        n = sum(c.values())
        tx, tc = c.most_common(1)[0]
        return n, tc / n, tx

    def write(self, code, nxt):
        c = self.cnt.get(code)
        if c is None:
            self.cnt[code] = collections.Counter({nxt: 1})
        else:
            c[nxt] += 1
        self.seen_pairs += 1

    def stats(self) -> dict:
        n2 = sum(1 for v in self.cnt.values() if sum(v.values()) >= NMIN)
        return {"ctx_total": len(self.cnt), "ctx_n2": n2, "pairs": self.seen_pairs}


def val_ppl(model, data, n=20, seed=7):
    rng = np.random.default_rng(seed)
    ls = []
    for _ in range(n):
        x, y = batch(data, rng)
        logits, _ = model.forward(x)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ls.append(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    return float(np.exp(sum(ls) / len(ls)))


def hybrid_eval(model, shelf: Shelf, data, n=6, seed=7, sufler=None):
    """S1-гибрид + S4: ppl гибрида, покрытие и acc полки, бины для S6.
    covered → вероятность полки (count/N, пол 1e-3); иначе модель.
    sufler (зона B) включает оценку маршрутизатора v3 (канон V1):
    covered_v3 = {N≥2, f≥0.90} ∪ {N≥2, f∈[0.70,0.90), top1_суфлёр == кандидат};
    S6' (s6p_*) — форма живой полосы [0.7,0.9) с согласованием суфлёра."""
    rng = np.random.default_rng(seed)
    ce, ce3 = [], []
    cov = hit = tot = 0
    cov3 = hit3 = 0
    s6_pass = s6_pass_ok = s6_ok = 0  # валидатор: passed, passed&ok, ok всего
    s6p_pass = s6p_pass_ok = s6p_ok = 0  # S6': полоса [0.7,0.9), pass=суфлёр==кандидат
    for _ in range(n):
        x, y = batch(data, rng)
        logits, _ = model.forward(x)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        codes = code8(x)
        top1_s = None
        if sufler is not None:
            logits_s, _ = sufler.forward(x)
            top1_s = logits_s.argmax(1)
        for j in range(len(y)):
            N, f, tx = shelf.query(int(codes[j]))
            tot += 1
            cnt = shelf.cnt.get(int(codes[j]), {})
            covered = N >= NMIN and f >= THETA
            in_band = N >= NMIN and 0.70 <= f < THETA and tx >= 0
            pass_s = bool(in_band and top1_s is not None and int(top1_s[j]) == tx)
            if covered:
                cov += 1
                hit += (tx == y[j])
            if covered or pass_s:
                cov3 += 1
                hit3 += (tx == y[j])
                ce3.append(-np.log(max(cnt.get(int(y[j]), 0) / N, 1e-3)))
            else:
                ce3.append(-np.log(np.clip(p[j, y[j]], 1e-12, 1)))
            if covered:
                ce.append(-np.log(max(cnt.get(int(y[j]), 0) / N, 1e-3)))
            else:
                ce.append(-np.log(np.clip(p[j, y[j]], 1e-12, 1)))
            # S6: кандидаты в полосе f∈[0.5,0.9), N>=2 — валидатор = модель (cand в top-3)
            if N >= NMIN and 0.5 <= f < 0.9 and tx >= 0:
                ok = (tx == y[j])
                s6_ok += ok
                rank = int((p[j] > p[j, tx]).sum())  # 0..2 ⇒ top-3
                if rank <= 2:
                    s6_pass += 1
                    s6_pass_ok += ok
            # S6' (канон V1): широкая полоса N≥2, f<0.90; pass = суфлёр==кандидат ∧ f≥0.70
            if N >= NMIN and f < THETA and tx >= 0:
                ok = (tx == y[j])
                s6p_ok += ok
                if f >= 0.70 and pass_s:
                    s6p_pass += 1
                    s6p_pass_ok += ok
    ppl = float(np.exp(np.mean(ce))) if ce else float("nan")
    out = dict(ppl_h=round(ppl, 4),
               shelf_cov=round(cov / max(tot, 1), 4), shelf_acc=round(hit / max(cov, 1), 4),
               s6_prec=round(s6_pass_ok / max(s6_pass, 1), 4),
               s6_rec=round(s6_pass_ok / max(s6_ok, 1), 4),
               s6_n_pass=s6_pass, s6_n_ok=int(s6_ok))
    if sufler is not None:
        out.update(ppl_h_v3=round(float(np.exp(np.mean(ce3))), 4) if ce3 else float("nan"),
                   duty_v3=round(cov3 / max(tot, 1), 4),
                   acc_v3u=round(hit3 / max(cov3, 1), 4),
                   v3_gain_pp=round(100 * (cov3 - cov) / max(tot, 1), 2),
                   s6p_prec=round(s6p_pass_ok / max(s6p_pass, 1), 4),
                   s6p_rec=round(s6p_pass_ok / max(s6p_ok, 1), 4),
                   s6p_n_pass=s6p_pass, s6p_n_ok=int(s6p_ok))
    return out


def load_sufler(path: Path) -> CharMLP:
    """суфлёр (веса руки B) из npz-чекпоинта; d_hid выводится из лога чекпоинта."""
    z = np.load(path, allow_pickle=True)
    d = 96
    if "log_json" in z.files:
        d = int(dict(json.loads(str(z["log_json"]))).get("d_hid", 96))
    m = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d, seed=42)
    m.load_arrays({k[2:]: z[k] for k in z.files if k.startswith("p_")})
    return m


def run(arm: str, steps: int, max_min: float, resume: bool, tag: str,
        stop_at: int, report_every: int, src: str = "m1", sufler: str = "") -> dict:
    d_hid = {"A": 96, "B": 96, "B256": 256, "B512": 512}[arm]
    use_shelf = arm == "A"
    lr0 = LR_MAP[d_hid]
    if src == "live":
        tok = CharTokenizer.load(ROOT / "data" / "tokenizer_wiki64.json")
        train_ids = load_ids(ROOT / "corpus_external" / "wikitext2" / "train.txt", tok)
        valid_ids = load_ids(ROOT / "corpus_external" / "wikitext2" / "valid.txt", tok)
    else:
        tok = CharTokenizer.load(ROOT / "data" / "tokenizer_char.json")
        train_ids = load_ids(ROOT / "data" / "corpus_train.txt", tok)
        valid_ids = load_ids(ROOT / "data" / "corpus_valid.txt", tok)
    suf = load_sufler(Path(sufler)) if sufler else None
    if suf is not None:
        print(f"   [h33] суфлёр ← {sufler} (маршрутизатор v3 в оценке)", flush=True)

    model = CharMLP(vocab=VOCAB, ctx=CTX, d_emb=D_EMB, d_hid=d_hid, seed=42)
    opt = AdamW(model.arrays(), lr=lr0)
    rng = np.random.default_rng(123)
    bus12 = BusSigmaDelta((B, d_hid), 0.05)
    bus21 = BusSigmaDelta((B, d_hid), 0.05)
    shelf = Shelf() if use_shelf else None
    log = dict(arm=arm, tag=tag, d_hid=d_hid, curve=[],
               duty_links=[], tokens_train=0, positions=0, covered=0)
    start = 0
    ckpt = RESULTS / f"ckpt_h33_{tag}.npz"
    if resume and ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        model.load_arrays({k[2:]: z[k] for k in z.files if k.startswith("p_")})
        for k in opt.m:
            opt.m[k] = z[f"m_{k}"]; opt.v[k] = z[f"v_{k}"]
        opt.t = int(z["t"]); start = int(z["step"])
        rng = np.random.default_rng(); rng.bit_generator.state = z["rng_state"].item()
        log = dict(json.loads(str(z["log_json"])))
        if use_shelf and "shelf_dump" in z:
            sd = z["shelf_dump"]
            shelf = Shelf()
            for c, x2, n in zip(sd[:, 0].tolist(), sd[:, 1].tolist(), sd[:, 2].tolist()):
                cc = shelf.cnt.setdefault(int(c), collections.Counter())
                cc[int(x2)] += int(n)
                shelf.seen_pairs += int(n)
        print(f"   [h33 resume {tag} @step {start}]", flush=True)

    t0 = time.perf_counter()
    link_pos = link_cov = 0
    partial = False
    for step in range(start + 1, steps + 1):
        x, y = batch(train_ids, rng)
        beta = max(0.1, 1.0 + (0.1 - 1.0) * min(1.0, step / steps))
        T_eff = int(round(32 + (64 - 32) * (1.0 - beta) / 0.9))
        mask = np.ones(len(y), dtype=bool)
        if use_shelf:
            codes = code8(x)
            for j in range(len(y)):
                c = int(codes[j])
                N, f, _ = shelf.query(c)
                cover = (N >= NMIN and f >= THETA)
                mask[j] = not cover         # duty-фильтр градиента
                shelf.write(c, int(y[j]))   # запись ВСЕГДА
                log["positions"] += 1
                log["covered"] += cover
                link_pos += 1; link_cov += cover
            log["tokens_train"] += int(mask.sum())
            if mask.sum() == 0:
                continue
            x2, y2 = x[mask], y[mask]
        else:
            x2, y2 = x, y
            log["tokens_train"] += len(y)
            log["positions"] += len(y)
            link_pos += len(y)
        b12 = bus12 if len(x2) == B else BusSigmaDelta((len(x2), d_hid), 0.05)
        b21 = bus21 if len(x2) == B else BusSigmaDelta((len(x2), d_hid), 0.05)
        g, st = model.pc_grads(x2, y2, beta=beta, method="bb", alpha=1.0, T=T_eff,
                               freeze=3e-3, eps=1e-2, bus12=b12, bus21=b21)
        opt.lr = lr0 * (0.3 if step > int(steps * 2 / 3) else 1.0)  # K2 step03
        opt.step(g)
        if step % report_every == 0 or step == steps or (stop_at and step == stop_at):
            wall = time.perf_counter() - t0
            rec = {"step": step, "val_ppl": round(val_ppl(model, valid_ids), 4),
                   "duty_link": round(link_cov / max(link_pos, 1), 4), "wall_s": round(wall, 1)}
            if use_shelf:
                he = hybrid_eval(model, shelf, valid_ids, sufler=suf)
                rec.update(he)
            log["curve"].append(rec)
            link_pos = link_cov = 0
            print(f"   [{tag}] {step}/{steps} ppl {rec['val_ppl']:.4f}"
                  + (f" гибр {rec.get('ppl_h')} полка {rec.get('shelf_cov')}/{rec.get('shelf_acc')}" if use_shelf else "")
                  + f" duty {rec['duty_link']:.3f} ({wall:.0f}с)", flush=True)
            sh = []
            if use_shelf:
                for c, cnt in shelf.cnt.items():
                    for x2, n in cnt.items():
                        sh.append((c, x2, n))
                sh = np.asarray(sh, dtype=np.int64) if sh else np.zeros((0, 3), dtype=np.int64)
            np.savez(ckpt, t=opt.t, step=step,
                     rng_state=np.asarray(rng.bit_generator.state, dtype=object),
                     log_json=json.dumps(log, ensure_ascii=False),
                     shelf_dump=sh,
                     **{f"p_{k}": v for k, v in model.arrays().items()},
                     **{f"m_{k}": v for k, v in opt.m.items()},
                     **{f"v_{k}": v for k, v in opt.v.items()})
            save_json(tag, log)
            # зеркало чекпоинта в git-папку для разборов (переживает снапшоты среды)
            import shutil
            (ROOT / "research/ckpts_h33").mkdir(exist_ok=True)
            shutil.copy(ckpt, ROOT / "research/ckpts_h33" / (ckpt.name + ".tmp").replace(".tmp", ""))
        if max_min and (time.perf_counter() - t0) / 60 >= max_min:
            print(f"   [{tag}] стоп звена --max-minutes {max_min} @step {step}; чекпоинт сохранён", flush=True)
            partial = True
            break
        if stop_at and step >= stop_at:
            print(f"   [{tag}] stop-at {step}", flush=True)
            partial = True
            break
    log["wall_s"] = round(time.perf_counter() - t0, 1)
    log["partial"] = partial
    log["val_ppl_full"] = round(val_ppl(model, valid_ids, n=20), 4)
    log["duty_total"] = round(log["covered"] / max(log["positions"], 1), 4)
    log["shelf"] = shelf.stats() if use_shelf else None
    log["src"] = src
    n_tok = max(log["tokens_train"], 1)
    log["ms_per_tok"] = round(1000 * log["wall_s"] / n_tok, 5)
    if use_shelf:
        log["hybrid_final"] = hybrid_eval(model, shelf, valid_ids, n=20, sufler=suf)
    save_json(tag, log)
    return log


def save_json(tag: str, rec: dict) -> None:
    fp = RESULTS / "results_exp17.json"
    out = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    out[f"h33_{tag}"] = rec
    fp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "B256", "B512"])
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="щадящая остановка звена (0 = до конца)")
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=300)
    ap.add_argument("--src", default="m1", choices=["m1", "live"],
                    help="m1 — стойка M1 (по умолчанию); live — wikitext2 + wiki64")
    ap.add_argument("--sufler", default="",
                    help="путь к npz-чекпоинту суфлёра (рука B) → маршрутизатор v3 в оценке A")
    args = ap.parse_args()
    tag = args.tag or f"{args.arm}_{args.steps}"
    r = run(args.arm, args.steps, args.max_minutes, bool(args.resume), tag,
            args.stop_at, args.report_every, src=args.src, sufler=args.sufler)
    print(f"   [готово {tag}] ppl {r['val_ppl_full']:.4f}"
          + (f" гибр {r['hybrid_final']['ppl_h']} S6 prec/rec {r['hybrid_final']['s6_prec']}/{r['hybrid_final']['s6_rec']}" if args.arm == "A" else "")
          + f" duty {r['duty_total']:.3f} мс/ток {r['ms_per_tok']} ({r['wall_s']} с, partial={r['partial']})", flush=True)


if __name__ == "__main__":
    main()
