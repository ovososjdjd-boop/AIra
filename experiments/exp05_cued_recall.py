#!/usr/bin/env python3
"""EXP-05 «Вспоминание по крючку» — первая проба ответа на C5 (BRAIN_CUES §4, H-21, H-22).

Эпизодический «мир фактов»: каждый факт — уникальный эпизод (уникальный тег-имя, аналог
DG-разделения), слоты существо/место/объект — плотный словарь 8 значений (слоты пересекаются
между эпизодами, теги — нет). Стор — автоассоциатор с одно-событийной записью.

Чтения:
  - DAM top-1: argmax сходства (непрерывный совр. Хопфилд-аналог). Оценка — по индексу
    возвращённого эпизода (точно и дёшево);
  - Hopfield-sign: 4 такта r←sign(C²r). Может сходиться к СМЕСИ эпизодов — это и мерим:
    декодирование слотов прямо из relaxed-кода (ищет интерференцию честно).
Крючки:
  - эпизодный (имя, уникальный) — проверка ёмкости (строгая accuracy@1);
  - содержательный (существо+объект, неоднозначный при росте n) — cue-match и интерференция.
Гейт фамильярности (H-22): max-сходство до достройки; ниже θ — «не знаю», не платим.

Энергия — прокси (явные цены): 1-бит MAC = 0.01 пДж, байт HBM = 15 пДж; референс KV-скана:
(15 токенов/факт)·n·768 · (2 MAC·int8 0.5 пДж + 2 Б·15 пДж).

Запуск: .venv/bin/python experiments/exp05_cued_recall.py [--quick]
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

from aira.hdc import compose_cue, encode_fact, make_codebook, scan_sims  # noqa: E402
from aira import energy as en  # noqa: E402

DIM = 2048
ROLES = ["имя", "существо", "место", "объект"]
FILLERS = {
    "существо": ["котёнок", "лисёнок", "ёжик", "зайчонок", "медвежонок", "сороконога",
                 "дракончик", "воробей"],
    "место": ["лес", "берег", "дуб", "поляна", "холм", "пещера", "деревня", "болото"],
    "объект": ["фонарик", "шар", "карта", "колокольчик", "семечко", "шапка", "ожерелье",
               "пищалка"],
}
MAX_FACTS = 12800
POOL_EXTRA = 512            # эпизоды-невидимки для гейта (никогда не пишутся)
E_BIT_MAC_PJ = 0.01
KV_TOKENS_PER_FACT = 15
KV_D_MODEL = 768


def name(i: int) -> str:
    return f"существо_{i:05d}"


def make_fact(i: int, rng: np.random.Generator) -> dict[str, str]:
    return {"имя": name(i),
            "существо": rng.choice(FILLERS["существо"]),
            "место": rng.choice(FILLERS["место"]),
            "объект": rng.choice(FILLERS["объект"])}


def relax_hopfield(c32: np.ndarray, q: np.ndarray, iters: int = 4) -> np.ndarray:
    r = q.astype(np.float32)
    for _ in range(iters):
        a = c32 @ r
        r = np.sign(c32.T @ a)
        r[r == 0] = 1
    return r.astype(np.int8)


def pj_scan(n: int) -> float:
    return n * DIM * E_BIT_MAC_PJ


def pj_hopfield_read(n: int) -> float:
    return 4 * 2 * pj_scan(n) + 34 * DIM * E_BIT_MAC_PJ + DIM * en.E_HBM_PJ


def pj_dam_read(n: int) -> float:
    return pj_scan(n) + 34 * DIM * E_BIT_MAC_PJ + DIM * en.E_HBM_PJ


def pj_kv_scan(n_facts: int) -> float:
    toks = n_facts * KV_TOKENS_PER_FACT
    return 2 * toks * KV_D_MODEL * (en.E_MAC_INT8_PJ + en.E_HBM_PJ)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seed = 42
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    labels = ([name(i) for i in range(MAX_FACTS + POOL_EXTRA)] +
              [f for v in FILLERS.values() for f in v] + ["role:" + r for r in ROLES])
    cb = make_codebook(labels, dim=DIM, seed=seed)
    facts = [make_fact(i, rng) for i in range(MAX_FACTS + POOL_EXTRA)]
    print(f"[world] вселенная эпизодов: {len(facts)}; роли: {ROLES}")

    scales = [100, 1000, 5120] if args.quick else [100, 500, 1000, 2500, 5120, 12800]
    n_queries = 250 if args.quick else 500
    n_novel = 128 if args.quick else 256
    n_hop_check = 100 if args.quick else 200

    results = []
    for n in scales:
        written = facts[:n]
        novel = facts[MAX_FACTS : MAX_FACTS + n_novel]
        codes = np.stack([encode_fact(f, cb) for f in written])
        c32 = codes.astype(np.float32)
        # матрица декодирования слота-имени (для relaxed-кодов): n×D связанных кодов
        name_bound32 = np.stack([cb["role:имя"] * cb[name(i)]
                                 for i in range(n)]).astype(np.float32)
        small_bound = {r: np.stack([cb["role:" + r] * cb[f]
                                    for f in FILLERS[r]]).astype(np.float32)
                       for r in ("существо", "место", "объект")}

        for cue_roles in (["имя"], ["существо", "объект"]):
            qids = rng.choice(n, size=min(n_queries, n), replace=False)
            dam_strict = dam_cuematch = 0
            g_known = []
            hop_codes = []
            for qi in qids:
                q = compose_cue({r: written[qi][r] for r in cue_roles}, cb)
                s = scan_sims(c32, q)
                j = int(np.argmax(s))
                g_known.append(float(s[j]))
                f = written[qi]
                dam_strict += int(written[j] == f)
                dam_cuematch += int(all(written[j][r] == f[r] for r in cue_roles))
                hop_codes.append(relax_hopfield(c32, q))

            dam_strict /= len(qids)
            dam_cuematch /= len(qids)

            # строгая проверка relaxed-кодов (подвыборка; декод всех 4 слотов)
            sub = rng.choice(len(qids), size=min(n_hop_check, len(qids)), replace=False)
            hop_strict = hop_cuematch = 0
            for k in sub:
                f = written[qids[k]]
                code = hop_codes[k].astype(np.float32)
                dec = {"имя": name(int(np.argmax(name_bound32 @ code)))}
                for r in ("существо", "место", "объект"):
                    dec[r] = FILLERS[r][int(np.argmax(small_bound[r] @ code))]
                hop_strict += int(all(dec[r] == f[r] for r in ROLES))
                hop_cuematch += int(all(dec[r] == f[r] for r in cue_roles))
            hop_strict /= len(sub)
            hop_cuematch /= len(sub)

            # гейт: известные крючки vs незнакомые (тот же тип крючка)
            g_novel = [float(np.max(scan_sims(
                c32, compose_cue({r: f[r] for r in cue_roles}, cb)))) for f in novel]
            g_known = np.asarray(g_known)
            theta = float(np.quantile(g_known, 0.05))
            known_accept = float(np.mean(g_known >= theta))
            novel_reject = float(np.mean(np.asarray(g_novel) < theta))

            e_full = pj_dam_read(n)
            e_gate_known = pj_scan(n) + known_accept * (e_full - pj_scan(n))
            e_gate_novel = pj_scan(n) + (1 - novel_reject) * (e_full - pj_scan(n))

            res = {
                "n_facts": n, "cue": "+".join(cue_roles),
                "acc_dam_strict": round(dam_strict, 4),
                "acc_dam_cuematch": round(dam_cuematch, 4),
                "acc_hop_strict": round(hop_strict, 4),
                "acc_hop_cuematch": round(hop_cuematch, 4),
                "gate_known_accept": round(known_accept, 4),
                "gate_novel_reject": round(novel_reject, 4),
                "gate_theta": round(theta, 4),
                "e_read_known_nogate_pj": round(e_full, 1),
                "e_read_known_gate_pj": round(e_gate_known, 1),
                "e_read_novel_gate_pj": round(e_gate_novel, 1),
                "e_kv_scan_pj": round(pj_kv_scan(n), 1),
            }
            results.append(res)
            print(f"[n={n:>5} | {'+'.join(cue_roles):18s}] "
                  f"DAM {dam_strict:.3f}/{dam_cuematch:.3f} "
                  f"Hopf {hop_strict:.3f}/{hop_cuematch:.3f} | "
                  f"гейт: пр.{known_accept:.2f}/отс.{novel_reject:.2f} | "
                  f"эн. извест. {e_gate_known:8.0f} vs без гейта {e_full:8.0f} пДж "
                  f"(KV {pj_kv_scan(n)/1e6:.1f} мкДж)")

    out = {"seed": seed, "dim": DIM,
           "prices": {"e_bit_mac_pj": E_BIT_MAC_PJ,
                      "kv_tokens_per_fact": KV_TOKENS_PER_FACT, "kv_d_model": KV_D_MODEL,
                      "e_mac_int8_pj": en.E_MAC_INT8_PJ, "e_hbm_pj": en.E_HBM_PJ},
           "acc_cols": "strict = все 4 слота; cuematch = слоты крючка",
           "rows": results, "wall_s": round(time.perf_counter() - t0, 1)}
    res_dir = ROOT / "experiments" / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "results_exp05.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] results_exp05.json за", round(time.perf_counter() - t0, 1), "с")

    # --- графики ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for cue, mark, lab in (("имя", "o", "эпизодный крючок (имя)"),
                           ("существо+объект", "s", "содержат. крючок (строгая)")):
        xs = [r["n_facts"] for r in results if r["cue"] == cue]
        a.plot(xs, [r["acc_dam_strict"] for r in results if r["cue"] == cue],
               mark + "-", label="DAM " + lab)
        b.plot(xs, [r["e_read_known_gate_pj"] for r in results if r["cue"] == cue],
               mark + "-", label="стор+гейт " + lab)
    xs_h = [r["n_facts"] for r in results if r["cue"] == "имя"]
    a.plot(xs_h, [r["acc_hop_strict"] for r in results if r["cue"] == "имя"],
           "x--", label="Hopfield-sign (контраст)")
    b.plot(xs_h, [r["e_kv_scan_pj"] for r in results if r["cue"] == "имя"],
           "k--", label="KV-скан (референс)")
    a.set_xscale("log")
    a.set_xlabel("фактов в сторе")
    a.set_ylabel("accuracy@1 (строгая)")
    a.set_title("Вспоминание по крючку: ёмкость")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("фактов в сторе")
    b.set_ylabel("пДж / запрос (прокси)")
    b.set_title("Цена одного вспоминания")
    b.grid(alpha=0.3, which="both")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res_dir / "exp05_cued_recall.png", dpi=130)
    print("[ok] exp05_cued_recall.png")


if __name__ == "__main__":
    main()
