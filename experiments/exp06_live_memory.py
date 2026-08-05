#!/usr/bin/env python3
"""EXP-06 «Живая память» — QA-батарея на живом тексте (обход C5, развитие EXP-05).

EXP-05 доказал ёмкость/гейт на синтетических фактах (случайные, некоррелированные).
Здесь — живые тексты (предложения коррелируют): «Кавказский пленник» Толстого как
«прочитанная история», «Война и мир» как фон-интерферент.

Стор (1-бит, 256 Б/эпизод): код предложения = sign(idf^0.5-взвешенный бандлинг
униграмм + биграмм prev⊗cur) над стеммами (грубый стеммер: первые 6 букв для слов
длиной ≥7). Запрос — НЕ квантованный presum-вектор (float32, транзиент, 8 КБ SRAM);
скоринг = косинус presum×1-бит-коды. Диагностика quick-прогона (зафиксирована в
EXP06_REPORT): sign-на-sign при idf-весах инвертировал ранжирование в пользу
коротких посторонних предложений (разложено численно: presum-косинус верен,
sign-dot инвертирован); веса — только на этапе кодирования, ХРАНЕНИЕ остаётся
1-бит. Чтение — DAM top-1, гейт фамильярности — max-сходство до достройки (H-22).

Сцены:
  S1 cloze: вопрос = предложение без редкого слова (частота стемма 1–5, длина ≥5);
     hit@1 «содержит ответ» и строгий (тот самый эпизод) — при шкалах фона
     [0, 1500, 6000, 16000] предложений ВиМ.
  S2 задержка: точность vs «сколько записано после эпизода» (разрыв бакетов ≤0.15).
  S3 гейт: свои vs чужие (ВиМ, не записано) vs семантические гибриды;
     θ* принимает ≥90% своих; критерий — отсев чужих ≥ 0.95; отчёт по AUC.
  S4 ручная батарея (12 вопросов по сюжету, regex-паттерны ответов с границами
     слов): hit если ответ в top-1/top-3.
Энергия — прокси EXP-05: бит-операция = 0.01 пДж (чтение 1 бита + add — честная
оговорка в отчёте), HBM 15 пДж/Б, int8 MAC 0.5 пДж; KV-бейзлайн: per-fact KV int8
d=768 (K+V) + скан с чтением.

Запуск: .venv/bin/python experiments/exp06_live_memory.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from aira import energy as en  # noqa: E402

DIM = 2048
SEED = 42
STOP_TOP = 40                 # стоп-лист = top-40 стемм «Кавказа» + вопросительный мусор
BG_SCALES = [0, 1500, 6000, 16000]
N_CLOZE = 600
N_ALIEN = 500
N_HYBRID = 300
IDF_ALPHA = 0.5               # веса = idf^alpha (0.5 — по свипу, см. отчёт)
E_BIT_MAC_PJ = 0.01
KV_D_MODEL = 768              # per-fact KV int8 (K+V), как в EXP-05
DELAY_BUCKETS = [(0, 700), (700, 2200), (2200, 6700), (6700, 10**9)]
DELAY_LABELS = ["0–700", "700–2200", "2200–6700", "6700+"]
Q_JUNK = ["кто", "что", "как", "какой", "какого", "какие", "сколько", "когда",
          "почему", "зачем", "куда", "откуда", "кого", "кому", "кем", "чем", "чей",
          "чьи", "чья", "чьё", "который", "которая", "которые", "ли", "это",
          "этот", "свой", "свои", "что-то", "кто-то"]

KAUKAZ = ROOT / "corpus_external" / "russian_lit" / "kavkazskiy_plennik.txt"
VIM = ROOT / "corpus_external" / "russian_lit" / "voina_i_mir.txt"

# ---- ручная батарея S4: (вопрос, regex-паттерн ответа) — основана на тексте ---
HAND = [
    ("Как звали офицера, который служил на Кавказе?", r"\bжилин[а-я]*"),
    ("Какой офицер подъезжает к Жилину с ружьем?", r"\bкостылин[а-я]*"),
    ("Кто подбежал, схватил куклу и убежал?", r"\bдин(а|ы|у|е|ой)\b"),
    ("Какие глаза виднелись над ямой, как у кошки, когда спустили шест?", r"\bдин(а|ы|у|е|ой)\b"),
    ("Что Жилин поставил солдатам на прощанье?", r"\bводк[а-я]*"),
    ("Каким хлебом плохо кормил хозяин: что давали пленникам?", r"\bлепеш[а-я]*"),
    ("Что Дина стала носить Жилину каждый день, крадучись?", r"\bмолок[а-я]*"),
    ("Кому красный татарин отдал Жилина?", r"\bабдул[а-я]*"),
    ("Кто взял Жилина в плен первым, красный или черный татарин?", r"\bмугамед[а-я]*"),
    ("Сколько монет выкупа просили татары поначалу?", r"\bтри тысячи"),
    ("Кто заробел и сомневался, бежать ли из плена?", r"\bкостылин[а-я]*"),
    ("Кого услыхали наши на крик «братцы, выручай»?", r"\bказак[а-я]*"),
]


# ---------------------------------------------------------------------------
def norm_tokens(s: str) -> list[str]:
    return re.findall(r"[а-яё]+", s.lower().replace("ё", "е"))


def stem(t: str) -> str:
    """Грубый стеммер: отсекает изменяемые окончания (первые 6 букв)."""
    return t[:6] if len(t) >= 7 else t


def feat_tokens(s: str) -> list[str]:
    return [stem(t) for t in norm_tokens(s)]


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[.!?…]+", text)]
    return [p for p in parts if 3 <= len(p.split()) <= 40 and len(norm_tokens(p)) >= 3]


class WordCodes:
    """Детерминированные коды стемм: word → {-1,+1}^DIM (int8), crc32^seed."""

    def __init__(self, dim: int = DIM, seed: int = SEED):
        self.dim, self.seed, self.cache = dim, seed, {}

    def __call__(self, w: str) -> np.ndarray:
        c = self.cache.get(w)
        if c is None:
            rng = np.random.default_rng((zlib.crc32(w.encode()) ^ self.seed) & 0xFFFFFFFF)
            c = rng.choice(np.int8([-1, 1]), size=self.dim)
            self.cache[w] = c
        return c


def presum_vec(toks: list[str], wc: WordCodes, stop: set[str],
               idf: dict[str, float], alpha: float = IDF_ALPHA) -> np.ndarray:
    """Float-presum: Σ idf^α·униграммы + Σ √(wᵢwⱼ)·(cᵢ⊗cⱼ) по не-стоп стеммам."""
    ts = [t for t in toks if t not in stop] or toks or ["пусто"]
    w = [idf.get(t, 1.0) ** alpha for t in ts]
    s = np.zeros(wc.dim, dtype=np.float32)
    for t, wi in zip(ts, w):
        s += wi * wc(t)
    for i in range(len(ts) - 1):
        s += (w[i] * w[i + 1]) ** 0.5 * wc(ts[i]) * wc(ts[i + 1])
    return s


def encode_store(toks: list[str], wc: WordCodes, stop: set[str],
                 idf: dict[str, float]) -> np.ndarray:
    """1-бит код для ХРАНЕНИЯ: sign(presum), нули → +1."""
    return np.where(presum_vec(toks, wc, stop, idf) >= 0, 1, -1).astype(np.int8)


def query_vec(toks: list[str], wc: WordCodes, stop: set[str],
              idf: dict[str, float]) -> np.ndarray:
    """L2-нормированный float-presum запроса (не квантуется)."""
    q = presum_vec(toks, wc, stop, idf)
    n = float(np.linalg.norm(q))
    return q / (n if n > 0 else 1.0)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)

    bg_scales = [0, 1500] if args.quick else BG_SCALES
    n_cloze, n_alien, n_hybrid = (200, 200, 100) if args.quick else (N_CLOZE, N_ALIEN, N_HYBRID)

    print(f"[data] читаю тексты…")
    kau_sents = split_sentences(KAUKAZ.read_text(encoding="utf-8"))
    vim_sents_all = split_sentences(VIM.read_text(encoding="utf-8"))
    print(f"[data] Кавказ: {len(kau_sents)} предложений; ВиМ: {len(vim_sents_all)}")

    kau_feats = [feat_tokens(s) for s in kau_sents]
    kau_tokens_raw = [norm_tokens(s) for s in kau_sents]
    freq = Counter(t for f in kau_feats for t in set(f))
    stop = {w for w, _ in freq.most_common(STOP_TOP)} | {stem(w) for w in Q_JUNK}
    print(f"[data] стоп-лист: {len(stop)} слов (top-{STOP_TOP} по Кавказу + вопрос-мусор)")

    # ВиМ: первые 75% — фон, последние 25% — чужие (дизъюнктно)
    cut = int(0.75 * len(vim_sents_all))
    bg_src, alien_pool = vim_sents_all[:cut], vim_sents_all[cut:]
    n_bg_max = min(max(bg_scales), len(bg_src))
    bg_pool = [bg_src[i] for i in rng.choice(len(bg_src), size=n_bg_max, replace=False)]
    aliens = [alien_pool[i] for i in rng.choice(len(alien_pool),
                                                size=min(n_alien, len(alien_pool)), replace=False)]
    print(f"[data] фон ВиМ: {len(bg_pool)}; чужие для гейта: {len(aliens)}")

    bg_feats = [feat_tokens(s) for s in bg_pool]

    # idf по стеммам ЗАПИСАННОГО потока (Кавказ + макс. фон; честно оговорено в отчёте)
    docs = [set(f) for f in kau_feats] + [set(f) for f in bg_feats]
    n_docs = len(docs)
    df = Counter()
    for d in docs:
        df.update(d)
    idf = {w: float(np.log(1.0 + n_docs / (1.0 + c))) for w, c in df.items()}
    print(f"[data] idf построен: {n_docs} документов, словарь {len(idf)} стемм, "
          f"веса idf^{IDF_ALPHA}")

    # --- S1 cloze: цели — редкие стеммы Кавказа -------------------------------
    docfreq_kau = Counter(t for f in kau_feats for t in set(f))
    cloze = []
    for i, toks in enumerate(kau_feats):
        for w in dict.fromkeys(toks):
            if len(w) >= 5 and 1 <= docfreq_kau[w] <= 5:
                q = toks.copy(); q.remove(w)
                if len([t for t in q if t not in stop]) >= 3:
                    cloze.append({"sent": i, "target": w, "q_tokens": q})
    rng.shuffle(cloze)
    cloze = cloze[:n_cloze]
    print(f"[S1] cloze-вопросов: {len(cloze)}")

    wc = WordCodes()
    print(f"[enc] кодирую Кавказ…")
    kau_codes = np.stack([encode_store(t, wc, stop, idf) for t in kau_feats])
    print(f"[enc] кодирую фон…")
    bg_codes = np.stack([encode_store(f, wc, stop, idf) for f in bg_feats]) if bg_feats \
        else np.zeros((0, DIM), np.int8)

    def make_alien_q(f: list[str]) -> list[str]:
        toks = list(f)
        cand = [k for k, t in enumerate(toks) if t not in stop and len(t) >= 4]
        if len(cand) >= 3:
            toks.pop(int(rng.choice(cand)))
        return toks

    print(f"[enc] кодирую вопросы…")
    alien_feats = [feat_tokens(s) for s in aliens]
    q_cloze = np.stack([query_vec(c["q_tokens"], wc, stop, idf) for c in cloze])
    q_alien = np.stack([query_vec(make_alien_q(f), wc, stop, idf) for f in alien_feats])

    # гибриды: предложение Кавказа с одним подменённым контентным словом
    content_vocab = [w for w, f in docfreq_kau.items() if f >= 3 and w not in stop and len(w) >= 4]
    hybrid_q = []
    for _ in range(n_hybrid):
        toks = list(kau_feats[int(rng.integers(len(kau_feats)))])
        cand = [k for k, t in enumerate(toks) if t not in stop and len(t) >= 4]
        if not cand:
            continue
        toks[int(rng.choice(cand))] = content_vocab[int(rng.integers(len(content_vocab)))]
        hybrid_q.append(toks)
    q_hybrid = np.stack([query_vec(t, wc, stop, idf) for t in hybrid_q])
    q_hand = np.stack([query_vec(feat_tokens(q), wc, stop, idf) for q, _ in HAND])

    # --- прогоны по шкалам ---------------------------------------------------
    kau_feat_sets = [set(f) for f in kau_feats]
    bg_feat_sets = [set(f) for f in bg_feats]
    results = {"config": {"dim": DIM, "seed": SEED, "stop_top": STOP_TOP,
                          "idf_alpha": IDF_ALPHA, "bg_scales": bg_scales,
                          "n_kau": len(kau_sents), "n_cloze": len(cloze),
                          "n_alien": len(aliens), "n_hybrid": len(hybrid_q),
                          "n_docs_idf": n_docs},
               "scales": []}

    for n_bg in bg_scales:
        codes = np.vstack([kau_codes, bg_codes[:n_bg]]) if n_bg else kau_codes
        c32n = codes.astype(np.float32) / np.sqrt(DIM)      # ‖code‖=√D → косинус
        n_total = codes.shape[0]
        ts = time.perf_counter()

        def feat_set(r: int) -> set[str]:
            return kau_feat_sets[r] if r < len(kau_sents) else bg_feat_sets[r - len(kau_sents)]

        def sent_text(r: int) -> str:
            return kau_sents[r] if r < len(kau_sents) else bg_pool[r - len(kau_sents)]

        # S1: скан всех cloze-вопросов
        sims = q_cloze @ c32n.T                               # (Q, n) косинусы
        top1 = np.argmax(sims, axis=1)
        maxsim_own = sims[np.arange(len(cloze)), top1]
        hits_contains = np.array([c["target"] in feat_set(r) for c, r in zip(cloze, top1)])
        hits_strict = np.array([c["sent"] == r for c, r in zip(cloze, top1)])
        acc_contains, acc_strict = float(hits_contains.mean()), float(hits_strict.mean())

        # S2: задержка/забывание — на макс. шкале зонды по ВСЕМУ потоку (иначе все
        # кавказские эпизоды старые и бакеты пусты): вопрос = предложение потока
        # без редкого стемма; задержка = сколько записано после эпизода-цели.
        delay_acc = None
        if n_bg == max(bg_scales):
            stream_feats = kau_feats + bg_feats[:n_bg]
            probe_idx = rng.choice(n_total, size=min(400, n_total), replace=False)
            probe_q, probe_ans, probe_delay = [], [], []
            for pos in probe_idx:
                toks = stream_feats[pos]
                # редкий стемм потока (df ≤ 20) — однозначная цель
                cand = [w for w in dict.fromkeys(toks)
                        if w not in stop and len(w) >= 5 and df.get(w, 10**9) <= 20]
                if not cand:
                    continue
                w = cand[int(rng.integers(len(cand)))]
                q = toks.copy(); q.remove(w)
                if len([t for t in q if t not in stop]) < 3:
                    continue
                probe_q.append(query_vec(q, wc, stop, idf))
                probe_ans.append(w)
                probe_delay.append(n_total - 1 - pos)
            if probe_q:
                pq = np.stack(probe_q)
                ptop = np.argmax(pq @ c32n.T, axis=1)
                phit = np.array([probe_ans[k] in feat_set(r) for k, r in enumerate(ptop)])
                pdel = np.array(probe_delay)
                delay_acc = {}
                for (lo, hi), lab in zip(DELAY_BUCKETS, DELAY_LABELS):
                    m = (pdel >= lo) & (pdel < hi)
                    if m.sum():
                        delay_acc[lab] = {"n": int(m.sum()),
                                          "hit_contains": float(phit[m].mean())}

        # S3: гейт (полностью на максимальной шкале)
        gate = None
        if n_bg == max(bg_scales):
            maxsim_alien = np.max(q_alien @ c32n.T, axis=1)
            maxsim_hybrid = np.max(q_hybrid @ c32n.T, axis=1)
            theta = float(np.quantile(maxsim_own, 0.10))          # принимаем ≥90% своих
            ranks = np.argsort(np.argsort(np.concatenate([maxsim_own, maxsim_alien])))
            n1, n2 = len(maxsim_own), len(maxsim_alien)
            auc = float((np.sum(ranks[:n1]) - n1 * (n1 - 1) / 2) / (n1 * n2))
            gate = {"theta": theta, "accept_own": float(np.mean(maxsim_own >= theta)),
                    "reject_alien": float(np.mean(maxsim_alien < theta)),
                    "reject_hybrid": float(np.mean(maxsim_hybrid < theta)), "auc": auc,
                    "sim_own_mean": float(np.mean(maxsim_own)),
                    "sim_alien_mean": float(np.mean(maxsim_alien))}

        # S4: ручная батарея (regex с границами слов)
        sims_h = q_hand @ c32n.T
        top_h = np.argsort(-sims_h, axis=1)[:, :3]
        hand_res = []
        for (q, pat), tops in zip(HAND, top_h):
            low = [re.sub(r"\s+", " ", sent_text(r).lower()) for r in tops]
            hit = [re.search(pat, t) is not None for t in low]
            hand_res.append({"q": q, "pat": pat, "top1_hit": hit[0], "top3_hit": any(hit),
                             "top1_sent": sent_text(tops[0])[:110]})
        hand_top1 = float(np.mean([h["top1_hit"] for h in hand_res]))
        hand_top3 = float(np.mean([h["top3_hit"] for h in hand_res]))

        # энергия (прокси, по формулам EXP-05)
        n_tok = sum(len(t) for t in kau_tokens_raw) + sum(len(norm_tokens(s)) for s in bg_pool[:n_bg])
        write_hdc_b = 256 * n_total                                  # 2048 бит = 256 Б
        write_kv_b = n_tok * 2 * KV_D_MODEL                          # int8 K+V
        read_hdc_pj = n_total * DIM * E_BIT_MAC_PJ
        read_kv_pj = n_tok * 2 * KV_D_MODEL * (en.E_MAC_INT8_PJ + 2 * en.E_HBM_PJ)
        energy = {"n_tokens_total": int(n_tok),
                  "write_hdc_bytes": int(write_hdc_b), "write_kv_bytes": int(write_kv_b),
                  "write_ratio": float(write_kv_b / write_hdc_b),
                  "read_hdc_pj": float(read_hdc_pj), "read_kv_pj": float(read_kv_pj),
                  "read_ratio": float(read_kv_pj / read_hdc_pj)}

        rec = {"n_bg": n_bg, "n_total": int(n_total),
               "S1": {"acc_contains": acc_contains, "acc_strict": acc_strict},
               "S2_delay": delay_acc, "S3_gate": gate,
               "S4": {"hand_top1": hand_top1, "hand_top3": hand_top3, "items": hand_res},
               "energy": energy,
               "wall_s": round(time.perf_counter() - ts, 2)}
        results["scales"].append(rec)
        print(f"[scale] фон {n_bg:>6} | всего {n_total:>6} | "
              f"cloze содерж {acc_contains:.3f} строгая {acc_strict:.3f} | "
              f"ручная top1 {hand_top1:.2f} top3 {hand_top3:.2f} | "
              f"read ×{energy['read_ratio']:.0f} | {rec['wall_s']}с")

    # --- вердикты по воротам -------------------------------------------------
    base = results["scales"][0]
    last = results["scales"][-1]
    gaps = [abs(last["S2_delay"][lab]["hit_contains"] - last["S2_delay"][DELAY_LABELS[0]]["hit_contains"])
            for lab in DELAY_LABELS
            if last["S2_delay"] and lab in last["S2_delay"] and DELAY_LABELS[0] in last["S2_delay"]]
    verdict = {
        "H-23a cloze@чистый (≥0.90 / ✗<0.70)": base["S1"]["acc_contains"],
        "H-23b cloze@фон×24 (≥0.80 / ✗<0.50)": last["S1"]["acc_contains"],
        "H-23c гейт: отсев чужих (≥0.95 / ✗<0.80)": last["S3_gate"]["reject_alien"] if last["S3_gate"] else None,
        "H-23d ручная top3 (≥8/12)": last["S4"]["hand_top3"],
        "H-23e забывание: разрыв бакетов (≤0.15)": max(gaps) if gaps else None,
    }
    results["verdicts"] = verdict
    print("\n=== ВЕРДИКТЫ ===")
    for k, v in verdict.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    out = ROOT / "experiments" / "results" / "results_exp06.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[save] {out}")

    # --- картинка -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    xs = [r["n_total"] for r in results["scales"]]
    ax = axes[0][0]
    ax.plot(xs, [r["S1"]["acc_contains"] for r in results["scales"]], "o-", label="содержит ответ")
    ax.plot(xs, [r["S1"]["acc_strict"] for r in results["scales"]], "s--", label="строгий top-1")
    ax.axhline(0.9, color="gray", lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_title("S1 cloze hit@1 vs масштаб стора"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0][1]
    labs = [lab for lab in DELAY_LABELS if last["S2_delay"] and lab in last["S2_delay"]]
    ax.bar(labs, [last["S2_delay"][lab]["hit_contains"] for lab in labs] if labs else [])
    ax.set_ylim(0, 1.02); ax.set_title("S2 hit vs «записано после» (макс. шкала)")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    if last["S3_gate"]:
        g = last["S3_gate"]
        ax.bar(["свои\n(принято)", "чужие\n(отсеяно)", "гибриды\n(отсеяно)"],
               [g["accept_own"], g["reject_alien"], g["reject_hybrid"]])
        ax.set_ylim(0, 1.02)
        ax.set_title(f"S3 гейт фамильярности (AUC={g['auc']:.3f})")
        ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(xs, [r["energy"]["read_ratio"] for r in results["scales"]], "o-", color="crimson")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("энергия чтения: KV-скан / HDC-скан (×)")
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("EXP-06 «Живая память» — Кавказ + фон ВиМ")
    fig.tight_layout()
    p = ROOT / "experiments" / "results" / "exp06_live_memory.png"
    fig.savefig(p, dpi=130)
    print(f"[save] {p}")
    print(f"[done] за {time.perf_counter() - t0:.1f} с")


if __name__ == "__main__":
    main()
