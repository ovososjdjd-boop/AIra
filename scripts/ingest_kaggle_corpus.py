#!/usr/bin/env python3
# Приёмник Kaggle-корпуса + замер замороженных прогнозов K1–K2 (+K3 по флажку).
# Прогнозы заморожены 07.08 (research/CALC07_SUPER.md §E, H-34) — вердикты по ним.
# Прицельный счёт (память < 1 ГБ), обучения нет.
import json, sys, time, collections, zipfile, argparse, re
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
BASE_CHARS = 9_702_393  # эталон ×1 (train wikitext2, 90%-сплит)

def find_corpus(arg_path=None):
    if arg_path:
        return Path(arg_path)
    p = ROOT / "corpus_external/wikitext-103-raw/wiki.train.raw"
    if p.exists():
        return p
    for z in (ROOT / "corpus_external").glob("*.zip"):
        with zipfile.ZipFile(z) as zp:
            names = [n for n in zp.namelist() if "train" in n and n.endswith((".raw", ".txt"))]
            if names:
                zp.extract(names[0], ROOT / "corpus_external")
                return ROOT / "corpus_external" / names[0]
    for t in sorted((ROOT / "corpus_external").rglob("*")):
        if t.is_file() and t.suffix in (".txt", ".raw") and t.stat().st_size > 200_000_000:
            return t
    return None

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="")
ap.add_argument("--mult", type=float, default=30.0, help="масштаб ×N к эталону")
ap.add_argument("--bpe", action="store_true", help="добавить заход K3 (BPE-w2, ~15–20 мин)")
args = ap.parse_args()

src = find_corpus(args.path or None)
if not src or not src.exists():
    print("Корпус не найден. Положите wikitext-103-raw-v1.zip (или большой .txt) в corpus_external/ и повторите.")
    sys.exit(1)
budget = int(BASE_CHARS * args.mult)

probe_txt = (ROOT / "corpus_external/wikitext2/train.txt").read_text(encoding="utf-8")
ho_txt = probe_txt[int(len(probe_txt) * 0.90):]
print(f"корпус: {src} ({src.stat().st_size/1e6:.0f} МБ) | бюджет ×{args.mult:.0f} = {budget/1e6:.0f}М симв  t={time.time()-T0:.0f}s", flush=True)

alpha = set(ho_txt)
with src.open(encoding="utf-8", errors="replace") as fh:
    while chunk := fh.read(8 << 20):
        alpha.update(chunk)
        if fh.tell() > budget:
            break
amap = {c: i for i, c in enumerate(sorted(alpha))}
V = len(amap)
print(f"V={V}  t={time.time()-T0:.0f}s", flush=True)

ho = [amap[c] for c in ho_txt]
LEVELS = (8, 6, 4)
mods = {n: V ** n for n in LEVELS}
targets = {n: set() for n in LEVELS}
codes_ho = {n: 0 for n in LEVELS}
ho_rows = []
for i in range(len(ho) - 1):
    for n in LEVELS:
        codes_ho[n] = (codes_ho[n] * V + ho[i]) % mods[n]
    ho_rows.append(({n: codes_ho[n] for n in LEVELS}, ho[i + 1]))
    if i >= 7:
        for n in LEVELS:
            targets[n].add(codes_ho[n])
ho_rows = ho_rows[7:]  # отбрасываем «недозревшие» (контекст < 8)
print(f"проба {len(ho_rows)} позиций, цели c8={len(targets[8])}  t={time.time()-T0:.0f}s", flush=True)

ctx = {n: collections.defaultdict(int) for n in LEVELS}
pair = {n: collections.defaultdict(int) for n in LEVELS}
codes = {n: 0 for n in LEVELS}
pos = 0
stop = False
with src.open(encoding="utf-8", errors="replace") as fh:
    while not stop:
        chunk = fh.read(2 << 20)
        if not chunk:
            break
        for ch in chunk:
            x = amap[ch]
            pos += 1
            if pos > 1:
                # контекст = codes (после предыдущего символа), продолжение = x
                for n in LEVELS:
                    c = codes[n]
                    if c in targets[n]:
                        ctx[n][c] += 1
                        pair[n][c * V + x] += 1
            for n in LEVELS:
                codes[n] = (codes[n] * V + x) % mods[n]
            if pos >= budget:
                stop = True
                break
print(f"проход по корпусу готов ({pos/1e6:.1f}М симв)  t={time.time()-T0:.0f}s", flush=True)

top = {n: {} for n in LEVELS}
for n in LEVELS:
    for k, c in pair[n].items():
        cd, xx = divmod(k, V)
        cur = top[n].get(cd)
        if cur is None or c > cur[0]:
            top[n][cd] = (c, xx)

out = {"corpus": str(src), "chars": pos, "V": V}
cov = hit = 0
for row, nxt in ho_rows:
    n = ctx[8].get(row[8], 0)
    if n >= 2:
        tc, tx = top[8][row[8]]
        if tc / n >= 0.90:
            cov += 1; hit += (tx == nxt)
d1, a1 = cov / len(ho_rows), hit / max(cov, 1)
v1 = "✓" if 0.43 <= d1 <= 0.55 and a1 >= 0.95 else "✗"
out["K1"] = dict(duty=round(d1, 4), acc=round(a1, 4), verdict=v1)
print(f"K1 char-c8 ×{args.mult:.0f}: duty {d1*100:.1f}% (прогноз 47–52, смерть вне [43,55]) acc {a1:.4f} → {v1}", flush=True)

cov = hit = 0
for row, nxt in ho_rows:
    for n in (8, 6, 4):
        nn = ctx[n].get(row[n], 0)
        if nn >= 5:
            tc, tx = top[n][row[n]]
            if tc / nn >= 0.90:
                cov += 1; hit += (tx == nxt)
                break
d2, a2 = cov / len(ho_rows), hit / max(cov, 1)
v2 = "✓" if d2 >= 0.50 and a2 >= 0.95 else "✗"
out["K2"] = dict(duty=round(d2, 4), acc=round(a2, 4), verdict=v2)
print(f"K2 каскад c8>c6>c4 ×{args.mult:.0f}: duty {d2*100:.1f}% (прогноз 55–60, смерть <50) acc {a2:.4f} → {v2}", flush=True)

if args.bpe:
    print("K3: BPE w2 прицельный замер (кодирование ~15–20 мин)…", flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    from aira.bpe import BPETokenizer
    WORD_RE = re.compile(r" |\n|[^\s]+")
    tok = BPETokenizer.load(ROOT / "experiments/results/bpe_tokenizer_bpe16k_glue.json")
    Vb = len(tok.vocab)
    cache: dict[str, list[int]] = {}
    def enc_words(line_iter):
        out_ids = []
        lead = False
        for m in line_iter:
            w = m.group(0)
            if w == " ":
                if lead: out_ids.append(tok.vocab["␣"])
                lead = True; continue
            if w == "\n":
                if lead: out_ids.append(tok.vocab["␣"])
                lead = False; out_ids.append(tok.vocab["␤"]); continue
            unit = " " + w if lead else w; lead = False
            ids = cache.get(unit)
            if ids is None:
                ids = [tok.vocab.get(p, tok.vocab["�"]) for p in tok._apply(tuple(unit))]
                cache[unit] = ids
            out_ids.extend(ids)
        return out_ids
    ho_b = enc_words(WORD_RE.finditer(ho_txt))
    ho_b = [x for x in ho_b if isinstance(x, int)]
    target_b = set()
    codeb = 0
    ho_b_rows = []
    for i in range(len(ho_b) - 1):
        codeb = (codeb * Vb + ho_b[i]) % (Vb * Vb)
        if i >= 1:
            target_b.add(codeb)
            ho_b_rows.append((codeb, ho_b[i + 1]))
    ctxb = collections.defaultdict(int); pairb = collections.defaultdict(int)
    codeb = 0
    nread = 0
    with src.open(encoding="utf-8", errors="replace") as fh:
        stop = False
        while not stop:
            chunk = fh.read(2 << 20)
            if not chunk: break
            ids = enc_words(WORD_RE.finditer(chunk))
            nread += len(chunk)
            for x in ids:
                if codeb >= 0 and codeb in target_b:
                    ctxb[codeb] += 1
                    pairb[codeb * Vb + x] += 1
                codeb = (codeb * Vb + x) % (Vb * Vb)
            if nread >= budget:
                stop = True
    topb = {}
    for k, c in pairb.items():
        cd, xx = divmod(k, Vb)
        cur = topb.get(cd)
        if cur is None or c > cur[0]:
            topb[cd] = (c, xx)
    cov = hit = 0
    for cd, nxt in ho_b_rows:
        n = ctxb.get(cd, 0)
        if n >= 2:
            tc, tx = topb[cd]
            if tc / n >= 0.90:
                cov += 1; hit += (tx == nxt)
    d3, a3 = cov / len(ho_b_rows), hit / max(cov, 1)
    v3 = "✓" if a3 >= 0.95 else "✗"
    out["K3"] = dict(duty=round(d3, 4), acc=round(a3, 4), tokens=sum(ctxb.values()), verdict=v3)
    print(f"K3 BPE-w2 ×{args.mult:.0f}: duty {d3*100:.1f}% acc {a3:.4f} (прогноз acc>=0.95) → {v3}  t={time.time()-T0:.0f}s", flush=True)

json.dump(out, open(ROOT / "research/KAGGLE_K13.json", "w"), ensure_ascii=False, indent=1)
print(f"json -> research/KAGGLE_K13.json  t={time.time()-T0:.0f}s")
