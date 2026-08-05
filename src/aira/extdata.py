"""Лоадер внешних корпусов (corpus_external/). Первая цель: opus46 (SFT с рассуждениями).

Нормализует диалоги OpenAI-формата в плоские записи {q, a, reasoning, category, difficulty},
выкидывая мёртвые строки (assistant=None/пустые). Детерминированно, без зависимостей.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OPUS46 = ROOT / "corpus_external" / "opus46" / "opus46_final.jsonl"


def _grab(messages: list[dict], role: str) -> str:
    for m in messages:
        if m.get("role") == role:
            return m.get("content") or ""
    return ""


def load_gsm8k(split: str = "train") -> list[dict]:
    """Реальный GSM8K (openai/grade-school-math через GitHub): {q, solution, answer}."""
    out = []
    with (ROOT / "corpus_external" / "gsm8k" / f"{split}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ans = r["answer"]
            final = ans.split("####")[-1].strip().replace(",", "")
            out.append({"q": r["question"], "solution": ans.split("####")[0].strip(),
                        "answer": final})
    return out


_QUAR_PATH = ROOT / "corpus_external" / "opus46" / "quarantine_mismatch.json"


def load_quarantine() -> set[int]:
    """Индексы строк opus46, расходящихся с золотым GSM8K (проверка 2026-08-06)."""
    if not _QUAR_PATH.exists():
        return set()
    return {r["row"] for r in json.loads(_QUAR_PATH.read_text(encoding="utf-8"))}


def load_opus46(path: Path = OPUS46, only_latin: bool = True,
                require_reasoning: bool = True,
                exclude_quarantined: bool = True) -> list[dict]:
    """Читает opus46_final.jsonl → плоские записи. Фильтры:
    - только строки с непустым user/assistant;
    - require_reasoning: отбрасывает строки без поля reasoning;
    - only_latin: отбрасывает вопросы с кириллицей (защита русской char-статистики).
    """
    out, dropped = [], {"dead": 0, "no_reasoning": 0, "cyrillic": 0, "quarantined": 0}
    quarantined = load_quarantine() if exclude_quarantined else set()
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if idx in quarantined:
                dropped["quarantined"] += 1
                continue
            r = json.loads(line)
            q = _grab(r["messages"], "user").strip()
            a = _grab(r["messages"], "assistant").strip()
            rs = next((m.get("reasoning") or "" for m in r["messages"]
                       if m.get("role") == "assistant"), "").strip()
            if not q or not a:
                dropped["dead"] += 1
                continue
            if require_reasoning and not rs:
                dropped["no_reasoning"] += 1
                continue
            if only_latin and re.search(r"[а-яА-Я]", q):
                dropped["cyrillic"] += 1
                continue
            md = r.get("metadata", {})
            out.append({"q": q, "a": a, "reasoning": rs,
                        "category": md.get("category", "?"),
                        "difficulty": md.get("difficulty", "?")})
    print(f"[extdata] opus46: загружено {len(out)} строк, выкинуто {dropped}")
    return out


if __name__ == "__main__":
    rows = load_opus46()
    ex = rows[3]
    print("Пример:", ex["category"], "|", ex["q"][:90], "→", ex["a"][:120])
